from typing import Optional, Tuple, Union

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import Module, ModuleList

import einx
from einx import rearrange


from beartype import beartype

from ring_attention_pytorch.ring import (
    all_ring_pass,
    is_distributed,
    get_rank,
    get_world_size
)
from ring_attention_pytorch.ring_flash_attention import (
    ring_flash_attn
)

from ring_attention_pytorch.distributed import (
    split_by_rank,
    AllGather
)
def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0
def default_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    causal: bool = False
):
    q = q * (q.shape[-1] ** -0.5)

    mask_value = -torch.finfo(q.dtype).max

    # similarity

    sim = einsum('b i h d, b j h d -> b h i j', q, k)

    # masking

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
        sim = torch.where(causal_mask, mask_value, sim)

    elif exists(mask):
        sim = einx.where('b j, b h i j, -> b h i j', mask, sim, mask_value)

    # attend

    attn = einx.softmax('b h i [j]', sim)

    # aggregate

    out = einsum('b h i j, b j h d -> b i h d', attn, v)

    return out

# rotary embeddings with modifications to support striped attention
class RingRotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        ring: bool = False,
        striped: bool = False,
        buckets: int = 1,        # in striped attention with flash buckets > 1, one needs to specify the number of buckets per machine
        theta = 10000
    ):
        super().__init__()
        self.ring = ring
        self.striped = striped
        self.buckets = buckets

        inv_freq = theta ** -(torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(enabled = False)
    def forward(
        self,
        seq_len: int,
        offset = 0
    ):
        device = self.device
        pos = None

        if self.ring:
            if self.striped:
                buckets = self.buckets
                ring_stride = get_world_size() * buckets
                ring_offset = buckets

                pos = torch.arange(seq_len // buckets, device = device)
                pos = rearrange('n -> n b', pos, b = buckets)

                pos = pos * ring_stride
                pos += torch.arange(buckets, device = device) + (get_rank() * buckets)
                pos = rearrange('n b -> (b n)', pos)

            else:
                pos = torch.arange(seq_len, device = device)
                pos += seq_len * get_rank()
        else:
            pos = torch.arange(seq_len, device = device)

        pos = pos.type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', pos, self.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim = -1)
        return torch.cat((-x2, x1), dim=-1)

    @autocast(enabled = False)
    def apply_rotary_pos_emb(pos, t):
        pos = rearrange('n d -> n 1 d', pos)
        return t * pos.cos() + rotate_half(t) * pos.sin()

# batch to sequence sharding and back

    def pad_to_multiple(
        x: Tensor,
        length: int,
        pad_value = 0
    ):
        seq_len = x.shape[-1]
        remainder = seq_len % length

        if remainder == 0:
            return x, 0

        pad_length = length - remainder
        return F.pad(x, (0, pad_length), value = pad_value), pad_length
    def maybe_pad_seq_and_mask(
        x: Tensor,
        mask: Optional[Tensor],
        seq_size: int
    ):
        orig_x, seq_len = x, x.shape[-1]

        # auto pad sequence and mask, as ring passing makes assumption tensor is all same shape

        x, pad_length = pad_to_multiple(x, seq_size)

        if pad_length == 0:
            return x, mask

        if not exists(mask):
            mask = torch.ones_like(orig_x).bool()

        mask, _ = pad_to_multiple(mask, seq_size, pad_value = False)

        return x, mask
    def sharded_batch_to_sharded_seq(
        x: Tensor,
        mask: Optional[Tensor],
        seq_size: int
    ):
        assert is_distributed()

        # all gather across batch

        all_gather = AllGather(dim = 0)

        x, sizes = all_gather(x)

        if exists(mask):
            mask, _ = all_gather(mask)

        # first make sure world size is divisible by the sequence size

        world_size = get_world_size()
        total_split_seq = x.shape[-1] // seq_size

        assert divisible_by(world_size, total_split_seq)

        num_sharded_batches = world_size // total_split_seq

        x = rearrange('(b s) n -> b (s n)', x, s = num_sharded_batches)

        # then split sequence across machines

        x = x.split(seq_size, dim = -1)

        x, _ = split_by_rank(x)

        if exists(mask):
            mask = rearrange('(b s) n -> b (s n)', mask, s = num_sharded_batches)
            mask = mask.split(seq_size, dim = -1)
            mask, _ = split_by_rank(mask)

        return (x, mask), sizes, num_sharded_batches
    def sharded_seq_to_sharded_batch(
        logits: Tensor,
        sizes,
        num_sharded_batches = 1
    ):
        all_gather = AllGather(dim = -2) # all gather across sequence

        logits, _ = all_gather(logits)

        logits = rearrange('b (s n) c -> (b s) n c', logits, s = num_sharded_batches)

        logits = logits.split(sizes.tolist(), dim = 0)

        logits, _ = split_by_rank(logits)

        return logits
class RingAttention(Module):
    @beartype
    def __init__(
        self,
        dim: int,
        *,
        dim_head: int = 64,
        heads: int = 8,
        causal: bool = False,
        eps: float = 1e-10,
        bucket_size: int = 512,
        ring_attn: bool = False,
        ring_seq_size: int = 512,
        max_lookback_seq_len: Optional[int] = None,
        striped_ring_attn: bool = False,
        auto_shard_seq: Optional[bool] = None,
        prenorm: bool = True,
        force_regular_attn: bool = False,
        rotary_embed: bool = False,
        rotary_embed_theta: int = 10000,
        use_cuda_kernel: bool = None
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        assert divisible_by(ring_seq_size, bucket_size)

        self.ring_attn = ring_attn
        self.max_lookback_seq_len = max_lookback_seq_len
        self.striped_ring_attn = striped_ring_attn

        self.force_regular_attn = force_regular_attn
        self.auto_shard_seq = default(auto_shard_seq, ring_attn) # this should be done at the transformer level on the token ids for efficiency, but for testing purposes
