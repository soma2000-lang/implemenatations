Explorations into Ring Attention, from Liu et al. at Berkeley AI.

It basically splits the data across the sequence dimension (instead of batch) and applies ring reduce to the processing of the tiles of the attention matrix, flash attention style.

I believe this is being used for the 1-10 million tokens for the latest Gemini. At least some form of it; the other possibility would be unpublished improvements on top of RMT.

In addition, the repository also contains the logic for Striped Attention, a follow up paper that permutes the sequence for better workload balancing for autoregressive transformers.
Todo
 make it work with derived causal mask based on rank and chunk sizes

 modify flash attention to output intermediates and figure out backwards with recompute and ring passes

 functions for splitting the sequence evenly among ranks, either within attention function, or in the external ring transformer wrapper

 basic test case with two processes and check for equivalent output and gradients

 testing

 make sure key padding mask works
 make sure causal mask works
 rotary embeddings, with proper key/value offset depending on ring rank
 striped attention

 add the permutating logic before and after transformer
 add causal masking logic - account for sub bucketing by flash attention