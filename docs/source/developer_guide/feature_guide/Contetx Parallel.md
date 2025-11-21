# Context Parallel

## Why We Need Context Parallel
Context parallel mainly solves the problem of serving long context requests. As prefill and decode present quite different characteristics and have quite different SLO (service level objectives), we need to implement context parallel separately for them. The major considerations are:

- For long context prefill, we need to control the TTFT (time to first token) by amortizing the computation time of the prefill across query tokens.
- For long context decode, we need more space for KV cache to increase the batchsize (and hence the throughput).

## How to use Context Parallel?

### Prefill Context Parallel

--prefill_context_parallel-size：The size of Prefill Context Parallel



### Decode Context Parallel
--decode_context_parallel_size: The size of Prefill Context Parallel
--cp-kv-cache-interleave-size:  The Kvcache store interval size


## How it works

### Prefill Context Parallel

During prefill, for a long request with T new tokens, we need to compute query/key/value tensors for these new tokens. Say we have N GPUs, we can split the request into N chunks, and each GPU computes one chunk of the query/key/value tensors. Concretely, Prefill Context Parallel primarily targets the Self-attention module for parallel computing along the sequence dimension. By splitting long sequences into segments across the context dimension and distributing these segments to different devices for parallel processing, Prefill Context Parallel reduces the first-token latency. The key implementation steps of PCP include:

- Each device computes its own attention, and the devices transfer KV (Key-Value) values in a ring manner to obtain the results of block-wise computation. The overall principle is similar to ring-attention.
- Block-wise computation is performed using the Flash-Attention 2 algorithm, and finally, the block results are corrected.

Prefill Context parallel(PCP) is a system optimization technique that improves the latency and scalability of inference, particularly for long contexts. Without modifying the underlying dense attention algorithms, CP offers several advantages for long-context LLM inference:

- **Compute parallelization**: CP distributes computation across multiple GPUs in order to reduce latency, in contrast with pipeline parallelization (PP) that improves throughput but not latency.
- **KV cache distribution**: Key and value (KV) embeddings grow linearly with context length. CP distributes
the storage of KV embeddings across multiple GPUs, enabling larger batch sizes with the addition of more CP ranks

### Decode Context Parallel

Due to the auto-regressive nature of decoding, every decoding step needs to compute a small amount of query tokens w.r.t. a large number of key/value tokens stored in the paged KV cache. The core of decode context parallel is how to shard the KV cache across GPUs.

For a model with H kv-heads, a request with T tokens in the context needs to store H * T key/value tensors in the KV cache.

- If one GPU can hold them all, and the performance is good enough, then no parallelization is needed.
- If one GPU cannot hold them all, or we want to hold more requests in the KV cache, we can first shard the KV cache along the H dimension, that's the plain tensor parallel sharding. It's as simple as adding -tp <num_gpus> to the command line.
- Since H is limited (determined by the model architecture), when we continue to increase the tensor parallel size, the KV cache for each GPU will be duplicated for tp_size / H times. Of course, duplication is not good for efficiency. Then we need to add decode context parallel to further shard the KV cache along the T dimension. This is as simple as adding -dcp <size> to the command line. Note that size does not increase the number of GPUs we need to launch, but just reduces the KV cache duplication. The dcp size should lie in the range of [1, tp_size/H]. With larger dcp size, the KV cache duplication is reduced, but the communication overhead increases.

Theoretically, it is possible to extend the dcp size beyond tp_size / H to further shard the KV cache and accelerate the decoding phase. However, since the number of query tokens is limited in decoding, it's unclear what should we do for the remaining dcp_size - tp_size / H GPUs for non-attention layers. For the sake of simplicity, dcp size is upper bounded by tp_size / H. If you want to further accelerate the decoding phase, you can consider increasing the tp_size first, and then increasing the dcp size.

Note that kv cache can grow during decoding, and the sharding strategy needs to be carefully implemented. We use an interleaving strategy to shard the KV cache along the T dimension, so that kv cache for future tokens can be naturally sharded along the T dimension.This is explained in details in https://arxiv.org/abs/2507.07120.

# DFX

## Integer Validation

Prefill Context Parallel Size must equal to ranks // TP_Size
Decode Context Parallel Size  lie in the range of [1, tp_size/H]

Say we have 16 NPUs, TP_size is 8 ,Then Prefill Contetx Parallel Size is 2.for Deepseek ,the kv_head is 1 ,So Decode Context Parallel size can be 8 or 4 or 2 or 1. for Qwen235B, the kv_head is 4, Decode Context Parallel Size is 8 // 4 =2.

# Limitation

Prefill Contetx Parallel does support cross-device

