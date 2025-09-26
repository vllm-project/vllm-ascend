# Introduction to Omni Attention

## What is Omni Attention?
Omni Attention is a key-value (KV) sparsification method developed by the omni-infer team. Unlike traditional approaches that focus on head-wise pruning, Omni Attention performs layer-wise KV compression. This key difference makes it particularly well-suited for tensor parallelism and MLA models.

At its core, Omni Attention employs a genetic algorithm to search for the optimal arrangement of full and compressed layers—referred to as the *pattern* . For compressible layers, only the first $M$ and last $N$ tokens in the KV cache are retained per request. This strategy not only accelerates attention computation but also allows for larger batch sizes by reducing memory usage.

Currently, our implementation supports integration with other features such as PD disaggregation, MTP, and EP placement. It supports DeepSeek-V3 and DeepSeek-R1 models, while more models will be included in the future.

## How to use?
To enable Omni Attention when starting a VLLM serving instance, simply add the following configuration:
```bash
vllm serve /path/to/model ... --additional-config '{"enable_omni_attn": true}'
```
