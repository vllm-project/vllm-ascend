# Weight Prefetch Guide

Weight prefetching optimizes memory usage by preloading weights into the cache before they are needed, minimizing delays caused by memory access during model execution. Linear layers sometimes exhibit relatively high MTE utilization. To address this, we create a separate pipeline specifically for weight prefetching, which runs in parallel with the original vector computation pipeline, such as quantize, MoE gating top_k, RMSNorm and SwiGlu. This approach allows the weights to be preloaded to L2 cache ahead of time, reducing MTE utilization during the linear layer computations and indirectly improving Cube computation efficiency by minimizing resource contention and optimizing data flow.

Since we use vector computations to hide the weight prefetching pipeline, it has effect on computation, if you prioritize low latency over high throughput, then it is best not to enable prefetching.

## Quick Start

With `--additional-config '{"weight_prefetch_config": {"enabled": true}}'` to open weight prefetch.

## Fine-tune Prefetch Ratio

Since weight prefetch use vector computations to hide the weight prefetching pipeline, the setting of the prefetch size is crucial. If the size is too small, the optimization benefits will not be fully realized, while a larger size may lead to resource contention, resulting in performance degradation. To accommodate different scenarios, we have added `prefetch_ratio` to allow for flexible size configuration based on the specific workload, details as follows:

With `prefetch_ratio` in `"weight_prefetch_config"` to custom the weight prefetch ratio for specific linear layers.

The “attn” and “moe” configuration options are used for MoE model, details as follows:

`"attn": { "qkv": 1.0,  "o": 1.0},  "moe": {"gate_up": 0.8}`

The “mlp” configuration option is used to optimize the performance of the Dense model, details as follows:

 `"mlp": {"gate_up": 1.0, "down": 1.0}`

Above value are the default config, the default value has a good performance for Qwen3-235B-A22B-W8A8 when `--max-num-seqs` is 144, for Qwen3-32B-W8A8 when `--max-num-seqs` is 72.

However, this may not be the optimal configuration for your scenario. For higher concurrency, you can try increasing the prefetch size. For lower concurrency, prefetching may not offer any advantages, so you can decrease the size or disable prefetching. Determine if the prefetch size is appropriate by collecting profiling data. Specifically, check if the time required for the prefetch operation (e.g., MLP Down Proj weight prefetching) overlaps with the time required for parallel vector computation operators (e.g., SwiGlu computation), and whether the prefetch operation is no later than the completion time of the vector computation operator. In the profiling timeline, a prefetch operation appears as a CMO operation on a single stream; this CMO operation is the prefetch operation.

Notes:

1) Weight prefetch of MLP `down` project prefetch depends on sequence parallel, if you want to open for mlp `down` please also enable sequence parallel.
2) Due to the current size of the L2 cache, the maximum prefetch cannot exceed 18MB. If `prefetch_ratio * linear_layer_weight_size >= 18 * 1024 * 1024` bytes, the backend will only prefetch 18MB.

## Example

1) For MoE model:

```shell
    --additional-config \
    '{
        "weight_prefetch_config": {
            "enabled": true,
            "prefetch_ratio": {
                "attn": {
                    "qkv": 1.0,
                    "o": 1.0
                },
                "moe": {
                    "gate_up": 0.8
                }
            }
        }
    }'
```

2) For dense model:

Following is the default configuration that can get a good performance for `--max-num-seqs` is 72 for Qwen3-32B-W8A8

```shell
    --additional-config \
    '{
        "weight_prefetch_config": {
            "enabled": true,
            "prefetch_ratio": {
                "mlp": {
                    "gate_up": 1.0,
                    "down": 1.0
                }
            }
        }
    }'
```
