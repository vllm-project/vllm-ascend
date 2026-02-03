# Weight Prefetch Guide

Weight prefetching optimizes memory usage by preloading weights into the cache before they are needed, minimizing delays caused by memory access during model execution. Linear layers sometimes exhibit relatively high MTE utilization. To address this, we create a separate pipeline specifically for weight prefetching, which runs in parallel with the original vector computation pipeline, such as quantize, MoE gating top_k, RMSNorm and SiLU. This approach allows the weights to be preloaded to L2 cache ahead of time, reducing MTE utilization during the linear layer computations and indirectly improving Cube computation efficiency by minimizing resource contention and optimizing data flow.

Since we use vector computations to hide the weight prefetching pipeline, it has effect on computation, if you prioritize low latency over high throughput, then it it best not to enable prefetching.

## How to Use

With `--additional-config '{"weight_prefetch_config": {"enabled": true}}'` to open weight prefetch.
With `prefetch_ratio` in `"weight_prefetch_config"` to custom the weight prefetch ratio for specify linear layers.
The “attn” and “moe” configuration options are used for MoE model, detail as following:
`"attn": { "qkv": 1.0,  "o": 1.0},  "moe": {"gate_up": 0.8}`
The “mlp” configuration option is used to optimize the performance of the Dense model, detail as following:
 `"mlp": {"gate_up": 1.0, "down": 1.0}`

Notices:

1) Due to the current size of the L2 cache, the maximum prefetch cannot exceed 18MB. If `prefetch_ration * lineaer_layer_weight_size >= 18 * 1024 * 1024` bytes, the backend will only prefetch 18MB.
2) Weight prefetch of MLP `down` project prefetch dependence sequence parallel, if you want open for mlp `down` please also enable sequence parallel.

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
