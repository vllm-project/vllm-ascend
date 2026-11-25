# Qwen3.5 Decode Weight Prefetch

## Overview

This experimental feature overlaps bounded weight prefetches with decode work for Qwen3.5-35B-A3B. It is disabled by default and should only be enabled after workload-specific correctness and performance testing.

The supported targets are:

- `next_input`: prefetch the next layer's fused GDN input or full-attention QKV weight while tensor-parallel all-reduce is queued.
- `gdn_out`: prefetch the current GDN layer's output projection during its attention work.
- `routed_w2`: warm a budgeted portion of active experts' `w2` weights before the down projection. This target requires the optional Ascend C extension described below.

## Compatibility

The current implementation is limited to Ascend A2/910B, Qwen3.5-35B-A3B's 40-layer text model, BF16, TP=2, DP=1, PP=1, with expert parallelism, quantization, LoRA, and speculative decoding disabled. Unsupported model or topology combinations leave the regular model path in use. Sleep mode, weight transfer, and weight reload are not supported while the feature is enabled because graph capture is tied to the loaded weight storage.

Configuration is read at model startup. Restart the service and use a fresh compilation cache when changing it.

## Configuration

Pass `qwen35_weight_prefetch` through `--additional-config`. For example:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 \
  --additional-config '{"qwen35_weight_prefetch":{"enabled":true,"targets":["next_input"],"next_input_bytes":2097152,"min_decode_concurrency":8}}'
```

The default is equivalent to `{"enabled": false}`. The supported fields are:

| Field | Default | Description |
|---|---:|---|
| `enabled` | `false` | Master switch. |
| `mode` | `"full"` | `"full"` issues prefetch transfers; `"control"` keeps stream/wait plumbing without data movement. |
| `targets` | `["next_input"]` | Any non-empty, duplicate-free combination of `next_input`, `gdn_out`, and `routed_w2`. |
| `next_input_bytes` | `2097152` | Bytes to prefetch for `next_input`, or `"full"` for the entire weight. |
| `gdn_out_bytes` | `2097152` | Byte budget for `gdn_out`. |
| `routed_w2_bytes` | `4194304` | Byte budget for `routed_w2`. |
| `layers` | `null` | Optional list of zero-based layer indices; unset selects all eligible layers. |
| `next_layer_types` | `["linear_attention", "full_attention"]` | Eligible next-layer attention types. |
| `routed_start` | `"after_dispatch"` | Start `routed_w2` prefetch after dispatch or after the first grouped matmul. |
| `vector_cores` | `2` | Ascend C vector cores used by the routed target; supports 1, 2, or 4. |
| `next_input_join` | `"immediate"` | Join after the dependency collective, or defer the join to the next linear-attention layer. |
| `min_decode_concurrency` | `8` | Skip prefetch below this pure-decode request count. |

Byte budgets must be positive and 32-byte aligned. `next_input_bytes` must not exceed the selected weight's storage; the same bound applies to the routed and GDN budgets.

Deferred `next_input_join` is applied only to linear-attention targets; full-attention targets use the immediate join.

## Building the routed extension

`next_input` and `gdn_out` use `torch_npu.npu_prefetch` directly. To select `routed_w2`, build and install its optional extension on the target Ascend A2 host before starting vLLM:

```bash
VLLM_ASCEND_ROOT=/path/to/vllm-ascend \
ASCEND_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh \
bash scripts/build_qwen35_routed_prefetch.sh
```

Stop inference services before running the script; it checks for running vLLM processes and installs the resulting extension under `vllm_ascend/ops/`.

## Validation guidance

The feature is opt-in and experimental. Compare prefetch-off and prefetch-on runs with the same model, prompt set, output length, graph mode, parallel settings, and cache state. Check output consistency, target-operator and consumer-wait timing, throughput, median and tail latency, and whether the dependency window regresses. Aggregate benchmark summaries alone do not establish that prefetch caused a change. Keep the feature disabled if the result is neutral, unstable, or slower.
