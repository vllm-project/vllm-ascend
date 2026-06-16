# API Parameter Support

vLLM Ascend is fully compatible with the upstream vLLM API. The vast majority of `vllm serve` CLI parameters work on Ascend NPU out of the box, with no additional configuration required. For general usage instructions, please refer to the vLLM community documentation:

- [vLLM Serving Documentation](https://docs.vllm.ai/en/latest/serving/online_serving.html)
- [vLLM CLI Reference](https://docs.vllm.ai/en/latest/cli/.html)
- [vLLM API Reference](https://docs.vllm.ai/en/latest/api/index.html)

This page **only** documents the parameters that are **not supported**, **silently reset**, or **partially limited** on Ascend NPU due to hardware differences or missing backend support. For any parameters not listed below, they are supported by default — please refer to the vLLM documentation above for usage instructions.

Some GPU-specific parameters have Ascend-native equivalents exposed via `--additional-config`. See [Additional Configuration](additional_config.md) for the full list of Ascend-specific options.

## Parameters Silently Reset by Ascend

When any of the following parameters are set, vLLM Ascend will log a warning and automatically reset them to a safe default. The service will still start normally. The reset logic is implemented in [`vllm_ascend/platform.py` → `_fix_incompatible_config`](https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/platform.py).

### ModelConfig

| Parameter | Default Reset | Reason |
|-----------|--------------|--------|
| `--disable-cascade-attn` | `False` | Cascade Attention is a GPU-specific feature not available on Ascend NPU. |

### CacheConfig

| Parameter | Default Reset | Reason |
|-----------|--------------|--------|
| `--cpu-kvcache-space` | `None` | CPU KV cache space configuration is tied to an incompatible memory backend. |

### MultiModalConfig

| Parameter | Default Reset | Reason |
|-----------|--------------|--------|
| `--mm-encoder-attn-backend` | `None` | Ascend uses a different mechanism for multi-modal encoder attention; this parameter is ignored. |

### ObservabilityConfig

| Parameter | Default Reset | Reason |
|-----------|--------------|--------|
| `--enable-layerwise-nvtx-tracing` | `False` | NVTX tracing is an NVIDIA-specific profiling tool not available on Ascend. |

### SchedulerConfig

| Parameter | Default Reset | Reason |
|-----------|--------------|--------|
| `--max-num-partial-prefills` | `1` | Partial prefill optimization targets ROCm and is not applicable on Ascend. |

### ParallelConfig

| Parameter | Default Reset | Reason |
|-----------|--------------|--------|
| `--ray-workers-use-nsight` | `False` | NVIDIA Nsight profiler is not available on Ascend NPU. |
| `--numa-bind` | `False` (converted) | GPU-to-NUMA topology detection is unavailable. Automatically converted to Ascend-native CPU binding, see [`enable_cpu_binding`](additional_config.md#configuration-options). |
| `--numa-bind-nodes` | `None` | Ignored on Ascend; Ascend-native CPU binding performs automatic topo-affinity allocation internally. |
| `--numa-bind-cpus` | `None` | Same as above. |
| `--enable-dbo` | `False` | Dynamic Batch Optimization is not yet supported on Ascend. |
| `--ubatch-size` | `0` | Micro-batch size tuning is not yet supported on Ascend. |

### AttentionConfig (`--attention-config`)

| Subfield | Default Reset | Reason |
|----------|--------------|--------|
| `use_prefill_decode_attention` | `False` | GPU-specific attention path. |
| `use_cudnn_prefill` | `False` | cuDNN is NVIDIA-only. |
| `use_trtllm_ragged_deepseek_prefill` | `False` | TensorRT-LLM feature, NVIDIA-only. |
| `use_trtllm_attention` | `False` | TensorRT-LLM feature, NVIDIA-only. |
| `disable_flashinfer_prefill` | `False` | FlashInfer is NVIDIA-specific; flag has no effect on Ascend. |
| `disable_flashinfer_q_quantization` | `False` | Same as above. |
| `flash_attn_version` | `None` | Ascend uses its own attention backend; FlashAttention version selection is ignored. |
| `flash_attn_max_num_splits_for_cuda_graph` | `32` | CUDA Graph split-point tuning is not applicable on Ascend. |

### KVTransferConfig (`--kv-transfer-config`)

| Subfield | Default Reset | Reason |
|----------|--------------|--------|
| `kv_buffer_size` | `1e9` | Buffer size tuning is optimized for NCCL/GPU backends. |
| `enable_permute_local_kv` | `False` | Tied to an incompatible KV transfer backend (NIXL). |

### SpeculativeConfig (`--speculative-config`)

| Subfield | Default Reset | Reason |
|----------|--------------|--------|
| `quantization` | `None` | Ascend automatically inherits the main model's quantization method; separate speculative quantization is ignored. |

---

## Summary

| Category | Count |
|----------|-------|
| Parameters silently reset (GPU/NVIDIA-specific) | 22 |

For parameters that are silently reset, the service starts normally — users may observe warnings in the logs.

For any other parameters not specified in this document, they are supported by default on Ascend NPU — please refer directly to the [vLLM community documentation](https://docs.vllm.ai/en/latest/) for usage instructions.

If you encounter a parameter not listed here that fails on Ascend, please open an issue at [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend/issues).
