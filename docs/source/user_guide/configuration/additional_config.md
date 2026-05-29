# Additional Configuration

Additional configuration is a mechanism provided by vLLM to allow plugins to control internal behavior by themselves. VLLM Ascend uses this mechanism to make the project more flexible.

## Migration Guide

Starting from PR #9064, vLLM Ascend is migrating **10 environment variables** to `--additional-config`.

### Important Notice

- **Current Support**: Both environment variables and `--additional-config` are supported during the transition period
- **Recommendation**: Use `--additional-config` for new deployments and migrate existing configurations
- **Future Plan**: Environment variables will be **removed** in a future release; only `--additional-config` will be supported

### Quick Reference

| Environment Variable | Config Key | Type Conversion |
|---------------------|------------|-----------------|
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `enable_balance_scheduling` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | `enable_flashcomm1` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` | `enable_matmul_allreduce` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` | `enable_flashcomm2_parallel_size` | Integer (unchanged) |
| `MSMONITOR_USE_DAEMON` | `msmonitor_use_daemon` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_MLAPO` | `enable_mlapo` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_NZ` | `weight_nz_mode` | Integer (unchanged, field name changed) |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | `enable_context_parallel` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `enable_fused_mc2` | Integer (unchanged) |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `enable_transpose_kv_cache_by_block` | `"1"` → `true`, `"0"` → `false` |

### Example Migration

**Before (environment variable):**

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
vllm serve Qwen/Qwen3-8B
```

**After (additional-config):**

```bash
vllm serve Qwen/Qwen3-8B --additional-config='{"enable_flashcomm1": true}'
```

## How to use

With either online mode or offline mode, users can use additional configuration. Take Qwen3 as an example:

**Online mode**:

```bash
vllm serve Qwen/Qwen3-8B --additional-config='{"config_key":"config_value"}'
```

**Offline mode**:

```python
from vllm import LLM

LLM(model="Qwen/Qwen3-8B", additional_config={"config_key":"config_value"})
```

### Configuration options

The following table lists additional configuration options available in vLLM Ascend:

| Name                                | Type | Default | Description                                                                                               |
|-------------------------------------|------|---------|-----------------------------------------------------------------------------------------------------------|
| `xlite_graph_config`                | dict | `{}`    | Configuration options for Xlite graph mode                                                                |
| `weight_prefetch_config`            | dict | `{}`    | Configuration options for weight prefetch                                                                 |
| `finegrained_tp_config`             | dict | `{}`    | Configuration options for module tensor parallelism                                                       |
| `ascend_compilation_config`         | dict | `{}`    | Configuration options for ascend compilation                                                              |
| `eplb_config`                       | dict | `{}`    | Configuration options for eplb |
| `sampling_config`                   | dict | `{}`    | Configuration options for model runner v1 sampling operator integration. |
| `refresh`                           | bool | `false` | Whether to refresh global Ascend configuration content. This is usually used by rlhf or ut/e2e test case. |
| `dump_config`                       | dict | `None`  | Inline msprobe dump configuration. vLLM-Ascend will materialize it to a temporary JSON file and pass that file to the debugger. |
| `dump_config_path`                  | str  | `None`  | Configuration file path for msprobe dump (compatible legacy option).                                      |
| `enable_async_exponential`          | bool | `False` | Whether to enable asynchronous exponential overlap. To enable asynchronous exponential, set this config to True.        |
| `enable_shared_expert_dp`           | bool | `False` | When the expert is shared in DP, it delivers better performance but consumes more memory. Currently only DeepSeek series models are supported. |
| `multistream_overlap_shared_expert` | bool | `False` | Whether to enable multi-stream shared expert. This option only takes effect on MoE models with shared experts. |
| `multistream_overlap_gate`          | bool | `False` | Whether to enable multi-stream overlap gate. This option only takes effect on MoE models with shared experts.  |
| `recompute_scheduler_enable`        | bool | `False` | Whether to enable the recompute scheduler. **Only valid in PD-disaggregated mode** (`kv_role` is `kv_producer` or `kv_consumer`). **Do not enable in PD-mixed mode** (no `kv_transfer_config`, or `kv_role` is `kv_both`); startup will fail with a clear error. |
| `enable_cpu_binding`                | bool | `True`  | Enables Ascend-native CPU binding on ARM servers. Set to `False` to disable. See [CPU Binding](../feature_guide/cpu_binding.md). |
| `SLO_limits_for_dynamic_batch`      | int  | `-1`    | SLO limits for dynamic batch. This is new scheduler to support dynamic batch feature                            |
| `enable_npugraph_ex`                | bool | `False` | Whether to enable npugraph_ex graph mode.                                                                 |
| `pa_shape_list`                     | list | `[]`    | The custom shape list of page attention ops.                                                              |
| `enable_kv_nz`                      | bool | `False` | Whether to enable KV cache NZ layout. This option only takes effects on models using MLA (e.g., DeepSeek).                                      |
| `layer_sharding`                    | dict | `{}`    | Configuration options for Layer Sharding Linear. Layer Sharding can only be enabled in PD-disaggregated's P node. |
| `enable_sparse_c8`                  | bool | `False` | Whether to enable KV cache C8 in DSA models (e.g., DeepSeekV3.2 and GLM5). Not supported on A5 devices now |
| `enable_mc2_hierarchy_comm`         | bool | `False` | Enable dispatch/combine op inter-node communication by ROCE. |
| `profiling_chunk_config`            | dict | `{}`    | Configuration options for dynamic chunked pipeline parallel. See [Dynamic Chunked Pipeline Parallel](../feature_guide/dynamic_chunk_pipeline_parallel.md) for details. |
| `enable_balance_scheduling`         | bool | `False` | Whether to enable balance scheduling. Can also be configured via `VLLM_ASCEND_BALANCE_SCHEDULING` environment variable (deprecated). |
| `enable_flashcomm1`                 | bool | `False` | Whether to enable FlashComm1 optimization. Can also be configured via `VLLM_ASCEND_ENABLE_FLASHCOMM1` environment variable (deprecated). |
| `enable_matmul_allreduce`           | bool | `False` | Whether to enable matmul allreduce optimization. Can also be configured via `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` environment variable (deprecated). |
| `flashcomm2_parallel_size`          | int  | `0`     | FlashComm2 parallel size. Can also be configured via `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` environment variable (deprecated). |
| `msmonitor_use_daemon`              | bool | `False` | Whether to use daemon mode for msmonitor. Can also be configured via `MSMONITOR_USE_DAEMON` environment variable (deprecated). |
| `enable_mlapo`                      | bool | `True`  | Whether to enable MLAPO (Model Layer-wise Adaptive Parallel Optimization). Can also be configured via `VLLM_ASCEND_ENABLE_MLAPO` environment variable (deprecated). |
| `weight_nz_mode`                    | int  | `1`     | Weight NZ mode. Can also be configured via `VLLM_ASCEND_ENABLE_NZ` environment variable (deprecated). |
| `enable_context_parallel`           | bool | `False` | Whether to enable context parallelism. Can also be configured via `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` environment variable (deprecated). |
| `enable_fused_mc2`                  | int  | `0`     | Fused MC2 configuration. Can also be configured via `VLLM_ASCEND_ENABLE_FUSED_MC2` environment variable (deprecated). |
| `enable_transpose_kv_cache_by_block`| bool | `True`  | Whether to enable transpose KV cache by block. Can also be configured via `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` environment variable (deprecated). |

The details of each configuration option are as follows:

**xlite_graph_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Whether to enable Xlite graph mode. Currently only Llama, Qwen dense series models, and Qwen3-VL are supported. |
| `full_mode` | bool | `False` | Whether to enable Xlite for both the prefill and decode stages. By default, Xlite is only enabled for the decode stage. |

**weight_prefetch_config**

| Name             | Type | Default                                                     | Description                        |
|------------------|------|-------------------------------------------------------------|------------------------------------|
| `enabled`        | bool | `False`                                                     | Whether to enable weight prefetch. |
| `prefetch_ratio` | dict | `{"attn": {"qkv": 1.0, "o": 1.0}, "moe": {"gate_up": 0.8}, "mlp": { "gate_up": 1.0,  "down": 1.0}}` | Prefetch ratio of each weight.     |

**finegrained_tp_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `lmhead_tensor_parallel_size`    | int  | `0` | The custom tensor parallel size of lm_head.    |
| `oproj_tensor_parallel_size`     | int  | `0` | The custom tensor parallel size of o_proj.     |
| `embedding_tensor_parallel_size` | int  | `0` | The custom tensor parallel size of embedding. |
| `mlp_tensor_parallel_size`       | int  | `0` | The custom tensor parallel size of mlp.       |

**ascend_compilation_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enable_npugraph_ex`               | bool | `True` | Whether to enable npugraph_ex backend.                                                 |
| `enable_static_kernel` | bool | `False` | Whether to enable static kernel. Suitable for scenarios where shape changes are minimal and some time is available for static kernel compilation. |
| `fuse_norm_quant`  | bool | `True` | Whether to enable fuse_norm_quant pass. |
| `fuse_qknorm_rope` | bool | `True` | Whether to enable fuse_qknorm_rope pass. If Triton is not in the environment, set it to False. |
| `fuse_allreduce_rms` | bool | `False` | Whether to enable fuse_allreduce_rms pass. It's set to False because of conflict with SP. |
| `fuse_muls_add` | bool | `True` | Whether to enable fuse_muls_add pass.|

**eplb_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `dynamic_eplb`                   | bool| `False`| Whether to enable dynamic EPLB. |
| `expert_map_path`                | str | `None` | When using expert load balancing for an MoE model, an expert map path needs to be passed in.|
| `expert_heat_collection_interval`| int | `400`  | Forward iterations when EPLB begins. |
| `algorithm_execution_interval`   | int | `30`   | The forward iterations when the EPLB worker will finish CPU tasks. |
| `expert_map_record_path`         | str | `None` | Save the expert load calculation results to a new expert table in the specified directory.|
| `num_redundant_experts`          | int | `0`    | Specify redundant experts during initialization. |

**sampling_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enable_sampling_v2` | bool | `False` | Whether to let `model_runner_v1` use the sampling operators introduced for `model_runner_v2`. This switch only controls the `model_runner_v1` integration path. If vLLM Ascend uses `model_runner_v2` directly in the future, this switch will have no effect on `model_runner_v2`. |
| `enable_reduced_sampling` | bool | `False` | Whether to reduce sampling communication by keeping local vocabulary logits and sampling from gathered top-k candidates. This is intended for top-k sampling workloads and is disabled by default. |

Online mode:

```bash
vllm serve Qwen/Qwen3-8B \
  --additional-config='{"sampling_config":{"enable_sampling_v2":true,"enable_reduced_sampling":true}}'
```

Offline mode:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-8B",
    additional_config={
        "sampling_config": {
            "enable_sampling_v2": True,
            "enable_reduced_sampling": True,
        }
    },
)
```

**profiling_chunk_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled`       | bool  | `False` | Whether to enable dynamic chunked pipeline parallel. Requires `pipeline-parallel-size > 1`. |
| `smooth_factor` | float | `1.0`   | Smoothing factor (0 < x ≤ 1.0). Higher values trust the dynamic prediction more; `0.0` disables dynamic adjustment. |
| `min_chunk`     | int   | `4096`  | Minimum chunk size for dynamic calculation. Should be smaller than `max-num-batched-tokens`. |
| `need_timing` | bool | True | Enable/disable Online Calibration |

### Example

An example of additional configuration is as follows:

```python
{
    "weight_prefetch_config": {
        "enabled": True,
        "prefetch_ratio": {
            "attn": {
                "qkv": 1.0,
                "o": 1.0,
            },
            "moe": {
                "gate_up": 0.8
            },
            "mlp": {
                "gate_up": 1.0,
                "down": 1.0
            }
        },
    },
    "finegrained_tp_config": {
        "lmhead_tensor_parallel_size": 8,
        "oproj_tensor_parallel_size": 8,
        "embedding_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8,
    },
    "enable_kv_nz": False,
    "multistream_overlap_shared_expert": True,
    "refresh": False
}
```
