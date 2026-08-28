# Additional Configuration

Additional configuration is a mechanism provided by vLLM to allow plugins to control internal behavior by themselves. VLLM Ascend uses this mechanism to make the project more flexible.

## Migration Guide

Starting from [PR #9064](https://github.com/vllm-project/vllm-ascend/pull/9064), vLLM Ascend is migrating supported environment variables to `--additional-config`.

### Important Notice

- **Current Support**: Both environment variables and `--additional-config` are supported during the transition period
- **Recommendation**: Use `--additional-config` for new deployments and migrate existing configurations
- **Future Plan**: Environment variables will be **removed** in a future release; only `--additional-config` will be supported

### Quick Reference

| Environment Variable | Config Key | Type Conversion |
|---------------------|------------|-----------------|
| `VLLM_ASCEND_BALANCE_SCHEDULING` | `scheduler_config.enable_balance_scheduling` | `"1"` → `true`, `"0"` → `false` |
| `MSMONITOR_USE_DAEMON` | `msmonitor_use_daemon` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_MLAPO` | `enable_mlapo` | `"1"` → `true`, `"0"` → `false` |
| `VLLM_ASCEND_ENABLE_NZ` | `weight_nz_mode` | Integer (unchanged, field name changed) |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | `enable_fused_mc2` | Integer (unchanged) |
| `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` | `enable_transpose_kv_cache_by_block` | `"1"` → `true`, `"0"` → `false` |

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
| `finegrained_tp_config`             | dict | `{}`    | Configuration options for module tensor parallelism                                                       |
| `ascend_compilation_config`         | dict | `{}`    | Configuration options for ascend compilation                                                              |
| `eplb_config`                       | dict | `{}`    | Runner-specific EPLB extensions. See [Expert Parallelism Load Balancer](../feature_guide/expert_parallelism_load_balancer.md). |
| `scheduler_config`                  | dict | `{}`    | Configuration options for Ascend scheduler extensions, including balance scheduling, recompute scheduling, DyntraLB, ShortRequestFirst, and dynamic chunked pipeline parallel. |
| `refresh`                           | bool | `false` | Whether to refresh global Ascend configuration content. This is usually used by rlhf or ut/e2e test case. |
| `dump_config`                       | dict | `None`  | Inline msprobe dump configuration. vLLM-Ascend will materialize it to a temporary JSON file and pass that file to the debugger. |
| `dump_config_path`                  | str  | `None`  | Configuration file path for msprobe dump (compatible legacy option).                                      |
| `enable_shared_expert_dp`           | bool | `False` | Replicate shared-expert weights across TP ranks and run the shared expert with data parallelism. This option is independent of upstream MoE sequence parallelism; either feature or both can be enabled. It improves performance but consumes more memory. |
| `multistream_overlap_shared_expert` | bool | `False` | Whether to enable multi-stream shared expert. This option only takes effect on MoE models with shared experts. |
| `enable_cpu_binding`                | bool | `True`  | Enables Ascend-native CPU binding on ARM servers. Set to `False` to disable. See [CPU Binding](../feature_guide/cpu_binding.md). |
| `pa_shape_list`                     | list | `[]`    | The custom shape list of page attention ops.                                                              |
| `enable_kv_nz`                      | bool | `False` | Whether to enable KV cache NZ layout. This option only takes effects on models using MLA (e.g., DeepSeek).                                      |
| `enable_sparse_sfa_c8`              | bool | `False` | Whether to enable the packed C8 KV cache for Sparse Flash Attention in DSA models (e.g., DeepSeek V3.2 and GLM5). This option is independent of `enable_sparse_li_c8`. SFA prefill context parallelism and Ascend 950 DCP are not supported. |
| `enable_sparse_li_c8`               | bool | `False` | Whether to enable the C8 key and scale caches for LightningIndexer in DSA models. This option is independent of `enable_sparse_sfa_c8` and only applies to eligible indexer layers from the model quantization config. SFA prefill context parallelism and Ascend 950 DCP are not supported. |
| `c8_enable_reshape_optim`           | bool | `False` | Whether to use the StoreKVBlock operator to accelerate LightningIndexer C8 cache writes. `enable_sparse_li_c8` must also be enabled. In the PD separation scenario, only the P node is enabled. |
| `mc2_comm_alg`                      | str  | `""`    | set dispatch/combine op's `comm_alg` param, only supports `""/"fullmesh"/"hierarchy"/"fullmesh_v2"`. `"hierarchy"` is only supported by A2/A3, and `"fullmesh_v2"` is only supported by A3 now. |
| `enable_mc2_hierarchy_comm`         | bool | `False` | Enable dispatch/combine op inter-node communication by ROCE. This param will be deprecated and be replaced by mc2_comm_alg = "hierarchy" |
| `enable_prefill_mc2`                | bool | `False` | Whether to reserve mc2_token_capacity for prefill batches. When enabled, `max_num_batched_tokens` is used to calculate the mc2_token_capacity instead of the decode-only capacity. In this scenario, the recommended maximum value of `max_num_batched_tokens` is `tp_size * 512`. This is a temporary switch; once MC2 operators are complete for all scenarios, this switch will be removed and MC2 will be enabled by default. |
| `mega_moe_max_tokens`               | int  | `65536` | Per-rank token capacity after dispatch in the mega moe (dispatch_ffn_combine) fused operator. When load imbalance causes a rank to receive more tokens than this limit, the excess tokens are dropped and skipped from computation, degrading accuracy. Do not set this too large: workspace memory scales linearly with this value. |
| `msmonitor_use_daemon`              | bool | `False` | Whether to use daemon mode for msmonitor. The legacy `MSMONITOR_USE_DAEMON` environment variable is no longer supported. |
| `enable_mlapo`                      | bool | `True`  | Whether to enable MLAPO (Model Layer-wise Adaptive Parallel Optimization). Can also be configured via the `VLLM_ASCEND_ENABLE_MLAPO` environment variable during the migration period. |
| `mlapo_keep_prefill_weights`        | bool | `False` | When True, keep MLAPO prefill weights on NPU instead of freeing them on kv_consumer (decode-only D) nodes. D nodes have normal local-prefill paths (recompute / fallback / preempt) that crash when the weights are freed (issue #11882). Enable this to trade NPU memory for stability. |
| `weight_nz_mode`                    | int  | `1`     | Weight NZ mode. Can also be configured via the `VLLM_ASCEND_ENABLE_NZ` environment variable during the migration period. |
| `enable_fused_mc2`                  | int  | `0`     | Fused MC2 configuration. `0` disables the fused path and `1` enables it when the model and parallel configuration support it. The legacy `VLLM_ASCEND_ENABLE_FUSED_MC2` environment variable is no longer supported. |
| `enable_transpose_kv_cache_by_block`| bool | `True`  | Whether to enable transpose KV cache by block. The legacy `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` environment variable is no longer supported. |
| `enable_dsa_cp`                     | bool | `False` | Whether to enable dsa_cp for DeepSeek V3.2, DeepSeek V4, and other models with the same architecture. This feature requires sequence parallelism to be enabled.|
| `rejection_sampler_config`          | dict | `{}`    | Configuration options for rejection sampler (block verify and entropy verify). |
| `dynamic_spec_config`               | dict | `{}`    | Configuration options for Dynamic Speculative Decoding. See [Dynamic Speculative Decoding](../feature_guide/speculative_decoding.md#dynamic-speculative-decoding). |
| `multistream_dsv4_dsa_overlap`      | bool | `True`  | Whether to enable dsa multi-stream overlap for DeepSeek V4.  |
| `rl_config`                        | dict | `{}`    | One-click RL mode configuration. See <a href="#rl_config">rl_config</a> for all fields, the two deployment modes, usage examples, and the migration guide. |
| `enable_reduce_sample`              | bool | `False` | Whether to enable reduce sample optimization to reduce communication and computation overheads in the tensor parallelism scenario. When enabled, logits are kept partitioned across TP ranks and only the small set of top-k candidate values/indices is communicated, instead of performing a full-vocabulary all-to-all/all-gather. **Note**: This is an experimental feature. **Limitations**: (1) Not supported on PD-disaggregated scenario. (2) Must be disabled when sampling logprobs are requested. When reduce sample is enabled, logprobs are silently computed over partitioned logits instead of the full vocabulary, producing incorrect logprob values and top-k rankings. (3) Cannot be enabled together with lmhead TP.|

The details of each configuration option are as follows:

**xlite_graph_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Whether to enable Xlite graph mode. See [Using XliteGraph](../feature_guide/graph_mode.md#using-xlitegraph) for the supported models, the decode-only vs. full-mode distinction, and examples. |
| `full_mode` | bool | `False` | Whether to enable Xlite for both the prefill and decode stages. By default, Xlite is only enabled for the decode stage, with prefill falling back to the runnable under ACLGraph. When `True`, xlite owns prefill and decode, ACLGraph capture is not used, and `--enforce-eager` is recommended (unless speculative decoding is configured, etc.). |

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
| `fuse_muls_add` | bool | `True` | Whether to enable fuse_muls_add pass.|

**eplb_config**

The accepted fields depend on the model runner:

- **Model Runner V2** accepts only `load_collection_phase` here. Configure
  upstream EPLB through `--enable-eplb` and `--eplb-config`, and set
  `--eplb-config.use_async false` on Ascend.
- **Model Runner V1** accepts the legacy fields below except
  `load_collection_phase`.
  MRv1 does not accept upstream `--enable-eplb` on Ascend.

Mixing the two schemas fails during startup instead of silently ignoring
configuration.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `dynamic_eplb`                   | bool| `False`| MRv1 only. Whether to enable legacy dynamic EPLB. |
| `expert_map_path`                | str | `None` | MRv1 only. Load a recorded static expert map. |
| `expert_heat_collection_interval`| int | `600`  | MRv1 only. Number of forward iterations used to collect expert heat. |
| `algorithm_execution_interval`   | int | `50`   | MRv1 only. Interval allowed for the EPLB worker to finish its CPU task. |
| `expert_map_record_path`         | str | `None` | MRv1 only. Save the calculated expert map to the specified JSON path. |
| `num_redundant_experts`          | int | `0`    | MRv1 only in this table. Configure the MRv2 value through upstream `--eplb-config`. |
| `eplb_policy_type`               | int | `2`    | MRv1 only. EPLB policy: `0`=Random, `1`=DefaultEplb, `2`=SwiftBalanceEplb, `3`=FlashLB. |
| `eplb_heat_collection_stage`     | str | `"all"`| MRv1 only. Select `"all"`, `"prefill"`, or `"decode"` heat collection. |
| `load_collection_phase`          | str | `"all"`| MRv2 only. Select `"all"`, `"prefill"`, or `"decode"` load submission. Any batch containing a prefill request is classified entirely as prefill. |

**scheduler_config**

The legacy top-level `enable_balance_scheduling`, `recompute_scheduler_enable`, `short_request_first_config`, and `profiling_chunk_config` keys remain supported during the migration period, but are deprecated. If both formats provide the same field, the value in `scheduler_config` takes precedence.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enable_balance_scheduling` | bool | `False` | Whether to enable balance scheduling. The legacy `VLLM_ASCEND_BALANCE_SCHEDULING` environment variable is no longer supported. |
| `recompute_scheduler_enable` | bool | `False` | Whether to enable the recompute scheduler. **Only valid on PD-disaggregated D nodes** (`kv_role` is `kv_consumer`). **Do not enable on P nodes or in PD-mixed mode** (no `kv_transfer_config`, `kv_role` is `kv_producer`, or `kv_role` is `kv_both`); startup will fail with a clear error. |
| `profiling_chunk_config` | dict | `{}` | Configuration options for dynamic chunked pipeline parallel. See [Dynamic Chunked Pipeline Parallel](../feature_guide/dynamic_chunk_pipeline_parallel.md) for details. |
| `short_request_first_config` | dict | `{}` | Configuration options for ShortRequestFirst prefill scheduling on FCFS synchronous or asynchronous, PD-prefill (P), or PD-mixed nodes. |
| `batch_job_sched_config` | dict | `{}` | Configuration options for the batch-job-aware scheduler. See [Batch-Job-Aware Scheduler](../feature_guide/batch_job_aware_scheduler.md) for details. |
| `dyntra_lb_config` | dict | `{}` | Configuration options for DyntraLB load balancing on PD-disaggregated decode nodes. |

**scheduler_config.profiling_chunk_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled`       | bool  | `False` | Whether to enable dynamic chunked pipeline parallel. Requires `pipeline-parallel-size > 1`. |
| `smooth_factor` | float | `1.0`   | Smoothing factor (0 < x ≤ 1.0). Higher values trust the dynamic prediction more; `0.0` disables dynamic adjustment. |
| `min_chunk`     | int   | `4096`  | Minimum chunk size for dynamic calculation. Should be smaller than `max-num-batched-tokens`. |
| `need_timing` | bool | True | Enable/disable Online Calibration |
| `max_fit_chunk` | int | 30 | Number of chunk-time data for Online Calibration |

**scheduler_config.dyntra_lb_config**

DyntraLB balances decode requests across data-parallel ranks. It is supported only on
PD-disaggregated decode nodes (`kv_role="kv_consumer"`) with `data_parallel_size > 1`.
`dyntra_lb_config.enabled` and `recompute_scheduler_enable` are independent sibling
settings; enabling both selects the combined DyntraLB recompute scheduler.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `False` | Enable DyntraLB and select a DyntraLB-aware scheduler. |
| `mode` | str | `"dynamic"` | Use `"static"` or `"dynamic"` activation. |
| `start_step` | int | `250` | First completed engine-step snapshot allowed to generate a plan. |
| `end_step` | int | `-1` | Exclusive final snapshot step; `-1` means no upper bound. |
| `bubble_threshold` | float | `5.0` | Minimum maximum-to-average rank-load difference required to modify scheduling. Values greater than or equal to `1` are KV-cache blocks; values below `1` are normalized ratios. |
| `long_req_block_threshold` | int | `700` | In dynamic mode, a newly added request above this block count activates balancing. The default threshold corresponds to approximately 89,600 tokens when `block_size=128`. |
| `dynamic_max_step` | int | `256` | Stop dynamic balancing after this many active steps without another newly added long request. |
| `enable_diagnostics` | bool | `False` | Enable verbose logs for feature validation and debugging only. It is disabled by default and should remain disabled in production. |

**rejection_sampler_config**

> **Note**: Both block verify and entropy verify improve speculative decoding performance (higher acceptance rate, lower latency) at the cost of reduced sampling precision. A larger `posterior_alpha` makes the adjustment more aggressive — it further lowers the acceptance threshold for high-entropy tokens, improving throughput but degrading output quality. Users should tune these parameters based on their specific model weights and application scenario to find the right trade-off between performance and precision.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enable_block_verify`   | bool  | `False` | Whether to enable block verify mode. Block verify evaluates all draft tokens as a block using cumulative probability products, which can improve acceptance rate. |
| `enable_entropy_verify` | bool  | `False` | Whether to enable entropy verify mode. Entropy verify adjusts the acceptance threshold based on the entropy of the target distribution — higher entropy (uncertain) tokens get a lower threshold (easier to accept), while lower entropy (confident) tokens get a stricter threshold. |
| `posterior_threshold`   | float | `0.95`  | Upper bound for the entropy-adjusted acceptance threshold. Must be in (0, 1]. The effective threshold is `min(exp(-entropy * posterior_alpha), posterior_threshold)`. |
| `posterior_alpha`       | float | `0.4`   | Scaling factor for entropy in the threshold computation. Must be >= 0. Higher values make the threshold more sensitive to entropy — high-entropy tokens become much easier to accept, improving performance but reducing precision. |

**dynamic_spec_config**

> **Note**: This is an exploratory feature for model runner v1. Supported methods are `"dspark"` (DSpark confidence head) and `"dflash"` (head-free; uses max-softmax over draft logits as a confidence proxy). You still need a matching `speculative_config` (`method: "dspark"` or `"dflash"`); `dynamic_spec_config` only controls how many drafted tokens are verified per request. See [Dynamic Speculative Decoding](../feature_guide/speculative_decoding.md#dynamic-speculative-decoding) for usage and limitations.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `method` | str | `None` | Dynamic method name. Supported values: `"dspark"`, `"dflash"`. Omit or set to `None` to disable. |
| `method_params` | dict | `{}` | Method-specific hyperparameters. When empty, each method falls back to its built-in defaults. |

**dynamic_spec_config.method_params** (when `method` is `"dspark"` or `"dflash"`)

`dspark` and `dflash` share the same scheduling hyperparameters. The difference is only how per-token acceptance confidence is estimated: DSpark uses its confidence head (`sigmoid`), while DFlash (head-free) uses `max(softmax(logits))` of the drafted token.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `initial_verify_budget_per_req` | int | `5` | Initial per-request verify budget before the first recompute. |
| `budget_update_interval` | int | `16` | Recompute the shared verify budget every N decode steps. |
| `budget_threshold` | float | `0.3` | Cumulative survival-probability threshold used when estimating the mean verify budget. |
| `min_verify_tokens` | int | `1` | Minimum number of draft tokens verified per request. |

**scheduler_config.short_request_first_config**

ShortRequestFirst is a waiting-queue policy for FCFS synchronous or asynchronous scheduling on prefill and PD-mixed paths. It does not support batch-job-aware, profiling-chunk, or PD-disaggregated D-node scheduling. See [ShortRequestFirst Prefill Scheduling](../feature_guide/short_request_first.md) for usage, behavior, and tuning guidance.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled`                | bool  | `False` | Whether to enable ShortRequestFirst scheduling. |
| `threshold`              | int   | `256`   | Prompt-length threshold (tokens). Requests with `num_prompt_tokens <= threshold` are treated as short prefills and prioritized over long prefills. |
| `long_max_wait_ms`       | float | `0.0`   | Maximum time a long prefill may wait behind short prefills before it can be promoted ahead of them. `0` disables long-request promotion and keeps strict short-request priority. |

**scheduler_config.batch_job_sched_config**

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `false` | Enable the batch-job-aware scheduler. |
| `max_jobs` | int | `20` | Maximum number of tracked jobs. `0` means unlimited. |
| `reserve_margin_blocks` | int | `2` | Extra block margin added to the KV cache reserve as safety buffer. |
| `reserve_max_blocks` | int | `8` | Maximum number of blocks that can be reserved. |
| `low_available_tokens_threshold` | int | `4096` | Threshold for prioritising long vs short decode jobs. When available tokens > threshold, long decode jobs are prioritised; when ≤ threshold, short decode jobs are prioritised. |
| `short_decode_token_threshold` | int | `32` | Threshold for classifying a job as "short decode". |

<span id="rl_config"></span>**rl_config**

`rl_config` is a one-click RL mode switch. When `enabled` is `true`, it refreshes the global Ascend configuration on every initialization, forces `AscendConfig.weight_nz_mode=0`, synchronizes `VLLM_ASCEND_ENABLE_NZ=0`, sets `VLLM_SERVER_DEV_MODE=1`, and removes the `expandable_segments` entry from `PYTORCH_NPU_ALLOC_CONF` with an informational log. These fixed RL behaviors are not configurable as `rl_config` sub-fields. When `enabled` is `false`, all other sub-fields are ignored.

When RL mode is enabled, its fixed NZ and developer-endpoint settings take precedence over top-level configuration and environment variables. `VLLM_BATCH_INVARIANT=1` remains enabled when `rl_config.enable_batch_invariant` is false.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `enabled` | bool | `false` | Master switch for RL mode. When `true`, all RL best-practice defaults below are applied. |
| `sleep_mode_extra_cleanup` | bool | `false` | Same-device mode. Enables HCCL process-group release + ACL graph workspace cleanup during sleep, returning more NPU memory to the trainer at the cost of increased wakeup latency. This option is available only under `rl_config`; the former top-level `enable_sleep_mode_extra_cleanup` key has been removed. |
| `enable_training_consistency` | bool | `false` | Both modes. Enables the FA3 attention backend used for training-inference consistency. Requires the `flash_attn_npu_v3` package and does not implicitly enable batch invariance. |
| `enable_batch_invariant` | bool | `false` | Both modes. Provides an additional way to enable batch-invariant deterministic computation: when `true`, sets `VLLM_BATCH_INVARIANT=1` before workers are started. Worker-side batch-invariant initialization then sets `HCCL_DETERMINISTIC=strict` and `LCCL_DETERMINISTIC=1`. When `false`, an existing `VLLM_BATCH_INVARIANT` environment setting is preserved. Requires building vllm-ascend from source with `COMPILE_CUSTOM_KERNELS=1`; RL mode already forces `weight_nz_mode=0`. |

**Deployment modes**

- **Same-device mode** (sleep/wake + IPC weight transfer): the inference engine and trainer share the same NPU card. Use `enable_sleep_mode=True` and optionally `sleep_mode_extra_cleanup=True` for extra memory.
- **Cross-device mode** (pause/resume + HCCL weight transfer): the inference engine and trainer use different NPU cards. Use `--weight-transfer-config '{"backend": "nccl"}'`.

**Example (online, same-device):**

```bash
vllm serve DeepSeek-V4 \
    --enable-sleep-mode \
    --enable-return-routed-experts \
    --weight-transfer-config '{"backend": "ipc"}' \
    --additional-config '{"rl_config": {"enabled": true, "enable_training_consistency": true, "enable_batch_invariant": true, "sleep_mode_extra_cleanup": true}}'
```

**Example (online, cross-device):**

```bash
vllm serve DeepSeek-V4 \
    --weight-transfer-config '{"backend": "nccl"}' \
    --additional-config '{"rl_config": {"enabled": true}}'
```

**Example (offline):**

```python
from vllm import LLM

llm = LLM(
    model="DeepSeek-V4",
    enable_sleep_mode=True,  # same-device mode only
    additional_config={
        "rl_config": {
            "enabled": True,
            "enable_training_consistency": True,
            "enable_batch_invariant": True,
        },
    },
)
```

**Migration guide**

| Before | After |
| ------ | ----- |
| `export VLLM_ASCEND_ENABLE_NZ=0` | `"rl_config": {"enabled": true}` |
| `export VLLM_SERVER_DEV_MODE=1` | `"rl_config": {"enabled": true}` |
| `export VLLM_BATCH_INVARIANT=1` | `"rl_config": {"enabled": true, "enable_batch_invariant": true}` |
| `--attention-backend FLASH_ATTN` for FA3 consistency mode | `"rl_config": {"enabled": true, "enable_training_consistency": true}` |
| top-level `"weight_nz_mode": 0` for RL | `"rl_config": {"enabled": true}` |
| top-level `"enable_sleep_mode_extra_cleanup": true` | `"rl_config": {"enabled": true, "sleep_mode_extra_cleanup": true}` |

### Example

An example of additional configuration is as follows:

```python
{
    "finegrained_tp_config": {
        "lmhead_tensor_parallel_size": 8,
        "oproj_tensor_parallel_size": 8,
        "embedding_tensor_parallel_size": 8,
        "mlp_tensor_parallel_size": 8,
    },
    "enable_kv_nz": False,
    "multistream_overlap_shared_expert": True,
    "rejection_sampler_config": {
        "enable_block_verify": True,
        "enable_entropy_verify": True,
        "posterior_threshold": 0.95,
        "posterior_alpha": 0.4,
    },
    "dynamic_spec_config": {
        "method": "dspark",
        "method_params": {
            "initial_verify_budget_per_req": 5,
            "budget_update_interval": 50,
            "budget_threshold": 0.7,
        },
    },
    "refresh": False
}
```
