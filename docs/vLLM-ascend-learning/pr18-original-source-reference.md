# PR18 原始源码摘录：文件、函数与历史行号

配套讲解：[PR18 学习详解](pr18-megamoe-design-and-code-walkthrough.md)。

**所有左侧行号均对应原始提交 `df1286cca8c1bfffa665ea4be57c6498bea1c49e`，不是当前工作区行号。** 本文从 Git 对象提取原始 PR18 修改处及讲解需要的上下文，覆盖原始 17 个修改文件；不包含 PR31、PR51 或当前适配代码。未摘录区间明确标出省略。

每行格式为 `原始行号 | 原始代码`。行号与竖线是阅读标记，不能直接复制为 Python 运行。为便于从讲解逐步跳转，部分连续代码按步骤分块展示，代码顺序、缩进和原始行号保持不变。

原始文件的完整内容仍以 `git show 'df1286cca:文件路径'` 为准；本页不是第二套运行代码。查看增加和删除请使用 [pr18-original.patch](pr18-original.patch)。父提交被删除的行不会伪装成修改后的源码。

| 原始文件 | 摘录入口 |
| --- | --- |
| `docs/source/user_guide/configuration/additional_config.md` | [带行号的源码](#additional-config) |
| `vllm_ascend/_310p/model_runner_310p.py` | [带行号的源码](#runner-310p) |
| `vllm_ascend/ascend_config.py` | [带行号的源码](#ascend-config) |
| `vllm_ascend/ascend_forward_context.py` | [带行号的源码](#forward-context) |
| `vllm_ascend/device/device_op.py` | [带行号的源码](#device-op) |
| `vllm_ascend/distributed/parallel_state.py` | [带行号的源码](#parallel-state) |
| `vllm_ascend/models/deepseek_v4/model.py` | [带行号的源码](#deepseek-model) |
| `vllm_ascend/ops/fused_moe/mega_moe.py` | [带行号的源码](#mega-moe) |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | [带行号的源码](#moe-comm) |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | [带行号的源码](#prepare-finalize) |
| `vllm_ascend/ops/fused_moe/routed_experts.py` | [带行号的源码](#routed-experts) |
| `vllm_ascend/ops/fused_moe/router/fused_topk_router.py` | [带行号的源码](#router) |
| `vllm_ascend/ops/fused_moe/shared_experts.py` | [带行号的源码](#shared-experts) |
| `vllm_ascend/patch/worker/patch_distributed.py` | [带行号的源码](#patch-distributed) |
| `vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py` | [带行号的源码](#w4a8-mxfp4) |
| `vllm_ascend/utils.py` | [带行号的源码](#utils) |
| `vllm_ascend/worker/model_runner_v1.py` | [带行号的源码](#model-runner) |

<a id="additional-config"></a>

## docs/source/user_guide/configuration/additional_config.md

原始完整文件共 325 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="additional-config-l63"></a>

### 原始第 63 行起

```text
   63 | | `enable_sparse_sfa_c8`              | bool | `False` | Whether to enable the packed C8 KV cache for Sparse Flash Attention in DSA models (e.g., DeepSeek V3.2 and GLM5). This option is independent of `enable_sparse_li_c8`. SFA prefill context parallelism and Ascend 950 DCP are not supported. |
   64 | | `enable_sparse_li_c8`               | bool | `False` | Whether to enable the C8 key and scale caches for LightningIndexer in DSA models. This option is independent of `enable_sparse_sfa_c8` and only applies to eligible indexer layers from the model quantization config. SFA prefill context parallelism and Ascend 950 DCP are not supported. |
   65 | | `c8_enable_reshape_optim`           | bool | `False` | Whether to use the StoreKVBlock operator to accelerate LightningIndexer C8 cache writes. `enable_sparse_li_c8` must also be enabled. In the PD separation scenario, only the P node is enabled. |
   66 | | `mc2_comm_alg`                      | str  | `""`    | set dispatch/combine op's `comm_alg` param, only supports `""/"fullmesh"/"hierarchy"/"fullmesh_v2"`. `"hierarchy"` is only supported by A2/A3, and `"fullmesh_v2"` is only supported by A3 now. |
   67 | | `enable_mc2_hierarchy_comm`         | bool | `False` | Enable dispatch/combine op inter-node communication by ROCE. This param will be deprecated and be replaced by mc2_comm_alg = "hierarchy" |
   68 | | `enable_prefill_mc2`                | bool | `False` | Whether to reserve mc2_token_capacity for prefill batches. When enabled, `max_num_batched_tokens` is used to calculate the mc2_token_capacity instead of the decode-only capacity. In this scenario, the recommended maximum value of `max_num_batched_tokens` is `tp_size * 512`. This is a temporary switch; once MC2 operators are complete for all scenarios, this switch will be removed and MC2 will be enabled by default. |
   69 | | `mega_moe_max_tokens`               | int  | `65536` | Token capacity for fused MoE operators. A3 passes it to `dispatch_ffn_combine` or uses it as CANN MegaMoE's receive capacity. A5 treats it as a global token ceiling and allocates a symmetric buffer for `min(mega_moe_max_tokens // expert_parallel_size, execution_tokens_per_rank)` tokens per rank. Workspace memory scales linearly with the allocated capacity. |
   70 | | `msmonitor_use_daemon`              | bool | `False` | Whether to use daemon mode for msmonitor. The legacy `MSMONITOR_USE_DAEMON` environment variable is no longer supported. |
   71 | | `enable_mlapo`                      | bool | `True`  | Whether to enable MLAPO (Model Layer-wise Adaptive Parallel Optimization). The legacy `VLLM_ASCEND_ENABLE_MLAPO` environment variable is no longer supported. |
   72 | | `mlapo_keep_prefill_weights`        | bool | `False` | When True, keep MLAPO prefill weights on NPU instead of freeing them on kv_consumer (decode-only D) nodes. D nodes have normal local-prefill paths (recompute / fallback / preempt) that crash when the weights are freed (issue #11882). Enable this to trade NPU memory for stability. |
   73 | | `weight_nz_mode`                    | int  | `1`     | Weight NZ mode. `0` disables NZ, `1` enables NZ only for quantized weights, and `2` also enables NZ for BF16/FP16 weights when supported. The legacy `VLLM_ASCEND_ENABLE_NZ` environment variable is no longer supported. |
   74 | | `enable_fused_mc2`                  | int  | `0`     | Fused MC2 configuration. `0` disables the fused path and `1` enables it when the model and parallel configuration support it. The legacy `VLLM_ASCEND_ENABLE_FUSED_MC2` environment variable is no longer supported. |
   75 | | `enable_transpose_kv_cache_by_block`| bool | `True`  | Whether to enable transpose KV cache by block. The legacy `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` environment variable is no longer supported. |
```

<a id="runner-310p"></a>

## vllm_ascend/_310p/model_runner_310p.py

原始完整文件共 986 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="runner-310p-l151"></a>

### 原始第 151 行起

```text
  151 |         allow_microbatching: bool = False,
  152 |         force_eager: bool = False,
  153 |         force_uniform_decode: bool | None = None,
  154 |         force_has_lora: bool | None = None,
  155 |         force_num_active_loras: int | None = None,
  156 |         num_encoder_reqs: int = 0,
  157 |         is_dummy_run: bool = False,
  158 |     ):
  159 |         is_all_decode = np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)
  160 |
  161 |         if self.attn_state in (AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit):
  162 |             force_eager = True
  163 |
```

省略原始第 164 至 187 行；下一块恢复原始行号。

<a id="runner-310p-l188"></a>

### 原始第 188 行起

```text
  188 |             allow_microbatching=allow_microbatching,
  189 |             force_eager=force_eager,
  190 |             force_uniform_decode=force_uniform_decode,
  191 |             force_has_lora=force_has_lora,
  192 |             force_num_active_loras=force_num_active_loras,
  193 |             num_encoder_reqs=num_encoder_reqs,
  194 |             is_dummy_run=is_dummy_run,
  195 |         )
  196 |
```

<a id="runner-310p-l197"></a>

### 原始第 197 行起

```text
  197 |     def _build_attention_metadata(self, *args: Any, **kwargs: Any):
  198 |         # Parent dummy_run assigns ChunkedPrefill for non-MLA MTP (910B FIA graph).
  199 |         # 310P must capture SpecDecoding + splitfuse for SpecDecoding uniform decode graphs.
  200 |         if self._spec_dummy_capture:
```

<a id="ascend-config"></a>

## vllm_ascend/ascend_config.py

原始完整文件共 1500 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="ascend-config-l30"></a>

### 原始第 30 行起

```text
   30 | if TYPE_CHECKING:
   31 |     from vllm.config import VllmConfig
   32 |
   33 | _MEGA_MOE_SUPPORTED = None
   34 |
   35 |
```

<a id="ascend-config-l36"></a>

### 原始第 36 行起

```text
   36 | def compute_mega_moe_buffer_tokens_per_rank(
   37 |     mega_moe_max_tokens: int,
   38 |     execution_tokens_per_rank: int,
   39 |     expert_parallel_size: int,
   40 | ) -> int:
   41 |     """Compute the per-rank token capacity for the A5 MegaMoE buffer."""
   42 |     if execution_tokens_per_rank <= 0:
   43 |         raise ValueError(f"execution_tokens_per_rank must be positive, got {execution_tokens_per_rank}")
   44 |     if expert_parallel_size <= 0:
   45 |         raise ValueError(f"expert_parallel_size must be positive, got {expert_parallel_size}")
   46 |
```

<a id="ascend-config-l47"></a>

### 原始第 47 行起

```text
   47 |     configured_tokens_per_rank = mega_moe_max_tokens // expert_parallel_size
   48 |     buffer_tokens_per_rank = min(configured_tokens_per_rank, execution_tokens_per_rank)
   49 |     if buffer_tokens_per_rank <= 0:
   50 |         raise ValueError(
   51 |             "MegaMoE per-rank token capacity must be positive: "
   52 |             f"mega_moe_max_tokens={mega_moe_max_tokens}, "
   53 |             f"execution_tokens_per_rank={execution_tokens_per_rank}, "
   54 |             f"expert_parallel_size={expert_parallel_size}."
   55 |         )
   56 |     return buffer_tokens_per_rank
   57 |
   58 |
```

<a id="ascend-config-l59"></a>

### 原始第 59 行起

```text
   59 | def is_mega_moe_supported() -> bool:
   60 |     """Whether the megamoe op is available at runtime.
   61 |
   62 |     Always read _MEGA_MOE_SUPPORTED through this accessor instead of
   63 |     ``from ascend_config import _MEGA_MOE_SUPPORTED``: the global is rebound
   64 |     during config init (AscendConfig._validate_user_input_ranges rolls back
```

省略原始第 65 至 419 行；下一块恢复原始行号。

<a id="ascend-config-l420"></a>

### 原始第 420 行起

```text
  420 |     enable_reduce_sample: bool = False
  421 |     enable_dsa_cp: bool = False
  422 |     enable_force_eplb: bool = False
  423 |     draft_window_size: int | None = None
  424 |     mix_placement: bool = False
  425 |     pa_shape_list: list[Any] = dataclasses.field(default_factory=list)
  426 |     # Global token ceiling for dispatch_ffn_combine. A5 MegaMoE derives its
  427 |     # per-rank symmetric-buffer capacity from this ceiling and the current
  428 |     # process's MC2 execution capacity.
  429 |     # This is a reference value: if the actual per-rank received token
  430 |     # count exceeds it, tokens may be truncated, causing precision
  431 |     # degradation. Do not set it too large because workspace memory scales
  432 |     # linearly with this value. Default 65536.
  433 |     mega_moe_max_tokens: int = 65536
  434 |     ascend_log_path: str = dataclasses.field(
```

<a id="forward-context"></a>

## vllm_ascend/ascend_forward_context.py

原始完整文件共 583 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="forward-context-l8"></a>

### 原始第 8 行起

```text
    8 | import vllm.envs as envs_vllm
    9 | from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
   10 | from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_world_size
   11 | from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
   12 | from vllm.logger import logger
   13 |
   14 | from vllm_ascend.ascend_config import (
   15 |     compute_mega_moe_buffer_tokens_per_rank,
   16 |     get_ascend_config,
   17 |     is_mega_moe_supported,
   18 | )
   19 | from vllm_ascend.quantization.quant_type import QuantType
   20 | from vllm_ascend.utils import (
   21 |     AscendDeviceType,
   22 |     get_ascend_device_type,
   23 |     has_layer_idx,
   24 |     is_moe_model,
   25 | )
```

省略原始第 26 至 145 行；下一块恢复原始行号。

<a id="forward-context-l146"></a>

### 原始第 146 行起

```text
  146 |         from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method
  147 |
  148 |         max_num_tokens = int(num_tokens_across_dp.max().item()) if num_tokens_across_dp is not None else num_tokens
  149 |         moe_comm_type = select_moe_comm_method(
  150 |             max_num_tokens,
  151 |             vllm_config,
  152 |             model_instance=model_instance,
  153 |         )
  154 |
  155 |         forward_context.moe_comm_type = moe_comm_type
  156 |         forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)
  157 |         forward_context.is_decode_only_node = _is_decode_only_node(vllm_config)
  158 |         forward_context.use_mega_moe = use_cann_megamoe(vllm_config)
```

省略原始第 159 至 263 行；下一块恢复原始行号。

<a id="forward-context-l264"></a>

### 原始第 264 行起

```text
  264 |
  265 |
```

<a id="forward-context-l266"></a>

### 原始第 266 行起

```text
  266 | def get_mc2_tokens_capacity():
  267 |     return _mc2_tokens_capacity
  268 |
  269 |
```

<a id="forward-context-l270"></a>

### 原始第 270 行起

```text
  270 | def get_a5_mega_moe_buffer_tokens_per_rank(
  271 |     vllm_config: VllmConfig,
  272 |     mc2_tokens_capacity: int | None = None,
  273 | ) -> int:
  274 |     """Resolve the A5 MegaMoE capacity for the current PD role."""
```

<a id="forward-context-l275"></a>

### 原始第 275 行起

```text
  275 |     kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
  276 |     kv_role = getattr(kv_transfer_config, "kv_role", None)
  277 |     is_kv_producer = bool(getattr(kv_transfer_config, "is_kv_producer", False))
  278 |     if kv_role in ("kv_producer", "kv_both") or (kv_role is None and is_kv_producer):
  279 |         execution_tokens_per_rank = vllm_config.scheduler_config.max_num_batched_tokens
  280 |     else:
  281 |         if mc2_tokens_capacity is None:
  282 |             mc2_tokens_capacity = get_mc2_tokens_capacity()
  283 |         if mc2_tokens_capacity is None:
  284 |             raise RuntimeError("MC2 token capacity must be initialized before using A5 MegaMoE.")
  285 |         execution_tokens_per_rank = mc2_tokens_capacity
  286 |
  287 |     ascend_config = get_ascend_config()
  288 |     expert_parallel_size = vllm_config.parallel_config.world_size_across_dp
  289 |     buffer_tokens_per_rank = compute_mega_moe_buffer_tokens_per_rank(
  290 |         ascend_config.mega_moe_max_tokens,
  291 |         execution_tokens_per_rank,
  292 |         expert_parallel_size,
  293 |     )
  294 |     return buffer_tokens_per_rank
  295 |
  296 |
```

<a id="forward-context-l297"></a>

### 原始第 297 行起

```text
  297 | def set_mc2_mask(vllm_config, device):
  298 |     global _reserved_mc2_mask
  299 |     if _reserved_mc2_mask is not None:
  300 |         return
  301 |     if is_moe_model(vllm_config):
  302 |         _reserved_mc2_mask = torch.zeros(
```

省略原始第 303 至 345 行；下一块恢复原始行号。

<a id="forward-context-l346"></a>

### 原始第 346 行起

```text
  346 |
  347 |
```

<a id="forward-context-l348"></a>

### 原始第 348 行起

```text
  348 | def _select_a5_moe_comm_method(
  349 |     num_tokens: int,
  350 |     vllm_config: VllmConfig,
  351 |     mc2_tokens_capacity: int,
  352 |     quant_type: QuantType | None,
  353 |     activation: str | None,
  354 |     group_size: int | None,
  355 | ) -> MoECommType:
  356 |     num_experts_per_tok = getattr(
  357 |         vllm_config.model_config.hf_text_config,
  358 |         "num_experts_per_tok",
  359 |         getattr(vllm_config.model_config.hf_text_config, "top_k_experts", 1),
  360 |     )
  361 |     world_size = vllm_config.parallel_config.world_size_across_dp
  362 |     ascend_config = get_ascend_config()
  363 |     eplb_config = ascend_config.eplb_config
  364 |     buffer_tokens_per_rank = (
  365 |         get_a5_mega_moe_buffer_tokens_per_rank(vllm_config, mc2_tokens_capacity)
  366 |         if ascend_config.enable_fused_mc2 == 1
  367 |         else 0
  368 |     )
```

<a id="forward-context-l369"></a>

### 原始第 369 行起

```text
  369 |     normalized_activation = None if activation is None else activation.lower().removeprefix("moeactivation.")
```

<a id="forward-context-l370"></a>

### 原始第 370 行起

```text
  370 |     use_mega_moe = (
  371 |         ascend_config.enable_fused_mc2 == 1
  372 |         and world_size > 1
  373 |         and num_tokens <= buffer_tokens_per_rank
  374 |         and quant_type in _A5_MEGA_MOE_QUANT_TYPES
  375 |         and normalized_activation in (None, "silu", "swiglu")
  376 |         and group_size in (None, _A5_MEGA_MOE_GROUP_SIZE)
  377 |         and not eplb_config.dynamic_eplb
  378 |         and eplb_config.num_redundant_experts == 0
  379 |         and not ascend_config.mix_placement
  380 |     )
  381 |     if use_mega_moe:
  382 |         return MoECommType.FUSED_MC2
  383 |     if (num_tokens is None or num_tokens <= mc2_tokens_capacity) and world_size > 1:
  384 |         return MoECommType.MC2
  385 |     if world_size <= num_experts_per_tok:
  386 |         return MoECommType.ALLGATHER
  387 |     return MoECommType.ALLTOALL
  388 |
  389 |
  390 | _A5_MEGA_MOE_QUANT_TYPES = {
  391 |     QuantType.W4A8MXFP,
  392 | }
  393 | _A5_MEGA_MOE_GROUP_SIZE = 32
  394 | _A5_MOE_QUANT_TYPES_BY_CONFIG_ID: dict[int, QuantType] = {}
  395 |
  396 |
```

<a id="forward-context-l397"></a>

### 原始第 397 行起

```text
  397 | def cache_a5_moe_quant_type(
  398 |     vllm_config: VllmConfig | None,
  399 |     quant_type: QuantType,
  400 | ) -> None:
  401 |     if vllm_config is None:
  402 |         return
  403 |     _A5_MOE_QUANT_TYPES_BY_CONFIG_ID[id(vllm_config)] = quant_type
  404 |
  405 |
```

<a id="forward-context-l406"></a>

### 原始第 406 行起

```text
  406 | def _get_a5_moe_quant_type(
  407 |     vllm_config: VllmConfig,
  408 |     model_instance: torch.nn.Module | None,
  409 | ) -> QuantType | None:
  410 |     if model_instance is not None:
  411 |         modules = model_instance.modules() if callable(getattr(model_instance, "modules", None)) else ()
  412 |         for module in modules:
  413 |             if not hasattr(module, "moe_config"):
  414 |                 continue
  415 |             quant_type = getattr(module, "quant_type", None)
  416 |             if isinstance(quant_type, QuantType) and quant_type != QuantType.NONE:
  417 |                 return quant_type
  418 |     return _A5_MOE_QUANT_TYPES_BY_CONFIG_ID.get(id(vllm_config))
  419 |
  420 |
```

<a id="forward-context-l421"></a>

### 原始第 421 行起

```text
  421 | def _get_a5_moe_activation(
  422 |     vllm_config: VllmConfig,
  423 |     model_instance: torch.nn.Module | None,
  424 | ) -> str | None:
  425 |     if model_instance is not None:
  426 |         modules = model_instance.modules() if callable(getattr(model_instance, "modules", None)) else ()
  427 |         for module in modules:
  428 |             if not hasattr(module, "moe_config"):
  429 |                 continue
  430 |             activation = getattr(module, "activation", None)
  431 |             if isinstance(activation, str):
  432 |                 return activation
  433 |             if isinstance(activation, Enum):
  434 |                 return activation.name
  435 |     hf_text_config = vllm_config.model_config.hf_text_config
  436 |     return getattr(hf_text_config, "hidden_act", getattr(hf_text_config, "hidden_activation", None))
  437 |
  438 |
```

<a id="forward-context-l439"></a>

### 原始第 439 行起

```text
  439 | def _get_a5_moe_group_size(vllm_config: VllmConfig) -> int | None:
  440 |     quant_description = getattr(getattr(vllm_config, "quant_config", None), "quant_description", None)
  441 |     if isinstance(quant_description, dict):
  442 |         return quant_description.get("group_size")
  443 |     return getattr(quant_description, "group_size", None)
  444 |
  445 |
```

<a id="forward-context-l446"></a>

### 原始第 446 行起

```text
  446 | def select_moe_comm_method(
  447 |     num_tokens: int,
  448 |     vllm_config: VllmConfig,
  449 |     model_instance: torch.nn.Module | None = None,
  450 | ) -> MoECommType | None:
  451 |     """Select the MoE communication method according to parallel settings,
  452 |     device generation, and token count.
  453 |
  454 |     1. Non-MoE models return `None`.
  455 |     2. Without expert parallel, fall back to all-gather.
  456 |     3. On A2 with expert parallel, pick MC2 when tokens fit the MC2 capacity
  457 |        and the DP size is large enough; otherwise use all-gather.
  458 |     4. On A3 with expert parallel, prefer fused MC2 when enabled and the EP
  459 |        group size is small enough; otherwise use MC2 within capacity or
  460 |        all-to-all.
  461 |     5. On 310P, always use all-gather.
  462 |     6. On A5 with expert parallel, use MegaMoE for supported MXFP layouts
  463 |        within its symmetric-buffer capacity; otherwise use MC2, all-gather,
  464 |        or all-to-all according to the existing token and EP constraints.
  465 |
  466 |     Args:
  467 |         num_tokens (int): The number of tokens in the current batch.
  468 |         vllm_config (VllmConfig): Runtime configuration for the model.
  469 |         model_instance (torch.nn.Module | None): Model instance used to
  470 |             resolve the A5 routed-expert quantization and activation.
  471 |
  472 |     Raises:
  473 |         ValueError: If the soc version is unsupported.
  474 |
  475 |     Returns:
  476 |         MoECommType | None: The selected MoE communication method.
```

省略原始第 477 至 494 行；下一块恢复原始行号。

<a id="forward-context-l495"></a>

### 原始第 495 行起

```text
  495 |         moe_comm_type = _select_a3_moe_comm_method(
  496 |             num_tokens,
  497 |             mc2_tokens_capacity,
  498 |             vllm_config,
  499 |         )
  500 |     elif soc_version == AscendDeviceType.A5:
  501 |         moe_comm_type = _select_a5_moe_comm_method(
  502 |             num_tokens,
  503 |             vllm_config,
  504 |             mc2_tokens_capacity,
  505 |             _get_a5_moe_quant_type(vllm_config, model_instance),
  506 |             _get_a5_moe_activation(vllm_config, model_instance),
  507 |             _get_a5_moe_group_size(vllm_config),
  508 |         )
  509 |     elif soc_version == AscendDeviceType._310P:
  510 |         moe_comm_type = MoECommType.ALLGATHER
  511 |
  512 |     else:
  513 |         raise ValueError(f"Unsupported soc_version: {soc_version}")
  514 |     logger.debug(
```

<a id="device-op"></a>

## vllm_ascend/device/device_op.py

原始完整文件共 1611 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="device-op-l1019"></a>

### 原始第 1019 行起

```text
 1019 |                 group_list=group_list,
 1020 |                 x_dtype=torch.float8_e4m3fn,
 1021 |                 weight_dtype=torch_npu.float4_e2m1fn_x2,
 1022 |                 output_dtype=torch.bfloat16,
 1023 |             )[0]
 1024 |             # DSV4 need swiglu_limit input
 1025 |             import custom_ops
 1026 |             import cann_ops_transformer
 1027 |             import cann_ops_nn
 1028 |
 1029 |             # Legacy CANN operator call kept for reference:
 1030 |             # out, out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
 1031 |             #     hidden_states,
 1032 |             #     topk_weight=None,
 1033 |             #     group_index=None,
 1034 |             #     dst_type=torch.float8_e4m3fn,
 1035 |             #     quant_mode=2,
 1036 |             #     clamp_value=swiglu_limit,
 1037 |             # )
 1038 |             swiglu_limit_args = {}
 1039 |             if swiglu_limit is not None:
 1040 |                 swiglu_limit_args["clamp_limit"] = swiglu_limit
```

<a id="device-op-l1041"></a>

### 原始第 1041 行起

```text
 1041 |             out, out_scale, _ = torch.ops.cann_ops_nn.swiglu_group_quant(
 1042 |                 hidden_states,
 1043 |                 dst_type=torch.float8_e4m3fn,
 1044 |                 round_scale=True,
 1045 |                 quant_mode=1,
 1046 |                 **swiglu_limit_args,
 1047 |             )
 1048 |         elif mxfp_quant_dtype == QuantType.W4A16MXFP:
 1049 |             hidden_states = torch_npu.npu_grouped_matmul(
 1050 |                 x=[x],
 1051 |                 weight=[weight],
 1052 |                 antiquant_scale=[weight_scale],
```

<a id="parallel-state"></a>

## vllm_ascend/distributed/parallel_state.py

原始完整文件共 273 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="parallel-state-l1"></a>

### 原始第 1 行起

```text
    1 | import torch
    2 | from vllm.config import ParallelConfig, get_current_vllm_config
    3 | from vllm.distributed.parallel_state import GroupCoordinator, get_world_group, init_model_parallel_group
    4 |
    5 | from vllm_ascend.ascend_config import get_ascend_config
    6 | from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
    7 |
    8 | # MegaMoE and MC2 operators need dedicated group coordinators.
    9 | _MEGA_MOE: GroupCoordinator | None = None
   10 | _MC2: GroupCoordinator | None = None
   11 |
   12 | # Module specific tensor parallel groups
   13 | _MLP_TP: GroupCoordinator | None = None
   14 | _OTP: GroupCoordinator | None = None
   15 | _LMTP: GroupCoordinator | None = None
```

省略原始第 16 至 16 行；下一块恢复原始行号。

<a id="parallel-state-l17"></a>

### 原始第 17 行起

```text
   17 |
   18 | _P_TP: GroupCoordinator | None = None
   19 |
   20 | _DYNAMIC_EPLB: GroupCoordinator | None = None
   21 |
   22 |
```

<a id="parallel-state-l23"></a>

### 原始第 23 行起

```text
   23 | def _initialize_mega_moe_communicator(group: GroupCoordinator) -> None:
   24 |     """Eagerly initialize the HCCL communicator required by SymmBuffer."""
   25 |     device_group = group.device_group
   26 |     rank_in_group = torch.distributed.get_rank(group=device_group)
   27 |     backend = device_group._get_backend(torch.device("npu"))
   28 |     group_name = backend.get_hccl_comm_name(rank_in_group)
   29 |     if not group_name:
   30 |         raise RuntimeError("Failed to initialize the MegaMoE HCCL communicator")
   31 |
   32 |
```

<a id="parallel-state-l33"></a>

### 原始第 33 行起

```text
   33 | def init_ascend_model_parallel(
   34 |     parallel_config: ParallelConfig,
   35 | ):
   36 |     if model_parallel_initialized():
   37 |         return
   38 |     assert torch.distributed.is_initialized()
```

省略原始第 39 至 91 行；下一块恢复原始行号。

<a id="parallel-state-l92"></a>

### 原始第 92 行起

```text
   92 |             global_dp_size * global_pcp_size * global_tp_size,
   93 |         )
   94 |         .unbind(0)
   95 |     )
   96 |     group_ranks = [x.tolist() for x in group_ranks]
   97 |
```

<a id="parallel-state-l98"></a>

### 原始第 98 行起

```text
   98 |     global _MC2, _MEGA_MOE
   99 |     _MC2 = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="mc2")
  100 |     if get_ascend_device_type() == AscendDeviceType.A5 and get_ascend_config().enable_fused_mc2 == 1:
```

<a id="parallel-state-l101"></a>

### 原始第 101 行起

```text
  101 |         _MEGA_MOE = init_model_parallel_group(
  102 |             group_ranks,
  103 |             get_world_group().local_rank,
  104 |             backend,
  105 |             group_name="mega_moe",
  106 |         )
  107 |         _initialize_mega_moe_communicator(_MEGA_MOE)
  108 |
  109 |     if get_ascend_config().eplb_config.dynamic_eplb:
  110 |         global _DYNAMIC_EPLB
  111 |         _DYNAMIC_EPLB = init_model_parallel_group(
  112 |             group_ranks, get_world_group().local_rank, backend, group_name="dynamic_eplb"
  113 |         )
```

省略原始第 114 至 159 行；下一块恢复原始行号。

<a id="parallel-state-l160"></a>

### 原始第 160 行起

```text
  160 |
```

<a id="parallel-state-l161"></a>

### 原始第 161 行起

```text
  161 | def get_mc2_group() -> GroupCoordinator:
  162 |     assert _MC2 is not None, "mc2 group is not initialized"
  163 |     return _MC2
  164 |
  165 |
```

<a id="parallel-state-l166"></a>

### 原始第 166 行起

```text
  166 | def get_mega_moe_group() -> GroupCoordinator:
  167 |     assert _MEGA_MOE is not None, "MegaMoE group is not initialized"
  168 |     return _MEGA_MOE
  169 |
  170 |
```

<a id="parallel-state-l171"></a>

### 原始第 171 行起

```text
  171 | def get_mlp_tp_group() -> GroupCoordinator:
  172 |     assert _MLP_TP is not None, "mlp group is not initialized"
  173 |     return _MLP_TP
  174 |
  175 |
```

<a id="parallel-state-l176"></a>

### 原始第 176 行起

```text
  176 | def get_otp_group() -> GroupCoordinator:
```

省略原始第 177 至 195 行；下一块恢复原始行号。

<a id="parallel-state-l196"></a>

### 原始第 196 行起

```text
  196 | def get_dynamic_eplb_group() -> GroupCoordinator:
  197 |     assert _DYNAMIC_EPLB is not None, "Dynamic eplb group is not initialized"
  198 |     return _DYNAMIC_EPLB
  199 |
  200 |
```

<a id="parallel-state-l201"></a>

### 原始第 201 行起

```text
  201 | def destroy_ascend_model_parallel():
  202 |     global _MEGA_MOE
  203 |     if _MEGA_MOE:
  204 |         _MEGA_MOE.destroy()
  205 |     _MEGA_MOE = None
  206 |
  207 |     global _MC2
  208 |     if _MC2:
  209 |         _MC2.destroy()
  210 |     _MC2 = None
  211 |
  212 |     global _MLP_TP
```

<a id="deepseek-model"></a>

## vllm_ascend/models/deepseek_v4/model.py

原始完整文件共 1336 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="deepseek-model-l374"></a>

### 原始第 374 行起

```text
  374 |     def forward(
  375 |         self,
  376 |         hidden_states: torch.Tensor,
  377 |         input_ids: torch.Tensor | None = None,
  378 |         hidden_states_fp32: torch.Tensor | None = None,
  379 |     ) -> torch.Tensor:
  380 |         if self.gate.tid2eid is not None and input_ids is None:
  381 |             raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
  382 |
  383 |         num_tokens, hidden_dim = hidden_states.shape
  384 |         hidden_states = hidden_states.view(-1, hidden_dim)
  385 |         if hidden_states_fp32 is not None:
  386 |             hidden_states_fp32 = hidden_states_fp32.view(-1, hidden_dim)
  387 |
  388 |         # Chunk the hidden states so they aren't replicated across TP ranks.
  389 |         # This avoids duplicate computation in self.experts.
  390 |         # TODO: We can replace the all_reduce at the end of attn with a
  391 |         # reduce_scatter instead of chunking here.
  392 |         if self.is_sequence_parallel:
  393 |             hidden_states = sequence_parallel_chunk(hidden_states)
  394 |             if hidden_states_fp32 is not None:
  395 |                 hidden_states_fp32 = sequence_parallel_chunk(hidden_states_fp32)
  396 |
  397 |         if self.experts.is_internal_router:
  398 |             # In this case, the gate/router runs inside the FusedMoEFactory class
  399 |             router_input = hidden_states if hidden_states_fp32 is None else hidden_states_fp32
  400 |             fused_moe_out = self.experts(
```

<a id="mega-moe"></a>

## vllm_ascend/ops/fused_moe/mega_moe.py

原始完整文件共 322 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="mega-moe-l1"></a>

### 原始第 1 行起

```text
    1 | #
    2 | # Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
    3 | # This file is a part of the vllm-ascend project.
    4 | #
    5 | # Licensed under the Apache License, Version 2.0 (the "License");
    6 | # you may not use this file except in compliance with the License.
    7 | # You may obtain a copy of the License at
    8 | #
    9 | #     http://www.apache.org/licenses/LICENSE-2.0
   10 | #
   11 | # Unless required by applicable law or agreed to in writing, software
   12 | # distributed under the License is distributed on an "AS IS" BASIS,
   13 | # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   14 | # See the License for the specific language governing permissions and
   15 | # limitations under the License.
   16 | #
   17 | from __future__ import annotations
   18 |
   19 | from dataclasses import dataclass
   20 | from typing import Any
   21 |
   22 | import torch
   23 | from vllm.config import get_current_vllm_config
   24 | from vllm.model_executor.layers.fused_moe import FusedMoEConfig
   25 |
   26 | from vllm_ascend.ascend_forward_context import get_a5_mega_moe_buffer_tokens_per_rank
   27 | from vllm_ascend.distributed.parallel_state import get_mega_moe_group
   28 | from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEFusedExpertsInput
   29 | from vllm_ascend.quantization.quant_type import QuantType
   30 |
   31 | _MEGA_MOE_SUPPORTED_QUANTS = {
   32 |     QuantType.W4A8MXFP,
   33 | }
   34 | _FP4_PACK_FACTOR = 2
   35 | _MXFP_SCALE_BLOCK_SIZE = 64
```

<a id="mega-moe-l36"></a>

### 原始第 36 行起

```text
   36 | _MXFP_SCALE_MULTIPLIER = 2
   37 | _TORCH_FLOAT8_E8M0FNU_DTYPE = getattr(torch, "float8_e8m0fnu", None)
   38 |
   39 |
```

<a id="mega-moe-l40"></a>

### 原始第 40 行起

```text
   40 | def _as_tensor_list(tensor_or_list: torch.Tensor | list[torch.Tensor] | None, name: str) -> list[torch.Tensor]:
   41 |     if tensor_or_list is None:
   42 |         raise ValueError(f"{name} is required for A5 MegaMoE.")
   43 |     if isinstance(tensor_or_list, list):
   44 |         if not tensor_or_list:
   45 |             raise ValueError(f"{name} cannot be an empty list for A5 MegaMoE.")
   46 |         return tensor_or_list
   47 |     return [tensor_or_list]
   48 |
   49 |
```

<a id="mega-moe-l50"></a>

### 原始第 50 行起

```text
   50 | def _view_mxfp_scales_as_e8m0(scales: list[torch.Tensor], name: str) -> list[torch.Tensor]:
   51 |     if _TORCH_FLOAT8_E8M0FNU_DTYPE is None:
   52 |         raise RuntimeError("A5 MegaMoE requires torch.float8_e8m0fnu to reinterpret MXFP weight scales.")
   53 |
   54 |     normalized_scales: list[torch.Tensor] = []
   55 |     for idx, scale in enumerate(scales):
```

<a id="mega-moe-l56"></a>

### 原始第 56 行起

```text
   56 |         if scale.dtype == _TORCH_FLOAT8_E8M0FNU_DTYPE:
   57 |             normalized_scales.append(scale)
   58 |         elif scale.dtype == torch.uint8:
   59 |             normalized_scales.append(scale.view(_TORCH_FLOAT8_E8M0FNU_DTYPE))
   60 |         else:
   61 |             raise RuntimeError(
   62 |                 f"A5 MegaMoE requires {name}[{idx}] to be FLOAT8_E8M0 weight scale, "
   63 |                 f"got dtype={scale.dtype}, shape={tuple(scale.shape)}."
   64 |             )
   65 |     return normalized_scales
   66 |
   67 |
```

<a id="mega-moe-l68"></a>

### 原始第 68 行起

```text
   68 | def _get_mega_moe_ops():
   69 |     try:
   70 |         from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
   71 |     except ImportError as exc:
   72 |         raise RuntimeError(
   73 |             "A5 MegaMoE requires cann_ops_transformer.ops. Install a package "
   74 |             "that provides mega_moe and get_symm_buffer_for_mega_moe."
   75 |         ) from exc
   76 |     return get_symm_buffer_for_mega_moe, mega_moe
   77 |
   78 |
   79 | @dataclass(frozen=True, slots=True)
```

<a id="mega-moe-l80"></a>

### 原始第 80 行起

```text
   80 | class _MegaMoEBufferKey:
   81 |     num_experts: int
   82 |     buffer_tokens_per_rank: int
   83 |     top_k: int
   84 |     hidden_size: int
   85 |     intermediate_hidden: int
   86 |     dispatch_quant_mode: int
   87 |     dispatch_quant_out_dtype: torch.dtype
   88 |
   89 |
   90 | @dataclass(frozen=True, slots=True)
```

<a id="mega-moe-l91"></a>

### 原始第 91 行起

```text
   91 | class _MegaMoESymmetricBufferState:
   92 |     key: _MegaMoEBufferKey
   93 |     buffer: Any
   94 |
   95 |
   96 | _MEGA_MOE_BUFFER_STATE_ATTR = "_mega_moe_symmetric_buffer_state"
   97 |
   98 |
```

<a id="mega-moe-l99"></a>

### 原始第 99 行起

```text
   99 | class MegaMoEBackend:
  100 |     """A5 MegaMoE wrapper for the logical FUSED_MC2 MoE path."""
  101 |
```

<a id="mega-moe-l102"></a>

### 原始第 102 行起

```text
  102 |     def __init__(self, moe_config: FusedMoEConfig):
  103 |         self.moe_config = moe_config
  104 |
  105 |     @staticmethod
```

<a id="mega-moe-l106"></a>

### 原始第 106 行起

```text
  106 |     def _validate_stacked_mxfp_layout(
  107 |         fused_experts_input: MoEFusedExpertsInput,
  108 |         w1: list[torch.Tensor],
  109 |         w2: list[torch.Tensor],
  110 |         w1_scale: list[torch.Tensor],
  111 |         w2_scale: list[torch.Tensor],
  112 |     ) -> int:
  113 |         tensors = {
  114 |             "w1": w1,
  115 |             "w2": w2,
  116 |             "w1_scale": w1_scale,
  117 |             "w2_scale": w2_scale,
  118 |         }
  119 |         for name, values in tensors.items():
  120 |             if len(values) != 1:
  121 |                 raise ValueError(
  122 |                     f"A5 MegaMoE requires {name} to contain one stacked tensor, got {len(values)} tensors."
  123 |                 )
  124 |             if not values[0].is_contiguous():
  125 |                 raise ValueError(
  126 |                     f"A5 MegaMoE requires contiguous {name}, got shape={tuple(values[0].shape)} "
  127 |                     f"and stride={values[0].stride()}."
  128 |                 )
  129 |
  130 |         weight1 = w1[0]
  131 |         weight2 = w2[0]
  132 |         scale1 = w1_scale[0]
  133 |         scale2 = w2_scale[0]
  134 |         if weight1.ndim != 3 or weight2.ndim != 3:
  135 |             raise ValueError(
  136 |                 "A5 MegaMoE expects stacked 3D expert weights, "
  137 |                 f"got w1={tuple(weight1.shape)} and w2={tuple(weight2.shape)}."
  138 |             )
  139 |         if scale1.ndim != 4 or scale2.ndim != 4:
  140 |             raise ValueError(
```

<a id="mega-moe-l141"></a>

### 原始第 141 行起

```text
  141 |                 "A5 MegaMoE expects stacked 4D MXFP scales, "
  142 |                 f"got w1_scale={tuple(scale1.shape)} and w2_scale={tuple(scale2.shape)}."
  143 |             )
  144 |
```

<a id="mega-moe-l145"></a>

### 原始第 145 行起

```text
  145 |         hidden_size = int(fused_experts_input.hidden_states.shape[-1])
  146 |         pack_factor = _FP4_PACK_FACTOR
  147 |         projected_hidden = int(weight1.shape[1])
  148 |         if projected_hidden % 2 != 0:
  149 |             raise ValueError(f"A5 MegaMoE expects w1.shape[1] to be even, got {projected_hidden}.")
  150 |         intermediate_hidden = projected_hidden // 2
  151 |         if hidden_size % pack_factor != 0 or intermediate_hidden % pack_factor != 0:
  152 |             raise ValueError(
  153 |                 "A5 MegaMoE FP4 dimensions must be divisible by the packing factor: "
  154 |                 f"hidden_size={hidden_size}, intermediate_hidden={intermediate_hidden}, "
  155 |                 f"pack_factor={pack_factor}."
  156 |             )
  157 |
```

<a id="mega-moe-l158"></a>

### 原始第 158 行起

```text
  158 |         expected_shapes = {
  159 |             "w1": (weight1.shape[0], projected_hidden, hidden_size // pack_factor),
  160 |             "w2": (weight1.shape[0], hidden_size, intermediate_hidden // pack_factor),
  161 |             "w1_scale": (
  162 |                 weight1.shape[0],
  163 |                 projected_hidden,
  164 |                 (hidden_size + _MXFP_SCALE_BLOCK_SIZE - 1) // _MXFP_SCALE_BLOCK_SIZE,
  165 |                 _MXFP_SCALE_MULTIPLIER,
  166 |             ),
  167 |             "w2_scale": (
  168 |                 weight1.shape[0],
  169 |                 hidden_size,
  170 |                 (intermediate_hidden + _MXFP_SCALE_BLOCK_SIZE - 1) // _MXFP_SCALE_BLOCK_SIZE,
  171 |                 _MXFP_SCALE_MULTIPLIER,
  172 |             ),
  173 |         }
  174 |         actual_shapes = {
  175 |             "w1": tuple(weight1.shape),
  176 |             "w2": tuple(weight2.shape),
  177 |             "w1_scale": tuple(scale1.shape),
  178 |             "w2_scale": tuple(scale2.shape),
  179 |         }
  180 |         if actual_shapes != expected_shapes:
  181 |             raise ValueError(
  182 |                 "A5 MegaMoE received an incompatible stacked MXFP layout: "
  183 |                 f"actual={actual_shapes}, expected={expected_shapes}."
  184 |             )
  185 |         return intermediate_hidden
  186 |
```

<a id="mega-moe-l187"></a>

### 原始第 187 行起

```text
  187 |     def _make_buffer_key(
  188 |         self,
  189 |         fused_experts_input: MoEFusedExpertsInput,
  190 |         intermediate_hidden: int,
  191 |         *,
  192 |         buffer_tokens_per_rank: int,
  193 |     ) -> _MegaMoEBufferKey:
  194 |         if fused_experts_input.hidden_states.shape[0] > buffer_tokens_per_rank:
  195 |             raise ValueError(
  196 |                 "A5 MegaMoE input exceeds the symmetric buffer token capacity: "
  197 |                 f"num_tokens={fused_experts_input.hidden_states.shape[0]}, "
  198 |                 f"mega_moe_buffer_tokens_per_rank={buffer_tokens_per_rank}."
  199 |             )
  200 |         act_dtype = torch.float8_e4m3fn
  201 |         if fused_experts_input.quant.mxfp is not None and fused_experts_input.quant.mxfp.act_quant_type is not None:
  202 |             act_dtype = fused_experts_input.quant.mxfp.act_quant_type
  203 |         return _MegaMoEBufferKey(
  204 |             num_experts=self.moe_config.num_experts,
  205 |             buffer_tokens_per_rank=buffer_tokens_per_rank,
  206 |             top_k=self.moe_config.experts_per_token,
  207 |             hidden_size=int(fused_experts_input.hidden_states.shape[-1]),
  208 |             intermediate_hidden=intermediate_hidden,
  209 |             dispatch_quant_mode=4,
  210 |             dispatch_quant_out_dtype=act_dtype,
  211 |         )
  212 |
```

<a id="mega-moe-l213"></a>

### 原始第 213 行起

```text
  213 |     def _get_sym_buffer(self, fused_experts_input: MoEFusedExpertsInput, intermediate_hidden: int):
  214 |         vllm_config = get_current_vllm_config()
  215 |         if vllm_config is None:
  216 |             raise RuntimeError("A5 MegaMoE requires a current vLLM configuration.")
  217 |         buffer_tokens_per_rank = get_a5_mega_moe_buffer_tokens_per_rank(vllm_config)
  218 |         key = self._make_buffer_key(
  219 |             fused_experts_input,
  220 |             intermediate_hidden,
  221 |             buffer_tokens_per_rank=buffer_tokens_per_rank,
  222 |         )
  223 |         mega_moe_group = get_mega_moe_group()
  224 |         state: _MegaMoESymmetricBufferState | None = getattr(
  225 |             mega_moe_group,
  226 |             _MEGA_MOE_BUFFER_STATE_ATTR,
  227 |             None,
  228 |         )
```

<a id="mega-moe-l229"></a>

### 原始第 229 行起

```text
  229 |         if state is not None:
  230 |             if state.key != key:
  231 |                 raise RuntimeError(
  232 |                     "A5 MegaMoE symmetric buffer is process-wide and cannot be replaced during inference: "
  233 |                     f"initialized={state.key}, requested={key}."
  234 |                 )
  235 |             return state.buffer
  236 |
  237 |         get_symm_buffer_for_mega_moe, _ = _get_mega_moe_ops()
```

<a id="mega-moe-l238"></a>

### 原始第 238 行起

```text
  238 |         buffer = get_symm_buffer_for_mega_moe(
  239 |             mega_moe_group.device_group,
  240 |             num_experts=key.num_experts,
  241 |             num_max_tokens_per_rank=key.buffer_tokens_per_rank,
  242 |             num_topk=key.top_k,
  243 |             hidden=key.hidden_size,
  244 |             intermediate_hidden=key.intermediate_hidden,
  245 |             dispatch_quant_mode=key.dispatch_quant_mode,
  246 |             dispatch_quant_out_dtype=key.dispatch_quant_out_dtype,
  247 |         )
  248 |         setattr(
  249 |             mega_moe_group,
  250 |             _MEGA_MOE_BUFFER_STATE_ATTR,
  251 |             _MegaMoESymmetricBufferState(key=key, buffer=buffer),
  252 |         )
  253 |         return buffer
  254 |
  255 |     @staticmethod
```

<a id="mega-moe-l256"></a>

### 原始第 256 行起

```text
  256 |     def _normalize_activation(activation: Any) -> str:
  257 |         activation_name = activation if isinstance(activation, str) else getattr(activation, "name", str(activation))
  258 |         activation_lower = activation_name.lower().removeprefix("moeactivation.")
  259 |         if activation_lower in ("silu", "swiglu"):
  260 |             return "swiglu"
  261 |         raise ValueError(
  262 |             f"A5 MegaMoE does not support activation={activation!r} without changing its semantics. "
  263 |             "Only SILU/SwiGLU is currently supported."
  264 |         )
  265 |
```

<a id="mega-moe-l266"></a>

### 原始第 266 行起

```text
  266 |     def fused_experts(self, fused_experts_input: MoEFusedExpertsInput) -> tuple[torch.Tensor, torch.Tensor | None]:
  267 |         if fused_experts_input.quant.quant_type not in _MEGA_MOE_SUPPORTED_QUANTS:
  268 |             raise RuntimeError(
  269 |                 f"A5 MegaMoE only supports MXFP MoE quantization, got {fused_experts_input.quant.quant_type}."
  270 |             )
  271 |         if not fused_experts_input.quant.is_mxfp:
  272 |             raise RuntimeError("A5 MegaMoE requires MXFP quant parameters.")
  273 |         if fused_experts_input.dynamic_eplb:
  274 |             raise RuntimeError("A5 MegaMoE does not support dynamic EPLB expert weight lists.")
  275 |         if fused_experts_input.routing.global_redundant_expert_num:
  276 |             raise RuntimeError("A5 MegaMoE does not support redundant physical experts.")
  277 |
  278 |         w1 = _as_tensor_list(fused_experts_input.weights.w1, "w1")
  279 |         w2 = _as_tensor_list(fused_experts_input.weights.w2, "w2")
  280 |         w1_scale = _view_mxfp_scales_as_e8m0(
  281 |             _as_tensor_list(fused_experts_input.weights.w1_scale, "w1_scale"),
  282 |             "w1_scale",
  283 |         )
  284 |         w2_scale = _view_mxfp_scales_as_e8m0(
  285 |             _as_tensor_list(fused_experts_input.weights.w2_scale, "w2_scale"),
  286 |             "w2_scale",
  287 |         )
  288 |         intermediate_hidden = self._validate_stacked_mxfp_layout(
  289 |             fused_experts_input,
  290 |             w1,
  291 |             w2,
  292 |             w1_scale,
  293 |             w2_scale,
  294 |         )
  295 |         activation = self._normalize_activation(fused_experts_input.activation)
  296 |         swiglu_limit = self.moe_config.swiglu_limit
  297 |         activation_clamp = None if swiglu_limit is None or swiglu_limit <= 0 else float(swiglu_limit)
  298 |         mxfp = fused_experts_input.quant.mxfp
  299 |         weight_type = None if mxfp is None else mxfp.weight_quant_type
  300 |
```

<a id="mega-moe-l301"></a>

### 原始第 301 行起

```text
  301 |         _, mega_moe = _get_mega_moe_ops()
```

<a id="mega-moe-l302"></a>

### 原始第 302 行起

```text
  302 |         mega_moe_kwargs = {
  303 |             "x": fused_experts_input.hidden_states,
  304 |             "topk_ids": fused_experts_input.topk_ids.to(torch.int32),
```

<a id="mega-moe-l305"></a>

### 原始第 305 行起

```text
  305 |             "topk_weights": fused_experts_input.topk_weights,
  306 |             "l1_weights": w1,
  307 |             "l1_weights_sf": w1_scale,
  308 |             "l2_weights": w2,
  309 |             "l2_weights_sf": w2_scale,
  310 |             "sym_buffer": self._get_sym_buffer(fused_experts_input, intermediate_hidden),
  311 |             "activation": activation,
  312 |             "activation_clamp": activation_clamp,
  313 |         }
  314 |         if weight_type is not None:
  315 |             mega_moe_kwargs["weight1_type"] = weight_type
  316 |             mega_moe_kwargs["weight2_type"] = weight_type
  317 |
```

<a id="mega-moe-l318"></a>

### 原始第 318 行起

```text
  318 |         output, expert_tokens = mega_moe(**mega_moe_kwargs)
  319 |         return output, expert_tokens
  320 |
  321 |
  322 | __all__ = ["MegaMoEBackend"]
```

<a id="moe-comm"></a>

## vllm_ascend/ops/fused_moe/moe_comm_method.py

原始完整文件共 517 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="moe-comm-l27"></a>

### 原始第 27 行起

```text
   27 | from vllm_ascend.distributed.parallel_state import get_mc2_group
   28 | from vllm_ascend.ops.fused_moe import moe_utils
   29 | from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEFusedExpertsInput
   30 | from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput, build_mlp_compute_input
   31 | from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
   32 | from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import build_token_dispatch_input
   33 | from vllm_ascend.ops.fused_moe.mega_moe import MegaMoEBackend
   34 | from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
   35 | from vllm_ascend.ops.fused_moe.prepare_finalize import (
   36 |     PrepareAndFinalize,
   37 |     PrepareAndFinalizeWithAll2All,
   38 |     PrepareAndFinalizeWithAllGather,
   39 |     PrepareAndFinalizeWithMC2,
   40 |     PrepareAndFinalizeWithMegaMoE,
   41 | )
   42 | from vllm_ascend.ops.fused_moe.token_dispatcher import (
   43 |     MoETokenDispatcher,
   44 |     TokenDispatcherWithAll2AllV,
   45 |     TokenDispatcherWithAllGather,
   46 |     TokenDispatcherWithMC2,
   47 | )
   48 | from vllm_ascend.quantization.quant_type import QuantType
   49 | from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
   50 |
   51 | _MoECommMethods: dict[MoECommType | None, MoECommMethod] = {}
   52 |
   53 |
```

<a id="moe-comm-l54"></a>

### 原始第 54 行起

```text
   54 | def get_moe_comm_method(moe_comm_type: MoECommType | None) -> MoECommMethod | None:
   55 |     return _MoECommMethods.get(moe_comm_type)
```

省略原始第 56 至 246 行；下一块恢复原始行号。

<a id="moe-comm-l247"></a>

### 原始第 247 行起

```text
  247 |         )
  248 |
```

<a id="moe-comm-l249"></a>

### 原始第 249 行起

```text
  249 |     def _get_prepare_finalize(self):
  250 |         return PrepareAndFinalizeWithAll2All(self.moe_config)
  251 |
  252 |
```

<a id="moe-comm-l253"></a>

### 原始第 253 行起

```text
  253 | class _MegaMoEBypassTokenDispatcher(MoETokenDispatcher[None]):
```

<a id="moe-comm-l254"></a>

### 原始第 254 行起

```text
  254 |     def token_dispatch(self, token_dispatch_input):
  255 |         raise RuntimeError("A5 MegaMoE bypasses token_dispatch; use MegaMoEBackend.fused_experts instead.")
  256 |
```

<a id="moe-comm-l257"></a>

### 原始第 257 行起

```text
  257 |     def token_combine(self, hidden_states, combine_metadata, bias=None):
  258 |         raise RuntimeError("A5 MegaMoE bypasses token_combine; use MegaMoEBackend.fused_experts instead.")
  259 |
  260 |
```

<a id="moe-comm-l261"></a>

### 原始第 261 行起

```text
  261 | class FusedMC2CommImpl(MoECommMethod):
  262 |     """This implementation is for the scenarios listed below:
  263 |     1. `enable_expert_parallel=True`.
  264 |     2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
  265 |     3. `enable_expert_parallel=False` is not supported.
  266 |
  267 |     This implementation uses the MC2 communication method, which is optimized for
  268 |     Communication and Computation parallelism on Ascend devices.
  269 |     """
  270 |
```

<a id="moe-comm-l271"></a>

### 原始第 271 行起

```text
  271 |     def __init__(self, moe_config):
  272 |         super().__init__(moe_config)
```

<a id="moe-comm-l273"></a>

### 原始第 273 行起

```text
  273 |         self._is_a5_backend = get_ascend_device_type() == AscendDeviceType.A5
  274 |         self._mega_moe_backend = MegaMoEBackend(moe_config) if self._is_a5_backend else None
  275 |         self.enable_fused_mc2 = get_ascend_config().enable_fused_mc2
  276 |         if self.enable_fused_mc2 == 1 and not self._is_a5_backend and is_mega_moe_supported():
  277 |             self.mega_moe_symm_buffer = None
  278 |             self.get_symm_buffer_for_mega_moe, self.mega_moe = moe_utils.load_cann_mega_moe_ops()
  279 |         if self.enable_fused_mc2 == 1 and not self._is_a5_backend:
  280 |             self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
  281 |         else:
  282 |             self.expert_token_nums = None
  283 |
  284 |         self.swiglu_limit = 0.0 if moe_config.swiglu_limit is None else moe_config.swiglu_limit
  285 |         self.swiglu_alpha = 1.0 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha
  286 |         self.swiglu_beta = 0.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta
  287 |
```

<a id="moe-comm-l288"></a>

### 原始第 288 行起

```text
  288 |     def pad_and_split_input_ids(self, input_ids):
  289 |         return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]
  290 |
```

<a id="moe-comm-l291"></a>

### 原始第 291 行起

```text
  291 |     def _get_token_dispatcher(self):
  292 |         if get_ascend_device_type() == AscendDeviceType.A5:
  293 |             return _MegaMoEBypassTokenDispatcher()
  294 |         return TokenDispatcherWithMC2()
  295 |
```

<a id="moe-comm-l296"></a>

### 原始第 296 行起

```text
  296 |     def _get_prepare_finalize(self):
  297 |         if get_ascend_device_type() == AscendDeviceType.A5:
```

<a id="moe-comm-l298"></a>

### 原始第 298 行起

```text
  298 |             return PrepareAndFinalizeWithMegaMoE(self.moe_config)
  299 |         return PrepareAndFinalizeWithMC2(self.moe_config)
  300 |
```

<a id="moe-comm-l301"></a>

### 原始第 301 行起

```text
  301 |     def _init_mega_moe_symm_buffer(
  302 |         self,
  303 |         dispatch_quant_mode: int = 0,
  304 |         dispatch_quant_out_dtype: torch.dtype | None = None,
  305 |         *,
  306 |         is_decode_only_node: bool,
  307 |     ):
  308 |         # FusedMC2CommImpl always builds a TokenDispatcherWithMC2 (see
  309 |         # setup_moe_comm_method), which is where global_bs / ep_world_size live.
  310 |         # Assert it so mypy resolves those attributes off the base dispatcher.
```

省略原始第 311 至 462 行；下一块恢复原始行号。

<a id="moe-comm-l463"></a>

### 原始第 463 行起

```text
  463 |         return out, expert_tokens
  464 |
```

<a id="moe-comm-l465"></a>

### 原始第 465 行起

```text
  465 |     def fused_experts(
  466 |         self,
  467 |         fused_experts_input: MoEFusedExpertsInput,
  468 |     ):
```

<a id="moe-comm-l469"></a>

### 原始第 469 行起

```text
  469 |         if self._is_a5_backend:
  470 |             assert self._mega_moe_backend is not None
  471 |             out, expert_tokens = self._mega_moe_backend.fused_experts(fused_experts_input)
  472 |             return FusedExpertsResult(
  473 |                 routed_out=out,
  474 |                 expert_tokens=expert_tokens,
  475 |             )
  476 |
  477 |         assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
  478 |             "token_dispatcher must be an instance of TokenDispatcherWithMC2."
  479 |         )
  480 |
  481 |         expert_tokens = None
  482 |         if get_ascend_config().enable_fused_mc2 == 1:
```

<a id="prepare-finalize"></a>

## vllm_ascend/ops/fused_moe/prepare_finalize.py

原始完整文件共 637 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="prepare-finalize-l17"></a>

### 原始第 17 行起

```text
   17 | from abc import ABC, abstractmethod
   18 |
   19 | import torch
   20 | import torch.distributed as dist
   21 | import torch.nn as nn
   22 | import torch_npu
   23 | from vllm.config import get_current_vllm_config
   24 | from vllm.distributed.parallel_state import (
   25 |     get_dp_group,
   26 |     get_pcp_group,
   27 |     get_tensor_model_parallel_rank,
   28 |     get_tensor_model_parallel_world_size,
   29 | )
   30 | from vllm.model_executor.layers.fused_moe import FusedMoEConfig
   31 | from vllm.model_executor.models.utils import sequence_parallel_chunk
   32 |
   33 | from vllm_ascend.ascend_forward_context import _EXTRA_CTX, get_a5_mega_moe_buffer_tokens_per_rank
   34 | from vllm_ascend.lora.fused_moe import prepare_lora_indices
   35 | from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
   36 | from vllm_ascend.quantization.quant_type import QuantType
   37 | from vllm_ascend.quantization.utils import get_dynamic_mx_quant_scale_alg
   38 |
   39 |
```

省略原始第 40 至 344 行；下一块恢复原始行号。

<a id="prepare-finalize-l345"></a>

### 原始第 345 行起

```text
  345 |             if self.tp_size > 1:
  346 |                 input_ids = torch.tensor_split(input_ids, self.tp_size, dim=0)
  347 |                 input_ids = input_ids[self.tp_rank]
  348 |         return input_ids
  349 |
  350 |
```

<a id="prepare-finalize-l351"></a>

### 原始第 351 行起

```text
  351 | class PrepareAndFinalizeWithMegaMoE(PrepareAndFinalize):
  352 |     """Keep each rank's local batch size while aligning A5 collective inputs."""
  353 |
```

<a id="prepare-finalize-l354"></a>

### 原始第 354 行起

```text
  354 |     def prepare(
  355 |         self,
  356 |         hidden_states: torch.Tensor,
  357 |         router_logits: torch.Tensor,
  358 |         replace_allreduce: bool = False,
  359 |         quant_type=QuantType.NONE,
  360 |     ) -> MoEPrepareOutput:
```

<a id="prepare-finalize-l361"></a>

### 原始第 361 行起

```text
  361 |         self.num_tokens = hidden_states.shape[0]
```

<a id="prepare-finalize-l362"></a>

### 原始第 362 行起

```text
  362 |         vllm_config = get_current_vllm_config()
```

<a id="prepare-finalize-l363"></a>

### 原始第 363 行起

```text
  363 |         if vllm_config is None:
  364 |             raise RuntimeError("A5 MegaMoE requires a current vLLM configuration.")
```

<a id="prepare-finalize-l365"></a>

### 原始第 365 行起

```text
  365 |         buffer_tokens_per_rank = get_a5_mega_moe_buffer_tokens_per_rank(vllm_config)
```

<a id="prepare-finalize-l366"></a>

### 原始第 366 行起

```text
  366 |         max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp
```

<a id="prepare-finalize-l367"></a>

### 原始第 367 行起

```text
  367 |         if max_tokens_across_dp is None:
  368 |             max_tokens_across_dp = self.num_tokens
```

<a id="prepare-finalize-l369"></a>

### 原始第 369 行起

```text
  369 |         if max_tokens_across_dp < self.num_tokens:
  370 |             raise ValueError(
  371 |                 "MegaMoE synchronized token count cannot be smaller than the local token count: "
  372 |                 f"max_across_dp={max_tokens_across_dp}, local={self.num_tokens}."
  373 |             )
```

<a id="prepare-finalize-l374"></a>

### 原始第 374 行起

```text
  374 |         if max_tokens_across_dp > buffer_tokens_per_rank:
  375 |             raise ValueError(
  376 |                 "MegaMoE input exceeds the symmetric buffer token capacity: "
  377 |                 f"capacity={buffer_tokens_per_rank}, max_across_dp={max_tokens_across_dp}, "
  378 |                 f"local={self.num_tokens}."
  379 |             )
  380 |
```

<a id="prepare-finalize-l381"></a>

### 原始第 381 行起

```text
  381 |         self.padded_num_tokens = max_tokens_across_dp
```

<a id="prepare-finalize-l382"></a>

### 原始第 382 行起

```text
  382 |         pad_size = self.padded_num_tokens - self.num_tokens
```

<a id="prepare-finalize-l383"></a>

### 原始第 383 行起

```text
  383 |         if pad_size > 0:
```

<a id="prepare-finalize-l384"></a>

### 原始第 384 行起

```text
  384 |             hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
```

<a id="prepare-finalize-l385"></a>

### 原始第 385 行起

```text
  385 |             router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
```

<a id="prepare-finalize-l386"></a>

### 原始第 386 行起

```text
  386 |         self._mega_moe_input_prepared = True
  387 |
```

<a id="prepare-finalize-l388"></a>

### 原始第 388 行起

```text
  388 |         return MoEPrepareOutput(
  389 |             hidden_states=hidden_states,
  390 |             router_logits=router_logits,
  391 |             mc2_mask=None,
  392 |             padded_hidden_states_shape=hidden_states.shape,
  393 |             pertoken_scale=None,
  394 |         )
  395 |
```

<a id="prepare-finalize-l396"></a>

### 原始第 396 行起

```text
  396 |     def pad_and_split_input_ids(self, input_ids):
```

<a id="prepare-finalize-l397"></a>

### 原始第 397 行起

```text
  397 |         if not getattr(self, "_mega_moe_input_prepared", False):
  398 |             return input_ids
  399 |         pad_size = self.padded_num_tokens - self.num_tokens
  400 |         if pad_size > 0:
```

<a id="prepare-finalize-l401"></a>

### 原始第 401 行起

```text
  401 |             input_ids = nn.functional.pad(input_ids, (0, pad_size))
  402 |         return input_ids
  403 |
```

<a id="prepare-finalize-l404"></a>

### 原始第 404 行起

```text
  404 |     def finalize(
  405 |         self,
  406 |         hidden_states: torch.Tensor,
  407 |         reduce_results: bool,
  408 |         padded_hidden_states_shape: torch.Size | None = None,
  409 |     ) -> torch.Tensor:
```

<a id="prepare-finalize-l410"></a>

### 原始第 410 行起

```text
  410 |         self._mega_moe_input_prepared = False
```

<a id="prepare-finalize-l411"></a>

### 原始第 411 行起

```text
  411 |         return hidden_states[: self.num_tokens]
  412 |
  413 |
```

<a id="prepare-finalize-l414"></a>

### 原始第 414 行起

```text
  414 | class PrepareAndFinalizeWithAllGather(PrepareAndFinalize):
  415 |     """
  416 |     MoE communication strategy using All-Gather + Reduce-Scatter on EP group.
  417 |     There are two sets of prepare and finalize:
  418 |     1. _prepare_with_dp_group/_finalize_with_dp_group: When sequence parallelism is not enabled,
  419 |     we gather inputs across DP ranks before MoE, scatter outputs after.
```

<a id="routed-experts"></a>

## vllm_ascend/ops/fused_moe/routed_experts.py

原始完整文件共 606 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="routed-experts-l27"></a>

### 原始第 27 行起

```text
   27 | from vllm.model_executor.layers.fused_moe import FusedMoERouter, RoutedExperts, SharedExperts
   28 | from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
   29 | from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
   30 | from vllm.model_executor.utils import replace_parameter
   31 |
   32 | from vllm_ascend.ascend_config import get_ascend_config
   33 | from vllm_ascend.ascend_forward_context import (
   34 |     _EXTRA_CTX,
   35 |     MoECommType,
   36 |     cache_a5_moe_quant_type,
   37 |     use_cann_megamoe,
   38 | )
   39 | from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
   40 | from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
   41 | from vllm_ascend.lora.fused_moe import sync_lora_context
   42 | from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
   43 | from vllm_ascend.ops.fused_moe.force_eplb import get_force_eplb_topk
   44 | from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult
```

省略原始第 45 至 253 行；下一块恢复原始行号。

<a id="routed-experts-l254"></a>

### 原始第 254 行起

```text
  254 |         self.global_redundant_expert_num: int = 0
  255 |         self.ascend_pertoken_scale: torch.Tensor | None = None
  256 |         self.ascend_mc2_mask: torch.Tensor | None = None
  257 |         if not self._use_v2_model_runner:
  258 |             self.init_eplb(n_shared_experts)
  259 |         self.return_with_event = False
```

<a id="routed-experts-l260"></a>

### 原始第 260 行起

```text
  260 |         if self.quant_type != QuantType.NONE:
  261 |             cache_a5_moe_quant_type(
  262 |                 vllm_config,
  263 |                 self.quant_type,
  264 |             )
  265 |
```

<a id="routed-experts-l266"></a>

### 原始第 266 行起

```text
  266 |     def get_expert_weights(self) -> Iterable[torch.Tensor]:
  267 |         try:
  268 |             get_weight_views = self.quant_method.get_eplb_weight_views
  269 |         except AttributeError as exc:
  270 |             raise NotImplementedError(
```

<a id="router"></a>

## vllm_ascend/ops/fused_moe/router/fused_topk_router.py

原始完整文件共 313 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="router-l23"></a>

### 原始第 23 行起

```text
   23 | from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
   24 | from vllm_ascend.device.device_op import DeviceOperator
   25 | from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
   26 |
   27 | DEEPSEEK_V4_IMAGE_SENTINEL_BASE_ID = 129257
   28 | DEEPSEEK_V4_IMAGE_SENTINEL_COUNT = 5
   29 | _VISION_GATING_OP_NAME = "npu_moe_gating_top_k"
   30 | _VISION_GATING_REQUIRED_ARGUMENTS = frozenset(("image_bias", "image_token_mask"))
   31 |
   32 |
```

<a id="router-l33"></a>

### 原始第 33 行起

```text
   33 | def _npu_moe_gating_vision_op() -> Callable | None:
   34 |     """Return the optional fused op only when its schema supports vision routing."""
   35 |     op = getattr(torch.ops._C_ascend, _VISION_GATING_OP_NAME, None)
   36 |     if op is None:
   37 |         return None
   38 |
   39 |     schemas_by_overload = getattr(op, "_schemas", None) or {}
   40 |     schemas = list(schemas_by_overload.values())
   41 |     schema = getattr(op, "_schema", None)
   42 |     if schema is not None:
   43 |         schemas.append(schema)
   44 |     default_overload = getattr(op, "default", None)
   45 |     default_schema = getattr(default_overload, "_schema", None)
   46 |     if default_schema is not None:
   47 |         schemas.append(default_schema)
   48 |
   49 |     for candidate in schemas:
   50 |         argument_names = {argument.name for argument in candidate.arguments}
   51 |         if argument_names >= _VISION_GATING_REQUIRED_ARGUMENTS:
   52 |             return op
   53 |     return None
   54 |
   55 |
```

<a id="router-l56"></a>

### 原始第 56 行起

```text
   56 | def select_deepseek_v4_vision_experts_with_fusion_op(
   57 |     vision_gating_op: Callable,
   58 |     router_logits: torch.Tensor,
   59 |     input_ids: torch.Tensor,
   60 |     tid2eid: torch.Tensor | None,
   61 |     bias_vl: torch.Tensor,
   62 |     text_bias: torch.Tensor | None,
   63 |     top_k: int,
   64 |     renormalize: bool,
   65 |     k_group: int,
   66 |     group_count: int,
   67 |     routed_scaling_factor: float = 1.0,
   68 |     eps: float = 1e-20,
   69 |     image_sentinel_lo: int = DEEPSEEK_V4_IMAGE_SENTINEL_BASE_ID,
   70 | ) -> tuple[torch.Tensor, torch.Tensor]:
   71 |     """Select DeepSeek V4 vision experts with the optional fused operator."""
   72 |     image_hi = image_sentinel_lo + DEEPSEEK_V4_IMAGE_SENTINEL_COUNT
   73 |     image_token_mask = (input_ids >= image_sentinel_lo) & (input_ids < image_hi)
   74 |     topk_weights, topk_ids, _ = vision_gating_op(
   75 |         router_logits,
   76 |         k=top_k,
   77 |         bias=text_bias,
   78 |         input_ids=input_ids,
   79 |         tid2eid=tid2eid,
   80 |         image_bias=bias_vl,
   81 |         image_token_mask=image_token_mask,
   82 |         k_group=k_group,
   83 |         group_count=group_count,
   84 |         routed_scaling_factor=1.0,
   85 |         eps=eps,
   86 |         group_select_mode=1,
   87 |         renorm=0,
   88 |         norm_type=2,
   89 |         out_flag=False,
   90 |     )
```

<a id="router-l91"></a>

### 原始第 91 行起

```text
   91 |     if renormalize:
   92 |         topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
   93 |             torch.finfo(topk_weights.dtype).tiny
   94 |         )
   95 |     if routed_scaling_factor != 1.0:
   96 |         topk_weights = topk_weights * routed_scaling_factor
   97 |     return topk_weights, topk_ids
   98 |
   99 |
```

<a id="router-l100"></a>

### 原始第 100 行起

```text
  100 | def select_deepseek_v4_vision_experts(
  101 |     router_logits: torch.Tensor,
  102 |     input_ids: torch.Tensor,
  103 |     tid2eid: torch.Tensor | None,
```

省略原始第 104 至 122 行；下一块恢复原始行号。

<a id="router-l123"></a>

### 原始第 123 行起

```text
  123 |     else:
  124 |         lookup_ids = torch.where(image_mask, 0, input_ids)
  125 |         text_ids = tid2eid[lookup_ids].to(torch.int64)
  126 |         topk_ids = torch.where(image_mask.unsqueeze(-1), dynamic_ids, text_ids)
  127 |     topk_weights = scores.gather(1, topk_ids)
  128 |     if renormalize:
  129 |         topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
  130 |             torch.finfo(topk_weights.dtype).tiny
  131 |         )
  132 |     if routed_scaling_factor != 1.0:
  133 |         topk_weights = topk_weights * routed_scaling_factor
  134 |     return topk_weights, topk_ids
  135 |
  136 |
```

<a id="router-l137"></a>

### 原始第 137 行起

```text
  137 | class AscendFusedTopKRouter(AscendGroupedTopKRouter):
```

省略原始第 138 至 171 行；下一块恢复原始行号。

<a id="router-l172"></a>

### 原始第 172 行起

```text
  172 |         self.routed_scaling_factor = routed_scaling_factor
  173 |         self.e_score_correction_bias = e_score_correction_bias
  174 |         self.num_logical_experts = num_logical_experts if num_logical_experts is not None else global_num_experts
  175 |         self.tid2eid = tid2eid
  176 |         self.bias_vl = bias_vl
  177 |         self.image_sentinel_lo = image_sentinel_lo
  178 |         self.vision_gating_op = _npu_moe_gating_vision_op() if bias_vl is not None else None
  179 |
```

<a id="router-l180"></a>

### 原始第 180 行起

```text
  180 |     def is_fused_supported(
  181 |         self,
  182 |         hidden_states: torch.Tensor,
  183 |     ) -> bool:
  184 |         topk_group = self.topk_group if self.topk_group is not None else 1
```

省略原始第 185 至 197 行；下一块恢复原始行号。

<a id="router-l198"></a>

### 原始第 198 行起

```text
  198 |     def _compute_routing(
  199 |         self,
  200 |         hidden_states: torch.Tensor,
  201 |         router_logits: torch.Tensor,
  202 |         indices_type: torch.dtype | None,
  203 |         *,
  204 |         input_ids: torch.Tensor | None = None,
  205 |     ) -> tuple[torch.Tensor, torch.Tensor]:
  206 |         if self.bias_vl is None and not self.is_fused_supported(hidden_states):
  207 |             return super()._compute_routing(
  208 |                 hidden_states=hidden_states,
  209 |                 router_logits=router_logits,
  210 |                 indices_type=indices_type,
  211 |                 input_ids=input_ids,
  212 |             )
  213 |
  214 |         topk_group = self.topk_group if self.topk_group is not None else 1
  215 |         num_expert_group = self.num_expert_group if self.num_expert_group is not None else 1
  216 |         renorm = int(self.renormalize)
  217 |         if self.scoring_func == "sqrtsoftplus":
```

<a id="router-l218"></a>

### 原始第 218 行起

```text
  218 |             if self.tid2eid is not None or self.bias_vl is not None:
  219 |                 if input_ids is None:
  220 |                     raise ValueError(
  221 |                         "DeepSeek V4 hash MoE routing requires input_ids; vision routing requires it as well."
  222 |                     )
  223 |                 input_ids = input_ids.to(torch.int64)
```

<a id="router-l224"></a>

### 原始第 224 行起

```text
  224 |                 tid2eid_ones = self.tid2eid.to(torch.int32) if self.tid2eid is not None else None
  225 |                 if _EXTRA_CTX.moe_comm_type == MoECommType.ALLGATHER:
  226 |                     prepare_finalize = _EXTRA_CTX.moe_comm_method.prepare_finalize
  227 |                     input_ids = prepare_finalize.all_gather_input_id_with_dp_group(input_ids)
  228 |                 else:
```

<a id="router-l229"></a>

### 原始第 229 行起

```text
  229 |                     input_ids = _EXTRA_CTX.moe_comm_method.pad_and_split_input_ids(input_ids)
  230 |                 if _EXTRA_CTX.moe_comm_type != MoECommType.ALLGATHER and input_ids.numel() != router_logits.shape[0]:
  231 |                     # Native MoE SP chunks hidden states before MC2/All2All,
  232 |                     # while their replace-allreduce paths retain full token
  233 |                     # ids. Apply the identical TP chunk only when communication
  234 |                     # has not already aligned ids with local router rows.
  235 |                     input_ids = sequence_parallel_chunk(input_ids.reshape(-1, 1)).reshape(-1)
```

<a id="router-l236"></a>

### 原始第 236 行起

```text
  236 |                 input_ids = torch.where(input_ids == -1, 0, input_ids)
  237 |             else:
  238 |                 input_ids = None
  239 |                 tid2eid_ones = None
  240 |             bias_vl = self.bias_vl
  241 |             if bias_vl is not None and bias_vl.dtype != router_logits.dtype:
  242 |                 bias_vl = bias_vl.to(router_logits.dtype)
  243 |             text_bias = self.e_score_correction_bias
  244 |             if text_bias is not None and text_bias.dtype != router_logits.dtype:
  245 |                 text_bias = text_bias.to(router_logits.dtype)
  246 |             if bias_vl is not None:
  247 |                 assert input_ids is not None
  248 |                 if self.vision_gating_op is not None:
```

<a id="router-l249"></a>

### 原始第 249 行起

```text
  249 |                     topk_weights, topk_ids = select_deepseek_v4_vision_experts_with_fusion_op(
  250 |                         vision_gating_op=self.vision_gating_op,
  251 |                         router_logits=router_logits,
  252 |                         input_ids=input_ids,
  253 |                         tid2eid=tid2eid_ones,
  254 |                         bias_vl=bias_vl,
  255 |                         text_bias=text_bias,
  256 |                         top_k=self.top_k,
  257 |                         renormalize=self.renormalize,
  258 |                         k_group=topk_group,
  259 |                         group_count=num_expert_group,
  260 |                         routed_scaling_factor=self.routed_scaling_factor,
  261 |                         image_sentinel_lo=self.image_sentinel_lo,
  262 |                     )
  263 |                 else:
  264 |                     topk_weights, topk_ids = select_deepseek_v4_vision_experts(
  265 |                         router_logits=router_logits,
  266 |                         input_ids=input_ids,
  267 |                         tid2eid=tid2eid_ones,
  268 |                         bias_vl=bias_vl,
  269 |                         text_bias=text_bias,
  270 |                         top_k=self.top_k,
  271 |                         renormalize=self.renormalize,
  272 |                         routed_scaling_factor=self.routed_scaling_factor,
  273 |                         image_sentinel_lo=self.image_sentinel_lo,
  274 |                     )
  275 |                 return topk_weights.to(torch.float32), topk_ids.to(
  276 |                     torch.int32 if indices_type is None else indices_type
  277 |                 )
```

<a id="router-l278"></a>

### 原始第 278 行起

```text
  278 |             topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
  279 |                 x=router_logits,
  280 |                 k=self.top_k,
  281 |                 bias=text_bias,
  282 |                 input_ids=input_ids,
  283 |                 tid2eid=tid2eid_ones,
  284 |                 k_group=topk_group,
  285 |                 group_count=num_expert_group,
  286 |                 routed_scaling_factor=self.routed_scaling_factor,
  287 |                 eps=1e-20,
  288 |                 group_select_mode=1,
  289 |                 # The hash custom op currently rejects renorm != 0. Apply
  290 |                 # norm_topk_prob in Python below before returning to MoE compute.
  291 |                 renorm=0,
  292 |                 norm_type=2,
  293 |                 out_flag=False,
  294 |             )
  295 |             return topk_weights.to(torch.float32), topk_ids.to(torch.int32 if indices_type is None else indices_type)
  296 |         norm_type = 0 if self.scoring_func == "softmax" else 1
  297 |         if self.e_score_correction_bias is not None and self.e_score_correction_bias.dtype != router_logits.dtype:
  298 |             self.e_score_correction_bias = self.e_score_correction_bias.to(router_logits.dtype)
  299 |         topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
  300 |             router_logits,
  301 |             k=self.top_k,
  302 |             k_group=topk_group,
  303 |             group_count=num_expert_group,
  304 |             group_select_mode=1,
  305 |             renorm=renorm,
  306 |             norm_type=norm_type,  # 0: softmax; 1: sigmoid
  307 |             out_flag=False,
  308 |             routed_scaling_factor=self.routed_scaling_factor,
  309 |             eps=1e-20,
  310 |             bias_opt=self.e_score_correction_bias,
  311 |         )
  312 |
```

<a id="router-l313"></a>

### 原始第 313 行起

```text
  313 |         return topk_weights, topk_ids
```

<a id="shared-experts"></a>

## vllm_ascend/ops/fused_moe/shared_experts.py

原始完整文件共 433 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="shared-experts-l375"></a>

### 原始第 375 行起

```text
  375 |                         beta=self.situ_activation.beta,
  376 |                         linear_beta=self.situ_activation.linear_beta or 0.0,
  377 |                         activate_left=True,
  378 |                         dst_type=SITU_MX_DST_TYPE_E4M3FN,
  379 |                     )
  380 |                 else:
  381 |                     import custom_ops
  382 |                     import cann_ops_transformer
```

<a id="shared-experts-l383"></a>

### 原始第 383 行起

```text
  383 |                     import cann_ops_nn
  384 |
  385 |                     # Legacy CANN operator call kept for reference:
  386 |                     # quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
  387 |                     #     hidden_states,
  388 |                     #     topk_weight=None,
  389 |                     #     group_index=None,
  390 |                     #     dst_type=torch.float8_e4m3fn,
  391 |                     #     quant_mode=2,
  392 |                     #     clamp_value=self.swiglu_limit,
  393 |                     # )
  394 |                     swiglu_limit_args = {}
  395 |                     if self.swiglu_limit is not None:
  396 |                         swiglu_limit_args["clamp_limit"] = self.swiglu_limit
```

<a id="shared-experts-l397"></a>

### 原始第 397 行起

```text
  397 |                     quantized_x, swiglu_out_scale, _ = torch.ops.cann_ops_nn.swiglu_group_quant(
  398 |                         hidden_states,
  399 |                         dst_type=torch.float8_e4m3fn,
  400 |                         round_scale=True,
  401 |                         quant_mode=1,
  402 |                         **swiglu_limit_args,
  403 |                     )
  404 |                 maybe_wait_event(down_projection_ready)
  405 |                 shared_out = self.layer.down_proj((quantized_x, swiglu_out_scale))[0]
  406 |             else:
  407 |                 # Ensure the shared experts wait for hidden_states to be ready.
  408 |                 torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
```

<a id="patch-distributed"></a>

## vllm_ascend/patch/worker/patch_distributed.py

原始完整文件共 254 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="patch-distributed-l35"></a>

### 原始第 35 行起

```text
   35 | def _normalize_backend(backend: str | Backend) -> str:
   36 |     return str(backend)
   37 |
   38 |
```

<a id="patch-distributed-l39"></a>

### 原始第 39 行起

```text
   39 | def _resolve_reuse_domain(group_name: str) -> str:
   40 |     group_base_name = group_name.split(":")[0]
   41 |     if "eplb" in group_base_name or group_base_name in ("mc2", "mega_moe"):
   42 |         return group_base_name
   43 |     return "shared"
   44 |
   45 |
```

<a id="patch-distributed-l46"></a>

### 原始第 46 行起

```text
   46 | def _create_device_group(
   47 |     ranks: list[int],
```

<a id="w4a8-mxfp4"></a>

## vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py

原始完整文件共 260 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="w4a8-mxfp4-l20"></a>

### 原始第 20 行起

```text
   20 |
   21 | import torch
   22 | import torch_npu
   23 | from vllm.config import get_current_vllm_config
   24 |
   25 | from vllm_ascend.ascend_config import get_ascend_config
   26 | from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
   27 | from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
   28 | from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: F401
   29 | from vllm_ascend.utils import FP8_METHOD, AscendDeviceType, get_ascend_device_type
   30 |
   31 | from ..base import (
   32 |     AscendLinearScheme,
   33 |     AscendMoEScheme,
   34 |     QuantType,
   35 |     TPWeightGatherSpec,
```

省略原始第 36 至 152 行；下一块恢复原始行号。

<a id="w4a8-mxfp4-l153"></a>

### 原始第 153 行起

```text
  153 |         shared_experts: Any | None,
  154 |         shared_experts_input: torch.Tensor | None,
  155 |     ) -> torch.Tensor:
  156 |         if x.dtype not in [torch.float8_e4m3fn]:
  157 |             topk_weights = topk_weights.to(x.dtype)
  158 |
```

<a id="w4a8-mxfp4-l159"></a>

### 原始第 159 行起

```text
  159 |         if (
  160 |             not self.dynamic_eplb
  161 |             and _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2
  162 |             and get_ascend_device_type() == AscendDeviceType.A5
  163 |         ):
```

<a id="w4a8-mxfp4-l164"></a>

### 原始第 164 行起

```text
  164 |             w1 = layer.w13_weight.transpose(1, 2)
  165 |             w2 = layer.w2_weight.transpose(1, 2)
  166 |             w1_scale = layer.w13_weight_scale.transpose(1, 2)
  167 |             w2_scale = layer.w2_weight_scale.transpose(1, 2)
  168 |         else:
  169 |             w1 = layer.w13_weight
  170 |             w2 = layer.w2_weight
  171 |             w1_scale = layer.w13_weight_scale
  172 |             w2_scale = layer.w2_weight_scale
  173 |
  174 |         moe_comm_method = _EXTRA_CTX.moe_comm_method
  175 |         return moe_comm_method.fused_experts(
  176 |             fused_experts_input=build_fused_experts_input(
  177 |                 hidden_states=x,
  178 |                 topk_weights=topk_weights,
  179 |                 topk_ids=topk_ids,
  180 |                 w1=w1,
  181 |                 w2=w2,
  182 |                 quant_type=self.quant_type,
  183 |                 dynamic_eplb=self.dynamic_eplb,
  184 |                 expert_map=layer.ascend_expert_map,
  185 |                 global_redundant_expert_num=layer.global_redundant_expert_num,
  186 |                 mc2_mask=layer.ascend_mc2_mask,
  187 |                 apply_router_weight_on_input=layer.apply_router_weight_on_input,
```

省略原始第 188 至 188 行；下一块恢复原始行号。

<a id="w4a8-mxfp4-l189"></a>

### 原始第 189 行起

```text
  189 |                 activation=layer.activation,
  190 |                 mxfp_act_quant_type=torch.float8_e4m3fn,
  191 |                 mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
  192 |                 mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
  193 |                 mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
  194 |                 mxfp_use_bf16=(x.dtype in [torch.bfloat16, torch.float8_e4m3fn]),
  195 |                 w1_scale=w1_scale,
  196 |                 w2_scale=w2_scale,
  197 |             )
  198 |         )
  199 |
```

<a id="w4a8-mxfp4-l200"></a>

### 原始第 200 行起

```text
  200 |     def process_weights_after_loading(self, layer):
  201 |         layer.w13_weight.data = torch_npu.npu_format_cast(
  202 |             layer.w13_weight.data, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
```

<a id="utils"></a>

## vllm_ascend/utils.py

原始完整文件共 1579 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="utils-l1083"></a>

### 原始第 1083 行起

```text
 1083 |
 1084 |
 1085 | # potential_max_tokens is computed once in the model runner __init__ and reused by
 1086 | # both the skip-allreduce decision and the o_proj static-exchange buffer sizing, so
 1087 | # neither path recomputes it.
 1088 | _potential_max_tokens: int | None = None
 1089 | _MIN_NONEMPTY_MOE_BATCH_TOKENS = 1
 1090 |
 1091 |
```

<a id="utils-l1092"></a>

### 原始第 1092 行起

```text
 1092 | def set_potential_max_tokens(vllm_config) -> None:
 1093 |     """Compute and cache potential_max_tokens once (called from model runner __init__)."""
 1094 |     global _potential_max_tokens
 1095 |     if _potential_max_tokens is not None:
```

省略原始第 1096 至 1104 行；下一块恢复原始行号。

<a id="utils-l1105"></a>

### 原始第 1105 行起

```text
 1105 |
```

<a id="utils-l1106"></a>

### 原始第 1106 行起

```text
 1106 | def should_skip_allreduce_across_dp_group(vllm_config: VllmConfig, is_draft_model: bool = False) -> bool:
 1107 |     """Decide whether to skip the all-reduce across the DP group.
 1108 |
 1109 |     Skipping is applicable for all dense models and for moe models only on ranks
 1110 |     that act as KV consumers. We skip the DP all-reduce when either:
 1111 |     - MegaMoE cannot be selected and both prefill and decode use regular MC2, or
 1112 |     - Decode requires MC2 and ascend_config.scheduler_config.recompute_scheduler_enable is True.
 1113 |
 1114 |     Skipping means each rank may have a different number of tokens, so MC2 needs
 1115 |     a non-zero global_bs and must NOT receive mc2_mask. MegaMoE requires
 1116 |     uniform token counts across ranks, so its FUSED_MC2 path cannot skip.
 1117 |
 1118 |     Returns False when hierarchy comm is enabled because hierarchy requires
 1119 |     global_bs=0 (uniform tokens), which is incompatible with skipping allreduce.
 1120 |
 1121 |     Recomputed per call (no memoization): potential_max_tokens is a set/get global
 1122 |     computed once in init, and select_moe_comm_method is just config lookups, so
 1123 |     this is cheap and avoids id-reuse / stale-cache / init-ordering hazards.
 1124 |     """
 1125 |     ascend_config = get_ascend_config()
 1126 |     # When mc2_comm_alg == "hierarchy", dispatch/combine op don't support dynamic global_bs,
 1127 |     # we need to do allreduce and pad token across dp every step.
 1128 |     if ascend_config.get_mc2_comm_alg() == "hierarchy":
 1129 |         return False
 1130 |
 1131 |     # For dense models, since we don't actually need dp communication, we simply skip it.
 1132 |     # This usually happens when main model is moe while eagle draft model is dense.
 1133 |     is_context_moe_model = is_drafter_moe_model(vllm_config) if is_draft_model else is_moe_model(vllm_config)
 1134 |     if not is_context_moe_model:
 1135 |         return True
 1136 |
 1137 |     # Only applicable to MoE models on KV consumer ranks.
 1138 |     is_kv_consumer = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_consumer
 1139 |     if not is_kv_consumer:
 1140 |         return False
```

<a id="utils-l1141"></a>

### 原始第 1141 行起

```text
 1141 |
 1142 |     from vllm_ascend.ascend_forward_context import select_moe_comm_method, use_cann_megamoe
 1143 |     from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommType
 1144 |
 1145 |     if use_cann_megamoe(vllm_config):
 1146 |         return False
 1147 |
```

<a id="utils-l1148"></a>

### 原始第 1148 行起

```text
 1148 |     def get_comm_method(n: int):
 1149 |         return select_moe_comm_method(n, vllm_config)
 1150 |
 1151 |     scheduler_config = vllm_config.scheduler_config
```

<a id="utils-l1152"></a>

### 原始第 1152 行起

```text
 1152 |     if get_comm_method(_MIN_NONEMPTY_MOE_BATCH_TOKENS) == MoECommType.FUSED_MC2:
 1153 |         return False
 1154 |
 1155 |     # potential_max_tokens is read from the set/get global (computed once in init).
 1156 |     # if mc2 is used in decode max potential tokens case, we can skip allreduce in decode only case.
 1157 |     decode_comm_method = get_comm_method(get_potential_max_tokens())
 1158 |     # For prefill, use the scheduler's max_num_batched_tokens for a single batch.
 1159 |     # if mc2 is used in prefill max potential tokens case and prefill and decode have the same cudagraph mode,
 1160 |     # we can skip allreduce in chunked prefill case.
 1161 |     prefill_comm_method = get_comm_method(scheduler_config.max_num_batched_tokens)
 1162 |     if MoECommType.FUSED_MC2 in {decode_comm_method, prefill_comm_method}:
 1163 |         return False
 1164 |
 1165 |     decode_can_skip = decode_comm_method == MoECommType.MC2
 1166 |     prefill_must_use_mc2 = prefill_comm_method == MoECommType.MC2
 1167 |     uniform_cudagraph_mode = not vllm_config.compilation_config.cudagraph_mode.separate_routine()
 1168 |     chunked_prefill_can_skip = prefill_must_use_mc2 and uniform_cudagraph_mode
 1169 |     return decode_can_skip and (chunked_prefill_can_skip or ascend_config.scheduler_config.recompute_scheduler_enable)
 1170 |
 1171 |
```

<a id="utils-l1172"></a>

### 原始第 1172 行起

```text
 1172 | def has_layer_idx(model_instance: torch.nn.Module) -> bool:
```

<a id="model-runner"></a>

## vllm_ascend/worker/model_runner_v1.py

原始完整文件共 5783 行；以下为修改处及相关上下文。[回到讲解](pr18-megamoe-design-and-code-walkthrough.md)。

<a id="model-runner-l244"></a>

### 原始第 244 行起

```text
  244 |
  245 | AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
  246 | # list when ubatching is enabled
  247 | PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict
  248 |
  249 | SEQ_LEN_WITH_MAX_PA_WORKSPACE = 6144
  250 | DP_METADATA_NUM_TOKENS = 0
  251 | DP_METADATA_CUDAGRAPH_MODE = 1
  252 | DP_METADATA_HAS_REAL_WORK = 2
  253 | DP_METADATA_FIELD_COUNT = 3
  254 |
  255 |
  256 | @dataclass
```

<a id="model-runner-l257"></a>

### 原始第 257 行起

```text
  257 | class GraphCaptureContext:
  258 |     stream: torch.npu.Stream
  259 |
```

省略原始第 260 至 740 行；下一块恢复原始行号。

<a id="model-runner-l741"></a>

### 原始第 741 行起

```text
  741 |     def _sync_metadata_across_dp(
  742 |         self,
  743 |         num_tokens: int,
  744 |         is_draft_model: bool = False,
  745 |         cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
  746 |         allow_dp_padding: bool = False,
  747 |         is_dummy_run: bool = False,
  748 |     ) -> tuple[int, torch.Tensor | None, CUDAGraphMode]:
  749 |         # TODO: In vLLM, the only thing that needs to be synced is num_tokens, but in
  750 |         # our case, we still need to sync the other two flags as well. So we need to
  751 |         # include them in the all_reduce operation, and more over, we CANNOT skip it
  752 |         # even if we are running in eager mode, which harms performance.
  753 |         # FIXME: Restore the `or self.vllm_config.model_config.enforce_eager` here
  754 |         # immediately once the other two flags are no longer needed.
  755 |         if self.dp_size == 1:
  756 |             return num_tokens, None, cudagraph_mode
  757 |
  758 |         if should_skip_allreduce_across_dp_group(self.vllm_config, is_draft_model):
  759 |             num_tokens_after_padding = torch.tensor([num_tokens] * self.dp_size, device="cpu", dtype=torch.int32)
  760 |             return num_tokens, num_tokens_after_padding, cudagraph_mode
  761 |
```

<a id="model-runner-l762"></a>

### 原始第 762 行起

```text
  762 |         packed_tensor = torch.zeros(DP_METADATA_FIELD_COUNT, self.dp_size, device="cpu", dtype=torch.int32)
  763 |         packed_tensor[DP_METADATA_NUM_TOKENS][self.dp_rank] = num_tokens
  764 |         packed_tensor[DP_METADATA_CUDAGRAPH_MODE][self.dp_rank] = cudagraph_mode.value
  765 |         packed_tensor[DP_METADATA_HAS_REAL_WORK][self.dp_rank] = not is_dummy_run
```

<a id="model-runner-l766"></a>

### 原始第 766 行起

```text
  766 |         dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)
  767 |
  768 |         # Unpack the results
  769 |         num_tokens_across_dp = packed_tensor[DP_METADATA_NUM_TOKENS, :]
```

<a id="model-runner-l770"></a>

### 原始第 770 行起

```text
  770 |         active_dp_mask = packed_tensor[DP_METADATA_HAS_REAL_WORK, :].bool()
  771 |         if not active_dp_mask.any():
  772 |             active_dp_mask = torch.ones_like(active_dp_mask)
  773 |
```

<a id="model-runner-l774"></a>

### 原始第 774 行起

```text
  774 |         max_tokens_across_dp = int(num_tokens_across_dp[active_dp_mask].max().item())
  775 |         synced_cudagraph_mode = CUDAGraphMode(_post_process_cudagraph_mode(packed_tensor, active_dp_mask))
  776 |
  777 |         # Dummy ranks enter the same collective shape and graph chosen by ranks
  778 |         # with real work without inflating the real ranks' batch size.
```

<a id="model-runner-l779"></a>

### 原始第 779 行起

```text
  779 |         num_tokens_across_dp[~active_dp_mask] = max_tokens_across_dp
  780 |         if logger.isEnabledFor(logging.DEBUG):
  781 |             logger.debug(
  782 |                 "DP metadata synchronized: rank=%d role=%s local_tokens=%d "
  783 |                 "active_mask=%s synced_tokens=%s aclgraph_mode=%s",
  784 |                 self.dp_rank,
  785 |                 "dummy" if is_dummy_run else "active",
  786 |                 num_tokens,
  787 |                 active_dp_mask.tolist(),
  788 |                 num_tokens_across_dp.tolist(),
  789 |                 synced_cudagraph_mode.name,
  790 |             )
  791 |
  792 |         # Create a tensor for num_tokens_after_padding
  793 |         if allow_dp_padding or is_draft_model:
  794 |             num_tokens_after_padding = torch.tensor(
  795 |                 [max_tokens_across_dp] * self.dp_size, device="cpu", dtype=torch.int32
  796 |             )
  797 |         else:
  798 |             num_tokens_after_padding = num_tokens_across_dp.cpu()
  799 |
  800 |         return max_tokens_across_dp, num_tokens_after_padding, synced_cudagraph_mode
  801 |
```

<a id="model-runner-l802"></a>

### 原始第 802 行起

```text
  802 |     def get_model(self) -> nn.Module:
  803 |         # get raw model out of the aclgraph wrapper.
  804 |         if isinstance(self.model, (ACLGraphWrapper, BreakableACLGraphWrapper)):
  805 |             return self.model.unwrap()
```

省略原始第 806 至 3115 行；下一块恢复原始行号。

<a id="model-runner-l3116"></a>

### 原始第 3116 行起

```text
 3116 |         # For cudagraph capture TODO(lucas): Refactor how we capture cudagraphs (will
 3117 |         # be improved in model runner v2)
 3118 |         force_uniform_decode: bool | None = None,
 3119 |         force_has_lora: bool | None = None,
 3120 |         force_num_active_loras: int | None = None,
 3121 |         num_encoder_reqs: int = 0,
 3122 |         is_dummy_run: bool = False,
 3123 |     ) -> tuple[CUDAGraphMode, BatchDescriptor, bool, torch.Tensor | None, CUDAGraphStat | None]:
 3124 |         num_tokens_padded = self._pad_for_sequence_parallelism(num_tokens)
 3125 |         # A stateful P/D handoff can use a uniform decode graph even at
 3126 |         # prompt_len - 1 computed tokens. Keep first-token prefills out.
 3127 |         has_initial_state = np.all(self.input_batch.num_computed_tokens_cpu[:num_reqs] > 0)
 3128 |         uniform_decode = (
```

省略原始第 3129 至 3171 行；下一块恢复原始行号。

<a id="model-runner-l3172"></a>

### 原始第 3172 行起

```text
 3172 |                 num_tokens=num_tokens_padded,
 3173 |                 cudagraph_mode=cudagraph_mode,
 3174 |                 allow_dp_padding=((cudagraph_mode != CUDAGraphMode.NONE)
 3175 |                                   or enable_sp(self.vllm_config)
 3176 |                                   or oproj_tp_enable()
 3177 |                                   or embedding_tp_enable()),
 3178 |                 is_dummy_run=is_dummy_run,
 3179 |             )
 3180 |
 3181 |             # Extract DP padding if there is any
 3182 |             if num_tokens_across_dp is not None:
 3183 |                 dp_rank = self.parallel_config.data_parallel_rank
 3184 |                 num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
```

省略原始第 3185 至 3704 行；下一块恢复原始行号。

<a id="model-runner-l3705"></a>

### 原始第 3705 行起

```text
 3705 |             force_uniform_decode=uniform_decode,
 3706 |             # `force_has_lora` is used for cudagraph capture; because LoRA is
 3707 |             # activated later in the context manager, but we need to know the
 3708 |             # LoRA state when determining the batch descriptor for capture
 3709 |             force_has_lora=num_active_loras > 0,
 3710 |             force_num_active_loras=num_active_loras,
 3711 |             is_dummy_run=True,
 3712 |         )
 3713 |         if self.use_dcp:
 3714 |             self.dcp_manager.init_batch_info(
 3715 |                 num_scheduled_tokens,
 3716 |                 num_reqs,
 3717 |                 self.input_batch.num_computed_tokens_cpu,
```

省略原始第 3718 至 5666 行；下一块恢复原始行号。

<a id="model-runner-l5667"></a>

### 原始第 5667 行起

```text
 5667 |             cache_dtype=self.cache_config.cache_dtype,
 5668 |             runner_only_attn_layers=self.runner_only_attn_layers,
 5669 |             static_forward_context=(self.compilation_config.static_forward_context),
 5670 |         )
 5671 |
 5672 |
```

<a id="model-runner-l5673"></a>

### 原始第 5673 行起

```text
 5673 | def _post_process_cudagraph_mode(tensor: torch.Tensor, active_dp_mask: torch.Tensor | None = None) -> int:
 5674 |     """
 5675 |     Synchronize cudagraph_mode across DP ranks by taking the minimum.
 5676 |     If any rank has NONE (0), all ranks use NONE.
 5677 |     This ensures all ranks send consistent values (all padded or all unpadded).
 5678 |     """
 5679 |     cudagraph_modes = tensor[DP_METADATA_CUDAGRAPH_MODE, :]
 5680 |     if active_dp_mask is not None:
 5681 |         cudagraph_modes = cudagraph_modes[active_dp_mask]
 5682 |     return int(cudagraph_modes.min().item())
 5683 |
 5684 |
```

<a id="model-runner-l5685"></a>

### 原始第 5685 行起

```text
 5685 | def _get_gpu_model_runner_module_name(model_runner) -> str:
 5686 |     """Return the module name of GPUModelRunner found in the MRO."""
 5687 |     gpu_model_runner_cls = next(
 5688 |         (cls for cls in model_runner.__class__.__mro__ if cls.__name__ == "GPUModelRunner"),
```

## 核对方式

源码摘录按 Git 对象的文件内容逐行生成；行号是修改后的历史文件行号。图解中的教学数字和 C 示意没有插入这些源码块。

```powershell
git show 'df1286cca:vllm_ascend/ops/fused_moe/prepare_finalize.py'
git diff 'df1286cca^' df1286cca -- vllm_ascend/ops/fused_moe/prepare_finalize.py
```
