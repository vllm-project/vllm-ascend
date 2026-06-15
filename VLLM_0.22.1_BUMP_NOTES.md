# vLLM v0.22.1 兼容层处置说明（临时文件，合入前转 PR 描述或删除）

承接 v0.21.0 → v0.22.1 升级。第一轮把所有 `vllm_version_is("0.21.0")` 同步改成 `"0.22.1"`（保留全部兼容层）后，CI 在 **v0.22.1 矩阵**首个 import 处级联失败（`determine_expert_map`）。本轮按 vLLM **v0.22.1 tag 实际 API**（`gh api .../contents/<path>?ref=v0.22.1` 逐一核实）决定每处兼容层「折叠删除」还是「保留」。

**关键事实**：vLLM `v0.22.1` 与 main-verified `967c5c3b` 是 **diverged**（v0.22.1 ahead 27 / behind 421），二者互不包含 → 真·混合。所以**不能一刀切**：v0.22.1 已对齐 main 的折叠，仍停留旧 API 的保留。

## A. 折叠删除（删 `if` 分支留 `else`，v0.22.1 == main）

| 处 | 依据（v0.22.1 实际） |
|----|----------------------|
| `eplb/core/eplb_utils.py` import | `determine_expert_map` 迁到 `fused_moe/expert_map_manager.py` |
| `models/deepseek_v4.py` import | 类迁到 `vllm.models.deepseek_v4.{attention,compressor}` |
| `ops/gdn.py` / `_310p/ops/fla/gdn_310.py` import | `GatedDeltaNetAttention` 迁到 `mamba/gdn/base.py`（旧 `gdn_linear_attn.py` 已删） |
| `patch/worker/patch_idex_310.py` / `patch_qwen3_5.py` | 用 `mamba/gdn/qwen_gdn_linear_attn.py::QwenGatedDeltaNetAttention` |
| `ops/fused_moe/fused_moe.py:52` | `get_compressed_expert_map` 已不在 `layer.py` → 用内联 def |
| `ops/fused_moe/fused_moe.py:182` 路由专家 capturer | 单例 API 已删 → 用 per-layer `_ascend_routed_experts_capturer` |
| `worker/model_runner_v1.py`（import + 1910/2461/2472/2506/3062） | `ModelRunnerOutput.routed_experts: RoutedExpertsLists` 为新字段；单例 capturer / `routed_experts_dict` 已删 |
| `worker/model_runner_v1.py:2102/2128` mamba bufs | v0.22.1 `_get_mamba_bufs()` 返回带 `postprocess_align` 的 `MambaBuffers` |
| `core/recompute_scheduler.py:849/929` 路由专家 | 同上，用 `model_runner_output.routed_experts` |
| `tests/.../test_fused_qkvzba_split_reshape_cat.py` import | 同 gdn：用 `mamba/gdn/base.py` |

**⚠️ 非纯 if/else 删除的一处（请重点确认）**：`worker/model_runner_v1.py::_bind_routed_experts_capturer` 里有个**未被 `vllm_version_is` 包裹**的 v0.22.1 兜底 `if capturer is None: capturer = get_global_experts_capturer()`，引用了 v0.22.1 已删的单例 API。已删除该兜底（main 流程恒由调用方传入 capturer）。这是本轮唯一超出「删分支」范围的改动。

## B. 保留不动（v0.22.1 仍用旧 API，第一轮改串后的 `if` 分支已正确）

| 处 | 依据（v0.22.1 实际） |
|----|----------------------|
| `ops/bailing_moe_linear_attn.py` / `patch_minimax_m2.py` / `patch_minimax_m2_linear_attn.py` | `mamba/linear_attn.py` 仍是扁平文件（含 3 个 linear_attention 函数 + `MiniMaxText01RMSNormTP`），无 `mamba/linear/` |
| `spec_decode/llm_base_proposer.py:56` | `parallel_state.patch_tensor_parallel_group` 仍在（main 已删，故 else 是 backport） |
| `patch/platform/patch_tool_choice_none_content.py:137` | `entrypoints/openai/engine/serving.py::_parse_tool_calls_from_content` 仍在 |
| `core/single_type_kv_cache_manager.py:203` / `cpu_kv_cache_manager.py:106` / `patch_kv_cache_coordinator.py:278` | v0.22.1 用 `use_eagle`（无 `drop_eagle_block`） |
| `single_type_kv_cache_manager.py:262` / `recompute_scheduler.py:69` / `patch_mamba_manager.py:56` | v0.22.1 有 `spec_manager_map` 字典（无 Registry） |
| `fused_moe.py:415` / `cpu_kv_cache_manager.py:80` / `patch_kv_cache_coordinator.py:123,374` | v0.22.1 无 `scheduler_block_size` 形参 |
| `ops/gdn.py:286/301`（`ba.chunk` vs `split_ba`）、`:333`（`gdn_attention_core` 算子名） | v0.22.1 `gdn/base.py` 无 `split_ba`，`_split_ba_for_tp` 已 `hasattr` 兜底 → CI 复核算子名 |
| `worker/worker.py:751` KV-transfer | `{tp_rank}`（v0.22.1 应无 PP 键）→ CI 复核 |
| 测试：`test_patch_glm47_tool_call_parser.py`、`test_compressed_prefix_cache.py:64` 等 | v0.22.1 仍有 `_WrappedParser`/旧 kwargs；函数级守卫，不影响收集 |

## C. v2 model runner —— 不做兼容（按确认，仅注释）

`patch/worker/__init__.py` 的 `_V2_MODEL_RUNNER_SUPPORTED = not vllm_version_is("0.22.1")` 在 v0.22.1 上为 `False`，所有 `patch_v2.*` / 路由专家 capture patch **不导入**，v2 worker 守卫全部休眠（`worker.py:149`、`worker/v2/*`、eagle）。已把注释改写为「v2 主线专属、v0.22.1 release 故意不兼容（用 v1 runner）」。

## 验证

- `ruff check` 全过、`ruff format` 无改动、`py_compile` 全过（本地装了仓库 pin 的 `ruff==0.14.0` 到 `.venv`）。
- 全仓库已无 `get_global_experts_capturer` / `extract_routed_experts_for_current_batch` / `issue_routing_d2h_copy` / `routed_experts_dict` / `_get_mamba_copy_bufs` 残留引用。
- 余下 `vllm_version_is("0.22.1")` 守卫即为上表 B/C 的保留集。
- 待 CI v0.22.1 矩阵复核 B 类运行期项（gdn 算子名、KV-transfer 键形状）；这些为运行期失败、不再级联，便于第二轮逐条定位。
