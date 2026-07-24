#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ----------------------------------------------------------------------------------
# This module manage the patch for vllm. There are two folders in this module:
# - platform: contains the patches applied before worker starts. It's called by
#             `vllm_ascend.utils.adapt_patch(is_global_patch=True)` in
#             `vllm_ascend.platform.NPUPlatform.pre_register_and_update()` function.
# - worker: contains the patches applied when worker starts. It's called by
#           `vllm_ascend.utils.adapt_patch(is_global_patch=False)` in
#           each worker's `__init__` function.
#
# Once a new patch is added in vllm-ascend, please add the patch description into this file as well.
# ----------------------------------------------------------------------------------

# What's Patched and how it works:
# --------------------------------
# * Platform Patch:
# =================
# Entries are listed in alphabetical order by file name.
#
# ** 1. File: platform/patch_balance_schedule.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.engine.core.EngineCoreProc.run_engine_core`
#      `vllm.v1.core.sched.scheduler.Scheduler`
#    Why:
#       vLLM v1 scheduling currently enables chunkedprefill by default, which processes prefill and decode
#       requests simultaneously in a single scheduling session. This can impact the overall system throughput
#       and performance in some scenarios.
#    How：
#       Set --additional-config '{"enable_balance_scheduling": true}' or
#       set environmental variable VLLM_ASCEND_BALANCE_SCHEDULING=1 (deprecated).
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/29721
#    Future Plan:
#       Remove this patch when vLLM merge the PR.
#
# ** 2. File: platform/patch_camem_allocator.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.config.model.is_cumem_allocator_available`
#    Why:
#       Upstream vLLM main enables and validates the CUDA/ROCm CuMem allocator
#       when `enable_sleep_mode=True`. Ascend implements sleep mode with its own
#       CaMem allocator, so the upstream CuMem-only availability check fails
#       during `ModelConfig` validation before Ascend worker code can run.
#    How:
#       Treat Ascend's platform sleep allocator as satisfying the allocator
#       availability check, while preserving the original vLLM CuMem check as
#       fallback.
#    Related PR (if no, explain why):
#       No, this maps an upstream CUDA/ROCm allocator validation to Ascend's
#       backend-specific CaMem implementation.
#    Future Plan:
#       Remove this patch if upstream exposes a platform allocator capability hook
#       for sleep mode validation.
#
# ** 3. File: platform/patch_distributed.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `torch.distributed.all_reduce`, `torch.distributed.broadcast`
#    Why:
#       tensor alignment for 310p
#    How：
#       rewrite all_reduce and broadcast in torch.distributed
#    Related PR (if no, explain why):
#       No, not ready yet.
#    Future Plan:
#       Find a better way to support tensor alignment for 310p without this patch.
#
# ** 4. File: platform/patch_dp_device_ids.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.core.dp_utils.get_physical_gpu_ids_for_local_dp_rank`
#    Why:
#       PR #45026 removed the per-process device isolation that older vLLM
#       versions performed internally. Application-level DP (e.g.
#       `offline_data_parallel.py`) now has to slice ASCEND_RT_VISIBLE_DEVICES
#       per rank itself, but the upstream helper still expects the env var to
#       contain ALL devices for ALL ranks and tries to read it with the
#       `local_dp_rank * world_size` offset. With a sharded env var, that
#       offset is out of range and the helper raises IndexError (wrapped in the
#       user-facing "Error computing device indices for ..." message).
#    How：
#       Patch `get_physical_gpu_ids_for_local_dp_rank` so it tolerates a
#       pre-sharded ASCEND_RT_VISIBLE_DEVICES env var (one slice per DP rank),
#       instead of unconditionally applying `local_dp_rank * world_size` as an
#       offset into it.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/45026
#    Future Plan:
#       Remove this patch once upstream `get_physical_gpu_ids_for_local_dp_rank`
#       handles a pre-sharded visible-devices env var, or vLLM-Ascend stops
#       relying on application-level device slicing for DP.
#
# ** 5. File: platform/patch_fused_moe.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.fused_moe.FusedMoE`
#    Why:
#       vllm's FusedMoE is a factory function (not a class). deepseek_v2 and
#       other models do `from vllm.model_executor.layers.fused_moe import FusedMoE`
#       and call it directly, so on Ascend we must redirect it to AscendMoERunner
#       before any model is imported.
#    How：
#       Patch the FusedMoE binding in both the package `__init__` and the layer
#       module so model imports pick up the Ascend runner. `worker/patch_fused_moe.py`
#       reuses this platform patch to avoid double-wrapping the factory during
#       worker initialization.
#    Related PR (if no, explain why):
#       No, vllm-ascend-specific MoE runner integration.
#    Future Plan:
#       Remove this patch once upstream exposes a backend dispatch / plugin hook
#       for selecting the MoE runner implementation.
#
# ** 6. File: platform/patch_kv_cache_coordinator.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.core.kv_cache_coordinator.HybridKVCacheCoordinator.find_longest_cache_hit_per_group`
#    Why:
#       In PD disaggregation with hybrid Mamba models, the D side receives
#       FullAttention KV blocks from the P side but has no local prefix-cache
#       hit for Mamba groups. Upstream's min-reduction across all KV groups
#       collapses the FullAttention hit length to 0, preventing partial
#       FullAttention-only prefix cache reuse on the D side.
#    How:
#       For Mamba hybrid models,
#       num_new_local_computed_tokens should be the FA hit
#       length. This value is passed to the connector's
#       get_num_new_matched_tokens which computes:
#       external = total - local_computed.
#       Using the FA hit skips re-transferring FA blocks
#       already cached on D-side.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/42524
#       https://github.com/vllm-project/vllm/pull/44243
#    Future Plan:
#       Remove this patch when vLLM PR #42524 and #44243 is included in the supported
#       upstream vLLM version.
#
# ** 7. File: platform/patch_kv_cache_utils.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes`
#      `vllm.v1.engine.core.resolve_kv_cache_block_sizes`
#    Why:
#       vLLM PR #40860 added a restriction that hybrid KV cache groups with
#       multiple block sizes do not support context parallelism (dcp/pcp > 1).
#       This restriction is correct for CUDA but not for Ascend, which
#       implements context parallelism for MLA and SWA-MLA layers separately.
#    How：
#       Monkey-patch resolve_kv_cache_block_sizes to handle the multiple-groups
#       + CP case by returning lcm(block_sizes) * dcp * pcp as scheduler_block_size
#       instead of raising ValueError.
#    Related PR (if no, explain why):
#       vLLM PR #40860 ([Feat] DeepSeek V4 Rebased).
#    Future Plan:
#       Remove this patch once upstream vLLM supports hybrid KV cache + CP for
#       non-CUDA backends, or exposes a platform hook for this behavior.
#
# ** 8. File: platform/patch_mamba_config.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config`
#    Why:
#       block size is set to 16 in vLLM which is not supported by Ascend.
#    How：
#       Set block size to 128 on npu.
#    Related PR (if no, explain why):
#       we'll fix this in vLLM soon.
#    Future Plan:
#       Remove this patch when vLLM merges the PR.
#
# ** 9. File: platform/patch_mamba_config_310.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config`
#    Why:
#       310P requires attention page size to be >= mamba page size and aligned to
#       the kernel block alignment (128). The upstream verify_and_update_config
#       default (block size 16) is not supported on 310P.
#    How：
#       On 310P, override verify_and_update_config to align mamba_block_size and
#       attention block size to the 128-token kernel alignment, ensuring the
#       attention page size is >= mamba page size. This is the 310P counterpart
#       of patch_mamba_config.py (loaded only when `is_310p()` is True).
#    Related PR (if no, explain why):
#       No, 310P-specific kernel alignment requirement.
#    Future Plan:
#       Remove this patch once upstream supports 310P-aligned mamba block sizing.
#
# ** 10. File: platform/patch_mamba_manager.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.core.single_type_kv_cache_manager.MambaManager`
#    Why:
#       1. Upstream hybrid prefix cache lookup does not support PCP/DCP.
#       2. Upstream MambaManager#get_num_blocks_to_allocate give the
#          wrong number of blocks when an external cache hit occurred
#    How:
#       1. Replace MambaManager with AscendMambaManager for prefix cache hit lookup
#          on hybrid Mamba paths (logical mamba block_size when caching is enabled).
#       2. Override the get_num_blocks_to_allocate method to fix the number of blocks
#          when hitting the external cache and loading synchronously
#    Related PR (if no, explain why):
#       1. https://github.com/vllm-project/vllm/pull/40996
#       2. https://github.com/vllm-project/vllm/pull/46892
#    Future Plan:
#       1. Upstream PR #40996 adds hybrid prefix cache lookup for DCP only; PCP is
#          not supported yet. Remove this patch once upstream supports both PCP and DCP.
#       2. Remove this patch once upstream accept 46892 pr or fixed the bug by other pr.
#
# ** 11. File: platform/patch_minimax_m2_config.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.config.model.ModelConfig._verify_quantization`
#    Why:
#       MiniMax-M2 fp8 checkpoints on NPU may fail upstream quantization validation.
#       vllm-ascend needs to disable fp8 quantization and load bf16 dequantized
#       weights in worker-side patches instead.
#    How：
#       Monkey-patch `_verify_quantization` and intercept platform quantization
#       verification to force `cfg.quantization=None` for MiniMax-M2 fp8 on NPU.
#    Related PR (if no, explain why):
#       No, upstream behavior differs across versions and needs discussion.
#    Future Plan:
#       Remove this patch once upstream supports MiniMax-M2 fp8 on NPU or provides
#       a backend-safe validation / override mechanism.
#
#   2. `vllm.config.model.ModelConfig._verify_cuda_graph`
#    Why:
#       For MiniMax-M2 on NPU with ACL graph capture enabled, HCCL op expansion
#       mode affects graph shape coverage. Users may forget to set it.
#    How：
#       If user doesn't set it, set `HCCL_OP_EXPANSION_MODE=AIV` for this model
#       and log a warning when a different value is detected.
#    Related PR (if no, explain why):
#       No, this is an environment-specific tuning knob.
#    Future Plan:
#       Remove this patch if upstream provides an official NPU graph-capture
#       guidance / auto-configuration path for HCCL.
#
#   3. `vllm.config.speculative.SpeculativeConfig._verify_args`
#    Why:
#       Upstream vLLM's eagle3/extract_hidden_states restricts target model types
#       via a whitelist. MiniMax-M2 should be allowed once the worker-side model
#       can emit auxiliary hidden states.
#    How：
#       Monkey-patch `_verify_args` to bypass only the whitelist ValueError for
#       MiniMax model_type when method is eagle3/extract_hidden_states.
#       SpeculativeConfig is a Pydantic dataclass (`@config`); init validation calls
#       `__pydantic_decorators__.model_validators["_verify_args"].func`, so that
#       `Decorator.func` must be replaced (not only `SpeculativeConfig._verify_args`),
#       then `rebuild_dataclass(SpeculativeConfig, force=True)`.
#       If `VllmConfig` was imported earlier, also `rebuild_dataclass(VllmConfig, ...)`
#       so nested `speculative_config` validation does not use a stale schema.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/37512
#    Future Plan:
#       Remove this patch once upstream whitelist includes MiniMax.
#
#   4. `vllm.model_executor.models.registry` (spec decode aliases)
#    Why:
#       Some Eagle3 draft checkpoints may declare a MiniMax-specific architecture
#       string while reusing the shared Eagle3 implementation.
#    How：
#       Register `Eagle3MiniMaxM2ForCausalLM` as an alias pointing to the
#       existing Eagle3 implementation in the speculative decoding registry.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/37512
#    Future Plan:
#       Drop the alias once upstream registry includes it or the checkpoint
#       standardizes architecture strings.
#
# ** 12. File: platform/patch_mla_prefill_backend.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.attention.backends.mla.common.get_mla_prefill_backend`
#    Why:
#       PR vllm-project/vllm#32623 introduced a new MLAPrefillBackend abstraction.
#       When MLAAttention.__init__ calls get_mla_prefill_backend(), the upstream
#       selector sees that Ascend NPU returns None for get_device_capability() and
#       falls back to FlashAttnPrefillBackend, which asserts flash_attn_varlen_func
#       is available — crashing on Ascend.
#    How：
#       Register a no-op AscendMLAPrefillBackend and patch get_mla_prefill_backend
#       so that MLAAttention.__init__ completes without error. Ascend's
#       AscendSFAImpl/AscendMLAImpl handles the full forward pass (including
#       prefill) via impl.forward(), so prefill_backend.run_prefill_* is never
#       called.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/32623
#    Future Plan:
#       Remove this patch once upstream `get_mla_prefill_backend` provides a
#       platform/device hook so Ascend can be selected (or skipped) without
#       monkey-patching.
#
# ** 13. File: platform/patch_multiproc_executor.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.executor.multiproc_executor.MultiprocExecutor`
#    Why:
#       vLLM create child process with daemon=True, which doesn't work with EPLB case, since EPLB will create
#       a new process which is not allowed by daemon=True.
#    How：
#       Set daemon=False in MultiprocExecutor.
#    Related PR (if no, explain why):
#       Find a way to support daemon=False in vLLM
#    Future Plan:
#       Remove this patch when vLLM fix the issue.
#
# ** 14. File: platform/patch_pp_mtp.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.outputs.ModelRunnerOutput`
#    Why:
#       PP + MTP mixed deployment needs the model runner to return the draft
#       tokens produced for the same scheduler output. Upstream output objects
#       do not carry `spec_token_ids` on all supported vLLM revisions.
#    How：
#       Add a backward-compatible `spec_token_ids` field to `ModelRunnerOutput`
#       and `EMPTY_MODEL_RUNNER_OUTPUT` when the field is missing.
#    Related PR (if no, explain why):
#       Backport of local vLLM PP+MTP branch changes.
#    Future Plan:
#       Remove this patch once the supported vLLM version carries PP-safe
#       speculative token metadata in `ModelRunnerOutput`.
#
#   2. `vllm.v1.engine.core.EngineCore.post_step`
#    Why:
#       With PP batch queue, synchronous scheduling can schedule the next batch
#       before the previous model output is consumed. Calling `post_step` in that
#       window updates `request.spec_token_ids` from live request state that may
#       already belong to the newer schedule step.
#    How：
#       In PP + MTP + batch queue + sync scheduling, skip `post_step` after model
#       execution and let scheduler output processing perform the spec token
#       writeback from the corresponding `ModelRunnerOutput`.
#    Related PR (if no, explain why):
#       Backport of local vLLM PP+MTP branch changes.
#    Future Plan:
#       Remove this patch when upstream makes spec token writeback output-owned
#       for PP batch queue.
#
#   3. `vllm.v1.core.sched.scheduler.Scheduler._update_after_schedule`
#      `vllm.v1.core.sched.scheduler.Scheduler.update_from_output`
#    Why:
#       PP async scheduling must not schedule the same decode request again
#       before the previous output has written sampled/spec tokens back. Without
#       this request-level in-flight fence, the next step may use stale sampled
#       tokens and produce incorrect target-model output. Intermediate prefill
#       chunks do not depend on sampled/spec writeback and should remain
#       schedulable to keep the PP pipeline filled.
#    How：
#       After scheduling, set a temporary decode fence for final prefill/decode
#       chunks in PP IPC mode. Release the fence in `update_from_output` after
#       the matching output is processed. For PP + MTP, also filter zero-token
#       placeholder requests before delegating to upstream scheduler accounting,
#       then write `request.spec_token_ids` from `model_runner_output.spec_token_ids`.
#    Related PR (if no, explain why):
#       Backport of local vLLM PP+MTP branch changes.
#    Future Plan:
#       Remove this patch once upstream supports request-level PP async fences
#       and output-owned spec token writeback.
#
#   4. `vllm.v1.core.sched.scheduler.Scheduler._make_cached_request_data`
#    Why:
#       Upstream PP async scheduling relies on direct PP-rank GPU broadcast for
#       sampled-token handoff and omits `new_token_ids` from cached request data.
#       vLLM Ascend routes PP sampled-token handoff through scheduler IPC, so
#       non-last PP ranks need the previous sampled token in both sync and async
#       scheduling. This also avoids copying draft tokens as confirmed tokens in
#       PP + MTP mixed deployment.
#    How：
#       Temporarily build cached request data with sync-PP semantics in PP IPC
#       mode, then fill the last confirmed output token for requests whose
#       clamped upstream slice is empty.
#    Related PR (if no, explain why):
#       Backport of local vLLM PP+MTP branch changes.
#    Future Plan:
#       Remove this patch once upstream provides a scheduler-owned PP sampled
#       token handoff path that works for both sync and async scheduling.
#
#   5. `vllm.config.model.ModelConfig.verify_with_parallel_config`
#    Why:
#       Local Eagle/MTP drafters are loaded on the last PP stage rather than
#       partitioned across all PP ranks. Upstream `ModelConfig.verify_with_parallel_config`
#       validates against `pipeline_parallel_size`, which fails for these drafters
#       since they run locally with effective PP=1.
#    How：
#       Monkey-patch `verify_with_parallel_config` to detect Eagle/MTP drafter
#       models (by `model_type` and `architectures`) when `runner="draft"` and
#       `pipeline_parallel_size > 1`. For such configs, call the original verify
#       with a patched `pipeline_parallel_size=1` copy, preserving normal target-model
#       validation for non-drafter models.
#    Related PR (if no, explain why):
#       Backport of local vLLM PP+MTP branch changes.
#    Future Plan:
#       Remove this patch once upstream vLLM's `ModelConfig.verify_with_parallel_config`
#       supports local drafter models with PP > 1, or moves the PP validation to a
#       separate hook that can be overridden per-model-type.
#
# ** 15. File: platform/patch_profiling_chunk.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.engine.core.EngineCore.__init__`
#   2. `vllm.v1.engine.core.EngineCoreProc.run_engine_core`
#   3. `Scheduler.update_from_output` (scheduler class, wrapped when profiling chunk is enabled)
#    Why:
#       Profiling-based dynamic chunk sizing needs to run a one-shot profiling pass
#       after `model_executor` is ready, and to feed per-step execution latency back
#       into `ProfilingChunkManager` so the history-aware chunk predictor can refine
#       online. In multiprocessing `spawn` mode the child process starts a fresh
#       interpreter, so monkey-patches applied in the parent are lost unless the
#       subprocess entry point re-applies them before any `EngineCore` is created.
#    How：
#       Replace `EngineCore.__init__` to call `scheduler.run_profiling_chunk_init`
#       when present, then wrap `scheduler.update_from_output` once per process to
#       read `model_output.execution_time_ms` and `scheduler_output` token/chunk
#       metadata and call `ProfilingChunkManager.record_batch_execution_time` (and
#       bootstrap target latency for the first chunk when needed). Replace
#       `EngineCoreProc.run_engine_core` so importing this module in the child
#       re-runs the idempotent patch helper before delegating to the original
#       implementation.
#    Related PR (if no, explain why):
#       No, vllm-ascend-specific profiling / scheduling integration.
#    Future Plan:
#       Remove or narrow this patch if upstream exposes stable hooks for backend
#       profiling startup and per-step timing callbacks without monkey-patching
#       `EngineCore` and the multiprocess entry point.
#
# ** 16. File: platform/patch_speculative_config.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.config.speculative.SpeculativeConfig.hf_config_override`
#    Why:
#       Several MTP draft models (DeepSeek-V3/V4, Pangu, MiMo, GLM4-MoE, ERNIE,
#       Nemotron-H, Qwen3-Next, Qwen3.5, Exaone, LongCat-Flash, Step3.5/3.7) are
#       loaded as the target model's MTP layer rather than a standalone model.
#       Upstream `SpeculativeConfig.hf_config_override` does not remap these
#       target model_types/architectures to their MTP variants, so spec decoding
#       cannot locate the drafter.
#    How：
#       Replace `SpeculativeConfig.hf_config_override` to detect the supported
#       target model_type/architecture, derive `n_predict` from
#       `num_nextn_predict_layers` (or `mtp_num_hidden_layers`), and rewrite
#       `model_type`/`architectures` to the corresponding MTP model. Also remaps
#       MistralLarge3 to its Eagle variant.
#    Related PR (if no, explain why):
#       No, vllm-ascend-specific MTP model type remapping.
#    Future Plan:
#       Remove this patch once upstream vLLM supports loading these MTP draft
#       models without a custom `hf_config_override`, or exposes a plugin hook
#       for MTP model_type/architecture remapping.
#
# ** 17. File: platform/patch_structured_output.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.sampling_params.SamplingParams._validate_structured_outputs`
#      `vllm.v1.structured_output.StructuredOutputManager.grammar_init`
#    Why:
#       V1 structured outputs use one engine-level backend, while `backend=auto`
#       resolves the backend per request. After one request initializes
#       `xgrammar`, a later request that resolves to `guidance` can still reach
#       the initialized `xgrammar` backend and crash during grammar compilation.
#    How:
#       Record the first resolved backend on the structured-output config and
#       reject later requests that resolve to a different backend. Also guard
#       `grammar_init` so requests that bypass API-side validation fail before
#       backend grammar compilation.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/issues/43920
#       https://github.com/vllm-project/vllm/pull/44401
#    Future Plan:
#       Remove this patch once upstream vLLM either enforces backend consistency
#       before grammar compilation or safely handles mixed-backend grammar
#       failures without killing the engine.
#
# ** 18. File: platform/patch_torch_accelerator.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `torch.accelerator.memory_stats`, `torch.accelerator.memory_reserved`,
#      `torch.accelerator.reset_peak_memory_stats`, `torch.accelerator.get_memory_info`,
#      `torch.accelerator.empty_cache`
#    Why:
#       Upstream vLLM (commit 747b068) replaced `current_platform.memory_stats()`
#       with `torch.accelerator.memory_stats()`, but `torch.accelerator` does not
#       route to the NPU backend, so memory APIs fail on Ascend. `empty_cache`
#       also needs an NPU-compatible no-op-friendly implementation.
#    How：
#       Monkey-patch `torch.accelerator` memory APIs to delegate to the
#       corresponding `torch.npu` implementations, and patch `empty_cache` to a
#       compatible wrapper.
#    Related PR (if no, explain why):
#       No, maps upstream torch.accelerator memory APIs to the NPU backend.
#    Future Plan:
#       Remove this patch once `torch.accelerator` correctly routes to the NPU
#       backend for these memory APIs.
#
# ** 19. File: platform/patch_tool_choice_none_content.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionResponse`
#      `vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionStreamResponse`
#    Why:
#       vLLM v0.23.0 can serialize empty `tool_calls: []` fields for content-only
#       OpenAI chat responses / streaming deltas, while OpenAI-compatible SDKs
#       expect those empty fields to be omitted so clients see `tool_calls=None`.
#    How：
#       Wrap `model_dump` / `model_dump_json` for chat response payloads and drop
#       empty `tool_calls` lists from `message` / `delta` objects.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/44105
#    Future Plan:
#       Remove this patch once the supported vLLM version contains PR #44105.
#
# ** 20. File: platform/patch_use_v2_model_runner.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.config.vllm.VllmConfig.use_v2_model_runner`
#    Why:
#       Upstream vLLM enables the v2 model runner not only via the
#       VLLM_USE_V2_MODEL_RUNNER env var but also based on model
#       architecture whitelists, Triton availability, and feature
#       compatibility checks. On Ascend the NPU v2 runner is not yet
#       compatible with all upstream-defaulted models and features, so
#       enabling by model architecture can crash. We override the
#       property to read only VLLM_USE_V2_MODEL_RUNNER, deferring
#       model/framework checks to the NPU runner itself.
#    How:
#       Monkey-patch VllmConfig.use_v2_model_runner to return
#       envs.VLLM_USE_V2_MODEL_RUNNER (defaulting to False when unset).
#       worker/patch_v2/patch_use_v2_model_runner.py reuses this platform
#       patch so EngineCore and worker processes share the same behavior.
#    Related PR (if no, explain why):
#       1. https://github.com/vllm-project/vllm-ascend/pull/11389
#    Future Plan:
#       Remove this patch once vllm-ascend fully supports the v2 model
#       runner and can rely on upstream's default enablement heuristics
#       (model architecture, Triton, feature checks) without crashes or
#       degraded functionality.
#
# ** 21. File: platform/patch_weight_transfer_engine.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.distributed.weight_transfer.factory.WeightTransferEngineFactory._registry["nccl"]`
#    Why:
#       Upstream vLLM's WeightTransferConfig.backend is a pydantic Literal["nccl", "ipc"]
#       which does not accept "hccl".  On Ascend NPU, NCCL is unavailable and HCCL must
#       be used for trainer-to-worker weight broadcasting.
#    How：
#       Replace the "nccl" factory entry with a lambda that returns
#       HCCLWeightTransferEngine.  Users pass the already-accepted "nccl" string
#       (e.g. --weight-transfer-config '{"backend": "nccl"}') and the factory
#       resolves it to the HCCL engine at runtime.
#    Related PR (if no, explain why):
#       No.  Adding "hccl" to the Literal requires modifying pydantic core schemas,
#       which is fragile across pydantic versions.
#    Future Plan:
#       Remove this patch when upstream vLLM relaxes the Literal type to str or
#       provides an extension point for out-of-tree weight transfer backends.
#   2. `vllm.distributed.weight_transfer.factory.WeightTransferEngineFactory._registry["ipc"]`
#    Why:
#       The "ipc" backend must resolve to NPUIPCWeightTransferEngine on Ascend NPU.
#       However, this patch runs during global plugin patching - extremely early in
#       startup, before any weight transfer backend is selected. Importing the IPC
#       engine eagerly pulls in vllm.distributed.weight_transfer.ipc_engine, which
#       does `import ray` at module top level. Since ray is an optional dependency,
#       its absence aborts the whole vllm_ascend plugin load and crashes every
#       `vllm serve` invocation - even workloads that never use weight transfer.
#    How：
#       Register a lazy loader function (instead of an eager import) that imports
#       and returns NPUIPCWeightTransferEngine only when create_engine() is invoked
#       for the "ipc" backend. This matches the factory's zero-arg-callable
#       lazy-loading contract, so the ray-importing module is loaded only when ipc
#       is actually requested. (HCCL keeps its eager import - it never imports ray.)
#    Related PR (if no, explain why):
#       No.  The eager `import ray` lives in upstream vLLM's ipc_engine module; the
#       lazy loader is a local workaround until upstream defers that import.
#    Future Plan:
#       Remove this workaround once upstream vLLM stops importing ray at module top
#       level in vllm.distributed.weight_transfer.ipc_engine (e.g. defers it into
#       the code path that actually needs ray), so importing the IPC engine no
#       longer requires the optional ray dependency.
#
# * Worker Patch:
# ===============
# Entries are listed in alphabetical order by file name.
#
# ** 1. File: worker/patch_cudagraph.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.cudagraph_dispatcher.CudagraphDispatcher._create_padded_batch_descriptor`
#    Why:
#       vllm's FULL mode will cause error, we use a patch to avoid it.
#       After that, FULL can be enable now.
#    How：
#       Dynamically replace the `_create_padded_batch_descriptor` function at runtime,
#       and change the condition of if.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/34880
#    Future Plan:
#       Remove this patch when vLLM merges the PR.
#
# ** 2. File: worker/patch_deepseek_mtp.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.deepseek_v2.get_spec_layer_idx_from_weight_name` and
#      `vllm.model_executor.models.deepseek_mtp.get_spec_layer_idx_from_weight_name`
#    Why:
#       When GLM5 uses rotary quant in vllm-ascend, the MTP layer needs to load an extra weight
#       named `rot.weight`.
#    How：
#       If weight name starts with `rot`, return `layer_id + i` like other tensors in MTP layer.
#    Related PR (if no, explain why):
#       Rotary quant is a unique feature of vllm-ascend.
#    Future Plan:
#       Remove this patch when vllm supports rotary quant or pluggable `MultiTokenPredictorLayer`.
#   2. `vllm.model_executor.models.deepseek_mtp.DeepSeekMultiTokenPredictorLayer`
#    Why:
#       When GLM5 uses rotary quant in vllm-ascend, the `previous_hidden_states` does not .
#    How：
#       If the target model uses rotary quant, a new linear operation is added before `ehnorm`.
#    Related PR (if no, explain why):
#       Rotary quant is a unique feature of vllm-ascend.
#    Future Plan:
#       Remove this patch when vllm supports rotary quant or pluggable `MultiTokenPredictorLayer`.
#   3. `vllm.model_executor.models.deepseek_mtp.DeepSeekMTP._rewrite_spec_layer_name`
#    Why:
#       Rename `rot.weight` to match the format of weights in `DeepSeekMTP`.
#    How：
#       If the weight name is `rot`, rename it to `model.layers.{spec_layer}.rot.weight`.
#    Related PR (if no, explain why):
#       Rotary quant is a unique feature of vllm-ascend.
#    Future Plan:
#       Remove this patch when vllm supports rotary quant or pluggable `MultiTokenPredictorLayer`.
#   4. `vllm.model_executor.models.deepseek_v2.GlmMoeDsaForCausalLM.load_weights`
#    Why:
#       After vllm PR #41706, GlmMoeDsaForCausalLM.load_weights uses `AutoWeightsLoader` which
#       does not skip `rot.weight`, and will cause ValueError while loading weights.
#    How：
#       Use the `skip_prefixes` parameter to skip certain weight tensors.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/41706
#    Future Plan:
#       Remove this patch when vllm supports rotary quant or pluggable `MultiTokenPredictorLayer`.
#
# ** 3. File: worker/patch_deepseek_v2.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.deepseek_v2.DeepseekV2MLAAttention.__init__`
#    Why:
#       GLM-5.2 checkpoints omit `Indexer` weights on shared-indexer layers,
#       while GLM-5.1 IndexCache overrides only skip top-k computation and keep
#       per-layer `Indexer` weights. Treating both layouts alike breaks GLM-5.1
#       weight loading.
#    How:
#       Skip `Indexer` construction only when the layer both skips top-k and is
#       explicitly marked `shared` in `indexer_types`. MTP layers always retain
#       a complete `Indexer`.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/45895
#    Future Plan:
#       Remove this patch when vLLM Ascend depends on a vLLM version that includes
#       PR #45895.
#
# ** 4. File: worker/patch_distributed.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.distributed.parallel_state.GroupCoordinator`
#    Why:
#       vllm doesn't support all_to_all for GroupCoordinator.
#    How：
#       Add all_to_all implementation for GroupCoordinator.
#    Related PR (if no, explain why):
#       No, we should use vlLM all2all manager to support all_to_all for npu.
#    Future Plan:
#       Remove this patch when the refactor of all2all manager is done.
#
# ** 5. File: worker/patch_draft_quarot.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.llama_eagle3.Eagle3LlamaForCausalLM.load_weights`
#    Why:
#       vllm-ascend reused the loading logic of drafter model from vllm,
#       but vllm doesn't need to apply to Ascend quantization.
#    How：
#       Dynamically replace the `load_weights` function at runtime,
#       and fix `target_config` into the new implementation with a closure.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/36225
#    Future Plan:
#       Remove this patch when vLLM merges the PR.
#
# ** 6. File: worker/patch_eagle3_init.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.llama_eagle3.Eagle3LlamaForCausalLM`,
#      `vllm.model_executor.models.deepseek_v2.Eagle3DeepseekV2ForCausalLM` /
#      `Eagle3DeepseekV3ForCausalLM`
#    Why:
#       Upstream Eagle3 draft models compute `target_layer_num` via
#       `model_config.get_num_layers(parallel_config)` which, under PP, returns the
#       per-PP-stage count. This value feeds into the draft model's
#       `start_layer_id` (used to build parameter name prefixes like
#       `model.layers.<start_layer_id + i>`). With PP>1 the prefixes collide with
#       the checkpoint (e.g. a 61-layer target + 2-way PP builds prefixes 31..34
#       while the checkpoint expects 61..64), breaking weight loading. Additionally,
#       `config.target_layer_count` (used to index `layer_types` for draft
#       attention) ends up wrong.
#    How：
#       Patch the affected Eagle3 draft models to use
#       `get_total_num_hidden_layers()` instead, which matches the checkpoint's
#       global layer indices and keeps `target_layer_count` correct.
#    Related PR (if no, explain why):
#       No, vllm-ascend-specific Eagle3 + PP weight-loading fix.
#    Future Plan:
#       Remove this patch once upstream Eagle3 draft models compute
#       `target_layer_num` from the total (not per-stage) hidden layer count.
#
# ** 7. File: worker/patch_eagle3_pp_aux.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. Eagle3 target inner models (`DeepseekV2Model`, EagleModelMixin-based models)
#      `forward`, `make_empty_intermediate_tensors`
#    Why:
#       In Eagle3 speculative decoding with Pipeline Parallelism (PP), auxiliary
#       hidden states are collected from specific target model layers (e.g.,
#       layers 2, N/2, N-3). When these layers span multiple PP stages, the last
#       PP rank (where the drafter runs) only sees a subset of aux states, causing
#       `combine_hidden_states` to fail with a k-axis shape mismatch.
#    How：
#       Wrap the inner model's `forward` and `make_empty_intermediate_tensors` to
#       transparently pass aux hidden states through IntermediateTensors across PP
#       stages. Each PP stage carries forward all aux states from previous stages,
#       and the last PP rank merges them into a single list for the drafter.
#    Related PR (if no, explain why):
#       No, vllm-ascend-specific Eagle3 + PP aux-state propagation.
#    Future Plan:
#       Remove this patch once upstream Eagle3 supports collecting aux hidden
#       states that span PP stages without cross-stage propagation.
#
# ** 8. File: worker/patch_fused_moe.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.fused_moe.FusedMoE`
#    Why:
#       The worker process re-imports the FusedMoE factory after the platform
#       patch has already redirected it to AscendMoERunner. Re-applying the
#       monkey-patch directly would wrap an already-patched factory.
#    How：
#       Reuse the platform patch (`platform/patch_fused_moe.py`) so the monkey
#       patch lives in a single module and is applied idempotently during worker
#       initialization.
#    Related PR (if no, explain why):
#       No, see platform/patch_fused_moe.py.
#    Future Plan:
#       Remove this module together with platform/patch_fused_moe.py once upstream
#       exposes a backend dispatch / plugin hook for selecting the MoE runner.
#
# ** 9. File: worker/patch_idex_310.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.fla.ops.index.prepare_chunk_indices`
#      `vllm.model_executor.layers.fla.ops.index.prepare_chunk_offsets`
#    Why:
#       310P uses Ascend-friendly chunk index helpers for Qwen GDN prefill.
#    How:
#       Replace upstream FLA chunk index helper functions with 310P implementations.
#
#   2. `vllm_ascend.spec_decode.llm_base_proposer.AscendSpecDecodeBaseProposer.set_inputs_first_pass`
#    Why:
#       310P needs to protect the tail slot during MTP input_ids shift to avoid
#       GatherV2 corruption from persistent drafter input buffers.
#    How:
#       Reuse the 310P proposer implementation for the first-pass input shift.
#
#   3. `vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.QwenGatedDeltaNetAttention`
#    Why:
#       Qwen GDN needs 310P-specific state helpers, forward core, state dtype,
#       and attention backend/builder wiring.
#    How:
#       Patch Qwen GDN methods to use Ascend GDN implementations and the 310P
#       GDN attention backend. RC devices also route upstream GDNAttentionBackend
#       to the 310P metadata builder.
#    Related PR (if no, explain why):
#       No, 310P custom operator and backend behavior are vllm-ascend specific.
#    Future Plan:
#       Remove this patch when upstream exposes stable hooks for 310P GDN
#       chunk metadata, spec-decode input layout, and backend selection.
#
# ** 10. File: worker/patch_kimi_k25.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.kimi_k25_vit.Learnable2DInterpPosEmbDivided_fixed.forward`
#    Why:
#       The forward method uses interpolate with ops not supported on NPU.
#    How：
#       Replace with a new forward that uses CPU for interpolate when shape mismatch,
#       and use get_rope_shape to handle the rope shape interpolation.
#    Future Plan:
#       Remove this patch when vLLM aligns with the latest main.
#
# ** 11. File: worker/patch_mamba_utils.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel`
#    Why:
#       Oringnal batch_memcpy_kernel implemented in vLLM might encounter bugs when running on
#       Ascend hardwares.
#    How：
#       patch to fix related bugs.
#    Future Plan:
#       Remove this patch when:
#       (1) oringnal batch_memcpy_kernel can run on Ascend hardware.
#       or
#       (2) design a dispatch mechanism for batch_memcpy_kernel.
#   2. `vllm.v1.worker.mamba_utils.batch_memcpy = batch_memcpy`
#    Why:
#       vLLM use BLOCK_SIZE 1024 for batch_memcpy_kernel. This results in suboptimal performance
#       on Ascend hardwares.
#    How：
#       patch to change BLOCK_SIZE to 8192.
#    Future Plan:
#       Remove this patch when:
#       design a dispatch mechanism for batch_memcpy_kernel.
#   3. `mamba_utils.preprocess_mamba = preprocess_mamba`
#    Why:
#       1. preprocess_mamba has a assert logic, cause kv transfer call fails
#       2. preprocess_mamba copy the state of previous step to the last block before kv transfer load
#    How:
#       1. patch to remove assert
#       2. path to only collect copy metadata in preprocess_mamba(and do actual copy after kv transfer load).
#    Future Plan:
#       Remove this patch when:
#       vLLM itself supports kv transfer for mamba
#
# ** 12. File: worker/patch_minimax_m2.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.minimax_m2.MiniMaxM2MoE.forward`
#    Why:
#       In TP mode, MiniMax-M2 MoE needs a backend-aware reduction path to avoid
#       unnecessary communication / maintain correctness on NPU.
#    How：
#       Replace the forward to call `experts.maybe_all_reduce_tensor_model_parallel`
#       when `tp_size > 1`.
#    Related PR (if no, explain why):
#       No, model-specific behavior.
#    Future Plan:
#       Move this behavior upstream once a generic MoE reduce hook exists.
#
#   2. `vllm.model_executor.models.minimax_m2.MiniMaxM2Attention.forward`
#    Why:
#       MiniMax-M2 attention benefits from the NPU fused split-qkv + RMSNorm + rope
#       kernel path.
#    How：
#       Replace `forward` to call `torch.ops.vllm.split_qkv_tp_rmsnorm_rope` before
#       the upstream attention and output projection steps.
#    Related PR (if no, explain why):
#       No, backend-specific fused kernel path.
#    Future Plan:
#       Remove this patch when upstream exposes a backend dispatch path for this
#       fused attention preparation.
#
#   3. `vllm.model_executor.models.minimax_m2.MiniMaxM2Model.load_weights`
#    Why:
#       MiniMax-M2 fp8 checkpoints may store fp8 weights with per-block inverse
#       scales. On NPU we load bf16 weights by dequantizing at load time.
#    How：
#       Inject fp8 dequant helpers and wrap `load_weights` to convert fp8 weight +
#       `weight_scale_inv` pairs into bf16 blocks before delegating to upstream.
#    Related PR (if no, explain why):
#       No, fp8 load format and backend constraints are model/backend specific.
#    Future Plan:
#       Remove this patch when upstream supports MiniMax-M2 fp8 loading on NPU.
#
# ** 13. File: worker/patch_minimax_m2_linear_attn.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.mamba.linear_attn.MiniMaxText01RMSNormTP.__init__`
#      `vllm.model_executor.layers.mamba.linear_attn.MiniMaxText01RMSNormTP.weight_loader`
#    Why:
#       MiniMax-M2 linear attention RMSNorm needs weight sharding that can follow
#       TP layout (and sometimes kv-head replication) on NPU.
#    How：
#       Override `__init__` to parameterize weight shard world/rank and install a
#       sharded `weight_loader` implementation.
#    Related PR (if no, explain why):
#       No, upstream API surface differs across versions.
#    Future Plan:
#       Remove this patch when upstream exposes stable sharding hooks for this layer.
#
#   2. `vllm.model_executor.layers.mamba.linear_attn.MiniMaxText01RMSNormTP.forward_qk`
#      (or older `_normalize_qk`)
#    Why:
#       q/k norm for linear attention is performance-sensitive. On NPU, a fused
#       rms_norm kernel is faster and TP needs a global rstd correction.
#    How：
#       Replace q/k normalization with NPU rms_norm fast path and TP-global rstd
#       correction; fall back to upstream implementation on non-NPU.
#    Related PR (if no, explain why):
#       No, backend-specific optimization.
#    Future Plan:
#       Remove this patch when upstream adds a backend dispatch path for q/k norm.
#
# ** 14. File: worker/patch_npugraph_ex_triton.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `npugraph_ex.core._concrete_graph.ValuePack`,
#      `npugraph_ex.npu_fx_compiler._unpack_meta`,
#      `npugraph_ex.npu_fx_compiler._NpuGraphConverter._unpack_npu`
#    Why:
#       In the Triton scenario, npugraph_ex backend needs to process the value pack of the input parameters.
#    How：
#       Supplement the relevant processing logic through patches.
#    Related PR (if no, explain why):
#       https://gitcode.com/Ascend/torchair/pull/2575
#    Future Plan:
#       Remove this patch when the PTA version used by vllm-ascend has been upgraded.
#
# ** 15. File: worker/patch_process_weights_after_loading.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.model_loader.utils.process_weights_after_loading`
#      `vllm.model_executor.model_loader.base_loader.process_weights_after_loading`
#      and imported references in vllm-ascend model loaders
#    Why:
#       DSA attention is implemented in vllm-ascend as the plugin layer
#       `DSAAttention`. Upstream vLLM only runs post-load attention weight
#       processing for built-in attention classes, so
#       `DSAAttention.process_weights_after_loading()` is skipped in the
#       original loader flow. DSV4 DSA-CP o-proj TP initialization must run in
#       this post-load phase rather than being initialized lazily in forward.
#    How:
#       Rebind the upstream `process_weights_after_loading` helper, including
#       already-imported loader references, so `DSAAttention` participates in
#       the same post-load traversal while preserving the original quant-method
#       and torchao reload behavior.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm-ascend/pull/10694
#       https://github.com/vllm-project/vllm/pull/46828
#    Future Plan:
#       Remove this patch once the supported vLLM version includes PR #46828.
#       Then register `DSAAttention` through vLLM's post-load weight-processing
#       registry instead of monkey-patching model-loader helpers.
#
# ** 16. File: worker/patch_qwen3_5.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.qwen3_5.Qwen3_5GatedDeltaNet._forward_core`
#    Why:
#       The class Qwen3_5GatedDeltaNet reuse the `_forward_core` method of Qwen3NextGatedDeltaNet,
#       but the ascendC ops of Qwen3NextGatedDeltaNet do not support ssm_state with float32 format.
#    How：
#       patch Qwen3_5GatedDeltaNet._forward_core to use triton ops like `fused_recurrent_gated_delta_rule`.
#    Future Plan:
#       Remove this patch when all ops in _forward_core support both Qwen3_5 and Qwen3Next.
#
# ** 17. File: worker/patch_qwen3_dflash.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.qwen3_dflash.DFlashQwen3Model.precompute_and_store_context_kv`
#    Why:
#       The function directly calls the ops.rms_norm and ops.rotary_imbedding operators,
#       but NPU does not have a corresponding implementation.
#    How：
#       Replace ops.* with the internal implementation of vllm-ascend.
#    Future Plan:
#       Remove this patch when vllm-ascend supports pattern matching for ops.*.
#
# ** 18. File: hunyuan_vl_processor_compat.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.hunyuan_vision.HunYuanVLProcessingInfo.get_hf_processor`
#      and the vLLM processor lazy registry
#    Why:
#       The supported vLLM refs currently straddle the HunyuanVL processor
#       migration. v0.23.0 still bundles the processor, while the verified
#       main ref uses the Transformers-native processor but predates the full
#       Transformers 5.13 registry and prompt-protocol cleanup.
#    How:
#       Preserve the bundled v0.23.0 processor protocol, translate its image
#       processor registration to Transformers 5.13, and complete the native
#       processor registry, loader, and tokenizer schema on the main ref.
#    Related PR:
#       1. https://github.com/vllm-project/vllm/pull/47872
#       2. https://github.com/vllm-project/vllm/pull/47867
#    Future Plan:
#       Remove this patch and delete the `install_hunyuan_vl_processor_compat()`
#       call from `vllm_ascend/__init__.py` only after both supported vLLM refs
#       contain `4a6440acefbd4d977620bdb6dfb7fb325cd9bda7` (vLLM PR #47867,
#       merged into vLLM main at 2026-07-11 07:56:14 UTC), or an equivalent
#       fix, and the supported HunyuanOCR tokenizer artifacts expose the named
#       special-token schema required by Transformers 5.13.
#
# ** 19. File: worker/patch_qwen3_next_mtp.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.utils.bind_kv_cache`
#    Why:
#       'bind_kv_cache' func will raise an exception when current_platform is npu.
#    How：
#       Replace with a new bind_kv_cache.
#       Skip the raise.
#    Related PR (if no, explain why):
#       It need discuss.
#    Future Plan:
#       Remove this patch after discussing with vllm community and adapting bind_kv_cache to npu.
#
# ** 20. File: worker/patch_qwen3vl.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.qwen3_vl.Qwen3VLForConditionalGeneration._get_deepstack_input_embeds`
#    Why:
#       support flash comm v1 for qwen3vl.
#    How：
#       override _get_deepstack_input_embeds method with the flash comm v1 implementation.
#    Future Plan:
#       Remove this patch when https://github.com/vllm-project/vllm-ascend/issues/5712 is completed.
#   2. `vllm.model_executor.models.qwen3_vl_moe.Qwen3MoeLLMForCausalLM.start_layer`,
#      `vllm.model_executor.models.qwen3_vl_moe.Qwen3MoeLLMForCausalLM.end_layer`
#    Why:
#       Qwen3-VL-MoE checks the language-model pipeline boundary on non-first
#       PP ranks, but Qwen3MoeLLMForCausalLM keeps start_layer/end_layer only
#       on the inner model object.
#    How:
#       Expose start_layer/end_layer properties on Qwen3MoeLLMForCausalLM and
#       forward them to the inner model.
#    Future Plan:
#       Remove this patch when upstream vLLM exposes these PP layer boundaries
#       on the Qwen3-VL-MoE language-model wrapper.
#   3. `vllm.model_executor.models.qwen3.Qwen3Attention.forward` and
#      `vllm.model_executor.models.qwen3_moe.Qwen3MoeAttention.forward`
#    Why:
#       support triton_split_qkv_rmsnorm_mrope fused kernel for Qwen3Attention and Qwen3MoeAttention.
#    How：
#       override forward method with the triton_split_qkv_rmsnorm_mrope fused kernel,
#       when using mrope.
#    Future Plan:
#       Remove this patch when vllm-ascend supports pattern matching for this fused kernel.
#
# ** 21. File: worker/patch_rejection_sampler.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.sample.rejection_sampler`
#    Why:
#       - some functions from `rejection_sampler` are not supported or slow on npu.
#    How：
#       - add npu_top_k_top_p to 'apply_sampling_constraints' func
#       - add custom triton kernel to `expand_batch_to_tokens` and `rejection_sample`
#    Related PR (if no, explain why):
#       Let vLLM support triton ops dispatch.
#    Future Plan:
#       1. make these functions as class func of RejectionSampler, create AscendRejectionSampler
#           to override them, then delete the patch file `worker/patch_rejection_sampler.py`.
#       2. make these functions as costom op, then remove AscendRejectionSampler
#
# ** 22. File: worker/patch_routed_experts_capture.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.fused_moe.routed_experts_capturer.RoutedExpertsCapturer.capture`
#    Why:
#       The upstream implementation doesn't support vllm-ascend specific MoE communication types
#       (ALLTOALL and MC2). In the SP + modular-kernel path, the original code cannot correctly
#       handle tensor splitting and all-gather operations on NPU, especially when tokens are
#       unevenly distributed across TP ranks or padded to max_tokens in MC2 mode.
#    How：
#       Override the capture method to add support for vllm-ascend's MoECommType:
#         - Check `_EXTRA_CTX.moe_comm_type` to determine if ALLTOALL or MC2 mode is active
#         - Calculate correct gather_topk_ids_shape based on communication type:
#           * ALLTOALL: uses actual token_num_per_dp for shape calculation
#           * MC2: uses padded max_tokens * tp_size for shape calculation
#         - Properly handle tensor_split and all_gather operations for NPU distributed communication
#    Future Plan:
#       Remove this patch when upstream vLLM supports MoE communication type abstraction that
#       can be extended by hardware plugins like vllm-ascend.
#
# ** 23. File: worker/patch_step3p5.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.step3p5.Step3p5Attention.forward`
#    Why:
#       Add split_qkv_rmsnorm_rope support for step3.5/3.7. This model can't use
#       fusion pass to enable this op because of 2 reasons: 1) Step uses
#       Gemmanorm instead of normal norm, so existing fusion pass can't be
#       matched. 2) Step has 2 kinds of attention layer (full attention and
#       sliding window attention), each has different number of q_head. Right
#       now, torch.compile will use same pattern to match different attention
#       layer so there will be shape mismatch in split op.
#    How:
#       Monkey-patch Step3p5Attention.forward to enable split_qkv_rmsnorm_rope.
#    Future Plan:
#       Remove this patch once torch.compile fully supports matching pattern from
#       op's params.
#
# ** 24. File: worker/patch_triton.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.mamba.ops`, `vllm.model_executor.layers.fla.ops`,
#      `vllm.v1.worker.gpu.sample.gumbel.gumbel_sample`
#    Why:
#       triton ops in vLLM perform not good on NPU. And there is no dispatch mechanism for triton ops.
#    How：
#       override triton ops in vLLM with ascend implementation
#    Related PR (if no, explain why):
#       Let vLLM support triton ops dispatch.
#    Future Plan:
#       Remove this patch when vLLM support the dispatch function.
#
#   2. `triton.next_power_of_2`
#    Why:
#       The Triton version bundled with torch_npu on Ascend NPU
#       does not include `next_power_of_2`, which is called by
#       upstream vLLM and vLLM-Ascend code in 94+ places.
#       Additionally, when Triton is not available (HAS_TRITON=False),
#       vLLM uses TritonPlaceholder which also lacks this function.
#    How：
#       Import `triton` from vllm.triton_utils (which handles both
#       real Triton and TritonPlaceholder) and inject `next_power_of_2`
#       onto the module, reusing `vllm.utils.math_utils.next_power_of_2`.
#    Related PR (if no, explain why):
#       No, torch_npu Triton compatibility issue.
#    Future Plan:
#       Remove this patch when torch_npu's Triton includes
#       next_power_of_2 or when vLLM no longer calls triton.next_power_of_2.
#
# ** 25. File: worker/patch_v2/patch_attn_utils.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.attn_utils.get_kv_cache_spec`
#    Why:
#       The current v2 worker still goes through the shared upstream v1 helper
#       to build KV cache specs. For Ascend MLA layers that helper returns the
#       generic `MLAAttentionSpec`, but NPU-side cache allocation and reshape
#       logic expects `AscendMLAAttentionSpec`.
#    How：
#       Monkey-patch `get_kv_cache_spec` so regular attention layers keep the
#       upstream behavior while MLA layers are rewritten to
#       `AscendMLAAttentionSpec`, including the FA-quant head-size adjustment.
#    Related PR (if no, explain why):
#       No. This is a plugin-side compatibility patch for the current upstream
#       helper path.
#    Future Plan:
#       Remove this patch once upstream adds a backend hook for KV cache spec
#       construction or v2 worker no longer depends on the shared v1 helper.
#
# ** 26. File: worker/patch_v2/patch_block_table.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.block_table.BlockTables`
#    Why:
##      vllm-ascend need to initialize slot mapping as torch.int32 dtype,
#       but vllm default is torch.int64 dtype.
#    How：
#       replace BlockTables with AscendBlockTables which initialize slot mapping
#       as torch.int32 dtype.
#    Future Plan:
#       remove this patch when vLLM-ascend's BlockTables can initialize
#       slot mapping as torch.int64 dtype.
#
# ** 27. File: worker/patch_v2/patch_dflash_speculator.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.spec_decode.dflash.speculator.DFlashCudaGraphManager`
#    Why:
#       The v2 DFlash spec-decode path uses upstream `DFlashCudaGraphManager`
#       (CUDA graph) for spec decoding, which is not usable on Ascend.
#    How：
#       Replace `DFlashCudaGraphManager` with the Ascend `DFlashAclGraphManager`
#       (ACL graph) on the upstream speculator module.
#    Related PR (if no, explain why):
#       No, vllm-ascend v2 spec-decode ACL graph integration.
#    Future Plan:
#       Remove this patch once upstream exposes a backend-dispatchable spec-decode
#       graph manager abstraction.
#
# ** 28. File: worker/patch_v2/patch_eagle_speculator.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.spec_decode.autoregressive.speculator.PrefillSpeculatorCudaGraphManager`,
#      `vllm.v1.worker.gpu.spec_decode.autoregressive.speculator.DecodeSpeculatorCudaGraphManager`
#    Why:
#       The v2 Eagle spec-decode path uses upstream CUDA graph managers for
#       prefill/decode spec decoding, which are not usable on Ascend.
#    How：
#       Replace the prefill/decode speculator graph managers on the upstream
#       module with `PrefillEagleAclGraphManager` / `DecodeEagleAclGraphManager`
#       (ACL graph).
#    Related PR (if no, explain why):
#       No, vllm-ascend v2 spec-decode ACL graph integration.
#    Future Plan:
#       Remove this patch once upstream exposes a backend-dispatchable spec-decode
#       graph manager abstraction.
#
# ** 29. File: worker/patch_v2/patch_input_batch.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.input_batch.InputBatch`
#    Why:
#       vllm use InputBatch to make dummy tensors. in `model_runner.py` and `cudagraph_utils.py`
#       which make it difficult to inherit from vllm methods.
#    How：
#       replace InputBatch with AscendInputBatch.
#    Future Plan:
#       remove this patch when vLLM-ascend's make_dummy behavior aligns with vLLM.
#
# ** 30. File: worker/patch_v2/patch_model_state.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.model_states.default.init_model_state`
#    Why:
##      vllm's prepare_attn in ModelState is different from vllm,
#       we need to override init_model_state.
#    How：
#       Define AscendModelState and initialize it in init_model_state.
#    Future Plan:
#       remove this when vllm-ascend's attention metadata is align with vllm.
#
# ** 31. File: worker/patch_v2/patch_triton.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.sample.logprob`, `vllm.v1.worker.gpu.sample.penalties.apply_penalties`,
#      `vllm.v1.worker.gpu.sample.gumbel.gumbel_sample`
#    Why:
#       triton ops in vLLM perform not good on NPU. And there is no dispatch mechanism for triton ops.
#    How：
#       override triton ops in vLLM with ascend implementation
#    Related PR (if no, explain why):
#       Let vLLM support triton ops dispatch.
#    Future Plan:
#       Remove this patch when vLLM support the dispatch function.
#
# ** 32. File: worker/patch_v2/patch_use_v2_model_runner.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.config.vllm.VllmConfig.use_v2_model_runner`
#    Why:
#       EngineCore subprocesses only load global/platform patches, while workers
#       also import this compatibility module. The actual monkey-patch is defined
#       in `platform/patch_use_v2_model_runner.py`.
#    How：
#       Reuse the platform patch so EngineCore and worker processes share the
#       same `use_v2_model_runner` behavior.
#    Related PR (if no, explain why):
#       See platform/patch_use_v2_model_runner.py.
#    Future Plan:
#       Remove this module together with the platform patch once vllm-ascend
#       fully supports the v2 model runner.
#
# ** 33. File: worker/patch_v2/patch_uva.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.states.UvaBuffer`
#    Why:
#       ASCEND NPUs do not support UVA yet, so we need to wrap it in vLLM.
#    How：
#       make UvaBuffer a dummy class, mimic the interface of vllm UvaBuffer.
#    Future Plan:
#       Remove this patch when NPU support UVA.
#
# ** 36. File: hunyuan_vl_processor_compat.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.hunyuan_vision.HunYuanVLProcessingInfo.get_hf_processor`
#      and the vLLM processor lazy registry
#    Why:
#       The supported vLLM refs currently straddle the HunyuanVL processor
#       cleanup. v0.25.1 still bundles the processor and retains stale
#       lazy-registry entries, while the verified main ref uses the native
#       Transformers processor and has the upstream registry cleanup.
#    How:
#       Normalize the processor registry, loader, and tokenizer schema on both
#       refs, and preserve the correct image-token protocol for each lane.
#    Related PR:
#       1. https://github.com/vllm-project/vllm/pull/47872
#       2. https://github.com/vllm-project/vllm/pull/47867
#    Future Plan:
#       Remove this patch and delete the `install_hunyuan_vl_processor_compat()`
#       call from `vllm_ascend/__init__.py` only after both supported vLLM refs
#       contain `4a6440acefbd4d977620bdb6dfb7fb325cd9bda7` (vLLM PR #47867,
#       merged into vLLM main at 2026-07-11 07:56:14 UTC), or an equivalent
#       fix, and the supported HunyuanOCR tokenizer artifacts expose the named
#       special-token schema required by Transformers 5.13.
