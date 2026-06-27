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
# ** 1. File: platform/patch_distributed.py**
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
# ** 2. File: platform/patch_mamba_config.py**
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
# ** 3. File: platform/patch_multiproc_executor.py**
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
# ** 5. File: platform/patch_balance_schedule.py**
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
# ** 6. File: platform/patch_minimax_m2_config.py**
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
# ** 7. File: platform/patch_chat_usage_accounting.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat`
#    Why:
#       Chat usage accounting needs to report
#       `completion_tokens_details.reasoning_tokens` for both streaming and
#       non-streaming chat completions.
#    How：
#       Extend `UsageInfo` and update chat usage construction to count reasoning
#       tokens from raw output token ids when the active reasoning parser
#       supports token counting.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/37955
#    Future Plan:
#       Remove this patch once the runtime vLLM version contains the upstream
#       reasoning usage-accounting support.
#
# ** 7a. File: platform/patch_minimax_usage_accounting.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.reasoning.minimax_m2_reasoning_parser`
#    Why:
#       MiniMax-M2 reasoning parsers need token-count support for the generic
#       chat usage-accounting wrapper.
#    How：
#       Monkey-patch MiniMax reasoning token counters so usage accounting can
#       derive reasoning-token counts from raw output token ids.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/37955
#    Future Plan:
#       Remove this patch once the runtime vLLM version contains the upstream
#       MiniMax reasoning-token counting fix.
#
# ** 7b. File: platform/patch_glm_tool_call_streaming.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat`
#    Why:
#       GLM tool-call streaming can emit final remaining-argument chunks with
#       repeated tool-call metadata, and can combine terminal argument bytes with
#       `finish_reason="tool_calls"` in the same SSE chunk.
#    How：
#       Monkey-patch remaining-argument delta construction to emit only argument
#       fragments by default, and split terminal argument chunks into an argument
#       chunk followed by an empty finish chunk.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/issues/44098
#       https://github.com/vllm-project/vllm/pull/44099
#       https://github.com/vllm-project/vllm-ascend/issues/8327
#       https://github.com/vllm-project/vllm-ascend/pull/8178
#    Future Plan:
#       Remove this patch once the supported vLLM version contains the upstream
#       GLM tool-call final chunk fixes.
#
# ** 7c. File: platform/patch_glm47_tool_call_parser.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.tool_parsers.glm47_moe_tool_parser.Glm47MoeModelToolParser`
#    Why:
#       vLLM's GLM47 streaming parser can drop complete inline zero-argument
#       tool calls such as `<tool_call>get_current_time</tool_call>`, while
#       non-streaming parses the same output correctly.
#    How：
#       Monkey-patch GLM47 tool-call region extraction so complete inline
#       zero-argument regions are normalized for the existing streaming name
#       extractor without emitting partial names for incomplete regions.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/issues/44326
#       https://github.com/vllm-project/vllm/pull/44327
#    Future Plan:
#       Remove this patch once the supported vLLM version contains the upstream
#       GLM47 inline zero-argument streaming parser fix.
#
# ** 7d. File: platform/patch_anthropic_system_message.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.entrypoints.anthropic.protocol.AnthropicMessage`
#      `vllm.entrypoints.anthropic.serving.AnthropicServingMessages`
#    Why:
#       Recent Claude Code clients can send `role: system` entries inside the
#       Anthropic Messages API `messages` array. The pinned vLLM rejects those
#       requests before inference starts.
#    How：
#       Monkey-patch Anthropic message role validation to accept `system`, merge
#       inline system messages with the top-level system prompt, and skip inline
#       system entries when converting the remaining chat history.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/issues/44000
#       https://github.com/vllm-project/vllm/pull/44283
#    Future Plan:
#       Remove this patch once the supported vLLM version contains PR #44283.
#
# ** 10a. File: platform/patch_kv_cache_utils.py**
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
# ** 10ab. File: worker/patch_v2/patch_attn_utils.py**
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
# ** 10. File: platform/patch_profiling_chunk.py**
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
# ** 10b. File: platform/patch_pp_mtp.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.config.model.ModelConfig.verify_with_parallel_config`
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
# ** 11. File: platform/patch_tool_choice_none_content.py**
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
# ** 12. File: platform/patch_deepseek_v4_tool_call_parser.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.tool_parsers.deepseekv4_tool_parser.DeepSeekV4ToolParser`
#    Why:
#       Upstream vLLM now includes DeepSeek V4 tokenizer/renderer/reasoning
#       registration, but its streaming tool-call delta parsing does not guarantee
#       incremental `arguments` emission for long argument payloads.
#    How:
#       Monkey-patch `DeepSeekV4ToolParser` stream parsing to emit tool-call
#       metadata in the first delta and stream argument fragments incrementally.
#    Related PR (if no, explain why):
#       Upstream vLLM main behavior as of current runtime.
#    Future Plan:
#       Remove this patch if upstream streaming behavior is updated to satisfy the
#       same DeepSeek DSML incrementality contract.
#
# ** 12a. File: platform/patch_minimax_m2_tool_call_parser.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.tool_parsers.minimax_m2_tool_parser.MinimaxM2ToolParser`
#    Why:
#       vLLM 0.21.0 only emits MiniMax-M2 tool-call arguments after a complete
#       `<invoke>...</invoke>` block, so long arguments are buffered instead of
#       streamed incrementally.
#    How:
#       Monkey-patch the MiniMax-M2 parser to emit the tool name once the
#       `<invoke name=...>` header is available and then stream partial
#       `<parameter>` values as JSON argument fragments.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/40253
#       https://github.com/vllm-project/vllm/pull/40298
#    Future Plan:
#       Remove this patch once the supported vLLM version contains the upstream
#       MiniMax-M2 incremental tool-call streaming fix.
#
# ** 13. File: platform/patch_camem_allocator.py**
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
# ** 14. File: platform/patch_scheduler.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.core.sched.scheduler.Scheduler._mamba_block_aligned_split`
#    Why:
#       Upstream vLLM has an assert logic, cause it fails when external KV connector hit
#    How:
#      remove the assert
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/43935
#    Future Plan:
#       Remove this patch if upstream streaming behavior is updated to support mamba external KV connector
# ** 15. File: platform/patch_weight_transfer_engine.py**
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
# ** 15. File: platform/patch_kv_cache_coordinator.py**
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
# * Worker Patch:
# ===============
#
# ** 1. File: worker/patch_distributed.py**
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
# ** 3. File: worker/patch_triton.py**
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
# ** 4. File: worker/patch_qwen3_next_mtp.py**
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
# ** 5. File: worker/patch_rejection_sampler.py**
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
## ** 6. File: worker/patch_module.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.attention.backends.gdn_attn.torch.argsort`
#    Why:
#       1. 'torch.argsort' func of npu does not support bool.
#       2. Without `stable=True`, the output will have a lot of redundant tokens.
#    How：
#       Replace with a new torch.argsort that will cast the input to torch.int32
#       and do stable sort.
#    Related PR (if no, explain why):
#       1. It depends on torch_npu.
#       2. https://github.com/vllm-project/vllm/pull/30632
#    Future Plan:
#       Remove this patch when bool is supported in 'torch.argsort' func of npu.
#       Make 'torch.argsort' in `vllm.v1.attention.backends.gdn_attn` be stable.
#   2. `vllm_ascend.ops.gdn_attn_builder.AscendGDNAttentionMetadataBuilder.build`
#    Why:
#       Qwen3.5/Qwen3Next GDN Decode/Specific Decode on NPU needs prebuilt varlen chunk metadata
#       to avoid forward-time host round-trips that break async scheduling.
#    How：
#       Override the GDN attention metadata builder for Ascend backend and attach
#       prebuilt device metadata bundle onto the returned attention metadata object.
#    Future Plan:
#       Remove this patch when upstream exposes a backend hook for extending GDN
#       metadata or when the optimization is accepted upstream directly.
#
# ** 8. File: worker/patch_qwen3_next.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.qwen3_next.Qwen3NextGatedDeltaNet.forward`
#    Why:
#       The Qwen3Next GatedDeltaNet forward cannot directly add custom operators.
#    How：
#       Add a branch in Qwen3NextGatedDeltaNet.forward to adapt to fused_qkvzba_split_reshape_cat.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/30863
#    Future Plan:
#       Remove this patch when vLLM support these operators.
#
#   2. `vllm.model_executor.models.qwen3_next.Qwen3NextGatedDeltaNet._forward_core`
#    Why:
#       triton ops fused_recurrent_gated_delta_rule and fused_gdn_gating in vLLM perform not good on NPU.
#    How：
#       add a new fused triton ops in vLLM with ascend implementation.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/30860
#    Future Plan:
#       Remove this patch when vLLM support these operators.
#
#   3. `vllm.model_executor.models.qwen3_next.Qwen3NextGatedDeltaNet._forward_core`
#    Why:
#       The Qwen3Next GatedDeltaNet _forward_core cannot directly add custom operators.
#    How：
#       Add a branch in Qwen3NextGatedDeltaNet._forward_core to adapt to fused_gdn_gating_patch.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/31002
#    Future Plan:
#       Remove this patch when vLLM support these operators.
#
# ** 10. File: worker/patch_qwen3vl.py**
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
#
# ** 11. File: worker/patch_npugraph_ex_triton.py**
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
# ** 12. File: worker/patch_v2/patch_uva.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.states.UvaBuffer`
#    Why:
#       ASCEND NPUs do not support UVA yet, so we need to wrap it in vLLM.
#    How：
#       make UvaBuffer a dummy class, mimic the interface of vllm UvaBuffer.
#    Future Plan:
#       Remove this patch when NPU support UVA.
#
# ** 13. File: worker/patch_kimi_k25.py**
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
# ** 14. File: worker/patch_draft_quarot.py**
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
# ** 15. File: worker/patch_minimax_m2.py**
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
# ** 16. File: worker/patch_minimax_m2_linear_attn.py**
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
# ** 17. File: worker/patch_qwen3_5.py**
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
# ** 18. File: worker/patch_cudagraph.py**
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
# ** 19. File: worker/patch_deepseek_mtp.py**
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
# ** 19b. File: worker/model_runner_v1.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `NPUModelRunner._check_and_update_cudagraph_mode`
#    Why:
#       The upstream `GPUModelRunner._check_and_update_cudagraph_mode` initializes
#       drafter cudagraph keys unconditionally, but in PP mode only the last rank
#       loads the drafter. The previous hacky workaround temporarily set
#       `self.speculative_config = None` to bypass super()'s drafter init, then
#       restored it and called a separate `_maybe_initialize_drafter_cudagraph_keys`
#       helper. This state-mutation pattern is fragile and hard to maintain.
#    How：
#       Directly inline the upstream cudagraph mode resolution logic with Ascend-specific
#       additions: wrap `resolve_cudagraph_mode_and_sizes` with `update_pass_config` for
#       `enable_sp`, add PP last-rank guard for drafter initialization, and call
#       `set_graph_params`/`set_draft_graph_params` for ACL graph params. Remove the
#       `_maybe_initialize_drafter_cudagraph_keys` helper entirely.
#    Related PR (if no, explain why):
#       No, cleaner PP+MTP support without speculative_config state mutation.
#    Future Plan:
#       Remove this override once upstream exposes a hook for drafter cudagraph key
#       initialization that respects PP rank boundaries.
# ** 20. File: worker/patch_mamba_utils.py**
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
# ** 21. File: worker/patch_weight_utils.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.deepseek_v2.DeepseekV2ForCausalLM.load_weights`
#    Why:
#       The C8 weight quantized by modelslim will modify the model structure,
#       and the scale and offset required for kvcache quantization will increase.
#       In addition, the names of the quantization parameters are different from
#       those in the community.
#    How：
#       we have enhanced the maybe_remap_kv_scale_name function.
#    Future Plan:
#       The maybe_remap_kv_scale_name function of the community is reconstructed to support
#       multiple backends.
# ** 22. File: worker/patch_v2/patch_input_batch.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.input_batch.InputBatch`
#    Why:
#       vllm use InputBatch to make dummy tensors. in `model_runner.py` and `cudagraph_utils.py`
#       which make it difficult to inherit from vllm methods.
#    How：
#       replace InputBatch with AscendInputBatch.
#    Future Plan:
#       remove this patch when vLLM-ascend's make_dummy behavior aligns with vLLM.
# ** 23. File: worker/patch_v2/patch_block_table.py**
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
# ** 24. File: worker/patch_v2/patch_model_state.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.model_states.default.init_model_state`
#    Why:
##      vllm's prepare_attn in ModelState is different from vllm,
#       we need to override init_model_state.
#    How：
#       Define AscendModelState and initialize it in init_model_state.
#    Future Plan:
#       remove this when vllm-ascend's attention metadata is align with vllm.
# ** 25. File: worker/patch_v2/patch_triton.py**
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
# ** 26. File: worker/patch_gqa_c8.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.qwen3.Qwen3ForCausalLM.load_weights`
#    Why:
#       The GQA W8A8C8 model stores per-channel KV cache scales and offsets
#       (k_cache_scale, k_cache_offset, v_cache_scale, v_cache_offset) under
#       weight names that AutoWeightsLoader does not recognise and would
#       silently discard.  Without these scales the INT8 KV cache cannot be
#       dequantised correctly at inference time.
#    How:
#       Wrap load_weights to intercept the C8 scale/offset tensors before they
#       reach the base loader.  Each intercepted tensor is routed to the
#       corresponding nn.Parameter via its weight_loader, then excluded from
#       the remaining weight stream so the base loader never sees it.
#    Related PR (if no, explain why):
#       This PR (Qwen3-32B and GLM4.7  W8A8C8 support).  Upstream vLLM's weight-loading
#       pipeline does not yet have a generic hook for hardware-plugin-defined
#       KV cache parameters.
#    Future Plan:
#       Remove this patch when vLLM provides a first-class extension point
#       for loading extra KV cache quantisation parameters in model load_weights,
#       or when the GQA model's weight names are aligned with the parameter
#       names expected by the quantisation backend.
# ** 27. File: worker/patch_qwen3vl.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.qwen3.Qwen3Attention.forward` and
#      `vllm.model_executor.models.qwen3_moe.Qwen3MoeAttention.forward`
#    Why:
#       support triton_split_qkv_rmsnorm_mrope fused kernel for Qwen3Attention and Qwen3MoeAttention.
#    How：
#       override forward method with the triton_split_qkv_rmsnorm_mrope fused kernel,
#       when using mrope.
#    Future Plan:
#       Remove this patch when vllm-ascend supports pattern matching for this fused kernel.
# ** 28. File: worker/patch_qwen3_dflash.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.qwen3_dflash.DFlashQwen3Model.precompute_and_store_context_kv`
#    Why:
#       The function directly calls the ops.rms_norm and ops.rotary_imbedding operators,
#       but NPU does not have a corresponding implementation.
#    How：
#       Replace ops.* with the internal implementation of vllm-ascend.
#    Future Plan:
#       Remove this patch when vllm-ascend supports pattern matching for ops.*.
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
# ** 29. File: platform/patch_mamba_manager.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.core.single_type_kv_cache_manager.MambaManager`
#    Why:
#       Upstream hybrid prefix cache lookup does not support PCP/DCP.
#    How:
#       Replace MambaManager with AscendMambaManager for prefix cache hit lookup
#       on hybrid Mamba paths (logical mamba block_size when caching is enabled).
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/40996
#    Future Plan:
#       Upstream PR #40996 adds hybrid prefix cache lookup for DCP only; PCP is
#       not supported yet. Remove this patch once upstream supports both PCP and DCP.
