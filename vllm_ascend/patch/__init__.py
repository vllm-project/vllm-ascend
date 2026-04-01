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
# ** 4. File: platform/patch_sched_yield.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.distributed.utils.USE_SCHED_YIELD`
#    Why:
#       os.sched_yield() doesn't work on Arm systems.
#    How：
#       avoid using os.sched_yield() on Arm systems.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/30228
#    Future Plan:
#       Remove this patch when vLLM merge the PR.
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
#       Set environmental variables VLLM_ASCEND_BALANCE_SCHEDULING=1 in startup script.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/29721
#    Future Plan:
#       Remove this patch when vLLM merge the PR.
#
# ** 6. File: platform/patch_fusion_matcher_compat_ops.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `torch.ops._C.rms_norm`, `torch.ops._C.fused_add_rms_norm`,
#    Why:
#       upstream vLLM initializes fusion matcher global operators at import time.
#       On Ascend environment these symbols may be absent and cause import failure.
#    How：
#       inject placeholders only when the symbols are missing so import can continue.
#    Related PR (if no, explain why):
#       temporary compatibility patch before upstream adjustment is merged.
#    Future Plan:
#       remove this patch once upstream no longer requires these global symbols or
#       provides a backend-safe initialization path.
#
# ** 7. File: platform/patch_minimax_m2_config.py**
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
# ** 8. File: platform/patch_kv_cache_interface.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.kv_cache_interface.MLAAttentionSpec`
#    Why:
#       The default `MLAAttentionSpec` is mainly built around `kv_lora_rank`
#       and `qk_rope_head_dim`. On NPU, we also use this class to describe DSA
#       models. Unlike the GPU path, where cache management is handled by an
#       additional indexer module, extending this class directly simplifies the
#       corresponding `model_runner` implementation on NPU.
#
#       This patch also adds Sparse C8 support for DSA models on NPU. As part
#       of that support, members such as `page_size_bytes` need to be adapted,
#       so they are overridden here as well to preserve overall readability.
#    How:
#       This patch subclasses the original implementation, overrides selected
#       methods, and adds DSA-specific attributes and helpers with default
#       values where needed.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/25896
#    Future Plan:
#       Remove this patch after the upcoming KV cache spec refactor.
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
# ** 2. File: worker/patch_multimodal_merge.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.utils._merge_multimodal_embeddings`
#    Why:
#       '_merge_multimodal_embeddings' func of vllm is incompatible with Ascend.
#    How：
#       Replace with CPU operation that can be executed asynchronously.
#    Related PR (if no, explain why):
#       This is a bug by Ascend only. It can' be fixed in vLLM.
#    Future Plan:
#       Identify this pattern in torch-npu and remove this patch.
#
# ** 3. File: worker/patch_bert.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.bert._encode_token_type_ids`
#      `vllm.model_executor.models.bert._decode_token_type_ids`
#    Why:
#       shift operation in `_encode_token_type_ids` and `_decode_token_type_ids` cannot run in ascend aclgraph mode
#    How：
#       Replace shift operation with multiplication and division.
#    Related PR (if no, explain why):
#       No, this need CANN add an aclnn shift operation
#    Future Plan:
#       Revert this when CANN support shift aclnn operation
#
# ** 4. File: worker/patch_triton.py**
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
# ** 5. File: worker/patch_qwen3_next_mtp.py**
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
# ** 6. File: worker/patch_rejection_sampler.py**
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
## ** 7. File: worker/patch_module.py**
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
#
# ** 7a. File: worker/patch_gdn_attn.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.attention.backends.gdn_attn.GDNAttentionMetadataBuilder.build`
#    Why:
#       Qwen3.5/Qwen3Next GDN prefill on NPU needs prebuilt varlen chunk metadata
#       to avoid forward-time host round-trips that break async scheduling.
#    How：
#       Monkey-patch the upstream builder in-place, keep upstream code untouched,
#       and attach prebuilt device metadata bundle onto the returned attention
#       metadata object for Ascend-specific consumers.
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
# ** 9. File: worker/patch_huanyuan_vl.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.transformers_utils.processors.hunyuan_vl.HunYuanVLProcessor.__call__`
#    Why:
#       The `add_special_tokens` parameter is not supported by default in the processor.
#    How：
#       Remove the `add_special_tokens` parameter from kwargs before calling the original method.
#    Future Plan:
#       Remove this patch when vLLM aligns with the latest processor implementation.
#
# ** 10. File: worker/patch_v2/patch_eagle.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.spec_decode.eagle.EagleSpeculator.propose`
#    Why:
#       `propose` method use torch.gather, but the gather operator will
#       pollute the arguments passed to it. the bug is reported to huawei
#       CANN team, but not fixed yet.
#    How：
#       clone the out attribute ahead of gather to avoid the bug.
#    Future Plan:
#       Remove this patch when cann fix the gather bug.
#
# ** 11. File: worker/patch_unquantized_gemm.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.utils.default_unquantized_gemm`
#    Why:
#       unquantized_gemm in vLLM not is a customop, which will make MatmulAllReduceAddRMSNorm fusion failure.
#    How：
#       make unquantized_gemm as a customop.
#    Future Plan:
#       Remove this patch when vLLM support the operator as customop.
#
# ** 12. File: worker/patch_npugraph_ex_triton.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `torchair.core._concrete_graph.ValuePack`,
#      `torchair.npu_fx_compiler._unpack_meta`,
#      `torchair.npu_fx_compiler._NpuGraphConverter._unpack_npu`
#    Why:
#       In the Triton scenario, npugraph_ex backend needs to process the value pack of the input parameters.
#    How：
#       Supplement the relevant processing logic through patches.
#    Related PR (if no, explain why):
#       https://gitcode.com/Ascend/torchair/pull/2575
#    Future Plan:
#       Remove this patch when the PTA version used by vllm-ascend has been upgraded.
#
# ** 13. File: worker/patch_v2/patch_uva.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.states.UvaBuffer`
#    Why:
#       ASCEND NPUs do not support UVA yet, so we need to wrap it in vLLM.
#    How：
#       make UvaBuffer a dummy class, mimic the interface of vllm UvaBuffer.
#    Future Plan:
#       Remove this patch when NPU support UVA.
#
# ** 14. File: worker/patch_kimi_k25.py**
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
# ** 15. File: worker/patch_routed_experts_capturer.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.fused_moe.routed_experts_capturer.RoutedExpertsCapturer.init_buffer`
#    Why:
#       The `_device_buffer` initialization in vLLM uses `device="cuda"` hardcoded,
#       which doesn't work on NPU.
#    How：
#       Replace `device="cuda"` with `device=current_platform.device_name` to support NPU.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/34336
#    Future Plan:
#       Remove this patch when vLLM merges the PR.
# ** 16. File: worker/patch_draft_quarot.py**
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
# ** 17. File: worker/patch_minimax_m2.py**
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
#   2. `vllm.model_executor.models.minimax_m2.MiniMaxM2Attention.__init__`
#    Why:
#       When total kv heads < TP world size, kv head replication happens and k_norm
#       weights should be sharded to match the replication layout.
#    How：
#       Add `num_kv_head_replicas` and create sharded `k_norm` via
#       `MiniMaxText01RMSNormTP(..., weight_shard_world_size=total_num_kv_heads, ...)`.
#    Related PR (if no, explain why):
#       No, depends on Ascend kernel behavior and TP layout.
#    Future Plan:
#       Remove this patch if upstream implements kv-head-aware norm sharding.
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
#   4. `vllm.model_executor.models.minimax_m2.MiniMaxM2Model.forward`
#    Why:
#       Eagle3 speculative decoding needs auxiliary hidden states from specific
#       transformer layers of the target model.
#    How：
#       Extend `MiniMaxM2Model.forward` to optionally collect and return
#       `(final_hidden_states, aux_hidden_states)` when `aux_hidden_state_layers`
#       is set by the runtime.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/37512
#    Future Plan:
#       Remove this patch once upstream MiniMax-M2 integrates Eagle3 support.
#
#   5. `vllm.model_executor.models.minimax_m2.MiniMaxM2ForCausalLM`
#    Why:
#       vLLM core uses SupportsEagle3-style methods to configure which layers
#       should emit auxiliary hidden states.
#    How：
#       Inject `set_aux_hidden_state_layers` and default-layer getters onto
#       `MiniMaxM2ForCausalLM` so vLLM can configure the target model.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/37512
#    Future Plan:
#       Remove this patch once upstream provides these methods on the model.
#
# ** 18. File: worker/patch_minimax_m2_linear_attn.py**
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
# ** 19. File: worker/patch_qwen3_5.py**
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
# ** 20. File: worker/patch_cudagraph.py**
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
# ** 21. File: worker/patch_deepseek_mtp.py**
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
# ** 22. File: worker/patch_mamba_utils.py**
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
#
# ** 23. File: worker/patch_weight_utils.py**
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
# ** 24. File: worker/patch_v2/patch_input_batch.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.input_batch.InputBatch`
#    Why:
#       vllm use InputBatch to make dummy tensors. in `model_runner.py` and `cudagraph_utils.py`
#       which make it difficult to inherit from vllm methods.
#    How：
#       replace InputBatch with AscendInputBatch.
#    Future Plan:
#       remove this patch when vLLM-ascend's make_dummy behavior aligns with vLLM.
# ** 25. File: worker/patch_v2/patch_block_table.py**
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
# ** 25. File: worker/patch_v2/patch_model_state.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.worker.gpu.model_states.default.init_model_state`
#    Why:
##      vllm's prepare_attn in ModelState is different from vllm,
#       we need to override init_model_state.
#    How：
#       Define AscendModelState and initialize it in init_model_state.
#    Future Plan:
#       remove this when vllm-ascend's attention metadata is align with vllm.
# ** 26. File: worker/patch_v2/patch_triton.py**
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
# ** 27. File: worker/patch_qwen3_c8.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.qwen3.Qwen3ForCausalLM.load_weights`
#    Why:
#       The Qwen3 W8A8C8 model stores per-channel KV cache scales and offsets
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
#       This PR (Qwen3-32B W8A8C8 support).  Upstream vLLM's weight-loading
#       pipeline does not yet have a generic hook for hardware-plugin-defined
#       KV cache parameters.
#    Future Plan:
#       Remove this patch when vLLM provides a first-class extension point
#       for loading extra KV cache quantisation parameters in model load_weights,
#       or when the Qwen3 model's weight names are aligned with the parameter
#       names expected by the quantisation backend.
