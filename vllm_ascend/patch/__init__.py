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
# ** 2. File: platform/patch_ec_connector.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.distributed.ec_transfer.ec_connector.shared_storage_connector.ECSharedStorageConnector.start_load_caches`
#    Why:
#       it's hard code to cuda
#    How：
#       change the cuda to npu
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/30225
#    Future Plan:
#       Remove this patch when vllm merges the PR.
#
# ** 3. File: platform/patch_mamba_config.py**
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
# ** 4. File: platform/patch_multiproc_executor.py**
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
# ** 5. File: platform/patch_sched_yield.py**
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
# ** 6. File: platform/patch_balance_schedule.py**
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
# ** 7. File: platform/patch_compile_backend.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.compilation.backends.PiecewiseCompileInterpreter`
#      `vllm.compilation.piecewise_backend.PiecewiseBackend`
#    Why:
#       vllm removed the compile graph for general shape, which caused operator fusion to fail.
#       This issue affects the performance of model inference on Ascend.
#    How：
#       recover the compiled graph for dynamic_shape in PiecewiseBackend.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/24252
#    Future Plan:
#       Remove this patch when fix the problem.
#
# * Worker Patch:
# ===============
#
# ** 1. File: worker/patch_deepseek.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `DeepseekV2Model.forward`
#    Why:
#       getattr(self.config, "llama_4_scaling", None) will raise AttributeError
#       on npu with graph mode.
#    How：
#       catch the AttributeError and set llama_4_scaling to None.
#    Related PR (if no, explain why):
#       No, this is a bug in vLLM Ascend
#    Future Plan:
#       Find the root cause of this bug and fix it in vLLM Ascend.
#
# ** 2. File: worker/patch_distributed.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ** 3. File: worker/patch_minicpm.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.minicpm.MiniCPMAttention.forward`
#    Why:
#       The forward func of MiniCPMAttention in vllm do a datatype convert
#       (original datatype --> float32) to ensure the precision on cuda.
#       However float32 is not supported in cann rope op, thus we keep this patch
#    How：
#       Removed the dtype convert operations in forward
#    Related PR (if no, explain why):
#       NO, only for npu due to rope op.
#    Future Plan:
#       Keep this patch in vllm-ascend.
#
# ** 4. File: worker/patch_multimodal_merge.py**
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
# ** 5. File: worker/patch_roberta.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.bert `
#    Why:
#       shift operation in `_encode_token_type_ids` and `_decode_token_type_ids` cannot run in ascend aclgraph mode
#    How：
#       Replace shift operation with multiplication and division.
#    Related PR (if no, explain why):
#       No, this need CANN add an aclnn shift operation
#    Future Plan:
#       Revert this when CANN support shift aclnn operation
#
# ** 6. File: worker/patch_triton.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.mamba.ops`, `vllm.model_executor.layers.fla.ops`
#    Why:
#       triton ops in vLLM perform not good on NPU. And there is no dispatch mechanism for triton ops.
#    How：
#       override triton ops in vLLM with ascend implementation
#    Related PR (if no, explain why):
#       Let vLLM support triton ops dispatch.
#    Future Plan:
#       Remove this patch when vLLM support the dispatch function.
#
# ** 7. File: worker/patch_weight_loader.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.linear.UnquantizedLinearMethod`
#    Why:
#       vLLM Ascend doesn't work with weight loader v2
#    How：
#       patch it to fix the bug.
#    Related PR (if no, explain why):
#       This is a bug by Ascend only.  We should fix it soon
#    Future Plan:
#       Remove this patch when the bug is fixed.
#
# ** 8. File: worker/patch_qwen3_next_mtp.py**
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
# ** 9. File: worker/patch_module.py**
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
# ** 10. File: worker/patch_rejection_sampler.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.sample.rejection_sampler`
#    Why:
#       - some functions from `rejection_sampler` are not supported or slow on npu.
#    How：
#       - add npu_top_k_top_p to 'apply_sampling_constraints' func
#       - add custom triton kernel to `expand_batch_to_tokens` and `rejection_sample`
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/874
#       https://github.com/vllm-project/vllm/pull/4849
#    Future Plan:
#       1. make these functions as class func of RejectionSampler, create AscendRejectionSampler
#           to override them, then delete the patch file `worker/patch_rejection_sampler.py`.
#       2. make these functions as costom op, then remove AscendRejectionSampler
#
# ** 11.File: worker/patch_qwen3_next.py**
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
# ** 12. File: worker/patch_qwen3_next.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.qwen3_next.Qwen3NextGatedDeltaNet._forward_core`
#    Why:
#       triton ops fused_recurrent_gated_delta_rule and fused_gdn_gating in vLLM perform not good on NPU.
#    How：
#       add a new fused triton ops in vLLM with ascend implementation.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/30860
#    Future Plan:
#       Remove this patch when vLLM support these operators.
#
#   2. `vllm.model_executor.models.qwen3_next.Qwen3NextGatedDeltaNet._forward_core`
#    Why:
#       The Qwen3Next GatedDeltaNet _forward_core cannot directly add custom operators.
#    How：
#       Add a branch in Qwen3NextGatedDeltaNet._forward_core to adapt to fused_gdn_gating_patch.
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/31002
#    Future Plan:
#       Remove this patch when vLLM support these operators.
#
