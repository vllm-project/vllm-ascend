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
# Then in each kind of patch, there are three folders:
# - patch_0_9_0: contains the patches applied when vllm version is 0.9.0.
# - patch_main: contains the patches applied when vllm version is main branch.
# - patch_common: contains the patches applied in both 0.9.0 and main branch.
#
# Once a new patch is added in vllm-ascend, please add the patch description into this file as well.
# ----------------------------------------------------------------------------------

# What's Patched and how it works:
# --------------------------------
# * Platform Patch:
# =================
# ** File: platform/patch_0_9_0/patch_distributed.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.distributed.utils.stateless_init_torch_distributed_process_group()`
#    Why:
#       vllm distributed use gloo backend by default to initialize stateless process group, but we want to use hccl here
#    How：
#       Add hccl backend to the `stateless_init_torch_distributed_process_group`
#    Related PR (if no, explain why):
#       https://github.com/vllm-project/vllm/pull/18763
#    Future Plan:
#       Remove this patch once vllm is upgraded to 0.9.1
# ** File: platform/patch_common/patch_distributed.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.distributed.parallel_state.destroy_model_parallel()`
#    Why:
#       vllm dose not support outside platform maintain its own `CoordinatorGroup`, vllm-ascend maintain EP and ETP
#       inside of the repo, and needs a common interface to destroy them, this patch add the interface of destroy
#       platform owned `CoordinatorGroup` to make sure all the CoordinateGroup can be properly destroyed
#    How：
#       Call `vllm_ascend.distributed.parallel_state method `destroy_platform_model_parallel` to destroy all the `CoordinateGroup`
#    Related PR (if no, explain why):
#    Future Plan:
#       Remove those patch when vllm merged them
#   2. `vllm.v1.engine.core.DPEngineCoreProc._init_data_parallel`
#    Why:
#       There is some bug for ASCEND_RT_VISIBLE_DEVICES usage.
#    How：
#       The ASCEND_RT_VISIBLE_DEVICES related code is dropped.
#    Related PR (if no, explain why):
#       No, this is a bug for vllm ascend
#    Future Plan:
#       Remove this patch once ASCEND_RT_VISIBLE_DEVICES bug is fixed.
#   3. `vllm.config.ParallelConfig.get_next_dp_init_port`
#    Why:
#       vllm doesn't support get port from environment.
#    How：
#       Add the logic to get port from environment.
#    Related PR (if no, explain why):
#       Need a PR to vllm to support get port from environment.
#    Future Plan:
#       Remove those patch when vllm merged them
#   4. `vllm.config.ParallelConfig.ParallelConfig.stateless_init_dp_group`
#    Why:
#       vLLM use gloo backend by default to initialize stateless dp process gourp, but we want to use hccl here to
#       get better performance
#    How：
#       adopt nccl backend to init process group.(Now we still use gloo, it's just a placeholder, we'll use nccl in the future)
#    Related PR (if no, explain why):
#       Need a PR to vllm to support more backend.
#    Future Plan:
#       Remove those patch when vllm support more backend.
#
# * Worker Patch:
# ===============
# ** File: worker/patch_common/patch_minicpm.py **
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
# ** File: worker/patch_common/patch_multi_step_worker.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.spec_decode.multi_step_worker.MultiStepWorker.sampler_output`
#    Why:
#       There are cuda hard code (current_platform.is_cuda_alike()) in
#       `MultiStepWorker.sampler_output`, and we need to use the patched `TP1DraftModelRunner` in it.
#    How：
#       Make speculative decoding extensible to different backends.
#       - support attention metadata register to the set supported spec decode
#       - offer a api in platform to determine whether spec decode is supported,
#         and deprecate is_cuda_alike in it.
#    Related PR (if no, explain why):
#       - https://github.com/vllm-project/vllm/pull/15195
#       - https://github.com/vllm-project/vllm-ascend/pull/395
#    Future Plan:
#       Revert it when the related pr is merged in vllm and vllm-ascend.
#
#   2. `vllm.spec_decode.multi_step_worker.MultiStepWorker.set_include_gpu_probs_tensor` and
#       `vllm.spec_decode.multi_step_worker.MultiStepWorker.set_should_modify_greedy_probs_inplace`
#    Why:
#       vLLM `Remove Sampler from Model Code` so vllm-ascend needs adapt to this change.
#    How：
#       Use vLLM 0.8.4 method to patch it.
#    Related PR (if no, explain why):
#       - https://github.com/vllm-project/vllm/pull/15195
#       - https://github.com/vllm-project/vllm-ascend/pull/395
#    Future Plan:
#       Remove it when we identify the reasons clearly.
#
# ** File: worker/patch_common/patch_spec_decode_worker.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.spec_decode.spec_decode_worker.SpecDecodeWorker.create_worker`
#    Why:
#       We need to use the patched `TP1DraftModelRunner` in `SpecDecodeWorker.create_worker`.
#       The mainly reason to overwrite `TP1DraftModelRunner`is the hard code of
#           `FlashAttentionMetadata`
#    How：
#       ditto
#    Related PR (if no, explain why):
#       - https://github.com/vllm-project/vllm/pull/15195
#       - https://github.com/vllm-project/vllm-ascend/pull/395
#    Future Plan:
#       Revert it when the related pr is merged in vllm and vllm-ascend.
#
# ** File: worker/patch_common/patch_eagle.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.spec_decode.eagle.prepare_inputs`
#    Why:
#       We need to use the patched `prepare_input_kernel` in `eagle.prepare_inputs`.
#       The mainly reason to overwrite `prepare_input_kernel` is this is a triton
#       kernel, ascend is now not support triton kernel.
#    How：
#       Re-implementation the `prepare_input_kernel` triton kernel by pytorch
#    Related PR (if no, explain why):
#       - Ascend doesn't support triton
#    Future Plan:
#       Revert it when the ascend support triton kernel.
#
# ** File: worker/patch_common/patch_sampler.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.sample.sampler.Sampler.apply_top_k_top_p`
#    Why:
#       We need to use the patched `apply_top_k_top_p` in `sample`.
#       The mainly reason to overwrite `apply_top_k_top_p` is
#       to improve performance.
#    How：
#       Re-implementation the `apply_top_k_top_p` function by pytorch
#    Related PR (if no, explain why):
#       - https://github.com/vllm-project/vllm-ascend/pull/970
#    Future Plan:
#       Revert it when the ascend scatter performance improves.
#
#   2. `vllm.v1.sample.sampler.Sampler.apply_min_p`
#    Why:
#       We need to use the patched `apply_min_p` in `sample`.
#       The mainly reason to overwrite `apply_min_p` is
#       to improve performance.
#    How：
#       Re-implementation the `apply_min_p` function by pytorch
#    Related PR (if no, explain why):
#       - https://github.com/vllm-project/vllm-ascend/pull/970
#    Future Plan:
#       Revert it when the ascend indexput performance improves.
#
# ** File: worker/patch_common/patch_distributed.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.distributed.parallel_state.GroupCoordinator`
#    Why:
#       vllm doesn't support all_to_all for GroupCoordinator.
#    How：
#       Add all_to_all implementation for GroupCoordinator.
#    Related PR (if no, explain why):
#       Need a PR to vllm to support all_to_all for GroupCoordinator.
#    Future Plan:
#       Remove this patch when vllm merged them.
#
# ** File: worker/patch_common/patch_utils.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.utils.direct_register_custom_op`
#    Why:
#       pytorch 2.7.o is not compatible with pytorch 2.5.1. While vllm is based on pytorch 2.7.0, but vllm ascend
#       is based on pytorch 2.5.1, so we need to use this patch to make vllm compatible with pytorch 2.5.1.
#    How：
#       patch __annotations__ check to make it compatible with pytorch 2.5.1.
#    Related PR (if no, explain why):
#       This is the problem in vllm-ascend
#    Future Plan:
#       Remove this patch once pytorch 2.7.0 is supported for vllm ascend.
