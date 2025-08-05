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
# - patch_0_10_0: contains the patches applied when vllm version is 0.10.0.
# - patch_main: contains the patches applied when vllm version is main branch.
# - patch_common: contains the patches applied in both 0.10.0 and main branch.
#
# Once a new patch is added in vllm-ascend, please add the patch description into this file as well.
# ----------------------------------------------------------------------------------

# What's Patched and how it works:
# --------------------------------
# * Platform Patch:
# =================
# ** File: platform/patch_common/patch_distributed.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.config.ParallelConfig.get_next_dp_init_port`
#    Why:
#       vllm doesn't support get port from environment.
#    How：
#       Add the logic to get port from environment.
#    Related PR (if no, explain why):
#       Need a PR to vllm to support get port from environment.
#    Future Plan:
#       Remove those patch when vllm merged them
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.v1.sample.sampler.Sampler.gather_logprobs`
#    Why:
#       We need to patch gather_logprobs to make sure call batched_count_greater_than
#       with backend=current_platform.simple_compile_backend
#    How：
#       Patch gather_logprobs call new batched_count_greater_than
#    Related PR (if no, explain why):
#       - https://github.com/vllm-project/vllm/pull/21591
#    Future Plan:
#       Revert it when vLLM merge #21591 and release new version
# ** File: worker/patch_common/patch_linear.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.layers.linear.RowParallelLinear`
#    Why:
#       We need to fuse matmul and allreuce in `RowParallelLinear`
#       to improve performance.
#    How：
#       Create a new class `AscendRowParallelLinear` that inherits from `RowParallelLinear`.
#       In this class, we override the `forward` method to use
#       torch_npu.npu_mm_all_reduce_base to replace matmul and allreduce.
#    Related PR (if no, explain why):
#       - https://github.com/vllm-project/vllm-ascend/pull/1926
#    Future Plan:
#       Validate more models in all kinds of scenario,
#       if performance is always improved, we can enable this patch by default and remove the env
#       variable `VLLM_ASCEND_ENABLE_FUSE_MATMUL_ALLREDUCE` in the future.
