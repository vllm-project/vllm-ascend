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
# - patch_0_8_4: contains the patches applied when vllm version is 0.8.4.
# - patch_main: contains the patches applied when vllm version is main branch.
# - patch_common: contains the patches applied in both 0.8.4 and main branch.
#
# In the future, with the vllm version upgrade, the new patch folder such as
# patch_0_8_5, patch_0_8_6, etc. will be added to manage the patch for different
# vllm version. And the patch_common will contain the patches applied in all the
# vllm version.
# Once the vllm version is too old that vllm-ascend will not support, the related
# patch folder will be removed as well.
#
# Once a new patch is added in vllm-ascend, please add the patch description into this file as well.
# ----------------------------------------------------------------------------------

# What's Patched and how it works:
# --------------------------------
# * Platform Patch:
# =================
# ** File: platform/patch_0_8_4/patch_config.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.config.ModelConfig.__init__()`
#    Why:
#       It is hard coded for sleep mode to support cuda platform only
#    How：
#       Using a new method to check if sleep mode is available
#    Related PR (if no, explain why): 1. refused by vllm. 2. vllm doesn't support 3. prepare to submit....
#       https://github.com/vllm-project/vllm/pull/16562
#    Future Plan:
#       This patch is only used for 084 and can't be revert. just keep as it is.
#
# ** File: platform/patch_common/patch_distributed.py**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.distributed.parallel_state.destroy_model_parallel()`
#    Why:
#       vllm dose not support outside platform maintain its own `CoordinatorGroup`, vllm-ascend maintain EP and ETP
#       inside of the repo, and needs a common interface to destroy them, this patch add the interface of destroy
#       platform owned `CoordinatorGroup` to make sure all the CoordinateGroup can be properly destroyed
#    How：
#       Call platform method `destroy_platform_model_parallel` to destroy all the `CoordinateGroup`
#    Related PR (if no, explain why): no related PR, we want add this ability into vllm
#    Future Plan:
#       Remove those patch when vllm merged them
#   2. `vllm.distributed.stateless_init_torch_distributed_process_group()`
#    Why:
#       The stateless process group can not be initialized except from gloo and nccl backend, vllm-ascend
#       needs to initialize its own stateless process group for communication, so we add the platform related
#       call to the `stateless_init_torch_distributed_process_group`, to enable other platform which may support
#       stateless process group initialize method
#    How：
#       Call platform method `platform_has_backend_register` to judge if there is a stateless process group initialize
#       method and call platform method `platform_register_backend` to initialize them
#    Related PR (if no, explain why): no related PR, we want add this ability into vllm
#    Future Plan:
#       Remove those patch when vllm merged them
#   3. `ParallelConfig.get_next_dp_init_port`
#    Why:
#       We want to get dp port from env variable, so the multi-node inference can be properly initialized and run.
#    How：
#       Get the dp port from env variable enable multi-mode dp inference
#    Related PR (if no, explain why): no related PR, we want add this ability into vllm
#    Future Plan:
#       Its a workaround in vllm-ascend to enable multi-node dp inference, maybe removed if vllm have better plan
#       on multi-node dp inference implementation
#   4. `ParallelConfig.stateless_init_dp_group`
#    Why:
#       vLLM use gloo backend by default to initialize stateless dp process gourp, but we want to use hccl here to
#       get better performance
#    How：
#       adopt nccl backend to init process group
#    Related PR (if no, explain why): no related PR, we want add this ability into vllm
#    Future Plan:
#       Remove those patch when vllm merged them
#
#
# * Worker Patch:
# ===============
# ** File: worker/patch_0_8_4/patch_metrics.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.spec_decode.metrics.AsyncMetricsCollector.init_tensors` and
#       `vllm.spec_decode.metrics.AsyncMetricsCollector._copy_rejsample_metrics_async`
#    Why:
#       There are cuda hard code (torch.cuda.Stream) in `AsyncMetricsCollector.init_tensors` and
#       `AsyncMetricsCollector._copy_rejsample_metrics_async`
#    How：
#       Replace it with the corresponding npu method
#    Related PR (if no, explain why): 1. refused by vllm. 2. vllm doesn't support 3. prepare to submit....
#       https://github.com/vllm-project/vllm/pull/14411
#    Future Plan:
#       Revert it when the related pr is merged in vllm.
#
# ** File: worker/patch_0_8_4/patch_spec_decode_worker.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.spec_decode.spec_decode_worker.SpecDecodeWorker._configure_model_sampler_for_spec_decode`
#    Why:
#       vLLM `Remove Sampler from Model Code` so vllm-ascend needs a patch to run in v0.8.4.
#    How：
#       Use vLLM 0.8.4 method tp patch it.
#    Related PR (if no, explain why): 1. refused by vllm. 2. vllm doesn't support 3. prepare to submit....
#       - https://github.com/vllm-project/vllm/pull/17084
#       - https://github.com/vllm-project/vllm-ascend/pull/636
#    Future Plan:
#       Follow v0.8.4 version strategy.
#
# ** File: worker/patch_common/patch_metrics.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.spec_decode.metrics.AsyncMetricsCollector.maybe_collect_rejsample_metrics`
#    Why:
#       There are cuda hard code (current_platform.is_cuda_alike()) in
#       `AsyncMetricsCollector.maybe_collect_rejsample_metrics`
#    How：
#       Change to use `current_platform.Event` to determine whether to return None
#    Related PR (if no, explain why): 1. refused by vllm. 2. vllm doesn't support 3. prepare to submit....
#       https://github.com/vllm-project/vllm/pull/14411
#    Future Plan:
#       Revert it when the related pr is merged in vllm.
#
# ** File: worker/patch_common/patch_minicpm.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.model_executor.models.minicpm.MiniCPMAttention.forward`
#    Why:
#       The forward func of MiniCPMAttention in vllm do a datatype convert
#       (original datatype --> float32) to ensure the precision on cuda.
#       However float32 is not supported in cann rope op, thus we keep this patch
#    How：
#       Removed the dtype convert operations in forward
#    Related PR (if no, explain why): 1. refused by vllm. 2. vllm doesn't support 3. prepare to submit....
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
#    Related PR (if no, explain why): 1. refused by vllm. 2. vllm doesn't support 3. prepare to submit....
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
#    Related PR (if no, explain why): 1. refused by vllm. 2. vllm doesn't support 3. prepare to submit....
#       - https://github.com/vllm-project/vllm/pull/15195
#       - https://github.com/vllm-project/vllm-ascend/pull/395
#    Future Plan:
#       Remove it when we identify the reasons clearly.
#
# ** File: worker/patch_common/patch_spec_decode_worker.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `vllm.spec_decode.spec_decode_worker.SpecDecodeWorker.create_worker`
#    Why:
#       We need to use the patched `TP1DraftModelRunner` in `SpecDecodeWorker.create_worker`.
#       The mainly reason to overwrite `TP1DraftModelRunner`is the hard code of
#           `FlashAttentionMetadata`
#    How：
#       ditto
#    Related PR (if no, explain why): 1. refused by vllm. 2. vllm doesn't support 3. prepare to submit....
#       - https://github.com/vllm-project/vllm/pull/15195
#       - https://github.com/vllm-project/vllm-ascend/pull/395
#    Future Plan:
#       Revert it when the related pr is merged in vllm and vllm-ascend.
#
# ** File: worker/patch_0_8_4/patch_tritonplaceholder.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. `triton` Module
#    Why:
#       Triton is not supported on npu currently, importing triton will break vllm-ascend
#    How：
#       ditto
#    Related PR (if no, explain why): 1. refused by vllm. 2. vllm doesn't support 3. prepare to submit....
#       TritonPlaceholder is only available in vllm>0.8.4
#    Future Plan:
#       Revert it when branch main doesn't maintain v0.8.4.
