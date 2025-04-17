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
#

# What's Patched and how it works:
# ** File: worker/patch_common/patch_metrics.py **
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
#   2. `vllm.spec_decode.metrics.AsyncMetricsCollector.maybe_collect_rejsample_metrics`
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
# ** File: worker/patch_common/patch_multi_step_worker.py **
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
# ** File: worker/patch_common/patch_multi_step_worker.py **
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

# current_platform.is_cuda_alike()
# 0.8.4 patch doc:
# platform-0.8.4 + platform-common + worker-0.8.4 + worker-common
# ...

import vllm_ascend.patch.worker.patch_common.patch_metrics  # noqa
import vllm_ascend.patch.worker.patch_common.patch_multi_step_worker  # noqa
import vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker  # noqa
