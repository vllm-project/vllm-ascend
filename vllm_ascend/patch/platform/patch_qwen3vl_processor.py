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
"""Disable HF image/video rescale+normalize on the API/frontend.

Device-side normalize is applied in worker/patch_qwen3vl.py. Without this
platform patch, the APIServer still normalizes on CPU and the Engine applies
it again, which changes tokens.
"""

from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo


def _disable_processor_rescale_normalize(processor):
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if hasattr(image_processor, "do_rescale"):
            image_processor.do_rescale = False
        if hasattr(image_processor, "do_normalize"):
            image_processor.do_normalize = False
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        if hasattr(video_processor, "do_rescale"):
            video_processor.do_rescale = False
        if hasattr(video_processor, "do_normalize"):
            video_processor.do_normalize = False
    return processor


_orig_get_hf_processor = Qwen3VLProcessingInfo.get_hf_processor


def _patched_get_hf_processor(self, **kwargs: object):
    processor = _orig_get_hf_processor(self, **kwargs)
    return _disable_processor_rescale_normalize(processor)


Qwen3VLProcessingInfo.get_hf_processor = _patched_get_hf_processor
