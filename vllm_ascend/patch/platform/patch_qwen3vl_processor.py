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

# Attribute holding the flags the HF processor was configured with before they
# were turned off here. The worker patch reads them so the device path applies
# exactly what HF would have applied, and so it can tell "HF was disabled by us"
# apart from "this patch never ran".
ORIG_PREPROCESS_FLAGS_ATTR = "_ascend_orig_preprocess_flags"


def _disable_sub_processor(sub_processor) -> None:
    if sub_processor is None:
        return
    if not hasattr(sub_processor, ORIG_PREPROCESS_FLAGS_ATTR):
        setattr(
            sub_processor,
            ORIG_PREPROCESS_FLAGS_ATTR,
            {
                "do_rescale": bool(getattr(sub_processor, "do_rescale", False)),
                "do_normalize": bool(getattr(sub_processor, "do_normalize", False)),
            },
        )
    if hasattr(sub_processor, "do_rescale"):
        sub_processor.do_rescale = False
    if hasattr(sub_processor, "do_normalize"):
        sub_processor.do_normalize = False


def _disable_processor_rescale_normalize(processor):
    _disable_sub_processor(getattr(processor, "image_processor", None))
    _disable_sub_processor(getattr(processor, "video_processor", None))
    return processor


_orig_get_hf_processor = Qwen3VLProcessingInfo.get_hf_processor


def _patched_get_hf_processor(self, **kwargs: object):
    processor = _orig_get_hf_processor(self, **kwargs)
    return _disable_processor_rescale_normalize(processor)


Qwen3VLProcessingInfo.get_hf_processor = _patched_get_hf_processor
