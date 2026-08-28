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
# mypy: ignore-errors
"""Front-end side of the Qwen3.5-VL image-preprocessing offload.

Turns off the CPU rescale/normalize stages of the HF image processor so raw
uint8 pixel values reach the worker untouched; the worker patch re-applies them
on the NPU as one fused affine op. Runs in the API-server / AsyncLLM front-end
process where multimodal input processing lives, and is gated to qwen3.5 by
``model_type`` so qwen2/2.5/3-VL keep the stock CPU path.
"""
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor


def _is_qwen3_5(info) -> bool:
    model_type = getattr(info.get_hf_config(), "model_type", "")
    return isinstance(model_type, str) and model_type.startswith("qwen3_5")


_ORIG_CALL_HF = Qwen3VLMultiModalProcessor._call_hf_processor


def _patched_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
    if not _is_qwen3_5(self.info):
        return _ORIG_CALL_HF(self, prompt, mm_data, mm_kwargs, tok_kwargs)
    # do_rescale / do_normalize are call-time image-processor kwargs: they get
    # filtered out of the processor's dynamic keys by from_pretrained, so they
    # must be injected here rather than set on the processor object.
    mm_kwargs = dict(mm_kwargs)
    mm_kwargs.setdefault("do_rescale", False)
    mm_kwargs.setdefault("do_normalize", False)
    return _ORIG_CALL_HF(self, prompt, mm_data, mm_kwargs, tok_kwargs)


Qwen3VLMultiModalProcessor._call_hf_processor = _patched_call_hf_processor
