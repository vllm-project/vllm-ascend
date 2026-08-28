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
import torch
from transformers.image_processing_utils import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor, smart_resize)

from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
from vllm.multimodal.inputs import MultiModalFieldConfig


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
    self.info.get_image_processor()._offload_preprocess = True
    mm_kwargs = dict(mm_kwargs)
    mm_kwargs.setdefault("do_rescale", False)
    mm_kwargs.setdefault("do_normalize", False)
    return _ORIG_CALL_HF(self, prompt, mm_data, mm_kwargs, tok_kwargs)


Qwen3VLMultiModalProcessor._call_hf_processor = _patched_call_hf_processor
# ---- _preprocess: skip resize+patchify, emit flat raw uint8 + original (H, W)
_ORIG_PREPROCESS = Qwen2VLImageProcessor._preprocess


def _preprocess_offload(self, images, do_resize, size, resample, do_rescale,
                        rescale_factor, do_normalize, image_mean, image_std,
                        patch_size, temporal_patch_size, merge_size,
                        disable_grouping, return_tensors, **kwargs):
    if not getattr(self, "_offload_preprocess", False):
        return _ORIG_PREPROCESS(self, images, do_resize, size, resample,
                                do_rescale, rescale_factor, do_normalize,
                                image_mean, image_std, patch_size,
                                temporal_patch_size, merge_size,
                                disable_grouping, return_tensors, **kwargs)
    flats, grids, hws = [], [], []
    for image in images:  # (C, H, W) uint8, still at original resolution
        _, height, width = image.shape
        resized_h, resized_w = smart_resize(height,
                                            width,
                                            factor=patch_size * merge_size,
                                            min_pixels=size.shortest_edge,
                                            max_pixels=size.longest_edge)
        # Target grid is pure arithmetic on (H, W), so it can be reported here
        # even though the resize itself happens later on the NPU.
        grids.append([1, resized_h // patch_size, resized_w // patch_size])
        hws.append([height, width])
        flats.append(image.reshape(-1).contiguous())
    return BatchFeature(data={
        "pixel_values": torch.cat(flats, dim=0),
        "image_grid_thw": torch.tensor(grids, dtype=torch.long),
        "image_hw": torch.tensor(hws, dtype=torch.long),
    },
                        tensor_type=return_tensors)


# ---- field config: split pixel_values by C*H*W, and carry image_hw ----------
_ORIG_MM_FIELDS = Qwen3VLMultiModalProcessor._get_mm_fields_config


def _patched_get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
    config = dict(_ORIG_MM_FIELDS(self, hf_inputs, hf_processor_mm_kwargs))
    if _is_qwen3_5(self.info):
        channel = self.info.get_hf_config().vision_config.in_channels
        image_hw = hf_inputs.get("image_hw",
                                 torch.empty((0, 2), dtype=torch.long))
        # Images are no longer patchified on CPU, so the per-item split size is
        # the raw C*H*W instead of a patch count.
        config["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
            "image", image_hw.prod(-1) * channel)
        config["image_hw"] = MultiModalFieldConfig.batched("image",
                                                           keep_on_cpu=True)
    return config


Qwen2VLImageProcessor._preprocess = _preprocess_offload
Qwen3VLMultiModalProcessor._get_mm_fields_config = _patched_get_mm_fields_config
