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

import torch

try:
    from vllm.model_executor.models.glm4_1v import Glm4vForConditionalGeneration
except ImportError:
    Glm4vForConditionalGeneration = None


def _normalize_vision_tensor(
    tensor: torch.Tensor | None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return tensor
    if device is not None and tensor.device != torch.device(device):
        tensor = tensor.to(device=device)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor.contiguous()


if Glm4vForConditionalGeneration is not None:
    _original_forward = Glm4vForConditionalGeneration.forward
    _original_process_image_input = (
        Glm4vForConditionalGeneration._process_image_input
    )
    _original_process_video_input = (
        Glm4vForConditionalGeneration._process_video_input
    )

    def _patched_forward(self, *args, **kwargs):
        if "pixel_values" in kwargs and kwargs["pixel_values"] is not None:
            kwargs["pixel_values"] = _normalize_vision_tensor(
                kwargs["pixel_values"],
                dtype=self.visual.dtype,
                device=self.visual.device,
            )
        if "pixel_values_videos" in kwargs and kwargs[
            "pixel_values_videos"
        ] is not None:
            kwargs["pixel_values_videos"] = _normalize_vision_tensor(
                kwargs["pixel_values_videos"],
                dtype=self.visual.dtype,
                device=self.visual.device,
            )
        return _original_forward(self, *args, **kwargs)

    def _patched_process_image_input(self, image_input):
        if image_input.get("type") != "image_embeds":
            pixel_values = image_input.get("pixel_values")
            if isinstance(pixel_values, torch.Tensor):
                image_input.pixel_values = _normalize_vision_tensor(
                    pixel_values,
                    dtype=self.visual.dtype,
                    device=self.visual.device,
                )
        return _original_process_image_input(self, image_input)

    def _patched_process_video_input(self, video_input):
        if video_input.get("type") != "video_embeds":
            pixel_values_videos = video_input.get("pixel_values_videos")
            if isinstance(pixel_values_videos, torch.Tensor):
                video_input.pixel_values_videos = _normalize_vision_tensor(
                    pixel_values_videos,
                    dtype=self.visual.dtype,
                    device=self.visual.device,
                )
        return _original_process_video_input(self, video_input)

    Glm4vForConditionalGeneration._process_image_input = (
        _patched_process_image_input
    )
    Glm4vForConditionalGeneration._process_video_input = (
        _patched_process_video_input
    )
    Glm4vForConditionalGeneration.forward = _patched_forward
