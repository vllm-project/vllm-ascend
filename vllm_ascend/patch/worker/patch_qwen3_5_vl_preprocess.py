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
"""Worker side of the Qwen3.5-VL image-preprocessing offload: rescale+normalize.

The front-end patch disables the CPU rescale/normalize stages, so
``pixel_values`` arrives as raw [0, 255] laid out (num_patches, C*tps*ps*ps);
this re-applies HF's ``(x * rescale_factor - mean) / std`` on the NPU.

Both stages fold into one affine op (mean/rescale, std/rescale); when mean and
std are uniform across channels (the Qwen3.5 default) that affine is scalar and
applies to the flat tensor with no reshape at all.
"""
import torch
from torchvision.transforms.v2 import functional as tv_functional

from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.multimodal import MULTIMODAL_REGISTRY


def _rescale_and_normalize(self, pixel_values):
    """Fused rescale+normalize on device; shape is preserved."""
    if self._uniform_norm:
        # Scalar affine, commutes with any reshape: apply it flat.
        return pixel_values * self._norm_scale + self._norm_bias
    if self._fused_mean is None:
        inv = 1.0 / self.rescale_factor
        device = pixel_values.device
        self._fused_mean = torch.tensor(self.image_mean, device=device) * inv
        self._fused_std = torch.tensor(self.image_std, device=device) * inv
    shape = pixel_values.shape
    # normalize() indexes mean/std on dim -3, so C must sit there.
    view = pixel_values.reshape(-1, self.channel,
                                self.temporal_patch_size * self.patch_size,
                                self.patch_size)
    out = tv_functional.normalize(view.to(torch.float32), self._fused_mean,
                                  self._fused_std)
    return out.to(pixel_values.dtype).reshape(shape)


def _ensure_img_pp_cfg(self):
    if getattr(self, "_img_pp_ready", False):
        return
    vision_config = self.config.vision_config
    self.channel = vision_config.in_channels
    self.patch_size = vision_config.patch_size
    self.temporal_patch_size = vision_config.temporal_patch_size
    image_processor = (MULTIMODAL_REGISTRY.create_processor(
        self.model_config).info.get_hf_processor().image_processor)
    self.rescale_factor = image_processor.rescale_factor
    self.image_mean = tuple(image_processor.image_mean)
    self.image_std = tuple(image_processor.image_std)
    mean, std = self.image_mean, self.image_std
    self._fused_mean = self._fused_std = None
    self._uniform_norm = len(set(mean)) == 1 and len(set(std)) == 1
    if self._uniform_norm:
        # HF computes (x * rescale_factor - mean) / std on raw [0, 255] input.
        self._norm_scale = self.rescale_factor / float(std[0])
        self._norm_bias = -float(mean[0]) / float(std[0])
    self._img_pp_ready = True


def _ascend_process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
    self._ensure_img_pp_cfg()
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        pixel_values = self._rescale_and_normalize(
            image_input["pixel_values"].type(self.visual.dtype))
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(self.visual,
                                                     pixel_values,
                                                     grid_thw.tolist(),
                                                     rope_type="rope_3d")
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    return image_embeds.split(sizes)


def _ascend_process_video_input(self, video_input) -> tuple[torch.Tensor, ...]:
    self._ensure_img_pp_cfg()
    grid_thw = video_input["video_grid_thw"]
    assert grid_thw.ndim == 2

    if video_input["type"] == "video_embeds":
        video_embeds = video_input["video_embeds"].type(self.visual.dtype)
    else:
        pixel_values_videos = self._rescale_and_normalize(
            video_input["pixel_values_videos"].type(self.visual.dtype))
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(self.visual,
                                                     pixel_values_videos,
                                                     grid_thw.tolist(),
                                                     rope_type="rope_3d")
        video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    return video_embeds.split(sizes)


Qwen3_5ForConditionalGeneration._ensure_img_pp_cfg = _ensure_img_pp_cfg
Qwen3_5ForConditionalGeneration._rescale_and_normalize = _rescale_and_normalize
Qwen3_5ForConditionalGeneration._process_image_input = _ascend_process_image_input
Qwen3_5ForConditionalGeneration._process_video_input = _ascend_process_video_input
