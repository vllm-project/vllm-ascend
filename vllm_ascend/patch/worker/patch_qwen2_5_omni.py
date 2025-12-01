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
import torch.nn as nn
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5_VLImageInputs, Qwen2_5_VLVideoInputs,
    Qwen2_5OmniThinkerForConditionalGeneration)

from vllm_ascend.ascend_forward_context import set_ascend_forward_context


class AscendQwen2_5OmniThinkerForConditionalGeneration(nn.Module):

    def _process_image_input(
            self,
            image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].type(self.visual.dtype)

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        with set_ascend_forward_context(None, self.vllm_config):
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(
        self,
        video_input: Qwen2_5_VLVideoInputs,
        video_hashes: list[str] | None = None,
        cached_video_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.visual.dtype)
        with set_ascend_forward_context(None, self.vllm_config):
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())


# NOTE: These will be removed after https://github.com/vllm-project/vllm/pull/29388 is merged.
Qwen2_5OmniThinkerForConditionalGeneration._process_image_input = AscendQwen2_5OmniThinkerForConditionalGeneration._process_image_input
Qwen2_5OmniThinkerForConditionalGeneration._process_video_input = AscendQwen2_5OmniThinkerForConditionalGeneration._process_video_input
