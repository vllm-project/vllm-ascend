#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/qwen2_5_vl.py
# Copyright 2023 The vLLM team.
#
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

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import \
    Qwen2_5_VLConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer, Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.ascend_forward_context import set_ascend_forward_context


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder)
class AscendQwen2_5_VLForConditionalGeneration(
        Qwen2_5_VLForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.visual = Qwen2_5_VisionTransformer(
            vision_config=config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            with set_ascend_forward_context(None, self.vllm_config):
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _process_video_input(self, video_input) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype)
            with set_ascend_forward_context(None, self.vllm_config):
                video_embeds = self.visual(pixel_values_videos,
                                           grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return video_embeds.split(sizes.tolist())
