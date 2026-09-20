#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Ascend Qwen3.5-VL: move image rescale+normalize onto the device.

Uses vLLM's native supports_mm_device_do_normalize mechanism (the same
approach as Qwen2/2.5-VL upstream): vLLM injects do_rescale=False /
do_normalize=False into the processor kwargs, so the front-end CPU skips
those steps and pixel_values arrive as raw uint8 in patched layout.
FusedInputNorm then applies (x * rescale_factor - mean) / std on device
before the vision tower.

This is the minimal version — resize and patchify stay on the CPU (they
can be moved in a follow-up PR). The payload is uint8 patched instead of
fp32 patched (÷4), and the CPU normalize pass is eliminated.

Registered through ModelRegistry; no monkey-patching.
"""
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeProcessingInfo,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.model_executor.models.vision import (
    FusedInputNorm,
    run_dp_sharded_mrope_vision_model,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


class _AscendDeviceNormMixin:
    """Apply rescale+normalize on device via vLLM's native mechanism."""

    supports_mm_device_do_normalize = True

    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.input_norm = FusedInputNorm.from_model_config(self.model_config)

    def _process_image_input(self, image_input):
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = self.input_norm(
                image_input["pixel_values"], self.visual.dtype
            )
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual,
                    pixel_values,
                    grid_thw.tolist(),
                    rope_type="rope_3d",
                )
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input):
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = self.input_norm(
                video_input["pixel_values_videos"], self.visual.dtype
            )
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual,
                    pixel_values_videos,
                    grid_thw.tolist(),
                    rope_type="rope_3d",
                )
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen3_5ForConditionalGeneration(
    _AscendDeviceNormMixin, Qwen3_5ForConditionalGeneration
):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen3_5MoeForConditionalGeneration(
    _AscendDeviceNormMixin, Qwen3_5MoeForConditionalGeneration
):
    pass
