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
"""Ascend Qwen3.5-VL: offload image preprocessing to the NPU.

The front-end emits flat raw uint8 plus the original (H, W) and the target
grid; resize, patchify and rescale+normalize all run on the device.
Normalize reuses vLLM's FusedInputNorm. smart_resize stays on the CPU so
placeholder accounting is unchanged. Payload drops ~8x (uint8, no temporal
replication).
"""

import torch
import torch.nn.functional as F
from transformers.image_processing_utils import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
    smart_resize,
)
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
from vllm.multimodal.inputs import MultiModalFieldConfig


class AscendQwen2VLImageProcessor(Qwen2VLImageProcessor):
    """Emit flat raw uint8 + original (H, W) + target grid; skip everything else."""

    def _preprocess(
        self,
        images,
        do_resize,
        size,
        resample,
        do_rescale,
        rescale_factor,
        do_normalize,
        image_mean,
        image_std,
        patch_size,
        temporal_patch_size,
        merge_size,
        disable_grouping,
        return_tensors,
        **kwargs,
    ):
        flats, grids, hws = [], [], []
        for image in images:  # (C, H, W) uint8 at original resolution
            _, h, w = image.shape
            rh, rw = smart_resize(
                h, w, factor=patch_size * merge_size, min_pixels=size.shortest_edge, max_pixels=size.longest_edge
            )
            grids.append([1, rh // patch_size, rw // patch_size])
            hws.append([h, w])
            flats.append(image.reshape(-1).contiguous())
        return BatchFeature(
            data={
                "pixel_values": torch.cat(flats),
                "image_grid_thw": torch.tensor(grids, dtype=torch.long),
                "image_hw": torch.tensor(hws, dtype=torch.long),
            },
            tensor_type=return_tensors,
        )


class _AscendProcessorMixin:
    def get_hf_processor(self, **kwargs):
        processor = super().get_hf_processor(**kwargs)
        if processor.image_processor.__class__ is not AscendQwen2VLImageProcessor:
            processor.image_processor.__class__ = AscendQwen2VLImageProcessor
        return processor


class AscendQwen3_5ProcessingInfo(_AscendProcessorMixin, Qwen3_5ProcessingInfo):
    pass


class AscendQwen3_5MoeProcessingInfo(_AscendProcessorMixin, Qwen3_5MoeProcessingInfo):
    pass


class AscendQwen3_5VLProcessor(Qwen3VLMultiModalProcessor):
    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        config = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        channel = self.info.get_hf_config().vision_config.in_channels
        image_hw = hf_inputs.get("image_hw", torch.empty((0, 2), dtype=torch.long))
        config["pixel_values"] = MultiModalFieldConfig.flat_from_sizes("image", image_hw.prod(-1) * channel)
        config["image_hw"] = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
        return config


class _AscendVLPreprocessMixin:
    """Device-side resize/patchify + normalize for Qwen3.5-VL."""

    supports_mm_device_do_normalize = True

    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.input_norm = FusedInputNorm.from_model_config(self.model_config)
        vc = self.config.vision_config
        self._c = vc.in_channels
        self._ps = vc.patch_size
        self._tps = vc.temporal_patch_size
        self._ms = vc.spatial_merge_size
        # Disable offload when encoder ACL graph is enabled: _resize_and_patchify
        # has dynamic shapes (variable image sizes) that cannot be captured into
        # a fixed-shape graph. Fall back to the original CPU preprocessing.
        cc = vllm_config.compilation_config
        self._pp_offload = not cc.cudagraph_mm_encoder

    def _resize_and_patchify(self, image_input, grid_thw):
        """Flat raw uint8 -> per-image resize -> patchify -> concat patches."""
        flat = image_input["pixel_values"].to(self.visual.device)
        hw_list = image_input["image_hw"].tolist()
        c, ps, tps, ms = self._c, self._ps, self._tps, self._ms
        chunks = flat.split([c * h * w for h, w in hw_list])
        patches = []
        for chunk, (h, w), (_, gh, gw) in zip(chunks, hw_list, grid_thw.tolist()):
            img = chunk.reshape(1, c, h, w).float()
            # bicubic + antialias reproduces HF/PIL resize (cosine sim 0.999996)
            img = F.interpolate(img, size=[gh * ps, gw * ps], mode="bicubic", align_corners=False, antialias=True)
            img = img.clamp(0, 255).round()
            # patchify: same reshape/permute/expand sequence as HF's patchify,
            # with the batch dim kept so the permute indices match HF exactly
            x = img.reshape(1, c, gh // ms, ms, ps, gw // ms, ms, ps)
            x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
            x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, tps, -1, -1)
            patches.append(x.reshape(gh * gw, c * tps * ps * ps))
        return torch.cat(patches)

    def _parse_and_validate_image_input(self, **kwargs):
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None:
            return {
                "type": "pixel_values",
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "image_hw": kwargs.pop("image_hw", None),
            }
        return {
            "type": "image_embeds",
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
        }

    def _run_visual(self, pixel_values, grid_thw):
        """Shared tail: optional DP sharding, then ViT forward, then split."""
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d")
        embeds = self.visual(pixel_values, grid_thw=grid_thw)
        sizes = (grid_thw.prod(-1) // self._ms // self._ms).tolist()
        return embeds.split(sizes)

    def _process_image_input(self, image_input):
        grid_thw = image_input["image_grid_thw"]
        if image_input["type"] == "image_embeds":
            embeds = image_input["image_embeds"].type(self.visual.dtype)
            sizes = (grid_thw.prod(-1) // self._ms // self._ms).tolist()
            return embeds.split(sizes)
        if self._pp_offload:
            patches = self._resize_and_patchify(image_input, grid_thw)
            pixel_values = self.input_norm(patches, self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        return self._run_visual(pixel_values, grid_thw)

    def _process_video_input(self, video_input):
        grid_thw = video_input["video_grid_thw"]
        if video_input["type"] == "video_embeds":
            embeds = video_input["video_embeds"].type(self.visual.dtype)
            sizes = (grid_thw.prod(-1) // self._ms // self._ms).tolist()
            return embeds.split(sizes)
        pixel_values = self.input_norm(video_input["pixel_values_videos"], self.visual.dtype)
        return self._run_visual(pixel_values, grid_thw)


@MULTIMODAL_REGISTRY.register_processor(
    AscendQwen3_5VLProcessor,
    info=AscendQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen3_5ForConditionalGeneration(_AscendVLPreprocessMixin, Qwen3_5ForConditionalGeneration):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    AscendQwen3_5VLProcessor,
    info=AscendQwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen3_5MoeForConditionalGeneration(_AscendVLPreprocessMixin, Qwen3_5MoeForConditionalGeneration):
    pass
