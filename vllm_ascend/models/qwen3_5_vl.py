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
"""Ascend variant of Qwen3.5-VL with NPU-side image preprocessing.

Inheritance-based replacement of the qwen3_5_vl_preprocess patches: the
front-end emits flat raw uint8 images plus the original (H, W), and the model
does resize/patchify and the fused rescale+normalize on the NPU. No vllm or
transformers class is monkey-patched; the subclasses below are registered into
vllm's ModelRegistry (overwriting the architecture mapping), following the
same pattern as AscendDeepseekV4ForConditionalGeneration / AscendKimiK3.
"""
import torch
from transformers.image_processing_utils import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
    smart_resize,
)
from torchvision.transforms.v2 import functional as tv_functional

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
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig


class AscendQwen2VLImageProcessor(Qwen2VLImageProcessor):
    """Image processor that emits flat raw uint8 plus the original (H, W).

    Resize and patchify are skipped entirely; the target grid is still
    computed here via smart_resize because it is pure arithmetic on (H, W),
    so prompt-length accounting and placeholder ranges are unchanged.
    """

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
        for image in images:  # (C, H, W) uint8, still at original resolution
            _, height, width = image.shape
            resized_h, resized_w = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=size.shortest_edge,
                max_pixels=size.longest_edge,
            )
            # Target grid is pure arithmetic on (H, W), so it can be reported
            # here even though the resize itself happens later on the NPU.
            grids.append([1, resized_h // patch_size, resized_w // patch_size])
            hws.append([height, width])
            flats.append(image.reshape(-1).contiguous())
        return BatchFeature(
            data={
                "pixel_values": torch.cat(flats, dim=0),
                "image_grid_thw": torch.tensor(grids, dtype=torch.long),
                "image_hw": torch.tensor(hws, dtype=torch.long),
            },
            tensor_type=return_tensors,
        )


class AscendQwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_processor(self, **kwargs):
        processor = super().get_hf_processor(**kwargs)
        # Instance-level class swap: the subclass adds no instance state, so
        # the layout is compatible. Scoped to the processor instance returned
        # by this info; the transformers class itself stays untouched.
        if processor.image_processor.__class__ is not AscendQwen2VLImageProcessor:
            processor.image_processor.__class__ = AscendQwen2VLImageProcessor
        return processor


class AscendQwen3_5MoeProcessingInfo(Qwen3_5MoeProcessingInfo):
    def get_hf_processor(self, **kwargs):
        processor = super().get_hf_processor(**kwargs)
        if processor.image_processor.__class__ is not AscendQwen2VLImageProcessor:
            processor.image_processor.__class__ = AscendQwen2VLImageProcessor
        return processor


class AscendQwen3_5VLProcessor(Qwen3VLMultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs,
    ):
        config = dict(
            super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)
        )
        channel = self.info.get_hf_config().vision_config.in_channels
        image_hw = hf_inputs.get(
            "image_hw", torch.empty((0, 2), dtype=torch.long)
        )
        # Images are not patchified on the CPU anymore, so the per-item split
        # size is the raw C*H*W instead of a patch count.
        config["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
            "image", image_hw.prod(-1) * channel
        )
        config["image_hw"] = MultiModalFieldConfig.batched(
            "image", keep_on_cpu=True
        )
        return config


class _AscendVLPreprocessMixin:
    """Device-side resize/patchify + fused rescale/normalize for Qwen3.5-VL."""

    # Lets vLLM inject do_rescale=False / do_normalize=False into the
    # processor kwargs natively (the same mechanism Qwen2/2.5-VL use). The
    # image path ignores those flags (raw uint8 is emitted regardless); the
    # video path needs them so pixel_values_videos arrive unnormalized and
    # the affine below is the only one applied.
    supports_mm_device_do_normalize = True

    def _ensure_img_pp_cfg(self):
        if getattr(self, "_img_pp_ready", False):
            return
        vision_config = self.config.vision_config
        self.channel = vision_config.in_channels
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.merge_size = vision_config.spatial_merge_size
        image_processor = (
            MULTIMODAL_REGISTRY.create_processor(
                self.model_config
            ).info.get_hf_processor().image_processor
        )
        self.rescale_factor = image_processor.rescale_factor
        self.image_mean = tuple(image_processor.image_mean)
        self.image_std = tuple(image_processor.image_std)
        mean, std = self.image_mean, self.image_std
        self._fused_mean = self._fused_std = None
        self._uniform_norm = len(set(mean)) == 1 and len(set(std)) == 1
        if self._uniform_norm:
            # HF computes (x * rescale_factor - mean) / std on raw [0, 255].
            self._norm_scale = self.rescale_factor / float(std[0])
            self._norm_bias = -float(mean[0]) / float(std[0])
        self._img_pp_ready = True

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
        view = pixel_values.reshape(
            -1,
            self.channel,
            self.temporal_patch_size * self.patch_size,
            self.patch_size,
        )
        out = tv_functional.normalize(
            view.to(torch.float32), self._fused_mean, self._fused_std
        )
        return out.to(pixel_values.dtype).reshape(shape)

    def _patchify_on_device(self, image):
        # (C, rh, rw) -> (gh*gw, C*tps*ps*ps), matching HF's patch layout.
        channel, resized_h, resized_w = image.shape
        ps, ms, tps = (
            self.patch_size,
            self.merge_size,
            self.temporal_patch_size,
        )
        grid_h, grid_w = resized_h // ps, resized_w // ps
        x = image.reshape(channel, grid_h // ms, ms, ps, grid_w // ms, ms, ps)
        x = x.permute(1, 4, 2, 5, 0, 3, 6)
        # A still image is replicated across the temporal patch dimension.
        x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, tps, -1, -1)
        return x.reshape(grid_h * grid_w, channel * tps * ps * ps)

    def _resize_and_patchify(self, image_input, grid_thw):
        """Flat raw uint8 -> per-image resize -> patchify -> concat patches."""
        flat = image_input["pixel_values"].to(self.visual.device)
        image_hw = image_input["image_hw"]
        channel, ps = self.channel, self.patch_size
        hw_list = image_hw.tolist() if hasattr(image_hw, "tolist") else image_hw
        chunks = flat.split([channel * h * w for h, w in hw_list])
        patches = []
        for chunk, (height, width), thw in zip(chunks, hw_list, grid_thw.tolist()):
            _, grid_h, grid_w = thw
            image = chunk.reshape(1, channel, height, width).float()
            # bicubic + antialias reproduces HF/PIL resize (cosine sim
            # 0.999996).
            image = torch.nn.functional.interpolate(
                image,
                size=[grid_h * ps, grid_w * ps],
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            # Images stay in raw [0, 255] here; rescale+normalize is applied
            # once to the concatenated patches below.
            image = image.clamp(0, 255).round().squeeze(0)
            patches.append(self._patchify_on_device(image))
        return torch.cat(patches, dim=0)

    def _parse_and_validate_image_input(self, **kwargs):
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None:
            # A plain dict so image_hw can ride along; the upstream
            # TypedDicts have no field for it.
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

    def _process_image_input(self, image_input):
        self._ensure_img_pp_cfg()
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            patches = self._resize_and_patchify(image_input, grid_thw)
            # Affine stays in float32 on the raw [0, 255] patches; casting to
            # the vision dtype first would lose mantissa bits on the multiply.
            pixel_values = self._rescale_and_normalize(patches).type(
                self.visual.dtype
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
        self._ensure_img_pp_cfg()
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = self._rescale_and_normalize(
                video_input["pixel_values_videos"].type(self.visual.dtype)
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
    AscendQwen3_5VLProcessor,
    info=AscendQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen3_5ForConditionalGeneration(
    _AscendVLPreprocessMixin, Qwen3_5ForConditionalGeneration
):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    AscendQwen3_5VLProcessor,
    info=AscendQwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen3_5MoeForConditionalGeneration(
    _AscendVLPreprocessMixin, Qwen3_5MoeForConditionalGeneration
):
    pass
