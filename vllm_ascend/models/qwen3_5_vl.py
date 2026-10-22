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

When active, the front-end emits flat raw uint8 plus the original (H, W) and
the target grid; resize, patchify and rescale+normalize all run on the device.
Normalize reuses vLLM's fused mm input norm. smart_resize stays on the CPU so
placeholder accounting is unchanged. Payload drops ~8x (uint8, no temporal
replication).

The raw-pixel path is only taken when the effective processor settings match
what the device pipeline implements (resize enabled, bicubic resample,
uint8-only input, and mm_device_do_normalize enabled so the CPU skips
rescale/normalize). Any other configuration -- device normalization disabled
(e.g. --multimodal-config mm_device_do_normalize=false), do_resize=False, a
non-bicubic resample filter, or non-uint8 (floating-point) pixels -- falls
back to the complete stock CPU pipeline, and the emitted keys (no
``image_hw``) tell the field config and the model to keep the stock
behaviour as well, so rescale+normalize is applied exactly once in every
configuration.

Encoder ACL graph path: sub-batch selection slices the raw payloads first and
preprocesses only the selected items, so resize/patchify/normalize run exactly
once per item no matter how many sub-batches or capture paths consume it.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers.image_processing_utils import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
    smart_resize,
)
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
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

# Marker stored in the mm_kwargs dicts produced by select_encoder_cudagraph_items
# once device-side preprocessing has been applied. The graph manager re-packs
# sub-batches (and the DP path re-selects rank-local subsets), so without this
# marker a nested selection would normalize the patches a second time.
_DEVICE_PREPROCESS_DONE = "_ascend_device_preprocess_done"


def _all_uint8(images) -> bool:
    """The device pipeline quantizes pixels with clamp+round, which would
    alter valid floating-point values (e.g. 128.4 -> 128); only uint8 inputs
    may take the raw path, everything else keeps the stock CPU pipeline."""
    for image in images:
        dtype = image.dtype
        if torch.is_tensor(image):
            if dtype != torch.uint8:
                return False
        elif str(dtype) != "uint8":  # numpy arrays
            return False
    return True


class AscendQwen2VLImageProcessor(Qwen2VLImageProcessor):
    """Emit flat raw uint8 + original (H, W) + target grid; skip everything else.

    Falls back to the stock CPU pipeline whenever the effective settings are
    not what the device pipeline implements (see the class docstring above).
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
        if (
            not do_resize
            or do_rescale
            or do_normalize
            or resample != Image.Resampling.BICUBIC
            or not _all_uint8(images)
        ):
            # Unsupported or non-default settings: do_resize disabled (the
            # grid must come from the original resolution), a non-bicubic
            # resample filter (the device path hardcodes bicubic+antialias),
            # device normalization disabled (mm_device_do_normalize off
            # leaves do_rescale/do_normalize enabled, so the CPU must run the
            # full pipeline), or non-uint8 (floating-point) pixels that the
            # device path's clamp+round quantization would alter. Keep the
            # stock behaviour so grids, values and token counts match
            # upstream exactly.
            return Qwen2VLImageProcessor._preprocess(
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
            )
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
        # Install the raw-pixel image processor unconditionally: its
        # _preprocess falls back to the stock pipeline per call whenever the
        # effective settings are not supported (mm_device_do_normalize off,
        # do_resize=False, non-bicubic resample, non-uint8 pixels), so the
        # swap is inert there. mm_device_do_normalize=false (per-engine,
        # shared by processor and worker) restores the complete stock CPU
        # pipeline; the worker follows the emitted keys.
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
        if "image_hw" not in hf_inputs:
            # Stock CPU pipeline (offload disabled, or unsupported processor
            # settings): keep the stock field mapping.
            return config
        channel = self.info.get_hf_config().vision_config.in_channels
        image_hw = hf_inputs["image_hw"]
        config["pixel_values"] = MultiModalFieldConfig.flat_from_sizes("image", image_hw.prod(-1) * channel)
        config["image_hw"] = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
        return config


class _AscendVLPreprocessMixin:
    """Device-side resize/patchify + normalize for Qwen3.5-VL."""

    supports_mm_device_do_normalize = True

    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.input_norm = build_mm_input_norm(self.model_config)
        vc = self.config.vision_config
        self._c = vc.in_channels
        self._ps = vc.patch_size
        self._tps = vc.temporal_patch_size
        self._ms = vc.spatial_merge_size

    @property
    def _device_do_normalize(self) -> bool:
        return bool(self.model_config.multimodal_config.mm_device_do_normalize)

    def _resize_and_patchify(self, image_input, grid_thw):
        """Flat raw uint8 -> per-image resize -> patchify -> concat patches."""
        # Upload the uint8 payload once and upcast in a single vectorized op:
        # CANN's antialias bicubic kernel rejects uint8 anyway (the float
        # kernel differs slightly from the HF/PIL one), and one cast kernel
        # beats one per image inside the loop.
        flat = image_input["pixel_values"].to(self.visual.device, non_blocking=True).float()
        hw_list = image_input["image_hw"].tolist()
        c, ps, tps, ms = self._c, self._ps, self._tps, self._ms
        sizes = [c * h * w for h, w in hw_list]
        chunks = flat.split(sizes) if sizes else []
        patches = []
        for chunk, (h, w), (_, gh, gw) in zip(chunks, hw_list, grid_thw.tolist()):
            img = chunk.reshape(1, c, h, w)
            # bicubic + antialias reproduces HF/PIL resize (cosine sim
            # 0.999996); clamp+round replicate the original uint8
            # quantization.
            img = F.interpolate(img, size=[gh * ps, gw * ps], mode="bicubic", align_corners=False, antialias=True)
            img = img.clamp(0, 255).round().squeeze(0)
            # patchify: (C, rh, rw) -> (gh*gw, C*tps*ps*ps), HF layout.
            # Drop the batch dim before unsqueeze/expand: keeping it (as HF's
            # batched patchify does) creates a 9-D tensor, which CANN rejects
            # ("self tensor cannot be larger than 8 dimensions").
            x = img.reshape(c, gh // ms, ms, ps, gw // ms, ms, ps)
            x = x.permute(1, 4, 2, 5, 0, 3, 6)
            x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, tps, -1, -1)
            patches.append(x.reshape(gh * gw, c * tps * ps * ps))
        if not patches:
            # Empty batch (e.g. profiling with no images): torch.cat([]) raises.
            return flat.new_empty((0, c * tps * ps * ps))
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

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        """Slice the raw payloads BEFORE preprocessing.

        With cudagraph_mm_encoder enabled, the graph manager packs sub-batches
        (and the DP path re-selects rank-local subsets), calling this hook once
        per sub-batch. Preprocessing inside _get_pixel_values_by_modality would
        re-run resize/patchify/normalize on the whole batch for every
        sub-batch. Instead, slice first and preprocess only the selected items,
        then mark the result so nested selections pass it through unchanged.
        Downstream consumers (replay buffers, eager forward) receive
        graph-ready patches, exactly like the stock selector's output.
        """
        if "image_hw" in mm_kwargs:
            # Raw flat uint8: element offsets are (C*H*W) per item, not patch
            # counts, so the stock slicing logic does not apply.
            flat = mm_kwargs["pixel_values"]
            hw_list = mm_kwargs["image_hw"].tolist()
            grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
            selected_hw = [hw_list[i] for i in indices]
            selected_grid = [grid_thw[i] for i in indices]
            if not indices:
                # Nothing selected: skip preprocessing entirely.
                return {
                    "pixel_values": torch.empty(
                        (0, self._c * self._tps * self._ps * self._ps),
                        dtype=self.visual.dtype,
                        device=self.visual.device,
                    ),
                    "image_grid_thw": [],
                    _DEVICE_PREPROCESS_DONE: True,
                }
            c = self._c
            sizes = [c * h * w for h, w in hw_list]
            cum = [0]
            for s in sizes:
                cum.append(cum[-1] + s)
            selected_flat = torch.cat([flat[cum[i] : cum[i + 1]] for i in indices])
            image_input = {
                "type": "pixel_values",
                "pixel_values": selected_flat,
                "image_grid_thw": torch.tensor(selected_grid, dtype=torch.long),
                "image_hw": torch.tensor(selected_hw, dtype=torch.long),
            }
            patches = self._resize_and_patchify(image_input, image_input["image_grid_thw"])
            pixel_values = self.input_norm(patches, self.visual.dtype)
            return {
                "pixel_values": pixel_values,
                "image_grid_thw": selected_grid,
                _DEVICE_PREPROCESS_DONE: True,
            }
        # Stock image patches / video patches: slice by patch counts exactly
        # like the upstream selector, then apply device-side normalization
        # once (with mm_device_do_normalize enabled the CPU processor ran
        # with do_rescale/do_normalize disabled).
        modality = self.get_input_modality(mm_kwargs)
        pv_key, grid_key = (
            ("pixel_values", "image_grid_thw") if modality == "image" else ("pixel_values_videos", "video_grid_thw")
        )
        pixel_values = mm_kwargs[pv_key]
        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
        patches_per_item = [t * h * w for t, h, w in grid_thw]
        cum_patches = [0]
        for p in patches_per_item:
            cum_patches.append(cum_patches[-1] + p)
        selected_pv = (
            torch.cat([pixel_values[cum_patches[i] : cum_patches[i + 1]] for i in indices])
            if indices
            else pixel_values[:0]
        )
        selected_grid = [grid_thw[i] for i in indices]
        if indices and not mm_kwargs.get(_DEVICE_PREPROCESS_DONE) and self._device_do_normalize:
            selected_pv = self.input_norm(selected_pv, self.visual.dtype)
        return {
            pv_key: selected_pv,
            grid_key: selected_grid,
            _DEVICE_PREPROCESS_DONE: True,
        }

    def _process_image_input(self, image_input):
        grid_thw = image_input["image_grid_thw"]
        if image_input["type"] == "image_embeds":
            embeds = image_input["image_embeds"].type(self.visual.dtype)
            sizes = (grid_thw.prod(-1) // self._ms // self._ms).tolist()
            return embeds.split(sizes)
        if image_input.get("image_hw") is not None:
            # Raw uint8 payload from the Ascend processor: resize, patchify
            # and normalize on device (exactly once).
            patches = self._resize_and_patchify(image_input, grid_thw)
            pixel_values = self.input_norm(patches, self.visual.dtype)
        else:
            # Stock patchified input. When mm_device_do_normalize is enabled
            # the CPU processor ran with do_rescale/do_normalize disabled, so
            # the normalizer (identity when the feature is off) runs here
            # instead -- rescale+normalize happens exactly once either way.
            pixel_values = self.input_norm(image_input["pixel_values"], self.visual.dtype)
        return self._run_visual(pixel_values, grid_thw)

    def _process_video_input(self, video_input):
        grid_thw = video_input["video_grid_thw"]
        if video_input["type"] == "video_embeds":
            embeds = video_input["video_embeds"].type(self.visual.dtype)
            sizes = (grid_thw.prod(-1) // self._ms // self._ms).tolist()
            return embeds.split(sizes)
        # With mm_device_do_normalize enabled the CPU video processor ran
        # with do_rescale/do_normalize disabled; the normalizer (identity
        # when the feature is off) runs here exactly once.
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
