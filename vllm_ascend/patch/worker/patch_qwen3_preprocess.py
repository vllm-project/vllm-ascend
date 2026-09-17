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

"""Worker-side Qwen3-family image preprocessing offload."""

import torch
import torch.nn.functional as F
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.ascend_config import get_ascend_config

_ORIGINAL_PARSE_IMAGE_INPUT = Qwen3VLForConditionalGeneration._parse_and_validate_image_input
_ORIGINAL_PROCESS_IMAGE_INPUT = Qwen3VLForConditionalGeneration._process_image_input


def _get_preprocess_offload_stage() -> int:
    return get_ascend_config().qwen3_preprocess_offload_stage


def _is_npu_preprocess_enabled() -> bool:
    return _get_preprocess_offload_stage() >= 1


def _is_rescale_normalize_offload_enabled() -> bool:
    return _get_preprocess_offload_stage() >= 2


def _is_resize_offload_enabled() -> bool:
    return _get_preprocess_offload_stage() >= 3


def _ensure_image_rescale_normalize_config(
    self: Qwen3VLForConditionalGeneration,
) -> None:
    if getattr(self, "_ascend_image_rescale_normalize_ready", False):
        return

    image_processor = MULTIMODAL_REGISTRY.create_processor(self.model_config).info.get_hf_processor().image_processor
    channels = int(self.config.vision_config.in_channels)
    do_rescale = bool(image_processor.do_rescale)
    do_normalize = bool(image_processor.do_normalize)
    rescale_factor = float(image_processor.rescale_factor) if do_rescale else 1.0
    if do_normalize:
        image_mean = tuple(float(value) for value in image_processor.image_mean)
        image_std = tuple(float(value) for value in image_processor.image_std)
    else:
        image_mean = (0.0,) * channels
        image_std = (1.0,) * channels
    if len(image_mean) != channels or len(image_std) != channels:
        raise ValueError("image normalization parameters do not match channels")

    # Match the HF Torchvision backend exactly. When rescale and normalize are
    # both enabled, HF folds the rescale factor into FP32 mean/std and then
    # evaluates `(x - mean) / std`; it does not use `x * scale + bias`.
    if do_rescale and do_normalize:
        inverse_rescale = 1.0 / rescale_factor
        mean_tensor = torch.tensor(image_mean) * inverse_rescale
        std_tensor = torch.tensor(image_std) * inverse_rescale
        image_mean = tuple(mean_tensor.tolist())
        image_std = tuple(std_tensor.tolist())
        do_rescale = False

    self._ascend_image_do_rescale = do_rescale
    self._ascend_image_rescale_factor = rescale_factor
    self._ascend_image_do_normalize = do_normalize
    self._ascend_image_mean = image_mean
    self._ascend_image_std = image_std
    self._ascend_image_rescale_normalize_ready = True


def _rescale_normalize_on_npu(
    self: Qwen3VLForConditionalGeneration,
    images: torch.Tensor,
) -> torch.Tensor:
    """Apply FP32 rescale and normalize before temporal patch expansion."""
    _ensure_image_rescale_normalize_config(self)
    channels = int(self.config.vision_config.in_channels)
    values = images.to(dtype=torch.float32)
    if self._ascend_image_do_normalize:
        mean = values.new_tensor(self._ascend_image_mean).view(1, channels, 1, 1)
        std = values.new_tensor(self._ascend_image_std).view(1, channels, 1, 1)
        return values.sub_(mean).div_(std)
    if self._ascend_image_do_rescale:
        return values.mul_(self._ascend_image_rescale_factor)
    return values


def _parse_and_validate_image_input_with_npu_preprocess(self, **kwargs: object):
    pixel_values = kwargs.get("pixel_values")
    if _is_npu_preprocess_enabled() and isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 1:
        # The upstream TensorSchema requires post-patchify 2-D input. Offload
        # intentionally transports flat pre-patchify CHW data, so retain the
        # keys in a plain dict and validate them in _patchify_on_npu.
        return {
            "type": "pixel_values",
            "pixel_values": pixel_values,
            "image_grid_thw": kwargs.get("image_grid_thw"),
            "image_hw": kwargs.get("image_hw"),
        }
    return _ORIGINAL_PARSE_IMAGE_INPUT(self, **kwargs)


def _resize_uint8_on_npu(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Match the HF Torchvision UINT8 bicubic resize recipe on the NPU."""
    if images.shape[-2:] == (target_height, target_width):
        return images

    values = images.to(dtype=torch.float32).div_(256.0)
    values = F.interpolate(
        values,
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    values.mul_(256.0)
    values = torch.where(values > 255, 255, values)
    values = torch.where(values < 0, 0, values)
    return values.round().to(dtype=torch.uint8)


def _patchify_on_npu(
    self: Qwen3VLForConditionalGeneration,
    flat_images: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_hw: torch.Tensor | None = None,
) -> torch.Tensor:
    vision_config = self.config.vision_config
    channels = int(vision_config.in_channels)
    patch_size = int(vision_config.patch_size)
    merge_size = int(vision_config.spatial_merge_size)
    temporal_patch_size = int(vision_config.temporal_patch_size)

    if image_grid_thw is None or image_grid_thw.ndim != 2:
        raise ValueError("image_grid_thw must have shape [num_images, 3]")
    grid_rows = image_grid_thw.tolist()
    if any(int(grid_t) != 1 for grid_t, _, _ in grid_rows):
        raise ValueError("preprocessing offload currently supports images only")
    if _is_resize_offload_enabled():
        if (
            not isinstance(image_hw, torch.Tensor)
            or image_hw.ndim != 2
            or image_hw.shape[1] != 2
            or image_hw.shape[0] != image_grid_thw.shape[0]
        ):
            raise ValueError("image_hw must have shape [num_images, 2]")
        image_hw_rows = image_hw.tolist()
    else:
        image_hw_rows = [[int(grid_h) * patch_size, int(grid_w) * patch_size] for _, grid_h, grid_w in grid_rows]

    patch_batches: list[torch.Tensor] = []
    offset = 0
    image_index = 0
    while image_index < len(grid_rows):
        _, grid_h_raw, grid_w_raw = grid_rows[image_index]
        grid_h, grid_w = int(grid_h_raw), int(grid_w_raw)
        input_height, input_width = (int(value) for value in image_hw_rows[image_index])
        if grid_h % merge_size or grid_w % merge_size:
            raise ValueError(
                f"image grid must be divisible by spatial_merge_size: grid=({grid_h}, {grid_w}), merge={merge_size}"
            )

        # Batch consecutive images with matching source and target shapes while
        # preserving the original request order without gathers.
        run_end = image_index + 1
        while run_end < len(grid_rows):
            _, next_h, next_w = grid_rows[run_end]
            next_input_h, next_input_w = (int(value) for value in image_hw_rows[run_end])
            if (int(next_h), int(next_w)) != (grid_h, grid_w) or (next_input_h, next_input_w) != (
                input_height,
                input_width,
            ):
                break
            run_end += 1
        batch_size = run_end - image_index

        target_height = grid_h * patch_size
        target_width = grid_w * patch_size
        image_numel = channels * input_height * input_width
        batch_end = offset + batch_size * image_numel
        if batch_end > flat_images.numel():
            raise ValueError("pixel_values is shorter than image_grid_thw describes")

        images = flat_images[offset:batch_end].reshape(batch_size, channels, input_height, input_width)
        if _is_resize_offload_enabled():
            if images.dtype != torch.uint8:
                raise ValueError(f"Qwen3 image preprocessing stage 3 expects UINT8 input, got {images.dtype}")
            images = _resize_uint8_on_npu(images, target_height, target_width)
        if _is_rescale_normalize_offload_enabled():
            if images.dtype == torch.uint8:
                images = _rescale_normalize_on_npu(self, images)
            elif not images.is_floating_point():
                raise ValueError(f"Qwen3 image preprocessing stage 2 or 3 expects UINT8 input, got {images.dtype}")
            # Floating inputs retain compatibility with startup profiling and
            # cached entries already normalized before stage 2 was enabled.

        patches = images.reshape(
            batch_size,
            channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Match Transformers Qwen2VLImageProcessor.patchify exactly:
        # [B, gh/m, gw/m, m, m, C, patch_h, patch_w], followed by temporal
        # replication for still images. Ascend Copy only accepts tensors with
        # at most 8 dimensions, so collapse patch_h/patch_w before inserting
        # the temporal dimension instead of constructing a 9-D view.
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        patches = patches.reshape(
            batch_size,
            grid_h // merge_size,
            grid_w // merge_size,
            merge_size,
            merge_size,
            channels,
            patch_size * patch_size,
        )
        patches = (
            patches.unsqueeze(6)
            .expand(
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                temporal_patch_size,
                -1,
            )
            .reshape(
                batch_size * grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )
            .contiguous()
        )
        patch_batches.append(patches)
        offset = batch_end
        image_index = run_end

    if offset != flat_images.numel():
        raise ValueError(
            "pixel_values has trailing data not described by image_grid_thw: "
            f"consumed={offset}, total={flat_images.numel()}"
        )
    return torch.cat(patch_batches, dim=0)


def _process_image_input_with_npu_preprocess(
    self: Qwen3VLForConditionalGeneration,
    image_input,
) -> tuple[torch.Tensor, ...]:
    pixel_values = image_input.get("pixel_values")
    if (
        not _is_npu_preprocess_enabled()
        or image_input.get("type") != "pixel_values"
        or not isinstance(pixel_values, torch.Tensor)
        or pixel_values.ndim != 1
    ):
        return _ORIGINAL_PROCESS_IMAGE_INPUT(self, image_input)

    patches = _patchify_on_npu(
        self,
        pixel_values,
        image_input["image_grid_thw"],
        image_input.get("image_hw"),
    )
    patched_input = dict(image_input)
    patched_input["pixel_values"] = patches
    # Reuse upstream processing for the target dtype conversion, data-parallel
    # vision dispatch, and per-image embedding split.
    return _ORIGINAL_PROCESS_IMAGE_INPUT(self, patched_input)


Qwen3VLForConditionalGeneration._parse_and_validate_image_input = _parse_and_validate_image_input_with_npu_preprocess
Qwen3VLForConditionalGeneration._process_image_input = _process_image_input_with_npu_preprocess
