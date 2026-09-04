# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Worker-side Qwen3-VL family patchify and dtype conversion offload."""

import torch
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)

from vllm_ascend.ascend_config import get_ascend_config

_ORIGINAL_PARSE_IMAGE_INPUT = Qwen3VLForConditionalGeneration._parse_and_validate_image_input
_ORIGINAL_PROCESS_IMAGE_INPUT = Qwen3VLForConditionalGeneration._process_image_input


def _is_npu_preprocess_enabled() -> bool:
    return get_ascend_config().enable_qwen3_preprocess_on_npu


def _parse_and_validate_image_input_with_npu_preprocess(self, **kwargs: object):
    pixel_values = kwargs.get("pixel_values")
    if _is_npu_preprocess_enabled() and isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 1:
        # The upstream TensorSchema requires post-patchify 2-D input. Offload
        # intentionally transports flat pre-patchify CHW data, so retain the
        # same keys in a plain dict and validate them in _patchify_on_npu.
        return {
            "type": "pixel_values",
            "pixel_values": pixel_values,
            "image_grid_thw": kwargs.get("image_grid_thw"),
        }
    return _ORIGINAL_PARSE_IMAGE_INPUT(self, **kwargs)


def _patchify_on_npu(
    self: Qwen3VLForConditionalGeneration,
    flat_images: torch.Tensor,
    image_grid_thw: torch.Tensor,
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

    patch_batches: list[torch.Tensor] = []
    offset = 0
    image_index = 0
    while image_index < len(grid_rows):
        _, grid_h_raw, grid_w_raw = grid_rows[image_index]
        grid_h, grid_w = int(grid_h_raw), int(grid_w_raw)
        if grid_h % merge_size or grid_w % merge_size:
            raise ValueError(
                f"image grid must be divisible by spatial_merge_size: grid=({grid_h}, {grid_w}), merge={merge_size}"
            )

        # Batch consecutive images with the same resized shape. This is the
        # common 5/10-image case and avoids launching one patchify sequence per
        # image while preserving the original request order without gathers.
        run_end = image_index + 1
        while run_end < len(grid_rows):
            _, next_h, next_w = grid_rows[run_end]
            if (int(next_h), int(next_w)) != (grid_h, grid_w):
                break
            run_end += 1
        batch_size = run_end - image_index

        height = grid_h * patch_size
        width = grid_w * patch_size
        image_numel = channels * height * width
        batch_end = offset + batch_size * image_numel
        if batch_end > flat_images.numel():
            raise ValueError("pixel_values is shorter than image_grid_thw describes")

        images = flat_images[offset:batch_end].reshape(batch_size, channels, height, width)
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

    patches = _patchify_on_npu(self, pixel_values, image_input["image_grid_thw"])
    patched_input = dict(image_input)
    patched_input["pixel_values"] = patches
    # Reuse upstream processing for the runtime target dtype conversion,
    # data-parallel vision dispatch, and per-image embedding split.
    return _ORIGINAL_PROCESS_IMAGE_INPUT(self, patched_input)


Qwen3VLForConditionalGeneration._parse_and_validate_image_input = _parse_and_validate_image_input_with_npu_preprocess
Qwen3VLForConditionalGeneration._process_image_input = _process_image_input_with_npu_preprocess
