# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

"""Qwen3-VL family image patchify and dtype conversion offload."""

from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

import torch
from transformers import BatchFeature
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import SizeDict
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
    smart_resize,
)
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing.context import InputProcessingContext

from vllm_ascend.ascend_config import get_ascend_config

_ORIGINAL_POSTPROCESS_OUTPUT = InputProcessingContext._postprocess_output
_ORIGINAL_APPLY_HF_PROCESSOR_MAIN = Qwen3VLMultiModalProcessor._apply_hf_processor_main
_ORIGINAL_GET_MM_FIELDS_CONFIG = Qwen3VLMultiModalProcessor._get_mm_fields_config
_ORIGINAL_IMAGE_PREPROCESS = Qwen2VLImageProcessor._preprocess
_VISUAL_PIXEL_KEYS = {"pixel_values"}
_SUPPORTED_MODEL_TYPES = {
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}
_NPU_PREPROCESS_ACTIVE = ContextVar("qwen3_npu_preprocess_active", default=False)


def _is_npu_preprocess_enabled() -> bool:
    return get_ascend_config().enable_qwen3_preprocess_on_npu


def _is_supported_context(context: InputProcessingContext) -> bool:
    model_type = getattr(context.model_config.hf_config, "model_type", None)
    return model_type in _SUPPORTED_MODEL_TYPES


def _is_supported_processor(processor: Qwen3VLMultiModalProcessor) -> bool:
    model_type = getattr(processor.info.get_hf_config(), "model_type", None)
    return model_type in _SUPPORTED_MODEL_TYPES


def _apply_hf_processor_main_with_npu_preprocess(
    self: Qwen3VLMultiModalProcessor,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> BatchFeature:
    # The offload is deliberately image-only. A mixed image/video request keeps
    # the upstream contract until video patchify is implemented as well.
    enabled = (
        _is_npu_preprocess_enabled()
        and _is_supported_processor(self)
        and mm_items.get_count("video", strict=False) == 0
    )
    token = _NPU_PREPROCESS_ACTIVE.set(enabled)
    try:
        return _ORIGINAL_APPLY_HF_PROCESSOR_MAIN(self, mm_items, hf_processor_mm_kwargs)
    finally:
        _NPU_PREPROCESS_ACTIVE.reset(token)


def _preprocess_with_npu_preprocess(
    self: Qwen2VLImageProcessor,
    images: list[torch.Tensor],
    do_resize: bool,
    size,
    resample,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: float | list[float] | None,
    image_std: float | list[float] | None,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    disable_grouping: bool | None,
    return_tensors,
    **kwargs,
) -> BatchFeature:
    if not _NPU_PREPROCESS_ACTIVE.get():
        return _ORIGINAL_IMAGE_PREPROCESS(
            self,
            images,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            disable_grouping=disable_grouping,
            return_tensors=return_tensors,
            **kwargs,
        )

    grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
    resized_images_grouped = {}
    for shape, stacked_images in grouped_images.items():
        if do_resize:
            height, width = stacked_images.shape[-2:]
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=size.shortest_edge,
                max_pixels=size.longest_edge,
            )
            stacked_images = self.resize(
                image=stacked_images,
                size=SizeDict(height=resized_height, width=resized_width),
                resample=resample,
            )
        resized_images_grouped[shape] = stacked_images

    resized_images = reorder_images(resized_images_grouped, grouped_images_index)
    grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
    processed_images_grouped = {}
    processed_grids = {}
    for shape, stacked_images in grouped_images.items():
        processed_images = self.rescale_and_normalize(
            stacked_images,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
        )
        batch_size, _, resized_height, resized_width = processed_images.shape
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size

        # Preserve one row per image for shape-group reordering. The final cat
        # below produces the flat CHW transport contract expected by vLLM's
        # MultiModalFlatField and the Worker-side patchify implementation.
        processed_images_grouped[shape] = processed_images.flatten(1)
        processed_grids[shape] = [[1, grid_h, grid_w]] * batch_size

    processed_images = reorder_images(processed_images_grouped, grouped_images_index)
    processed_grids_ordered = reorder_images(processed_grids, grouped_images_index)
    pixel_values = torch.cat(processed_images, dim=0)
    image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)

    return BatchFeature(
        data={
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        },
        tensor_type=return_tensors,
    )


def _get_mm_fields_config_with_npu_preprocess(
    self: Qwen3VLMultiModalProcessor,
    hf_inputs: BatchFeature,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> Mapping[str, MultiModalFieldConfig]:
    configs = dict(_ORIGINAL_GET_MM_FIELDS_CONFIG(self, hf_inputs, hf_processor_mm_kwargs))
    pixel_values = hf_inputs.get("pixel_values")
    image_grid_thw = hf_inputs.get("image_grid_thw")
    if (
        not _is_npu_preprocess_enabled()
        or not _is_supported_processor(self)
        or not isinstance(pixel_values, torch.Tensor)
        or pixel_values.ndim != 1
        or not isinstance(image_grid_thw, torch.Tensor)
    ):
        return configs

    vision_config = self.info.get_hf_config().vision_config
    channels = int(vision_config.in_channels)
    patch_size = int(vision_config.patch_size)
    size_per_image = image_grid_thw[:, 1:].prod(-1) * channels * patch_size * patch_size
    configs["pixel_values"] = MultiModalFieldConfig.flat_from_sizes("image", size_per_image)
    return configs


def _postprocess_output_with_npu_preprocess(
    self: InputProcessingContext,
    output,
):
    if not _is_npu_preprocess_enabled() or not _is_supported_context(self) or not _NPU_PREPROCESS_ACTIVE.get():
        return _ORIGINAL_POSTPROCESS_OUTPUT(self, output)

    keep_on_device = self.model_config.get_multimodal_config().mm_tensor_ipc == "torch_shm"

    def postprocess(value: Any, *, keep_visual_dtype: bool = False):
        if isinstance(value, dict):
            return {
                key: postprocess(
                    item,
                    keep_visual_dtype=(keep_visual_dtype or key in _VISUAL_PIXEL_KEYS),
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [postprocess(item, keep_visual_dtype=keep_visual_dtype) for item in value]
        if isinstance(value, tuple):
            return tuple(postprocess(item, keep_visual_dtype=keep_visual_dtype) for item in value)
        if not isinstance(value, torch.Tensor):
            return value

        tensor = value
        if tensor.is_floating_point() and not keep_visual_dtype:
            tensor = tensor.to(dtype=self.model_config.dtype)
        if not tensor.is_cpu and not keep_on_device:
            tensor = tensor.cpu()
        return tensor

    return postprocess(output)


InputProcessingContext._postprocess_output = _postprocess_output_with_npu_preprocess
Qwen3VLMultiModalProcessor._apply_hf_processor_main = _apply_hf_processor_main_with_npu_preprocess
Qwen3VLMultiModalProcessor._get_mm_fields_config = _get_mm_fields_config_with_npu_preprocess
Qwen2VLImageProcessor._preprocess = _preprocess_with_npu_preprocess
