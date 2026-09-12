# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""DeepSeek V4.1 multimodal preprocessing.

V4.1 deliberately does not reuse V4's five-sentinel N-layout. The checkpoint
uses one token id for every image-span position and carries the
start/image/newline/end role in a separate tensor. vLLM adds at most one
reserved-token row before the span so ratio-2 compressor groups start at a
stable phase.
"""

import copy
import math
import threading
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import BatchFeature
from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)

from vllm_ascend.deepseek_v41_config import DeepseekV41Config

IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
COMPRESS_PAD_TO = 2

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
IMAGE_TOKEN_ID = 129264
# A reserved in-vocabulary token used only for vLLM's leading compressor pad.
IMAGE_PAD_ID = 129265
IMAGE_PAD_TOKEN_NAME = "<|place_holder_mm_span_0436|>"

_TOKENIZER_THREAD_LOCAL = threading.local()


def _get_thread_local_tokenizer(tokenizer):
    cached = getattr(_TOKENIZER_THREAD_LOCAL, "tokenizer", None)
    source_id = getattr(_TOKENIZER_THREAD_LOCAL, "source_id", None)
    if cached is None or source_id != id(tokenizer):
        cached = copy.deepcopy(tokenizer)
        _TOKENIZER_THREAD_LOCAL.tokenizer = cached
        _TOKENIZER_THREAD_LOCAL.source_id = id(tokenizer)
    return cached


def image_sentinel_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """Return image-span and compressor-pad positions."""
    return (token_ids == IMAGE_TOKEN_ID) | (token_ids == IMAGE_PAD_ID)


def validate_image_sentinel_ids(tokenizer) -> None:
    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if image_id != IMAGE_TOKEN_ID:
        raise ValueError(
            f"Image placeholder {IMAGE_PLACEHOLDER!r} has id {image_id}, "
            f"expected {IMAGE_TOKEN_ID}."
        )
    pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN_NAME)
    if pad_id != IMAGE_PAD_ID:
        raise ValueError(
            f"Image pad token {IMAGE_PAD_TOKEN_NAME!r} has id {pad_id}, "
            f"expected {IMAGE_PAD_ID}."
        )


def llm_grid(best_height, best_width, patch_size, downsample_ratio):
    return (
        math.ceil((best_height // patch_size) / downsample_ratio),
        math.ceil((best_width // patch_size) / downsample_ratio),
    )


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def leading_compressor_pad(start_pos: int) -> int:
    """Rows needed to align an image span to the V4.1 CR2 phase."""
    return COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(
        math.floor(max_w_float) * cell / width,
        math.floor(max_h_float) * cell / height,
    )
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def safe_resize(
    height,
    width,
    best_height,
    best_width,
    patch_size,
    downsample_ratio,
    max_n_token,
):
    # Reserve the maximum one-token leading pad injected after tokenization.
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w = llm_grid(
        best_height, best_width, patch_size, downsample_ratio
    )
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height,
            width,
            patch_size,
            downsample_ratio,
            max_n_token,
        )
        n_llm_h, n_llm_w = llm_grid(
            best_height, best_width, patch_size, downsample_ratio
        )
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def load_image(
    image: Image.Image,
    *,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
    min_pixels: int,
    max_wh_ratio: float | None,
):
    p = patch_size
    image = image.convert("RGB")
    width, height = image.size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        downsample_ratio,
        max_n_token,
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(
            image,
            (best_width, best_height),
            color=(127, 127, 127),
        )
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, n_vit_h, p, n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, p, p)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Reference reading-order V4.1 image-span roles."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


class DeepseekV41VLImageProcessor:
    def __init__(self, config: DeepseekV41Config) -> None:
        self.patch_size = config.vision_patch_size
        self.downsample_ratio = config.vision_downsample_ratio
        self.max_n_token = config.vision_max_n_token
        self.min_pixels = config.vision_min_pixels
        self.max_wh_ratio = config.vision_max_wh_ratio

    def __call__(self, image: Image.Image):
        return load_image(
            image,
            patch_size=self.patch_size,
            downsample_ratio=self.downsample_ratio,
            max_n_token=self.max_n_token,
            min_pixels=self.min_pixels,
            max_wh_ratio=self.max_wh_ratio,
        )


class DeepseekV41VLProcessor:
    def __init__(self, config: DeepseekV41Config) -> None:
        self.config = config
        self.image_processor = DeepseekV41VLImageProcessor(config)

    def __call__(
        self,
        text: str | None = None,
        images: Sequence[Image.Image] | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        del text, return_tensors, kwargs
        patches_list = []
        vit_grid = []
        llm_grid_list = []
        types_list = []
        for image in images or []:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = (
                self.image_processor(image)
            )
            patches_list.append(patches)
            vit_grid.append((n_vit_h, n_vit_w))
            llm_grid_list.append((n_llm_h, n_llm_w))
            types_list.append(image_token_types(n_llm_h, n_llm_w))

        if not patches_list:
            return BatchFeature({})

        return BatchFeature(
            {
                "patches": torch.cat(patches_list),
                "vit_grid": torch.tensor(vit_grid, dtype=torch.int64),
                "llm_grid": torch.tensor(llm_grid_list, dtype=torch.int64),
                "types": torch.cat(types_list),
            }
        )


class DeepseekV41VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> DeepseekV41Config:
        return self.ctx.get_hf_config(DeepseekV41Config)

    def get_hf_processor(self, **kwargs: object) -> DeepseekV41VLProcessor:
        if kwargs:
            raise ValueError(f"Unexpected processor kwargs: {sorted(kwargs)}")
        return DeepseekV41VLProcessor(self.get_hf_config())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        del seq_len, mm_counts
        return {
            "image": self.get_hf_config().vision_max_n_token
            + COMPRESS_PAD_TO
            - 1
        }

    def get_image_placeholder_token_id(self) -> int:
        token_id = self.get_tokenizer().convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if token_id is None:
            raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
        return token_id

    def get_image_size_with_most_features(self) -> ImageSize:
        config = self.get_hf_config()
        budget = config.vision_max_n_token - (COMPRESS_PAD_TO - 1)
        side = budget * config.vision_patch_size * config.vision_downsample_ratio
        best_h, best_w = solve_resize_ratio(
            side,
            side,
            config.vision_patch_size,
            config.vision_downsample_ratio,
            budget,
        )
        return ImageSize(width=best_w, height=best_h)


class DeepseekV41VLDummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekV41VLProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        del seq_len
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=cast(
                    ImageDummyOptions | None,
                    mm_options.get("image"),
                ),
            ),
        }


class DeepseekV41VLMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV41VLProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object] | None = None,
    ) -> BatchFeature:
        if tok_kwargs is None:
            tok_kwargs = {}
        processor = self.info.get_hf_processor(**mm_kwargs)
        processed = processor(
            text=prompt,
            images=cast(Sequence[Image.Image] | None, mm_data.get("images")),
            return_tensors="pt",
        )
        tokenizer = _get_thread_local_tokenizer(self.info.get_tokenizer())
        processed["input_ids"] = tokenizer(
            prompt,
            return_tensors="pt",
            **tok_kwargs,
        )["input_ids"]
        return processed

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        del prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        vit_grid = hf_inputs.get("vit_grid")
        llm_grid = hf_inputs.get("llm_grid")
        if vit_grid is None or llm_grid is None:
            empty = torch.empty(0, dtype=torch.long)
            patch_sizes = types_sizes = empty
        else:
            patch_sizes = vit_grid.prod(-1)
            n_llm_h, n_llm_w = llm_grid[:, 0], llm_grid[:, 1]
            types_sizes = n_llm_h * (n_llm_w + 1) + 2
        return {
            "patches": MultiModalFieldConfig.flat_from_sizes(
                "image", patch_sizes
            ),
            "vit_grid": MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            ),
            "llm_grid": MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            ),
            "types": MultiModalFieldConfig.flat_from_sizes(
                "image", types_sizes, keep_on_cpu=True
            ),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del mm_items, hf_processor_mm_kwargs
        image_token_id = self.info.get_image_placeholder_token_id()
        validate_image_sentinel_ids(self.info.get_tokenizer())

        def get_image_replacement(item_idx: int) -> PromptUpdateDetails:
            types: torch.Tensor = out_mm_kwargs["image"][item_idx]["types"].data
            full = [image_token_id] * types.numel()
            return PromptUpdateDetails.select_token_id(full, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_image_replacement,
            )
        ]

    def _apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[
        list[int],
        Mapping[str, list[PlaceholderFeaturesInfo]],
    ]:
        """Apply replacements and prepend the position-dependent CR2 pad."""
        new_token_ids, base_placeholders = super()._apply_prompt_updates(
            token_ids,
            mm_prompt_updates,
        )
        placeholders: dict[str, list[PlaceholderFeaturesInfo]] = {
            modality: [] for modality in base_placeholders
        }
        ordered = sorted(
            (
                placeholder.start_idx,
                modality,
                placeholder,
            )
            for modality, items in base_placeholders.items()
            for placeholder in items
        )
        inserted = 0
        for _, modality, placeholder in ordered:
            start_idx = placeholder.start_idx + inserted
            tokens = list(placeholder.tokens)
            is_embed = placeholder.is_embed
            if modality == "image":
                compress_pad = leading_compressor_pad(start_idx)
                new_token_ids[start_idx:start_idx] = [IMAGE_PAD_ID] * compress_pad
                tokens = [IMAGE_PAD_ID] * compress_pad + tokens
                original_mask = (
                    is_embed
                    if is_embed is not None
                    else torch.ones(len(placeholder.tokens), dtype=torch.bool)
                )
                is_embed = torch.cat(
                    [
                        torch.zeros(compress_pad, dtype=torch.bool),
                        original_mask,
                    ]
                )
                inserted += compress_pad
            placeholders[modality].append(
                PlaceholderFeaturesInfo(
                    modality=modality,
                    item_idx=placeholder.item_idx,
                    start_idx=start_idx,
                    tokens=tokens,
                    is_embed=is_embed,
                )
            )
        return new_token_ids, placeholders
