# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from transformers import AutoImageProcessor, HunYuanVLProcessor

from vllm_ascend.utils import vllm_version_is

_HUNYUAN_VL_EXTRA_SPECIAL_TOKENS = {
    "image_start_token": "<｜hy_place▁holder▁no▁100｜>",
    "image_end_token": "<｜hy_place▁holder▁no▁101｜>",
    "image_token": "<｜hy_place▁holder▁no▁102｜>",
}
_HUNYUAN_VL_SPECIAL_TOKENS = {
    **_HUNYUAN_VL_EXTRA_SPECIAL_TOKENS,
    "pad_token": "<｜hy_▁pad▁｜>",
}
_HUNYUAN_VL_SPECIAL_TOKEN_IDS = {
    "image_start_token": 120118,
    "image_end_token": 120119,
    "image_token": 120120,
    "pad_token": 120002,
}


def _register_hunyuan_tokenizer_special_tokens(tokenizer: Any) -> None:
    """Restore named image tokens missing from older HunyuanOCR snapshots."""
    missing_tokens = {
        name: token
        for name, token in _HUNYUAN_VL_EXTRA_SPECIAL_TOKENS.items()
        if getattr(tokenizer, name, None) is None
    }
    if missing_tokens:
        tokenizer._set_model_specific_special_tokens(special_tokens=missing_tokens)

    actual_tokens = {
        name: (getattr(tokenizer, name, None), getattr(tokenizer, f"{name}_id", None))
        for name in _HUNYUAN_VL_SPECIAL_TOKENS
    }
    expected_tokens = {
        name: (token, _HUNYUAN_VL_SPECIAL_TOKEN_IDS[name]) for name, token in _HUNYUAN_VL_SPECIAL_TOKENS.items()
    }
    if actual_tokens != expected_tokens:
        raise ValueError(
            "HunyuanVL tokenizer special-token schema does not match the model vocabulary: "
            f"expected {expected_tokens!r}, got {actual_tokens!r}"
        )


class _HunYuanVLProcessorCompat(HunYuanVLProcessor):
    """Register the tokenizer schema before native processor initialization."""

    def __init__(
        self,
        image_processor: Any = None,
        tokenizer: Any = None,
        chat_template: Any = None,
        cat_extra_token: bool = True,
        **kwargs: Any,
    ) -> None:
        _register_hunyuan_tokenizer_special_tokens(tokenizer)
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            cat_extra_token=cat_extra_token,
            **kwargs,
        )


def _import_v025_hunyuan_vision() -> Any:
    """Import v0.25.1 while skipping its obsolete processor registration."""
    original_descriptor = AutoImageProcessor.__dict__["register"]
    original_register = AutoImageProcessor.register

    def register(config_class: Any, *args: Any, **kwargs: Any) -> Any:
        if config_class == "HunYuanVLImageProcessor":
            return None
        return original_register(config_class, *args, **kwargs)

    AutoImageProcessor.register = staticmethod(register)  # type: ignore[method-assign]
    try:
        return importlib.import_module("vllm.model_executor.models.hunyuan_vision")
    finally:
        AutoImageProcessor.register = original_descriptor  # type: ignore[method-assign]


def _patch_hunyuan_processor_loader(hunyuan_vision: Any) -> None:
    def get_hf_processor(self: Any, **kwargs: object) -> Any:
        kwargs.pop("use_fast", None)
        kwargs.setdefault("backend", "pil")
        return self.ctx.get_hf_processor(_HunYuanVLProcessorCompat, **kwargs)

    hunyuan_vision.HunYuanVLProcessingInfo.get_hf_processor = get_hf_processor


def _patch_image_token_wrapping(hunyuan_vision: Any) -> None:
    def call_hf_processor(
        self: Any,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Any:
        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        if mm_data.get("images") is not None and prompt:
            image_token = hf_processor.image_token
            wrapped_token = f"{hf_processor.image_start_token}{image_token}{hf_processor.image_end_token}"
            if image_token in prompt and wrapped_token not in prompt:
                prompt = prompt.replace(image_token, wrapped_token)
        return self.info.ctx.call_hf_processor(
            hf_processor,
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )

    hunyuan_vision.HunYuanVLMultiModalProcessor._call_hf_processor = call_hf_processor


def install_hunyuan_vl_processor_compat() -> None:
    """Patch supported vLLM versions for older HunyuanOCR tokenizers."""
    is_v025 = vllm_version_is("0.25.1")
    if is_v025:
        hunyuan_vision = _import_v025_hunyuan_vision()
    else:
        hunyuan_vision = importlib.import_module("vllm.model_executor.models.hunyuan_vision")

    _patch_hunyuan_processor_loader(hunyuan_vision)
    if is_v025:
        _patch_image_token_wrapping(hunyuan_vision)
