# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformers import HunYuanVLProcessor

from vllm_ascend.utils import vllm_version_is

_STALE_PROCESSOR_MODULES = {
    "HunYuanVLProcessor": "vllm.transformers_utils.processors.hunyuan_vl",
    "HunYuanVLImageProcessor": "vllm.transformers_utils.processors.hunyuan_vl_image",
}

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
    """Restore the named-token schema required by Transformers 5.13."""
    missing_tokens = {
        name: token
        for name, token in _HUNYUAN_VL_EXTRA_SPECIAL_TOKENS.items()
        if tokenizer is not None and getattr(tokenizer, name, None) is None
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
    """Native processor with the legacy HunyuanOCR token schema restored."""

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


def _remove_stale_registry_entries() -> bool:
    """Backport the lazy-registry cleanup from vLLM PR #47867."""
    import vllm.transformers_utils.processors as vllm_processors

    class_to_module = vllm_processors._CLASS_TO_MODULE
    exported_names = vllm_processors.__all__
    entries_to_remove = []
    for class_name, stale_module in _STALE_PROCESSOR_MODULES.items():
        registered_module = class_to_module.get(class_name)
        if registered_module is None:
            continue
        if registered_module != stale_module:
            raise RuntimeError(f"Unexpected vLLM processor registry entry for {class_name}: {registered_module!r}")
        if class_name not in exported_names:
            raise RuntimeError(f"Missing vLLM processor export for {class_name}")

        entries_to_remove.append(class_name)

    for class_name in entries_to_remove:
        del class_to_module[class_name]
        exported_names.remove(class_name)

    return bool(entries_to_remove)


def _patch_hunyuan_processor_loader(hunyuan_vision: Any) -> None:
    """Use the native processor with the complete Hunyuan tokenizer schema."""

    def get_hf_processor(self: Any, **kwargs: object) -> Any:
        kwargs.pop("use_fast", None)
        kwargs.setdefault("backend", "pil")
        return self.ctx.get_hf_processor(_HunYuanVLProcessorCompat, **kwargs)

    hunyuan_vision.HunYuanVLProcessingInfo.get_hf_processor = get_hf_processor


def _patch_image_token_wrapping(hunyuan_vision: Any) -> None:
    """Wrap bare image tokens with start/end tokens for HunYuanVLProcessor.

    Transformers' ``HunYuanVLProcessor.validate_inputs`` requires every image
    placeholder to be surrounded by the image start/end tokens, i.e.
    ``<no_100><no_102><no_101>``. vLLM main folds this wrapping into
    ``hunyuan_vision._call_hf_processor`` natively, but vLLM v0.25.0 does not,
    so prompts built with a bare image token (e.g. the dummy mm batch during
    ``profile_run``) are rejected by ``validate_inputs``. Backport the wrapping
    on v0.25.0 only; the ``wrapped not in prompt`` guard keeps it idempotent.
    """

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
    """Align both supported vLLM refs with Transformers 5.13 Hunyuan APIs."""
    if not _remove_stale_registry_entries():
        return
    from vllm.model_executor.models import hunyuan_vision as main_hunyuan_vision

    _patch_hunyuan_processor_loader(main_hunyuan_vision)
    # vLLM v0.25.0 does not wrap bare image tokens with start/end tokens inside
    # ``_call_hf_processor`` (that landed on main only), so backport it here.
    # Main already does this natively, so the patch is v0.25.0-only.
    if vllm_version_is("0.25.0"):
        _patch_image_token_wrapping(main_hunyuan_vision)
