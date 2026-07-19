# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any, cast

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


def _import_v024_hunyuan_vision() -> Any:
    """Import a bundled release model with native processors from vLLM PR #47872."""
    from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import (
        HunYuanVLImageProcessor,
        smart_resize,
    )

    aliases = {
        _STALE_PROCESSOR_MODULES["HunYuanVLProcessor"]: {
            "HunYuanVLProcessor": HunYuanVLProcessor,
        },
        _STALE_PROCESSOR_MODULES["HunYuanVLImageProcessor"]: {
            "HunYuanVLImageProcessor": HunYuanVLImageProcessor,
            "smart_resize": smart_resize,
        },
    }
    missing = object()
    previous_modules = {name: sys.modules.get(name, missing) for name in aliases}
    parent_module = sys.modules.get("vllm.transformers_utils.processors")
    previous_attributes = {
        name.rpartition(".")[2]: (
            vars(parent_module).get(name.rpartition(".")[2], missing) if parent_module is not None else missing
        )
        for name in aliases
    }

    alias_modules = {}
    for module_name, exports in aliases.items():
        module = ModuleType(module_name)
        module.__package__ = module_name.rpartition(".")[0]
        module.__dict__.update(exports)
        module.__dict__["__all__"] = list(exports)
        alias_modules[module_name] = module
        sys.modules[module_name] = module

    try:
        return importlib.import_module("vllm.model_executor.models.hunyuan_vision")
    finally:
        for module_name, alias_module in alias_modules.items():
            previous_module = previous_modules[module_name]
            if previous_module is missing:
                if sys.modules.get(module_name) is alias_module:
                    del sys.modules[module_name]
            else:
                sys.modules[module_name] = cast(ModuleType, previous_module)

        current_parent_module = sys.modules.get("vllm.transformers_utils.processors")
        if current_parent_module is not None:
            for attribute_name, previous_attribute in previous_attributes.items():
                if previous_attribute is missing:
                    if vars(current_parent_module).get(attribute_name) in alias_modules.values():
                        delattr(current_parent_module, attribute_name)
                else:
                    setattr(current_parent_module, attribute_name, previous_attribute)


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


def _patch_v024_processor_methods(hunyuan_vision: Any) -> None:
    """Backport the Transformers 5.13 call protocol from vLLM PR #47872."""

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


def _patch_xdrope_input_positions(hunyuan_vision: Any) -> None:
    """Guard IndexError when image_start_indices > image_grid_thw entries."""

    def get_xdrope_input_positions(
        self: Any,
        input_tokens: list[int],
        mm_features: list[Any],
    ) -> Any:
        from vllm.multimodal.inputs import MultiModalFeatureSpec

        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw"},
        )
        image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]

        hf_config = self.config
        image_start_token_id = hf_config.image_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        xd_num = len(hf_config.rope_scaling["xdrope_section"])

        import torch

        input_tokens_tensor = torch.tensor(input_tokens)
        image_start_indices = torch.argwhere(input_tokens_tensor == image_start_token_id).squeeze(1)

        p_index = torch.arange(len(input_tokens_tensor))
        w_index = torch.arange(len(input_tokens_tensor))
        h_index = torch.arange(len(input_tokens_tensor))
        t_index = torch.arange(len(input_tokens_tensor))
        for image_index in range(len(image_start_indices)):
            if image_index >= len(image_grid_thw):
                continue
            pos = image_start_indices[image_index] + 2
            t, h, w = image_grid_thw[image_index]
            _, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )

            token_num = (llm_grid_w + 1) * llm_grid_h
            w_index[pos : pos + token_num].copy_(
                torch.arange(0, llm_grid_w + 1).reshape(1, -1).expand(llm_grid_h, -1).reshape(-1)
            )
            h_index[pos : pos + token_num].copy_(
                torch.arange(0, llm_grid_h).reshape(-1, 1).expand(-1, llm_grid_w + 1).reshape(-1)
            )
            t_index[pos : pos + token_num] = image_index

        if xd_num == 4:
            llm_positions = torch.stack([p_index, w_index, h_index, t_index])
        elif xd_num == 3:
            llm_positions = torch.stack([w_index, h_index, t_index])

        return llm_positions

    hunyuan_vision.HunYuanVLForConditionalGeneration.get_xdrope_input_positions = get_xdrope_input_positions


def install_hunyuan_vl_processor_compat() -> None:
    """Align both supported vLLM refs with Transformers 5.13 Hunyuan APIs."""
    # Keep each target's native, image-token-only prompt replacement. The
    # cached processor path applies it inside an existing start/image/end
    # wrapper; using a full-wrapper replacement here would duplicate wrappers.
    if vllm_version_is("0.24.0"):
        v024_hunyuan_vision = _import_v024_hunyuan_vision()
        _remove_stale_registry_entries()
        _patch_hunyuan_processor_loader(v024_hunyuan_vision)
        _patch_v024_processor_methods(v024_hunyuan_vision)
        return

    _remove_stale_registry_entries()
    from vllm.model_executor.models import hunyuan_vision as main_hunyuan_vision

    _patch_hunyuan_processor_loader(main_hunyuan_vision)
    _patch_xdrope_input_positions(main_hunyuan_vision)
