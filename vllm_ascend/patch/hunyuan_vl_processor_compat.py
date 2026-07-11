# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from functools import partial
from types import ModuleType
from typing import Any, cast

from vllm_ascend.utils import vllm_version_is

_STALE_PROCESSOR_MODULES = {
    "HunYuanVLProcessor": "vllm.transformers_utils.processors.hunyuan_vl",
    "HunYuanVLImageProcessor": "vllm.transformers_utils.processors.hunyuan_vl_image",
}


def _import_v023_hunyuan_vision() -> Any:
    """Import v0.23's model with the native processors from vLLM PR #47872."""
    from transformers import HunYuanVLProcessor
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


def _patch_v023_processor_methods(hunyuan_vision: Any) -> None:
    """Backport the Transformers 5.13 call protocol from vLLM PR #47872."""
    from transformers import HunYuanVLProcessor

    def get_hf_processor(self: Any, **kwargs: object) -> Any:
        kwargs.pop("use_fast", None)
        kwargs.setdefault("backend", "pil")
        return self.ctx.get_hf_processor(HunYuanVLProcessor, **kwargs)

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

    hunyuan_vision.HunYuanVLProcessingInfo.get_hf_processor = get_hf_processor
    hunyuan_vision.HunYuanVLMultiModalProcessor._call_hf_processor = call_hf_processor


def _patch_prompt_updates(hunyuan_vision: Any) -> None:
    """Backport the complete Hunyuan image placeholder from vLLM PR #47867."""
    import torch
    from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails

    def get_prompt_updates(
        self: Any,
        mm_items: Any,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: Any,
    ) -> list[Any]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        token_ids = {
            "image": hf_processor.image_token_id,
            "image_start": hf_processor.image_start_token_id,
            "image_end": hf_processor.image_end_token_id,
        }
        merge_size = image_processor.merge_size

        def get_replacement_hunyuan_vl(item_idx: int, modality: str) -> Any:
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            _, grid_h, grid_w = grid_thw
            num_tokens = (int(grid_h) // merge_size) * (int(grid_w) // merge_size + 1) + 2
            tokens = (
                [token_ids[f"{modality}_start"]] + [token_ids[modality]] * num_tokens + [token_ids[f"{modality}_end"]]
            )
            return PromptUpdateDetails.select_token_id(tokens, token_ids[modality])

        return [
            PromptReplacement(
                modality=modality,
                target=[token_ids[modality]],
                replacement=partial(get_replacement_hunyuan_vl, modality=modality),
            )
            for modality in ("image",)
        ]

    hunyuan_vision.HunYuanVLMultiModalProcessor._get_prompt_updates = get_prompt_updates


def install_hunyuan_vl_processor_compat() -> None:
    """Align both supported vLLM refs with Transformers 5.13 Hunyuan APIs."""
    if vllm_version_is("0.23.0"):
        v023_hunyuan_vision = _import_v023_hunyuan_vision()
        _remove_stale_registry_entries()
        _patch_v023_processor_methods(v023_hunyuan_vision)
        _patch_prompt_updates(v023_hunyuan_vision)
        return

    if not _remove_stale_registry_entries():
        return
    from vllm.model_executor.models import hunyuan_vision as main_hunyuan_vision

    _patch_prompt_updates(main_hunyuan_vision)
