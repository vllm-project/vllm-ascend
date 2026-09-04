# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""vLLM v0.27 compatibility for DeepSeek-V4 vision."""

import inspect
from collections import Counter
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig

_REGISTERED = False


def _is_deepseek_v4_vision_model(model_config: object) -> bool:
    """Return whether a model config belongs to the DeepSeek-V4 vision path."""
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "deepseek_v4" and getattr(hf_config, "vision_n_layers", 0) > 0


def _make_multimodal_parser_patch(original):
    signature = inspect.signature(original)

    @wraps(original)
    def patched(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        tracker = bound.arguments.get("mm_tracker")
        if tracker is not None and _is_deepseek_v4_vision_model(tracker.model_config):
            # The upstream encoding keeps image blocks in their input order and
            # joins every content block with two newlines.
            bound.arguments["interleave_strings"] = True
            if "multimodal_content_part_separator" in signature.parameters:
                bound.arguments["multimodal_content_part_separator"] = "\n\n"
        return original(*bound.args, **bound.kwargs)

    patched.__dict__["_deepseek_v4_vision_parser_patch"] = True
    return patched


def _make_multimodal_prompt_patch(original):
    signature = inspect.signature(original)

    @wraps(original)
    def patched(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not bound.arguments["interleave_strings"]:
            return original(*bound.args, **bound.kwargs)

        separator = bound.arguments.get("multimodal_content_part_separator", "\n")
        if separator != "\n\n":
            return original(*bound.args, **bound.kwargs)

        # Let the v0.27 helper validate counts and consume interleaved
        # placeholders, then rebuild from the original content-part strings.
        # This keeps newlines inside a text part intact.
        placeholder_counts = Counter(
            placeholder
            for placeholders in bound.arguments["placeholder_storage"].values()
            for placeholder in placeholders
        )
        bound.arguments["multimodal_content_part_separator"] = "\n"
        original(*bound.args, **bound.kwargs)
        texts = bound.arguments["texts"]
        missing_placeholders = []
        for placeholder, count in placeholder_counts.items():
            remaining = count - sum(text.count(placeholder) for text in texts)
            missing_placeholders.extend([placeholder] * remaining)
        return separator.join(missing_placeholders + list(texts))

    patched.__dict__["_deepseek_v4_vision_prompt_patch"] = True
    return patched


def _patch_deepseek_v4_multimodal_parser() -> None:
    """Preserve DeepSeek-V4 image block order in the v0.27 string parser."""
    try:
        from vllm.entrypoints import chat_utils

        original = chat_utils._parse_chat_message_content_parts
        original_prompt = chat_utils._get_full_multimodal_text_prompt
    except (AttributeError, ImportError, TypeError):
        return

    if not getattr(original_prompt, "_deepseek_v4_vision_prompt_patch", False):
        chat_utils._get_full_multimodal_text_prompt = _make_multimodal_prompt_patch(original_prompt)
    if getattr(original, "_deepseek_v4_vision_parser_patch", False):
        return
    chat_utils._parse_chat_message_content_parts = _make_multimodal_parser_patch(original)


def register_deepseek_v4_vision_config_convertor() -> None:
    """Route vision checkpoints to the Ascend multimodal wrapper.

    Keep vLLM config imports out of module scope. Global patches are imported
    while spawned engine processes unpickle their state, which can happen while
    ``model_arch_config_convertor`` itself is only partially initialized.
    """
    global _REGISTERED
    _patch_deepseek_v4_multimodal_parser()
    if _REGISTERED:
        return

    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )

    from vllm_ascend.utils import vllm_version_is

    class AscendDeepseekV4ModelArchConfigConvertor(ModelArchConfigConvertorBase):
        """Route vision checkpoints to the Ascend multimodal wrapper."""

        def __init__(
            self,
            hf_config: "PretrainedConfig",
            hf_text_config: "PretrainedConfig",
            revision: str | None = None,
        ) -> None:
            if getattr(hf_config, "vision_n_layers", 0) > 0:
                hf_config.architectures = ["DeepseekV4ForConditionalGeneration"]
                hf_config.mm_prefix_clamp_sliding_window = True
                hf_config.mm_prefix_span_leading_pad_modulus = 4
            if vllm_version_is("0.27.1"):
                super().__init__(hf_config, hf_text_config)
            else:
                super().__init__(hf_config, hf_text_config, revision)

        def is_mm_prefix_lm(self, supports_multimodal: bool = True) -> bool:
            return supports_multimodal and (getattr(self.hf_config, "vision_n_layers", 0) > 0)

    MODEL_ARCH_CONFIG_CONVERTORS["deepseek_v4"] = AscendDeepseekV4ModelArchConfigConvertor
    _REGISTERED = True


_patch_deepseek_v4_multimodal_parser()
