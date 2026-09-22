# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Register DeepSeek V4.1 frontend fallbacks missing from vLLM v0.29."""

import importlib.util

_REGISTERED = False


_NATIVE_FRONTEND_MODULES = (
    "vllm.transformers_utils.configs.deepseek_v41",
    "vllm.tokenizers.deepseek_v41",
    "vllm.parser.deepseek_v41",
)


def _has_native_frontend() -> bool:
    """Only skip the fallback when the complete upstream frontend is present."""
    return all(importlib.util.find_spec(module) is not None for module in _NATIVE_FRONTEND_MODULES)


def _register_config() -> None:
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    from .config import DeepseekV41Config

    _CONFIG_REGISTRY.setdefault("deepseek_v41", DeepseekV41Config)


def _register_tokenizer_and_renderer() -> None:
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    if "deepseek_v41" not in TokenizerRegistry.tokenizers:
        TokenizerRegistry.register(
            "deepseek_v41",
            "vllm_ascend.compat.deepseek_v41.tokenizer",
            "DeepseekV41Tokenizer",
        )
    if "deepseek_v41" not in RENDERER_REGISTRY.renderers:
        RENDERER_REGISTRY.register(
            "deepseek_v41",
            "vllm.renderers.deepseek_v4",
            "DeepseekV4Renderer",
        )


def _register_parsers() -> None:
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    module = "vllm_ascend.compat.deepseek_v41.adapters"
    if "deepseek_v41" not in ReasoningParserManager.list_registered():
        ReasoningParserManager.register_lazy_module("deepseek_v41", module, "DeepSeekV41ParserReasoningAdapter")
    if "deepseek_v41" not in ToolParserManager.list_registered():
        ToolParserManager.register_lazy_module("deepseek_v41", module, "DeepSeekV41EngineToolParser")


def _register_structural_tag() -> None:
    import vllm.tool_parsers.structural_tag_registry as registry

    if "deepseek_v41" not in registry._VLLM_STRUCTURAL_TAG_REGISTRY:
        from . import structural_tag  # noqa: F401

    registry.VLLM_BUILTIN_STRUCTURAL_TAG_MODELS = registry.VLLM_BUILTIN_STRUCTURAL_TAG_MODELS | {"deepseek_v41"}
    registry.SUPPORTED_STRUCTURAL_TAG_MODELS = (
        registry.XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS | registry.VLLM_BUILTIN_STRUCTURAL_TAG_MODELS
    )


def _patch_model_config() -> None:
    from vllm.config.model import ModelConfig

    original = ModelConfig.__post_init__
    if getattr(original, "_vllm_ascend_deepseek_v41", False):
        return

    def __post_init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.tokenizer_mode == "auto" and self._architecture == "DeepseekV41ForCausalLM":
            self.tokenizer_mode = "deepseek_v41"

    __post_init__._vllm_ascend_deepseek_v41 = True  # type: ignore[attr-defined]
    ModelConfig.__post_init__ = __post_init__


def _patch_model_arch_config() -> None:
    from vllm.transformers_utils.model_arch_config_convertor import (
        ModelArchConfigConvertorBase,
    )

    original = ModelArchConfigConvertorBase.is_deepseek_mla
    if getattr(original, "_vllm_ascend_deepseek_v41", False):
        return

    def is_deepseek_mla(self) -> bool:
        model_type = getattr(self.hf_text_config, "model_type", None)
        if model_type in ("deepseek_v41", "deepseek_v41_text"):
            if hasattr(self.hf_text_config, "compress_ratios"):
                return getattr(self.hf_text_config, "head_dim", None) is not None
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None
        return original(self)

    is_deepseek_mla._vllm_ascend_deepseek_v41 = True  # type: ignore[attr-defined]
    ModelArchConfigConvertorBase.is_deepseek_mla = is_deepseek_mla


def register_deepseek_v41_compat() -> None:
    """Install only the frontend pieces absent from the active vLLM."""
    global _REGISTERED
    if _REGISTERED or _has_native_frontend():
        return

    _register_config()
    _register_tokenizer_and_renderer()
    _register_parsers()
    _register_structural_tag()
    _patch_model_config()
    _patch_model_arch_config()
    _REGISTERED = True


__all__ = ["register_deepseek_v41_compat"]
