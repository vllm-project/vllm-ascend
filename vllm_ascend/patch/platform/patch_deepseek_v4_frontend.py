# SPDX-License-Identifier: Apache-2.0

import inspect
from functools import wraps
from types import MethodType
from typing import Any

from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.tokenizers import deepseek_v4, deepseek_v4_encoding
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

REASONING_EFFORT_PROMPTS = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively "
        "decompose the problem to resolve the root cause, rigorously "
        "stress-testing your logic against all potential paths, edge cases, "
        "and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting "
        "every intermediate step, considered alternative, and rejected "
        "hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and "
        "uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely "
        "nothing to chance: exhaustively decompose the problem into its most "
        "fundamental components, trace every causal chain to its root, and "
        "resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the "
        "solution from multiple angles and are certain that no assumption "
        "remains unchecked and no error remains undiscovered.\n\n"
    ),
}

_original_get_deepseek_v4_tokenizer = deepseek_v4.get_deepseek_v4_tokenizer
_original_parser_init = DeepSeekV4Parser.__init__
_original_parser_init_signature = inspect.signature(_original_parser_init)


def _uses_preview_reasoning_effort_mapping(tokenizer: Any) -> bool:
    """The pre-0731 checkpoint has no ``dspark_*`` config fields."""
    model_name_or_path = getattr(tokenizer, "name_or_path", None)
    if not model_name_or_path:
        return True
    try:
        config = get_hf_file_to_dict("config.json", model_name_or_path)
    except Exception:
        return True
    return not (config and any(key.startswith("dspark_") for key in config))


def _needs_legacy_renderer_patch() -> bool:
    """v0.27.1 exposes only ``REASONING_EFFORT_MAX``."""
    return not hasattr(deepseek_v4_encoding, "REASONING_EFFORT_PROMPTS")


def _needs_legacy_parser_patch() -> bool:
    """Detect the old parser default without keying the patch to a version."""

    class _ProbeTokenizer:
        def get_vocab(self) -> dict[str, int]:
            return {}

    parser = DeepSeekV4Parser(
        _ProbeTokenizer(),
        chat_template_kwargs={},
    )
    return parser.parser_engine_config.initial_state.name != "REASONING"


_USES_LEGACY_RENDERER = _needs_legacy_renderer_patch()
_USES_LEGACY_PARSER = _needs_legacy_parser_patch()


def _patched_render_message(
    index: int,
    messages: list[dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    reasoning_effort = reasoning_effort or "low"
    if reasoning_effort not in REASONING_EFFORT_PROMPTS:
        raise ValueError(
            f"Invalid reasoning effort: {reasoning_effort}, expected one of {list(REASONING_EFFORT_PROMPTS)}"
        )
    prompt = _original_render_message(
        index,
        messages,
        thinking_mode,
        drop_thinking,
        reasoning_effort="high",
    )
    if index == 0 and thinking_mode == "thinking":
        return REASONING_EFFORT_PROMPTS[reasoning_effort] + prompt
    return prompt


def _normalize_reasoning_effort(
    reasoning_effort: Any,
    thinking_enabled: bool,
    uses_preview_mapping: bool,
) -> tuple[str, str | None]:
    thinking_mode = "thinking" if thinking_enabled else "chat"
    if not isinstance(reasoning_effort, str):
        canonical = "low" if uses_preview_mapping else "high"
        return thinking_mode, canonical if thinking_enabled else None
    if reasoning_effort == "none":
        return "chat", None
    if uses_preview_mapping:
        canonical = "high" if reasoning_effort in ("max", "xhigh") else "low"
    elif reasoning_effort == "max":
        canonical = "max"
    elif reasoning_effort in ("low", "minimal", "medium"):
        canonical = "low"
    else:
        canonical = "high"
    return thinking_mode, canonical


def _attach_tools(messages: Any, tools: list[dict[str, Any]] | None) -> Any:
    if not tools:
        return messages
    conversation = list(messages)
    system_index = next(
        (index for index, message in enumerate(conversation) if message.get("role") == "system"),
        None,
    )
    if system_index is None:
        conversation.insert(0, {"role": "system", "tools": tools})
    else:
        system_message = conversation[system_index].copy()
        system_message["tools"] = tools
        conversation[system_index] = system_message
    return conversation


def _patched_get_deepseek_v4_tokenizer(tokenizer: deepseek_v4.HfTokenizer):
    dsv4_tokenizer = _original_get_deepseek_v4_tokenizer(tokenizer)
    uses_preview_mapping = _uses_preview_reasoning_effort_mapping(tokenizer)
    original_apply = type(dsv4_tokenizer).apply_chat_template

    @wraps(original_apply)
    def apply_chat_template(
        self: Any,
        messages: Any,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str | list[int]:
        conversation = kwargs.get("conversation", messages)
        adjusted_messages = _attach_tools(conversation, tools)
        thinking = kwargs.get("thinking")
        enable_thinking = kwargs.get("enable_thinking")
        thinking_enabled = bool(thinking) or bool(enable_thinking)
        if "thinking" not in kwargs and "enable_thinking" not in kwargs:
            thinking_enabled = True

        thinking_mode, canonical_effort = _normalize_reasoning_effort(
            kwargs.get("reasoning_effort"),
            thinking_enabled,
            uses_preview_mapping,
        )
        prompt = deepseek_v4.encode_messages(
            adjusted_messages,
            thinking_mode=thinking_mode,
            drop_thinking=kwargs.get("drop_thinking", True),
            reasoning_effort=canonical_effort,
        )
        if kwargs.get("tokenize", True):
            tokenizer_kwargs = {key: kwargs[key] for key in ("truncation", "max_length") if key in kwargs}
            return self.encode(
                prompt,
                add_special_tokens=False,
                **tokenizer_kwargs,
            )
        return prompt

    dsv4_tokenizer.apply_chat_template = MethodType(
        apply_chat_template,
        dsv4_tokenizer,
    )
    return dsv4_tokenizer


@wraps(_original_parser_init)
def _patched_parser_init(self: DeepSeekV4Parser, *args: Any, **kwargs: Any) -> None:
    bound = _original_parser_init_signature.bind(self, *args, **kwargs)
    extra_kwargs = bound.arguments.get("kwargs", {})
    chat_kwargs = extra_kwargs.get("chat_template_kwargs") or {}
    if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
        extra_kwargs["chat_template_kwargs"] = {
            **chat_kwargs,
            "enable_thinking": True,
        }
    _original_parser_init(*bound.args, **bound.kwargs)


if _USES_LEGACY_RENDERER:
    _original_render_message = deepseek_v4_encoding.render_message
    deepseek_v4_encoding.REASONING_EFFORT_PROMPTS = REASONING_EFFORT_PROMPTS
    deepseek_v4_encoding.render_message = _patched_render_message

deepseek_v4.get_deepseek_v4_tokenizer = _patched_get_deepseek_v4_tokenizer
if _USES_LEGACY_PARSER:
    DeepSeekV4Parser.__init__ = _patched_parser_init
