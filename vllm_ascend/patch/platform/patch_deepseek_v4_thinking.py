#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
from typing import Any

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
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
        "Reasoning Effort: Beyond maximum \u2014 exhaustive, relentless, and "
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
DEFAULT_REASONING_EFFORT = "low"

_original_render_message = deepseek_v4_encoding.render_message
_original_get_deepseek_v4_tokenizer = deepseek_v4.get_deepseek_v4_tokenizer
_original_deepseek_v4_parser_init = DeepSeekV4Parser.__init__


def _uses_preview_reasoning_effort_mapping(tokenizer: deepseek_v4.HfTokenizer) -> bool:
    model_name_or_path = getattr(tokenizer, "name_or_path", None)
    if not model_name_or_path:
        return True

    config = get_hf_file_to_dict("config.json", model_name_or_path)
    return not (config and any(key.startswith("dspark_") for key in config))


def _patched_render_message(
    index: int,
    messages: list[dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
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


def _patched_get_deepseek_v4_tokenizer(tokenizer: deepseek_v4.HfTokenizer):
    uses_preview_mapping = _uses_preview_reasoning_effort_mapping(tokenizer)
    dsv4_tokenizer = _original_get_deepseek_v4_tokenizer(tokenizer)
    tokenizer_cls = type(dsv4_tokenizer)

    def apply_chat_template(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str | list[int]:
        thinking = kwargs.get("thinking")
        enable_thinking = kwargs.get("enable_thinking")
        thinking_enabled = bool(thinking) or bool(enable_thinking)
        if "thinking" not in kwargs and "enable_thinking" not in kwargs:
            thinking_enabled = True
        thinking_mode = "thinking" if thinking_enabled else "chat"

        conversation = kwargs.get("conversation", messages)
        messages = conversation.copy()
        if tools is not None and len(tools) > 0:
            system_index = next(
                (index for index, message in enumerate(messages) if message.get("role") == "system"),
                None,
            )
            if system_index is None:
                messages.insert(0, {"role": "system", "tools": tools})
            else:
                system_message = messages[system_index].copy()
                system_message["tools"] = tools  # type: ignore[typeddict-unknown-key]
                messages[system_index] = system_message

        reasoning_effort = kwargs.get("reasoning_effort")
        if not isinstance(reasoning_effort, str):
            if thinking_enabled:
                reasoning_effort = "low" if uses_preview_mapping else "high"
            else:
                reasoning_effort = None
        elif reasoning_effort == "none":
            thinking_mode = "chat"
            reasoning_effort = None
        elif not uses_preview_mapping:
            if reasoning_effort == "max":
                reasoning_effort = "max"
            elif reasoning_effort in ("low", "minimal", "medium"):
                reasoning_effort = "low"
            else:
                reasoning_effort = "high"
        elif reasoning_effort in ("max", "xhigh"):
            reasoning_effort = "high"
        else:
            reasoning_effort = "low"

        prompt_str = deepseek_v4.encode_messages(
            messages,
            thinking_mode=thinking_mode,
            drop_thinking=kwargs.get("drop_thinking", True),
            reasoning_effort=reasoning_effort,
        )

        if kwargs.get("tokenize", True):
            tokenizer_kwargs = {key: kwargs[key] for key in ("truncation", "max_length") if key in kwargs}
            return self.encode(
                prompt_str,
                add_special_tokens=False,
                **tokenizer_kwargs,
            )

        return prompt_str

    tokenizer_cls.apply_chat_template = apply_chat_template
    return dsv4_tokenizer


@wraps(_original_deepseek_v4_parser_init)
def _patched_deepseek_v4_parser_init(
    self: DeepSeekV4Parser,
    tokenizer: Any,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> None:
    chat_kwargs = kwargs.get("chat_template_kwargs") or {}
    if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
        chat_kwargs = dict(chat_kwargs)
        chat_kwargs["enable_thinking"] = True
        kwargs["chat_template_kwargs"] = chat_kwargs

    _original_deepseek_v4_parser_init(self, tokenizer, tools, **kwargs)


if not hasattr(deepseek_v4_encoding, "REASONING_EFFORT_PROMPTS"):
    deepseek_v4_encoding.REASONING_EFFORT_PROMPTS = REASONING_EFFORT_PROMPTS
    deepseek_v4_encoding.render_message = _patched_render_message
    deepseek_v4.get_deepseek_v4_tokenizer = _patched_get_deepseek_v4_tokenizer
    DeepSeekV4Parser.__init__ = _patched_deepseek_v4_parser_init
