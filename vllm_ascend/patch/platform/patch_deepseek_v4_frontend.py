# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from types import MethodType
from typing import Any

from vllm.tokenizers import deepseek_v4

_original_get_deepseek_v4_tokenizer = deepseek_v4.get_deepseek_v4_tokenizer


def _attach_tools(
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages = list(conversation)
    system_index = next(
        (index for index, message in enumerate(messages) if message.get("role") == "system"),
        None,
    )
    if system_index is None:
        messages.insert(0, {"role": "system", "tools": tools})
    else:
        system_message = messages[system_index].copy()
        system_message["tools"] = tools
        messages[system_index] = system_message
    return messages


def _patched_get_deepseek_v4_tokenizer(tokenizer: deepseek_v4.HfTokenizer):
    dsv4_tokenizer = _original_get_deepseek_v4_tokenizer(tokenizer)
    original_apply = type(dsv4_tokenizer).apply_chat_template

    @wraps(original_apply)
    def apply_chat_template(
        self: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str | list[int]:
        if tools:
            conversation = kwargs.get("conversation", messages)
            kwargs["conversation"] = _attach_tools(conversation, tools)
            tools = None
        return original_apply(self, messages, tools=tools, **kwargs)

    dsv4_tokenizer.apply_chat_template = MethodType(
        apply_chat_template,
        dsv4_tokenizer,
    )
    return dsv4_tokenizer


deepseek_v4.get_deepseek_v4_tokenizer = _patched_get_deepseek_v4_tokenizer
