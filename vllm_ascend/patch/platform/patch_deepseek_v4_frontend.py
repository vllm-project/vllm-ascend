# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from typing import Any

from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.tokenizers import deepseek_v4

_original_get_deepseek_v4_tokenizer = deepseek_v4.get_deepseek_v4_tokenizer
_original_deepseek_v4_parser_init = DeepSeekV4Parser.__init__


@wraps(_original_get_deepseek_v4_tokenizer)
def _patched_get_deepseek_v4_tokenizer(tokenizer: deepseek_v4.HfTokenizer):
    dsv4_tokenizer = _original_get_deepseek_v4_tokenizer(tokenizer)
    tokenizer_cls = type(dsv4_tokenizer)
    original_apply_chat_template = tokenizer_cls.apply_chat_template

    @wraps(original_apply_chat_template)
    def apply_chat_template(
        self,
        messages,
        tools=None,
        **kwargs,
    ):
        if tools:
            conversation = kwargs.get("conversation", messages).copy()
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
            kwargs["conversation"] = conversation
            tools = None
        return original_apply_chat_template(self, messages, tools=tools, **kwargs)

    tokenizer_cls.apply_chat_template = apply_chat_template
    return dsv4_tokenizer


@wraps(_original_deepseek_v4_parser_init)
def _patched_deepseek_v4_parser_init(
    self: DeepSeekV4Parser,
    *args: Any,
    **kwargs: Any,
) -> None:
    chat_kwargs = kwargs.get("chat_template_kwargs") or {}
    if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
        chat_kwargs = dict(chat_kwargs)
        chat_kwargs["enable_thinking"] = True
        kwargs["chat_template_kwargs"] = chat_kwargs

    _original_deepseek_v4_parser_init(self, *args, **kwargs)


deepseek_v4.get_deepseek_v4_tokenizer = _patched_get_deepseek_v4_tokenizer
DeepSeekV4Parser.__init__ = _patched_deepseek_v4_parser_init
