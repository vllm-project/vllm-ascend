# SPDX-License-Identifier: Apache-2.0
"""OpenAI request adapter for the checkpoint's standalone encoder."""

import copy

from transformers import TokenizersBackend
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm.tokenizers.protocol import TokenizerLike

from .encoding import encode_messages, render_reasoning_effort


def thinking_enabled(kwargs):
    if kwargs.get("reasoning_effort") == "none":
        return False
    if "thinking" not in kwargs and "enable_thinking" not in kwargs:
        return True
    return bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))


def get_deepseek_v41_tokenizer(tokenizer):
    wrapped = copy.copy(tokenizer)

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore[name-defined]
        def apply_chat_template(self, messages, tools=None, **kwargs):
            # Keep original content blocks: vLLM's flattened conversation loses
            # reference separators, image positions and reasoning_content.
            messages = copy.deepcopy(messages)
            for message in messages:
                if "reasoning_content" not in message and "reasoning" in message:
                    message["reasoning_content"] = message["reasoning"]

            response_format = kwargs.get("response_format")
            if response_format is not None and hasattr(response_format, "model_dump"):
                response_format = response_format.model_dump(by_alias=True)
            schema = None
            if response_format and response_format.get("type") == "json_schema":
                schema = response_format["json_schema"]["schema"]
            if tools or schema is not None:
                if not messages or messages[0]["role"] != "system":
                    messages.insert(0, {"role": "system", "content": ""})
                if tools:
                    # OpenAI request validation inserts absent optional fields
                    # as None; those are not part of the reference tool schema.
                    messages[0]["tools"] = [
                        {
                            **tool,
                            "function": {key: value for key, value in tool["function"].items() if value is not None},
                        }
                        for tool in tools
                    ]
                if schema is not None:
                    messages[0]["response_format"] = schema

            effort = kwargs.get("reasoning_effort")
            effort = None if effort == "none" else effort
            try:
                render_reasoning_effort(0, "chat", effort)
            except (AssertionError, TypeError) as error:
                raise ValueError(str(error)) from error
            prompt = encode_messages(
                messages,
                thinking_mode="thinking" if thinking_enabled(kwargs) else "chat",
                reasoning_effort=effort,
                drop_thinking=kwargs.get("drop_thinking", True),
                context=kwargs.get("context"),
                add_default_bos_token=kwargs.get("add_default_bos_token", True),
            )
            if not kwargs.get("tokenize", True):
                return prompt
            return self.encode(
                prompt,
                add_special_tokens=False,
                **{key: kwargs[key] for key in ("truncation", "max_length") if key in kwargs},
            )

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer,)

    wrapped.__class__ = _DeepseekV41Tokenizer
    return wrapped


class DeepseekV41Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
