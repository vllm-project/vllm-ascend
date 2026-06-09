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
#
# DeepSeek V4 thinking compatibility with newer vLLM request/tokenizer behavior.
#

from __future__ import annotations

import copy
from typing import Any, Literal

from transformers import PreTrainedTokenizerFast
from vllm.entrypoints.openai.chat_completion import protocol as chat_protocol
from vllm.renderers.params import ChatParams
from vllm.tokenizers import deepseek_v4 as deepseek_v4_tokenizer

DeepSeekV4ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None


def _rebuild_model_field(model_cls, field_name: str, annotation) -> None:
    model_cls.__annotations__[field_name] = annotation
    model_cls.model_fields[field_name].annotation = annotation
    model_cls.model_rebuild(force=True)


_rebuild_model_field(
    chat_protocol.ChatCompletionRequest,
    "reasoning_effort",
    DeepSeekV4ReasoningEffort,
)

_original_build_chat_params = chat_protocol.ChatCompletionRequest.build_chat_params


def _patched_build_chat_params(
    self: chat_protocol.ChatCompletionRequest,
    default_template: str | None,
    default_template_content_format,
) -> ChatParams:
    params = _original_build_chat_params(
        self,
        default_template,
        default_template_content_format,
    )
    user_kwargs = self.chat_template_kwargs or {}
    if self.reasoning_effort is None or "enable_thinking" in user_kwargs:
        return params

    chat_template_kwargs = dict(params.chat_template_kwargs)
    chat_template_kwargs["enable_thinking"] = self.reasoning_effort != "none"
    return ChatParams(
        chat_template=params.chat_template,
        chat_template_content_format=params.chat_template_content_format,
        chat_template_kwargs=chat_template_kwargs,
        media_io_kwargs=params.media_io_kwargs,
        mm_processor_kwargs=params.mm_processor_kwargs,
    )


chat_protocol.ChatCompletionRequest.build_chat_params = _patched_build_chat_params


def _patched_get_deepseek_v4_tokenizer(tokenizer: deepseek_v4_tokenizer.HfTokenizer):
    dsv4_tokenizer = copy.copy(tokenizer)

    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[chat_protocol.ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            thinking = kwargs.get("thinking", False)
            enable_thinking = kwargs.get("enable_thinking", False)
            thinking = thinking or enable_thinking
            thinking_mode = "thinking" if thinking else "chat"

            conversation = kwargs.get("conversation", messages)
            messages = conversation.copy()
            if tools is not None and len(tools) > 0:
                messages.insert(0, {"role": "system"})
                messages[0]["tools"] = tools  # type: ignore[typeddict-unknown-key]

            reasoning_effort = kwargs.get("reasoning_effort")
            if not isinstance(reasoning_effort, str):
                reasoning_effort = None
            elif reasoning_effort == "none":
                thinking_mode = "chat"
                reasoning_effort = None
            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"

            prompt_str = deepseek_v4_tokenizer.encode_messages(
                messages,
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {k: kwargs[k] for k in ("truncation", "max_length") if k in kwargs}
                return self.encode(
                    prompt_str,
                    add_special_tokens=False,
                    **tokenizer_kwargs,
                )

            return prompt_str

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return _patched_get_deepseek_v4_tokenizer, (tokenizer,)

    _DeepseekV4Tokenizer.__name__ = f"DSV4{tokenizer.__class__.__name__}"

    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer
    return dsv4_tokenizer


def _patched_deepseek_v4_from_pretrained(cls, *args, **kwargs):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)
    return deepseek_v4_tokenizer.get_cached_tokenizer(_patched_get_deepseek_v4_tokenizer(tokenizer))


deepseek_v4_tokenizer.get_deepseek_v4_tokenizer = _patched_get_deepseek_v4_tokenizer
deepseek_v4_tokenizer.DeepseekV4Tokenizer.from_pretrained = classmethod(_patched_deepseek_v4_from_pretrained)
