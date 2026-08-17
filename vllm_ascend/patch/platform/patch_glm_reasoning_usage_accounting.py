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
# GLM reasoning usage: backport chat usage reporting and implicit-start counts.
#

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import Any

from vllm.entrypoints.openai.chat_completion import protocol as chat_protocol
from vllm.entrypoints.openai.chat_completion import serving as chat_serving
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine import protocol as engine_protocol
from vllm.parser.glm47_moe import THINK_END, THINK_START, Glm47MoeParser
from vllm.reasoning.glm47_moe_reasoning_parser import (
    Glm47MoeParserReasoningAdapter,
)

_ORIGINAL_COUNT_REASONING_TOKENS = Glm47MoeParser.count_reasoning_tokens


def _count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
    start_id = self._reasoning_start_token_id
    end_id = self._reasoning_end_token_id
    if start_id is None or end_id is None or start_id in token_ids:
        return _ORIGINAL_COUNT_REASONING_TOKENS(self, token_ids)

    try:
        return token_ids.index(end_id)
    except ValueError:
        return len(token_ids)


Glm47MoeParser.count_reasoning_tokens = _count_reasoning_tokens


class CompletionTokenUsageInfo(engine_protocol.OpenAIBaseModel):
    reasoning_tokens: int | None = None
    audio_tokens: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None


class UsageInfo(engine_protocol.UsageInfo):
    completion_tokens_details: CompletionTokenUsageInfo | None = None


CompletionTokenUsageInfo.__module__ = engine_protocol.__name__
UsageInfo.__module__ = engine_protocol.__name__
engine_protocol.CompletionTokenUsageInfo = CompletionTokenUsageInfo
engine_protocol.UsageInfo = UsageInfo
chat_protocol.UsageInfo = UsageInfo
chat_serving.CompletionTokenUsageInfo = CompletionTokenUsageInfo
chat_serving.UsageInfo = UsageInfo


def _rebuild_model_field(model_cls, field_name: str, annotation) -> None:
    model_cls.__annotations__[field_name] = annotation
    model_cls.model_fields[field_name].annotation = annotation
    model_cls.model_rebuild(force=True)


_rebuild_model_field(chat_protocol.ChatCompletionResponse, "usage", UsageInfo)
_rebuild_model_field(chat_protocol.ChatCompletionStreamResponse, "usage", UsageInfo | None)
_rebuild_model_field(
    engine_protocol.RequestResponseMetadata,
    "final_usage_info",
    UsageInfo | None,
)


@dataclass
class _IncrementalReasoningCounter:
    start_token_id: int | None
    end_token_id: int | None
    enabled: bool
    count: int = 0
    depth: int = 0
    saw_start: bool = False
    implicit_end_seen: bool = False

    def update(self, token_ids: Sequence[int]) -> None:
        if not self.enabled or self.start_token_id is None or self.end_token_id is None:
            return

        for token_id in token_ids:
            if token_id == self.start_token_id:
                if not self.saw_start:
                    self.saw_start = True
                    self.count = 0
                    self.depth = 1
                else:
                    self.depth += 1
                continue

            if token_id == self.end_token_id:
                if self.saw_start:
                    self.depth = max(0, self.depth - 1)
                else:
                    self.implicit_end_seen = True
                continue

            if self.saw_start:
                if self.depth > 0:
                    self.count += 1
            elif not self.implicit_end_seen:
                self.count += 1


@dataclass
class _StreamUsageState:
    counters: list[_IncrementalReasoningCounter]


@dataclass
class _FullUsageState:
    final_res: Any = None


def _thinking_enabled(chat_template_kwargs: dict[str, Any] | None) -> bool:
    chat_kwargs = chat_template_kwargs or {}
    thinking = chat_kwargs.get("thinking", None)
    enable_thinking = chat_kwargs.get("enable_thinking", None)
    if thinking is None and enable_thinking is None:
        return True
    return bool(thinking) or bool(enable_thinking)


def _create_stream_usage_state(
    request,
    tokenizer,
    chat_template_kwargs: dict[str, Any] | None,
) -> _StreamUsageState:
    vocab = tokenizer.get_vocab()
    num_choices = 1 if request.n is None else request.n
    counters = [
        _IncrementalReasoningCounter(
            start_token_id=vocab.get(THINK_START),
            end_token_id=vocab.get(THINK_END),
            enabled=_thinking_enabled(chat_template_kwargs),
        )
        for _ in range(num_choices)
    ]
    return _StreamUsageState(counters=counters)


async def _tracked_stream_results(
    result_generator: AsyncIterator,
    state: _StreamUsageState,
):
    async for res in result_generator:
        for output in res.outputs:
            if 0 <= output.index < len(state.counters):
                state.counters[output.index].update(chat_serving.as_list(output.token_ids))
        yield res


async def _tracked_full_results(
    result_generator: AsyncIterator,
    state: _FullUsageState,
):
    async for res in result_generator:
        state.final_res = res
        yield res


def _reasoning_tokens_for_stream_chunk(
    state: _StreamUsageState,
    chunk: dict[str, Any],
) -> int:
    choices = chunk.get("choices") or []
    if choices:
        choice_index = choices[0].get("index", 0)
        if 0 <= choice_index < len(state.counters):
            return state.counters[choice_index].count
    return sum(counter.count for counter in state.counters)


def _inject_stream_usage_details(data: str, state: _StreamUsageState) -> str:
    prefix = "data: "
    suffix = "\n\n"
    if not data.startswith(prefix):
        return data

    payload = data[len(prefix) :]
    if payload.endswith(suffix):
        payload = payload[: -len(suffix)]
    if payload == "[DONE]":
        return data

    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return data

    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return data

    completion_details = usage.setdefault("completion_tokens_details", {})
    completion_details["reasoning_tokens"] = _reasoning_tokens_for_stream_chunk(state, chunk)
    return f"{prefix}{json.dumps(chunk, ensure_ascii=False)}{suffix}"


def _set_usage_details(usage, reasoning_tokens: int) -> None:
    if usage is not None:
        usage.completion_tokens_details = CompletionTokenUsageInfo(reasoning_tokens=reasoning_tokens)


def _count_full_response_reasoning_tokens(
    parser,
    final_res,
) -> int:
    if parser is None or final_res is None:
        return 0
    reasoning_parser = parser.reasoning_parser
    if reasoning_parser is None:
        return 0
    return sum(
        reasoning_parser.count_reasoning_tokens(chat_serving.as_list(output.token_ids)) for output in final_res.outputs
    )


async def _wrapped_chat_completion_stream_generator(
    self,
    request,
    result_generator,
    request_id,
    model_name,
    conversation,
    tokenizer,
    request_metadata,
    chat_template_kwargs=None,
    mm_token_counts=None,
    **extra_kwargs,
):
    state = _create_stream_usage_state(
        request,
        tokenizer,
        chat_template_kwargs,
    )
    original = self._ascend_original_chat_completion_stream_generator
    async for data in original(
        request,
        _tracked_stream_results(result_generator, state),
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        chat_template_kwargs=chat_template_kwargs,
        mm_token_counts=mm_token_counts,
        **extra_kwargs,
    ):
        yield _inject_stream_usage_details(data, state)

    _set_usage_details(
        request_metadata.final_usage_info,
        sum(counter.count for counter in state.counters),
    )


async def _wrapped_chat_completion_full_generator(
    self,
    request,
    result_generator,
    request_id,
    model_name,
    conversation,
    tokenizer,
    request_metadata,
    parser=None,
    mm_token_counts=None,
    **extra_kwargs,
):
    state = _FullUsageState()
    original = self._ascend_original_chat_completion_full_generator
    response = await original(
        request,
        _tracked_full_results(result_generator, state),
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        parser=parser,
        mm_token_counts=mm_token_counts,
        **extra_kwargs,
    )

    if not isinstance(response, chat_protocol.ChatCompletionResponse):
        return response

    reasoning_tokens = _count_full_response_reasoning_tokens(parser, state.final_res)
    _set_usage_details(response.usage, reasoning_tokens)
    _set_usage_details(request_metadata.final_usage_info, reasoning_tokens)
    return response


def _is_glm_parser_cls(parser_cls) -> bool:
    reasoning_parser_cls = getattr(parser_cls, "reasoning_parser_cls", None)
    return isinstance(reasoning_parser_cls, type) and issubclass(
        reasoning_parser_cls,
        Glm47MoeParserReasoningAdapter,
    )


def _patch_chat_usage_instance(self) -> None:
    if getattr(self, "_ascend_glm_reasoning_usage_patched", False):
        return
    self._ascend_original_chat_completion_stream_generator = self.chat_completion_stream_generator
    self._ascend_original_chat_completion_full_generator = self.chat_completion_full_generator
    self.chat_completion_stream_generator = MethodType(
        _wrapped_chat_completion_stream_generator,
        self,
    )
    self.chat_completion_full_generator = MethodType(
        _wrapped_chat_completion_full_generator,
        self,
    )
    self._ascend_glm_reasoning_usage_patched = True


class _ParserClsDescriptor:
    def __init__(self, default_value=None):
        self.default_value = default_value

    def __get__(self, instance, owner=None):
        if instance is None:
            return self.default_value
        return instance.__dict__.get("_ascend_parser_cls", self.default_value)

    def __set__(self, instance, value) -> None:
        instance.__dict__["_ascend_parser_cls"] = value
        if _is_glm_parser_cls(value):
            _patch_chat_usage_instance(instance)


_current_parser_cls = OpenAIServingChat.__dict__.get("parser_cls")
if not isinstance(_current_parser_cls, _ParserClsDescriptor):
    OpenAIServingChat.parser_cls = _ParserClsDescriptor(_current_parser_cls)
