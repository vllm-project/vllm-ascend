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
# OpenAI chat usage accounting: backport MiniMax reasoning token accounting.
#

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.openai.chat_completion import protocol as chat_protocol
from vllm.entrypoints.openai.chat_completion import serving as chat_serving
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine import protocol as engine_protocol
from vllm.reasoning import minimax_m2_reasoning_parser as minimax_parser


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
_rebuild_model_field(engine_protocol.RequestResponseMetadata, "final_usage_info", UsageInfo | None)


def _count_minimax_reasoning_tokens(
    token_ids: Sequence[int],
    end_token_id: int | None,
) -> int:
    if end_token_id is None:
        return 0

    for idx, token_id in enumerate(token_ids):
        if token_id == end_token_id:
            return idx
    return len(token_ids)


def _patched_count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
    return _count_minimax_reasoning_tokens(token_ids, self.end_token_id)


minimax_parser.MiniMaxM2ReasoningParser.count_reasoning_tokens = _patched_count_reasoning_tokens
minimax_parser.MiniMaxM2AppendThinkReasoningParser.count_reasoning_tokens = _patched_count_reasoning_tokens


def _count_reasoning_tokens_for_usage(
    token_ids: Sequence[int],
    reasoning_parser,
) -> int | None:
    if reasoning_parser is None:
        return None
    return reasoning_parser.count_reasoning_tokens(token_ids)


def _clamp_reasoning_tokens(
    reasoning_tokens: int | None,
    completion_tokens: int,
) -> int | None:
    if reasoning_tokens is None:
        return None
    return max(0, min(reasoning_tokens, completion_tokens))


def _make_usage_info(
    self,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    num_cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> UsageInfo:
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    reasoning_tokens = _clamp_reasoning_tokens(reasoning_tokens, completion_tokens)
    if reasoning_tokens is not None:
        usage.completion_tokens_details = CompletionTokenUsageInfo(reasoning_tokens=reasoning_tokens)
    if self.enable_prompt_tokens_details and num_cached_tokens:
        usage.prompt_tokens_details = chat_serving.PromptTokenUsageInfo(cached_tokens=num_cached_tokens)
    return usage


OpenAIServingChat._count_reasoning_tokens_for_usage = staticmethod(_count_reasoning_tokens_for_usage)
OpenAIServingChat._make_usage_info = _make_usage_info


@dataclass
class _UsageTrackingState:
    completion_tokens: list[int]
    raw_output_token_ids: list[list[int]]
    reasoning_parser: Any
    num_prompt_tokens: int = 0
    num_cached_tokens: int | None = None
    final_res: Any = None


def _create_usage_tracking_state(
    num_choices: int,
    reasoning_parser,
) -> _UsageTrackingState:
    return _UsageTrackingState(
        completion_tokens=[0] * num_choices,
        raw_output_token_ids=[[] for _ in range(num_choices)],
        reasoning_parser=reasoning_parser,
    )


def _update_usage_tracking_state(
    state: _UsageTrackingState,
    res,
) -> None:
    if res.prompt_token_ids is not None:
        num_prompt_tokens = len(res.prompt_token_ids)
        if res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(res.encoder_prompt_token_ids)
        state.num_prompt_tokens = num_prompt_tokens

    if state.num_cached_tokens is None:
        state.num_cached_tokens = res.num_cached_tokens

    state.final_res = res

    for output in res.outputs:
        if 0 <= output.index < len(state.completion_tokens):
            token_ids = chat_serving.as_list(output.token_ids)
            state.completion_tokens[output.index] += len(token_ids)
            state.raw_output_token_ids[output.index].extend(token_ids)


async def _tracked_result_generator(
    result_generator: AsyncIterator,
    state: _UsageTrackingState,
):
    async for res in result_generator:
        _update_usage_tracking_state(state, res)
        yield res


def _sum_reasoning_tokens_for_usage(
    raw_output_token_ids: list[list[int]],
    reasoning_parser,
) -> int | None:
    if reasoning_parser is None:
        return None
    return sum(
        _count_reasoning_tokens_for_usage(token_ids, reasoning_parser) or 0 for token_ids in raw_output_token_ids
    )


def _reasoning_tokens_for_choice(
    state: _UsageTrackingState,
    choice_index: int,
) -> int | None:
    if state.reasoning_parser is None:
        return None
    if not 0 <= choice_index < len(state.raw_output_token_ids):
        return None
    return _count_reasoning_tokens_for_usage(
        state.raw_output_token_ids[choice_index],
        state.reasoning_parser,
    )


def _make_full_response_usage(
    self,
    state: _UsageTrackingState,
) -> UsageInfo | None:
    if state.final_res is None:
        return None

    return self._make_usage_info(
        prompt_tokens=state.num_prompt_tokens,
        completion_tokens=sum(state.completion_tokens),
        num_cached_tokens=state.num_cached_tokens,
        reasoning_tokens=_sum_reasoning_tokens_for_usage(
            state.raw_output_token_ids,
            state.reasoning_parser,
        ),
    )


def _usage_reasoning_tokens_for_stream_chunk(
    state: _UsageTrackingState,
    chunk: dict[str, Any],
    completion_tokens: int,
) -> int | None:
    if state.reasoning_parser is None:
        return None

    choices = chunk.get("choices") or []
    if choices:
        choice_index = choices[0].get("index", 0)
        reasoning_tokens = _reasoning_tokens_for_choice(state, choice_index)
    else:
        reasoning_tokens = _sum_reasoning_tokens_for_usage(
            state.raw_output_token_ids,
            state.reasoning_parser,
        )
    return _clamp_reasoning_tokens(reasoning_tokens, completion_tokens)


def _inject_stream_usage_details(
    data: str,
    state: _UsageTrackingState,
) -> str:
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

    completion_tokens = usage.get("completion_tokens") or 0
    reasoning_tokens = _usage_reasoning_tokens_for_stream_chunk(
        state,
        chunk,
        completion_tokens,
    )
    if reasoning_tokens is None:
        return data

    usage["completion_tokens_details"] = {
        "reasoning_tokens": reasoning_tokens,
    }
    return f"{prefix}{json.dumps(chunk, ensure_ascii=False)}{suffix}"


if not hasattr(OpenAIServingChat, "_ascend_original_chat_completion_stream_generator"):
    OpenAIServingChat._ascend_original_chat_completion_stream_generator = (
        OpenAIServingChat.chat_completion_stream_generator
    )

if not hasattr(OpenAIServingChat, "_ascend_original_chat_completion_full_generator"):
    OpenAIServingChat._ascend_original_chat_completion_full_generator = OpenAIServingChat.chat_completion_full_generator


async def _wrapped_chat_completion_stream_generator(
    self,
    request: chat_protocol.ChatCompletionRequest,
    result_generator: AsyncIterator,
    request_id: str,
    model_name: str,
    conversation,
    tokenizer,
    request_metadata: engine_protocol.RequestResponseMetadata,
    reasoning_parser=None,
):
    num_choices = 1 if request.n is None else request.n
    state = _create_usage_tracking_state(num_choices, reasoning_parser)

    original_stream_generator = self._ascend_original_chat_completion_stream_generator
    async for data in original_stream_generator(
        request,
        _tracked_result_generator(result_generator, state),
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        reasoning_parser,
    ):
        yield _inject_stream_usage_details(data, state)

    usage = _make_full_response_usage(self, state)
    if usage is not None:
        request_metadata.final_usage_info = usage


async def _wrapped_chat_completion_full_generator(
    self,
    request: chat_protocol.ChatCompletionRequest,
    result_generator: AsyncIterator,
    request_id: str,
    model_name: str,
    conversation,
    tokenizer,
    request_metadata: engine_protocol.RequestResponseMetadata,
    reasoning_parser=None,
):
    num_choices = 1 if request.n is None else request.n
    state = _create_usage_tracking_state(num_choices, reasoning_parser)

    original_full_generator = self._ascend_original_chat_completion_full_generator
    response = await original_full_generator(
        request,
        _tracked_result_generator(result_generator, state),
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        reasoning_parser,
    )

    if not isinstance(response, chat_protocol.ChatCompletionResponse):
        return response

    usage = _make_full_response_usage(self, state)
    if usage is None:
        return response

    response.usage = usage
    request_metadata.final_usage_info = usage
    return response


_wrapped_chat_completion_stream_generator.__module__ = OpenAIServingChat.__module__
_wrapped_chat_completion_stream_generator.__qualname__ = (
    f"{OpenAIServingChat.__qualname__}.chat_completion_stream_generator"
)
_wrapped_chat_completion_full_generator.__module__ = OpenAIServingChat.__module__
_wrapped_chat_completion_full_generator.__qualname__ = (
    f"{OpenAIServingChat.__qualname__}.chat_completion_full_generator"
)

OpenAIServingChat.chat_completion_stream_generator = _wrapped_chat_completion_stream_generator
OpenAIServingChat.chat_completion_full_generator = _wrapped_chat_completion_full_generator
