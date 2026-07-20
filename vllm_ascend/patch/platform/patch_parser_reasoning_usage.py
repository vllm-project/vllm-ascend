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
# Parser reasoning usage accounting: backport
# https://github.com/vllm-project/vllm/pull/45802.
#

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from vllm.entrypoints.openai.chat_completion import protocol as chat_protocol
from vllm.entrypoints.openai.chat_completion import serving as chat_serving
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine import protocol as engine_protocol
from vllm.logger import logger

from vllm_ascend import envs


class _AscendCompletionTokenUsageInfo(engine_protocol.OpenAIBaseModel):
    reasoning_tokens: int = 0


class _AscendUsageInfo(engine_protocol.UsageInfo):
    completion_tokens_details: _AscendCompletionTokenUsageInfo | None = None


if (
    hasattr(engine_protocol, "CompletionTokenUsageInfo")
    and "completion_tokens_details" in engine_protocol.UsageInfo.model_fields
):
    _completion_token_usage_info_cls: Any = engine_protocol.CompletionTokenUsageInfo
    _usage_info_cls: Any = engine_protocol.UsageInfo
else:
    _AscendCompletionTokenUsageInfo.__module__ = engine_protocol.__name__
    _AscendUsageInfo.__module__ = engine_protocol.__name__

    engine_protocol.CompletionTokenUsageInfo = _AscendCompletionTokenUsageInfo
    engine_protocol.UsageInfo = _AscendUsageInfo

    _completion_token_usage_info_cls = _AscendCompletionTokenUsageInfo
    _usage_info_cls = _AscendUsageInfo


chat_protocol.UsageInfo = _usage_info_cls
chat_serving.UsageInfo = _usage_info_cls
chat_serving.CompletionTokenUsageInfo = _completion_token_usage_info_cls

UsageInfo = _usage_info_cls
CompletionTokenUsageInfo = _completion_token_usage_info_cls


def _make_completion_tokens_details(
    reasoning_tokens: int,
) -> Any:
    return _completion_token_usage_info_cls(reasoning_tokens=reasoning_tokens)


chat_serving._make_completion_tokens_details = _make_completion_tokens_details


def _rebuild_model_field(model_cls, field_name: str, annotation) -> None:
    if not hasattr(model_cls, "model_fields"):
        return
    if field_name not in model_cls.model_fields:
        return
    model_cls.__annotations__[field_name] = annotation
    model_cls.model_fields[field_name].annotation = annotation
    model_cls.model_rebuild(force=True)


_rebuild_model_field(chat_protocol.ChatCompletionResponse, "usage", _usage_info_cls)
_rebuild_model_field(chat_protocol.ChatCompletionStreamResponse, "usage", _usage_info_cls | None)
_rebuild_model_field(engine_protocol.RequestResponseMetadata, "final_usage_info", _usage_info_cls | None)


def _as_list(value) -> list[int]:
    return chat_serving.as_list(value)


def _call_with_supported_kwargs(func: Callable, *args, **kwargs):
    signature = inspect.signature(func)
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(*args, **kwargs)


def _serialize_chat_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload

    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json", exclude_none=True)
    elif hasattr(payload, "dict"):
        payload = payload.dict(exclude_none=True)

    return json.dumps(payload, ensure_ascii=False, default=str)


def _log_chat_completion_request(
    request_id: str,
    model_name: str,
    request: chat_protocol.ChatCompletionRequest,
) -> None:
    if not envs.VLLM_ASCEND_LOG_CHAT_COMPLETIONS:
        return
    logger.info(
        "Chat completion request. request_id=%s, model=%s, payload=%s",
        request_id,
        model_name,
        _serialize_chat_payload(request),
    )


def _log_chat_completion_response(
    request_id: str,
    model_name: str,
    response: Any,
) -> None:
    if not envs.VLLM_ASCEND_LOG_CHAT_COMPLETIONS:
        return
    logger.info(
        "Chat completion response. request_id=%s, model=%s, payload=%s",
        request_id,
        model_name,
        _serialize_chat_payload(response),
    )


def _coerce_reasoning_token_count(value: object) -> int:
    if value is None:
        return 0
    return max(0, int(cast(Any, value)))


def _count_reasoning_tokens(parser, token_ids: Sequence[int] | None = None) -> int:
    count_reasoning_tokens = getattr(parser, "count_reasoning_tokens", None)
    if count_reasoning_tokens is None:
        return 0
    try:
        return _coerce_reasoning_token_count(count_reasoning_tokens(token_ids or ()))
    except TypeError:
        return _coerce_reasoning_token_count(count_reasoning_tokens())


def _patch_parser_count_methods() -> None:
    try:
        from vllm.parser import abstract_parser
    except ImportError:
        return

    parser_cls = abstract_parser.Parser
    if not hasattr(parser_cls, "count_reasoning_tokens"):

        def _parser_count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
            return 0

        parser_cls.count_reasoning_tokens = _parser_count_reasoning_tokens

    delegating_parser_cls = abstract_parser.DelegatingParser
    parse = delegating_parser_cls.parse
    try:
        parse_source = inspect.getsource(parse)
    except (OSError, TypeError):
        parse_source = ""

    if "engine_based_streaming" not in parse_source:
        delegating_parser_cls._ascend_original_parse_for_reasoning_usage = parse

        def _delegating_parse(
            self,
            model_output: str,
            request,
            enable_auto_tools: bool = False,
            model_output_token_ids: Sequence[int] = (),
        ):
            reasoning_parser = getattr(self, "_reasoning_parser", None)
            if (
                reasoning_parser is not None
                and getattr(reasoning_parser, "engine_based_streaming", False)
                and model_output_token_ids
            ):
                delta = self.extract_reasoning_streaming(
                    previous_text="",
                    current_text=model_output,
                    delta_text=model_output,
                    previous_token_ids=[],
                    current_token_ids=model_output_token_ids,
                    delta_token_ids=model_output_token_ids,
                )
                finish = getattr(reasoning_parser, "finish_streaming", None)
                finish_delta = finish() if finish is not None else None

                reasoning_parts: list[str] = []
                content_parts: list[str] = []
                for message in (delta, finish_delta):
                    if message is None:
                        continue
                    if message.reasoning:
                        reasoning_parts.append(message.reasoning)
                    if message.content:
                        content_parts.append(message.content)
                reasoning = "".join(reasoning_parts) or None
                content = "".join(content_parts) or None
            else:
                reasoning, content = self.extract_reasoning(model_output, request)

            tool_calls, content = self._extract_tool_calls(
                content=content,
                request=request,
                enable_auto_tools=enable_auto_tools,
            )
            return reasoning, content, tool_calls

        _delegating_parse.__module__ = delegating_parser_cls.__module__
        _delegating_parse.__qualname__ = f"{delegating_parser_cls.__qualname__}.parse"
        delegating_parser_cls.parse = _delegating_parse

    def _delegating_count_reasoning_tokens(
        self,
        token_ids: Sequence[int] | None = None,
    ) -> int:
        reasoning_parser = getattr(self, "_reasoning_parser", None)
        if reasoning_parser is None:
            return 0
        return _count_reasoning_tokens(reasoning_parser, token_ids)

    _delegating_count_reasoning_tokens.__module__ = delegating_parser_cls.__module__
    _delegating_count_reasoning_tokens.__qualname__ = f"{delegating_parser_cls.__qualname__}.count_reasoning_tokens"
    delegating_parser_cls.count_reasoning_tokens = _delegating_count_reasoning_tokens


def _patch_minimax_append_think_reasoning_token_count() -> None:
    try:
        from vllm.reasoning.minimax_m2_reasoning_parser import (
            MiniMaxM2AppendThinkReasoningParser,
        )
    except ImportError:
        return

    def _count_append_think_reasoning_tokens(
        self,
        token_ids: Sequence[int],
    ) -> int:
        end_token_id = getattr(self, "end_token_id", None)
        if end_token_id is None:
            return 0
        for index, token_id in enumerate(token_ids):
            if token_id == end_token_id:
                return index
        return len(token_ids)

    _count_append_think_reasoning_tokens.__module__ = MiniMaxM2AppendThinkReasoningParser.__module__
    _count_append_think_reasoning_tokens.__qualname__ = (
        f"{MiniMaxM2AppendThinkReasoningParser.__qualname__}.count_reasoning_tokens"
    )
    MiniMaxM2AppendThinkReasoningParser.count_reasoning_tokens = _count_append_think_reasoning_tokens


def _patch_parser_engine_reasoning_token_count() -> None:
    try:
        from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
        from vllm.parser.engine.events import EventType
        from vllm.parser.engine.parser_engine import ParserEngine
    except ImportError:
        return

    try:
        count_source = inspect.getsource(ParserEngine.count_reasoning_tokens)
    except (OSError, TypeError):
        count_source = ""
    if "reasoning_token_count" in count_source:
        return

    original_reset = ParserEngine._reset
    original_feed = ParserEngine._feed

    def _reasoning_tokens_in_delta(self, delta_token_ids: Sequence[int]) -> int:
        engine = self._engine
        state = engine.state
        resolved_token_ids = getattr(engine, "_resolved_token_ids", {})
        drop_token_ids: set[int] = getattr(engine, "_drop_token_ids", set())
        transitions = self.parser_engine_config.transitions
        content_events = self.parser_engine_config.content_events

        reasoning_tokens = 0
        for token_id in delta_token_ids:
            if token_id in drop_token_ids:
                continue

            terminal = resolved_token_ids.get(token_id)
            if terminal is None:
                if content_events.get(state) == EventType.REASONING_CHUNK:
                    reasoning_tokens += 1
                continue

            transition = transitions.get((state, terminal))
            if transition is None:
                if content_events.get(state) == EventType.REASONING_CHUNK:
                    reasoning_tokens += 1
                continue
            state = transition.next_state

        return reasoning_tokens

    def _patched_reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        self._ascend_reasoning_token_count = 0
        return result

    def _patched_feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ):
        reasoning_tokens = _reasoning_tokens_in_delta(self, delta_token_ids)
        result = original_feed(self, delta_text, delta_token_ids)
        self._ascend_reasoning_token_count = getattr(self, "_ascend_reasoning_token_count", 0) + reasoning_tokens
        return result

    def _patched_count_reasoning_tokens(
        self,
        token_ids: Sequence[int] | None = None,
    ) -> int:
        return _coerce_reasoning_token_count(getattr(self, "_ascend_reasoning_token_count", 0))

    def _adapter_count_reasoning_tokens(
        self,
        token_ids: Sequence[int] | None = None,
    ) -> int:
        return self._parser_engine.count_reasoning_tokens(token_ids)

    _patched_reset.__module__ = ParserEngine.__module__
    _patched_reset.__qualname__ = f"{ParserEngine.__qualname__}._reset"
    _patched_feed.__module__ = ParserEngine.__module__
    _patched_feed.__qualname__ = f"{ParserEngine.__qualname__}._feed"
    _patched_count_reasoning_tokens.__module__ = ParserEngine.__module__
    _patched_count_reasoning_tokens.__qualname__ = f"{ParserEngine.__qualname__}.count_reasoning_tokens"
    _adapter_count_reasoning_tokens.__module__ = ParserEngineReasoningAdapter.__module__
    _adapter_count_reasoning_tokens.__qualname__ = f"{ParserEngineReasoningAdapter.__qualname__}.count_reasoning_tokens"

    ParserEngine._reset = _patched_reset
    ParserEngine._feed = _patched_feed
    ParserEngine.count_reasoning_tokens = _patched_count_reasoning_tokens
    ParserEngineReasoningAdapter.count_reasoning_tokens = _adapter_count_reasoning_tokens


_patch_parser_count_methods()
_patch_minimax_append_think_reasoning_token_count()
_patch_parser_engine_reasoning_token_count()


@dataclass
class _UsageTrackingState:
    completion_tokens: list[int]
    reasoning_tokens: list[int]
    counting_parsers: list[Any | None]
    parser_failed: list[bool]
    output_token_ids: list[list[int]] = field(default_factory=list)
    num_prompt_tokens: int = 0
    num_cached_tokens: int | None = None
    final_res: Any = None


def _get_history_tool_calls_cnt(conversation) -> int:
    get_history_tool_calls_cnt = getattr(chat_serving, "get_history_tool_calls_cnt", None)
    if get_history_tool_calls_cnt is None:
        return 0
    return get_history_tool_calls_cnt(conversation)


def _make_parser(parser_cls, tokenizer, tools, chat_template_kwargs):
    try:
        return parser_cls(
            tokenizer,
            tools,
            chat_template_kwargs=chat_template_kwargs,
        )
    except TypeError:
        return parser_cls(tokenizer, tools)


def _make_counting_parsers(
    self,
    request,
    tokenizer,
    conversation,
    num_choices: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[Any | None]:
    parser_cls = getattr(self, "parser_cls", None)
    if parser_cls is None or tokenizer is None:
        return [None] * num_choices

    try:
        parsers = [
            _make_parser(
                parser_cls,
                tokenizer,
                getattr(request, "tools", None),
                chat_template_kwargs,
            )
            for _ in range(num_choices)
        ]
        history_tool_call_cnt = (
            _get_history_tool_calls_cnt(conversation) if getattr(self, "tool_call_id_type", None) == "kimi_k2" else 0
        )
        for parser in parsers:
            stream_state = getattr(parser, "_stream_state", None)
            if stream_state is None:
                continue
            stream_state.tool_call_id_type = getattr(self, "tool_call_id_type", "random")
            stream_state.history_tool_call_cnt = history_tool_call_cnt
        return parsers
    except Exception:
        return [None] * num_choices


def _create_usage_tracking_state(
    self,
    request,
    tokenizer,
    conversation,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> _UsageTrackingState:
    num_choices = 1 if getattr(request, "n", None) is None else request.n
    return _UsageTrackingState(
        completion_tokens=[0] * num_choices,
        reasoning_tokens=[0] * num_choices,
        counting_parsers=_make_counting_parsers(
            self,
            request,
            tokenizer,
            conversation,
            num_choices,
            chat_template_kwargs,
        ),
        parser_failed=[False] * num_choices,
        output_token_ids=[[] for _ in range(num_choices)],
    )


def _update_prompt_usage(state: _UsageTrackingState, res) -> None:
    if res.prompt_token_ids is not None:
        num_prompt_tokens = len(res.prompt_token_ids)
        if res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(res.encoder_prompt_token_ids)
        state.num_prompt_tokens = num_prompt_tokens

    if state.num_cached_tokens is None:
        state.num_cached_tokens = res.num_cached_tokens

    state.final_res = res


def _update_stream_reasoning_tokens(
    state: _UsageTrackingState,
    request,
    res,
    output,
) -> None:
    choice_index = output.index
    if not 0 <= choice_index < len(state.counting_parsers):
        return
    if state.parser_failed[choice_index]:
        return

    parser = state.counting_parsers[choice_index]
    if parser is None:
        return

    token_ids = _as_list(output.token_ids)
    state.output_token_ids[choice_index].extend(token_ids)
    try:
        parser.parse_delta(
            delta_text=output.text,
            delta_token_ids=token_ids,
            request=request,
            prompt_token_ids=res.prompt_token_ids,
            finished=output.finish_reason is not None,
        )
        state.reasoning_tokens[choice_index] = _count_reasoning_tokens(
            parser,
            state.output_token_ids[choice_index],
        )
    except Exception:
        state.parser_failed[choice_index] = True
        state.reasoning_tokens[choice_index] = 0


def _update_usage_tracking_state(
    state: _UsageTrackingState,
    request,
    res,
) -> None:
    _update_prompt_usage(state, res)

    for output in res.outputs:
        if not 0 <= output.index < len(state.completion_tokens):
            continue
        token_ids = _as_list(output.token_ids)
        state.completion_tokens[output.index] += len(token_ids)
        _update_stream_reasoning_tokens(state, request, res, output)


async def _tracked_result_generator(
    result_generator: AsyncIterator,
    state: _UsageTrackingState,
    request,
):
    async for res in result_generator:
        _update_usage_tracking_state(state, request, res)
        yield res


def _usage_reasoning_tokens_for_stream_chunk(
    state: _UsageTrackingState,
    chunk: dict[str, Any],
) -> int:
    choices = chunk.get("choices") or []
    if choices:
        choice_index = choices[0].get("index", 0)
        if 0 <= choice_index < len(state.reasoning_tokens):
            return state.reasoning_tokens[choice_index]
    return sum(state.reasoning_tokens)


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

    usage["completion_tokens_details"] = {
        "reasoning_tokens": _usage_reasoning_tokens_for_stream_chunk(state, chunk),
    }
    return f"{prefix}{json.dumps(chunk, ensure_ascii=False)}{suffix}"


def _set_usage_details(usage, reasoning_tokens: int) -> None:
    if usage is None:
        return
    usage.completion_tokens_details = _make_completion_tokens_details(reasoning_tokens)


def _set_stream_final_usage(
    request_metadata,
    state: _UsageTrackingState,
) -> None:
    reasoning_tokens = sum(state.reasoning_tokens)
    usage = request_metadata.final_usage_info
    if usage is None:
        completion_tokens = sum(state.completion_tokens)
        usage = _usage_info_cls(
            prompt_tokens=state.num_prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=state.num_prompt_tokens + completion_tokens,
            completion_tokens_details=_make_completion_tokens_details(reasoning_tokens),
        )
        request_metadata.final_usage_info = usage
        return
    _set_usage_details(usage, reasoning_tokens)


def _parse_full_output_for_usage(
    parser,
    request,
    output,
    enable_auto_tools: bool,
) -> int:
    token_ids = _as_list(output.token_ids)
    try:
        parser.parse(
            output.text,
            request,
            enable_auto_tools=enable_auto_tools,
            model_output_token_ids=token_ids,
        )
    except TypeError:
        parser.parse(
            output.text,
            request,
            enable_auto_tools=enable_auto_tools,
        )
    return _count_reasoning_tokens(parser, token_ids)


def _count_full_response_reasoning_tokens(
    self,
    request,
    final_res,
    tokenizer,
    conversation,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int:
    if final_res is None:
        return 0

    num_choices = max((output.index for output in final_res.outputs), default=-1) + 1
    parsers = _make_counting_parsers(
        self,
        request,
        tokenizer,
        conversation,
        num_choices,
        chat_template_kwargs,
    )

    reasoning_tokens = 0
    for output in final_res.outputs:
        if not 0 <= output.index < len(parsers):
            continue
        parser = parsers[output.index]
        if parser is None:
            continue
        try:
            reasoning_tokens += _parse_full_output_for_usage(
                parser,
                request,
                output,
                getattr(self, "enable_auto_tools", False),
            )
        except Exception:
            continue
    return reasoning_tokens


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
    chat_template_kwargs: dict[str, Any] | None = None,
    **extra_kwargs: Any,
):
    original_stream_generator = self._ascend_original_chat_completion_stream_generator_for_reasoning_usage
    _log_chat_completion_request(request_id, model_name, request)
    state = _create_usage_tracking_state(
        self,
        request,
        tokenizer,
        conversation,
        chat_template_kwargs,
    )

    kwargs = dict(extra_kwargs)
    kwargs["chat_template_kwargs"] = chat_template_kwargs
    async for data in _call_with_supported_kwargs(
        original_stream_generator,
        request,
        _tracked_result_generator(result_generator, state, request),
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        **kwargs,
    ):
        response = _inject_stream_usage_details(data, state)
        _log_chat_completion_response(request_id, model_name, response)
        yield response

    _set_stream_final_usage(request_metadata, state)


async def _wrapped_chat_completion_full_generator(
    self,
    request: chat_protocol.ChatCompletionRequest,
    result_generator: AsyncIterator,
    request_id: str,
    model_name: str,
    conversation,
    tokenizer,
    request_metadata: engine_protocol.RequestResponseMetadata,
    parser=None,
    chat_template_kwargs: dict[str, Any] | None = None,
    **extra_kwargs: Any,
):
    original_full_generator = self._ascend_original_chat_completion_full_generator_for_reasoning_usage
    _log_chat_completion_request(request_id, model_name, request)
    state = _UsageTrackingState(
        completion_tokens=[],
        reasoning_tokens=[],
        counting_parsers=[],
        parser_failed=[],
    )

    async def tracked_full_result_generator():
        async for res in result_generator:
            _update_prompt_usage(state, res)
            yield res

    kwargs = dict(extra_kwargs)
    kwargs["parser"] = parser
    kwargs["chat_template_kwargs"] = chat_template_kwargs
    response = await _call_with_supported_kwargs(
        original_full_generator,
        request,
        tracked_full_result_generator(),
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
        **kwargs,
    )

    if not isinstance(response, chat_protocol.ChatCompletionResponse):
        _log_chat_completion_response(request_id, model_name, response)
        return response

    reasoning_tokens = _count_full_response_reasoning_tokens(
        self,
        request,
        state.final_res,
        tokenizer,
        conversation,
        chat_template_kwargs,
    )
    _set_usage_details(response.usage, reasoning_tokens)
    _set_usage_details(request_metadata.final_usage_info, reasoning_tokens)
    _log_chat_completion_response(request_id, model_name, response)
    return response


_wrapped_chat_completion_stream_generator.__module__ = OpenAIServingChat.__module__
_wrapped_chat_completion_stream_generator.__qualname__ = (
    f"{OpenAIServingChat.__qualname__}.chat_completion_stream_generator"
)
_wrapped_chat_completion_full_generator.__module__ = OpenAIServingChat.__module__
_wrapped_chat_completion_full_generator.__qualname__ = (
    f"{OpenAIServingChat.__qualname__}.chat_completion_full_generator"
)


def _patch_chat_usage_generators() -> None:
    if getattr(OpenAIServingChat, "_ascend_reasoning_usage_patched", False):
        return
    OpenAIServingChat._ascend_original_chat_completion_stream_generator_for_reasoning_usage = (
        OpenAIServingChat.chat_completion_stream_generator
    )
    OpenAIServingChat._ascend_original_chat_completion_full_generator_for_reasoning_usage = (
        OpenAIServingChat.chat_completion_full_generator
    )
    OpenAIServingChat.chat_completion_stream_generator = _wrapped_chat_completion_stream_generator
    OpenAIServingChat.chat_completion_full_generator = _wrapped_chat_completion_full_generator
    OpenAIServingChat._ascend_reasoning_usage_patched = True


_patch_chat_usage_generators()
