#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

"""Register the upstream Kimi K3 parsers on the pinned vLLM core."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

import vllm_ascend.patch.platform.kimi_k3_structural_tag  # noqa: F401
from vllm_ascend.patch.platform.kimi_k3_parser import KimiK3Parser
from vllm_ascend.patch.platform.kimi_k3_protocol import (
    ARGUMENT_END,
    CALL_END,
    END_OF_MSG_TOKEN,
    JSON_END,
    MESSAGE_END,
    RESPONSE_END,
    RESPONSE_START,
    SEP_TOKEN,
    THINK_END,
    THINK_START,
    TOOLS_END,
    TOOLS_START,
)
from vllm_ascend.patch.platform.kimi_k3_reasoning_parser import (
    KimiK3ReasoningParser,
)
from vllm_ascend.patch.platform.kimi_k3_tool_parser import KimiK3ToolParser

_ORIGINAL_GET_PARSER_ATTR = "_ascend_original_kimi_k3_get_parser"
_ORIGINAL_CHAT_FULL_ATTR = "_ascend_original_kimi_k3_chat_completion_full_generator"
_ORIGINAL_CHAT_STREAM_ATTR = "_ascend_original_kimi_k3_chat_completion_stream_generator"
_ORIGINAL_RESPONSES_ATTR = "_ascend_original_kimi_k3_create_responses"

__all__ = [
    "ARGUMENT_END",
    "CALL_END",
    "END_OF_MSG_TOKEN",
    "JSON_END",
    "MESSAGE_END",
    "RESPONSE_END",
    "RESPONSE_START",
    "SEP_TOKEN",
    "THINK_END",
    "THINK_START",
    "TOOLS_END",
    "TOOLS_START",
    "KimiK3Parser",
    "KimiK3ReasoningParser",
    "KimiK3ToolParser",
]


def _named_tool_choice(request: Any) -> str | None:
    choice = getattr(request, "tool_choice", None)
    function = choice.get("function") if isinstance(choice, dict) else getattr(choice, "function", None)
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None)


async def _capture_final_outputs(
    result_generator,
    final_token_ids: dict[int, Sequence[int]],
    finish_reasons: dict[int, str],
    output_indices: list[int],
):
    async for result in result_generator:
        final_token_ids.clear()
        finish_reasons.clear()
        output_indices[:] = [output.index for output in result.outputs]
        for output in result.outputs:
            final_token_ids[output.index] = output.token_ids
            if output.finish_reason is not None:
                finish_reasons[output.index] = output.finish_reason
        yield result


def _is_remote_decode_request(request: ChatCompletionRequest) -> bool:
    kv_transfer_params = request.kv_transfer_params
    return bool(kv_transfer_params and kv_transfer_params.get("do_remote_decode") is True)


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
):
    original = getattr(self, _ORIGINAL_CHAT_FULL_ATTR)
    if isinstance(parser, KimiK3Parser) and _is_remote_decode_request(request):
        # The P-side token is internal transfer output, not a complete K3 response.
        return await original(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            None,
            mm_token_counts,
        )
    if not isinstance(parser, KimiK3Parser):
        return await original(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser,
            mm_token_counts,
        )

    final_token_ids: dict[int, Sequence[int]] = {}
    finish_reasons: dict[int, str] = {}
    output_indices: list[int] = []
    parsed_contents: list[str | None] = []
    parse_index = 0
    original_parse = parser.parse

    def parse_with_token_ids(
        model_output,
        parsed_request,
        enable_auto_tools=False,
        model_output_token_ids=(),
    ):
        nonlocal parse_index
        token_ids = model_output_token_ids
        if not token_ids and parse_index < len(output_indices):
            token_ids = final_token_ids.get(output_indices[parse_index], ())
        parse_index += 1
        parsed = original_parse(
            model_output,
            parsed_request,
            enable_auto_tools=enable_auto_tools,
            model_output_token_ids=token_ids,
        )
        parsed_contents.append(parsed[1])
        return parsed

    parser.parse = parse_with_token_ids  # type: ignore[method-assign]
    try:
        response = await original(
            request,
            _capture_final_outputs(
                result_generator,
                final_token_ids,
                finish_reasons,
                output_indices,
            ),
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser,
            mm_token_counts,
        )
    finally:
        del parser.parse

    if isinstance(response, ChatCompletionResponse):
        for choice in response.choices:
            engine_reason = finish_reasons.get(choice.index)
            if choice.finish_reason == "tool_calls" and engine_reason not in (
                None,
                "stop",
            ):
                choice.finish_reason = engine_reason
        if request.tool_choice == "required" or _named_tool_choice(request) is not None:
            for choice, content in zip(response.choices, parsed_contents, strict=False):
                choice.message.content = content or ""
    return response


async def _capture_finish_reasons(result_generator, finish_reasons: dict[int, str]):
    async for result in result_generator:
        for output in result.outputs:
            if output.finish_reason is not None:
                finish_reasons[output.index] = output.finish_reason
        yield result


def _restore_engine_finish_reason(data: str, finish_reasons: dict[int, str]) -> str:
    if not finish_reasons or '"finish_reason":"tool_calls"' not in data:
        return data

    prefix = "data: "
    suffix = "\n\n"
    if not data.startswith(prefix) or not data.endswith(suffix):
        return data

    payload = data[len(prefix) : -len(suffix)]
    if payload == "[DONE]":
        return data
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return data

    changed = False
    for choice in chunk.get("choices") or []:
        engine_reason = finish_reasons.get(choice.get("index"))
        if choice.get("finish_reason") == "tool_calls" and engine_reason not in (
            None,
            "stop",
        ):
            choice["finish_reason"] = engine_reason
            changed = True
    if not changed:
        return data
    return f"{prefix}{json.dumps(chunk, ensure_ascii=False, separators=(',', ':'))}{suffix}"


async def _wrapped_chat_completion_stream_generator(
    self,
    request,
    result_generator,
    *args,
    **kwargs,
):
    original = getattr(self, _ORIGINAL_CHAT_STREAM_ATTR)
    if not _is_kimi_k3_parser_class(self.parser_cls):
        async for data in original(request, result_generator, *args, **kwargs):
            yield data
        return

    finish_reasons: dict[int, str] = {}
    async for data in original(
        request,
        _capture_finish_reasons(result_generator, finish_reasons),
        *args,
        **kwargs,
    ):
        yield _restore_engine_finish_reason(data, finish_reasons)


async def _wrapped_create_responses(self, request, raw_request=None):
    if _is_kimi_k3_parser_class(self.parser):
        return self.create_error_response("Kimi K3 supports Chat Completions only; the Responses API is not supported.")
    original = getattr(self, _ORIGINAL_RESPONSES_ATTR)
    return await original(request, raw_request)


if not hasattr(OpenAIServingChat, _ORIGINAL_CHAT_FULL_ATTR):
    setattr(
        OpenAIServingChat,
        _ORIGINAL_CHAT_FULL_ATTR,
        OpenAIServingChat.chat_completion_full_generator,
    )
    setattr(
        OpenAIServingChat,
        _ORIGINAL_CHAT_STREAM_ATTR,
        OpenAIServingChat.chat_completion_stream_generator,
    )
    OpenAIServingChat.chat_completion_full_generator = _wrapped_chat_completion_full_generator
    OpenAIServingChat.chat_completion_stream_generator = _wrapped_chat_completion_stream_generator

if not hasattr(OpenAIServingResponses, _ORIGINAL_RESPONSES_ATTR):
    setattr(
        OpenAIServingResponses,
        _ORIGINAL_RESPONSES_ATTR,
        OpenAIServingResponses._create_responses,
    )
    OpenAIServingResponses._create_responses = _wrapped_create_responses


# vLLM d02 predates the K3 ParserManager special case added in f5a7cce9b.
if not hasattr(ParserManager, _ORIGINAL_GET_PARSER_ATTR):
    setattr(
        ParserManager,
        _ORIGINAL_GET_PARSER_ATTR,
        ParserManager.get_parser.__func__,
    )


def _is_kimi_k3_parser_class(parser_cls: Any) -> bool:
    return isinstance(parser_cls, type) and issubclass(parser_cls, KimiK3Parser)


def _get_parser_with_kimi_k3(
    cls,
    tool_parser_name: str | None = None,
    reasoning_parser_name: str | None = None,
    enable_auto_tools: bool = False,
    model_name: str | None = None,
    is_harmony: bool = False,
):
    uses_kimi_k3 = tool_parser_name == "kimi_k3" or reasoning_parser_name == "kimi_k3"
    if uses_kimi_k3:
        reasoning_parser_cls = cls.get_reasoning_parser(reasoning_parser_name)
        tool_parser_cls = cls.get_tool_parser(
            tool_parser_name,
            enable_auto_tools,
            model_name,
        )
        if reasoning_parser_cls is None and tool_parser_cls is None:
            return None

        if is_harmony:
            original = getattr(ParserManager, _ORIGINAL_GET_PARSER_ATTR)
            return original(
                cls,
                tool_parser_name=tool_parser_name,
                reasoning_parser_name=reasoning_parser_name,
                enable_auto_tools=enable_auto_tools,
                model_name=model_name,
                is_harmony=True,
            )

        reasoning_cls = reasoning_parser_cls
        tool_cls = tool_parser_cls

        class _KimiK3Parser(KimiK3Parser):
            reasoning_parser_cls = reasoning_cls
            tool_parser_cls = tool_cls

        return _KimiK3Parser

    original = getattr(ParserManager, _ORIGINAL_GET_PARSER_ATTR)
    return original(
        cls,
        tool_parser_name=tool_parser_name,
        reasoning_parser_name=reasoning_parser_name,
        enable_auto_tools=enable_auto_tools,
        model_name=model_name,
        is_harmony=is_harmony,
    )


ParserManager.get_parser = classmethod(_get_parser_with_kimi_k3)


if "kimi_k3" not in ReasoningParserManager.list_registered():
    ReasoningParserManager.register_module(
        name="kimi_k3",
        module=KimiK3ReasoningParser,
        force=False,
    )

if "kimi_k3" not in ToolParserManager.list_registered():
    ToolParserManager.register_module(
        name="kimi_k3",
        module=KimiK3ToolParser,
        force=False,
    )
