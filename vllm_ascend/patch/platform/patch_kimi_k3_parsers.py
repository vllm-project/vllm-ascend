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
"""Kimi K3 reasoning and tool adapters backed by strict XTML parsers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.abs_reasoning_parsers import (
    ReasoningParser,
    ReasoningParserManager,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
    ToolParserManager,
)

from vllm_ascend.patch.platform.kimi_k3_xtml import (
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
    KimiK3ParseDelta,
    KimiK3XTMLParseError,
    KimiK3XTMLParser,
    ToolMode,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

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


def _find_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern or len(pattern) > len(values):
        return -1
    for index in range(len(values) - len(pattern), -1, -1):
        if list(values[index : index + len(pattern)]) == list(pattern):
            return index
    return -1


def _encode_marker(tokenizer: TokenizerLike, marker: str) -> list[int]:
    encoded = tokenizer.encode(marker)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids", [])
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _template_uses_thinking(template_kwargs: dict[str, Any]) -> bool:
    return bool(template_kwargs.get("thinking", True))


def adjust_kimi_k3_request(request: Any):
    """Preserve K3's adjacent control/text markers during detokenization."""

    if hasattr(request, "skip_special_tokens"):
        request.skip_special_tokens = False
    if hasattr(request, "spaces_between_special_tokens"):
        request.spaces_between_special_tokens = False
    return request


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return function.get("name")
        return getattr(function, "name", None)
    function = getattr(tool, "function", None)
    return getattr(function, "name", None)


def _named_tool_choice(request: Any) -> str | None:
    choice = getattr(request, "tool_choice", None)
    function = choice.get("function") if isinstance(choice, dict) else getattr(choice, "function", None)
    if isinstance(function, dict):
        return function.get("name")
    return getattr(function, "name", None)


def _protocol_parser_for_request(
    request: ChatCompletionRequest,
    *,
    thinking_enabled: bool,
    tokenizer: TokenizerLike | None = None,
) -> KimiK3XTMLParser:
    tools = request.tools or []
    allowed_tool_names = frozenset(name for tool in tools if (name := _tool_name(tool)))
    choice = request.tool_choice
    named_tool = _named_tool_choice(request)
    tool_mode: ToolMode

    if named_tool:
        tool_mode = "named"
    elif choice == "required":
        tool_mode = "required"
    elif choice == "auto":
        tool_mode = "auto"
    elif choice == "none" or (choice is None and not tools):
        tool_mode = "none"
    elif choice is None:
        tool_mode = "auto"
    else:
        raise KimiK3XTMLParseError(f"Unsupported K3 tool_choice: {choice!r}.")

    if tool_mode != "none" and not allowed_tool_names:
        raise KimiK3XTMLParseError(f"K3 tool_choice={tool_mode!r} requires declared tools.")
    if named_tool is not None and named_tool not in allowed_tool_names:
        raise KimiK3XTMLParseError(f"Named K3 tool choice {named_tool!r} is not declared.")

    return KimiK3XTMLParser(
        thinking_enabled=thinking_enabled,
        tool_mode=tool_mode,
        allowed_tool_names=allowed_tool_names,
        named_tool=named_tool,
        tokenizer=tokenizer,
    )


def _to_delta_message(
    delta: KimiK3ParseDelta,
    call_ids: list[str],
    *,
    include_reasoning: bool,
) -> DeltaMessage | None:
    tool_deltas: list[DeltaToolCall] = []
    for call in delta.tool_calls:
        function_kwargs: dict[str, str] = {}
        if call.name is not None:
            function_kwargs["name"] = call.name
        if call.arguments is not None:
            function_kwargs["arguments"] = call.arguments

        tool_call_kwargs: dict[str, Any] = {
            "index": call.index,
            "function": DeltaFunctionCall(**function_kwargs),
        }
        if call.name is not None:
            while len(call_ids) <= call.index:
                call_ids.append(make_tool_call_id())
            tool_call_kwargs["id"] = call_ids[call.index]
            tool_call_kwargs["type"] = "function"
        tool_deltas.append(DeltaToolCall(**tool_call_kwargs))

    message_kwargs: dict[str, Any] = {}
    if include_reasoning and delta.reasoning:
        message_kwargs["reasoning"] = delta.reasoning
    if delta.content:
        message_kwargs["content"] = delta.content
    if tool_deltas:
        message_kwargs["tool_calls"] = tool_deltas
    return DeltaMessage(**message_kwargs) if message_kwargs else None


class KimiK3ReasoningParser(ReasoningParser):
    """Expose K3 reasoning boundaries to vLLM's scheduler and adapters."""

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        if not self.model_tokenizer:
            raise ValueError("KimiK3ReasoningParser requires a tokenizer.")
        self.think_start_token_ids = _encode_marker(tokenizer, THINK_START)
        self.think_end_token_ids = _encode_marker(tokenizer, THINK_END)
        if not self.think_start_token_ids or not self.think_end_token_ids:
            raise RuntimeError("Unable to encode Kimi K3 reasoning markers.")
        template_kwargs = kwargs.get("chat_template_kwargs") or {}
        self._thinking_enabled = _template_uses_thinking(template_kwargs)
        self._reasoning_end_tail: list[int] = []

    @property
    def reasoning_start_str(self) -> str:
        return THINK_START

    @property
    def reasoning_end_str(self) -> str:
        return THINK_END

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        last_start = _find_subsequence(input_ids, self.think_start_token_ids)
        last_end = _find_subsequence(input_ids, self.think_end_token_ids)
        return last_start < 0 or last_end > last_start

    def is_reasoning_end_streaming(
        self,
        input_ids: Sequence[int],
        delta_ids: Iterable[int],
    ) -> bool:
        if not self._thinking_enabled:
            return True
        candidate = self._reasoning_end_tail + [int(token_id) for token_id in delta_ids]
        ended = _find_subsequence(candidate, self.think_end_token_ids) >= 0
        tail_size = max(0, len(self.think_end_token_ids) - 1)
        self._reasoning_end_tail = candidate[-tail_size:] if tail_size else []
        return ended

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        end_index = _find_subsequence(input_ids, self.think_end_token_ids)
        if end_index >= 0:
            return input_ids[end_index + len(self.think_end_token_ids) :]
        if _find_subsequence(input_ids, self.think_start_token_ids) < 0:
            return input_ids
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        del model_output, request
        raise ValueError("Kimi K3 parsing supports Chat Completions only.")

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        del (
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        raise ValueError("Kimi K3 parsing supports Chat Completions only.")

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self._thinking_enabled:
            return 0
        start_index = _find_subsequence(token_ids, self.think_start_token_ids)
        content_start = start_index + len(self.think_start_token_ids) if start_index >= 0 else 0
        end_index = _find_subsequence(token_ids, self.think_end_token_ids)
        if end_index < content_start:
            end_index = len(token_ids)
        return max(0, end_index - content_start)

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        return adjust_kimi_k3_request(request)


class KimiK3ToolParser(ToolParser):
    """K3 request adapter; unified parsing lives in KimiK3Parser."""

    supports_required_and_named = False

    def adjust_request(self, request: ChatCompletionRequest):
        return adjust_kimi_k3_request(request)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ):
        del model_output, request
        raise ValueError("Kimi K3 parsing supports Chat Completions only.")

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del (
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
        raise ValueError("Kimi K3 parsing supports Chat Completions only.")


class KimiK3Parser(DelegatingParser):
    """K3-local unified parser used by Chat Completions in vLLM 0.23."""

    reasoning_parser_cls = KimiK3ReasoningParser
    tool_parser_cls = KimiK3ToolParser

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(tokenizer, tools, *args, **kwargs)
        template_kwargs = kwargs.get("chat_template_kwargs") or {}
        self._thinking_enabled = _template_uses_thinking(template_kwargs)
        self._stream_call_ids: list[str] = []
        self._stream_parser: KimiK3XTMLParser | None = None

    def _get_stream_parser(self, request: ChatCompletionRequest) -> KimiK3XTMLParser:
        if self._stream_parser is None:
            self._stream_parser = _protocol_parser_for_request(
                request,
                thinking_enabled=self._thinking_enabled,
                tokenizer=self.model_tokenizer,
            )
        return self._stream_parser

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        del enable_auto_tools
        if not isinstance(request, ChatCompletionRequest):
            raise ValueError("Kimi K3 parsing supports Chat Completions only.")

        parser = _protocol_parser_for_request(
            request,
            thinking_enabled=self._thinking_enabled,
            tokenizer=self.model_tokenizer,
        )
        parser.feed(model_output, model_output_token_ids)
        parser.finish()
        snapshot = parser.snapshot
        tool_calls = [
            FunctionCall(
                id=make_tool_call_id(),
                name=call.name,
                arguments=call.arguments,
            )
            for call in snapshot.tool_calls
        ]
        return snapshot.reasoning or None, snapshot.content or None, tool_calls

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        if not isinstance(request, ChatCompletionRequest):
            raise ValueError("Kimi K3 parsing supports Chat Completions only.")

        del prompt_token_ids
        parser = self._get_stream_parser(request)
        protocol_delta = parser.feed(delta_text, delta_token_ids)
        if finished:
            protocol_delta = protocol_delta.merged(parser.finish())
        return _to_delta_message(
            protocol_delta,
            self._stream_call_ids,
            include_reasoning=bool(request.include_reasoning),
        )


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
    if self.parser_cls is not KimiK3Parser:
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
    if self.parser is KimiK3Parser:
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


# vLLM 0.23 can only synthesize a generic DelegatingParser. K3 needs the final
# stream flag for strict envelope validation and must parse tool_choice=none
# through the same XTML state machine instead of passing raw tags through.
if not hasattr(ParserManager, _ORIGINAL_GET_PARSER_ATTR):
    setattr(
        ParserManager,
        _ORIGINAL_GET_PARSER_ATTR,
        ParserManager.get_parser.__func__,
    )


def _get_parser_with_kimi_k3(
    cls,
    tool_parser_name: str | None = None,
    reasoning_parser_name: str | None = None,
    enable_auto_tools: bool = False,
    model_name: str | None = None,
):
    uses_kimi_k3 = tool_parser_name == "kimi_k3" or reasoning_parser_name == "kimi_k3"
    if uses_kimi_k3:
        if tool_parser_name != "kimi_k3" or reasoning_parser_name != "kimi_k3" or not enable_auto_tools:
            raise ValueError(
                "Kimi K3 requires --enable-auto-tool-choice together with "
                "--reasoning-parser kimi_k3 and --tool-call-parser kimi_k3."
            )
        return KimiK3Parser

    original = getattr(ParserManager, _ORIGINAL_GET_PARSER_ATTR)
    return original(
        cls,
        tool_parser_name=tool_parser_name,
        reasoning_parser_name=reasoning_parser_name,
        enable_auto_tools=enable_auto_tools,
        model_name=model_name,
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
