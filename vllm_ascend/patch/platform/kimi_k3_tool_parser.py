# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tool-call parser for the Kimi K3 (XTML) chat format.

Vendored from vLLM commit f5a7cce9b6a61f4d995629a7418c7ea822e34a64.
"""

import json
from collections.abc import Sequence

import regex as re
from openai.types.responses import ToolChoiceFunction
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.logger import logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

_O, _C, _S = r"<\|open\|>", r"<\|close\|>", r"<\|sep\|>"
_TEXT_UNTIL_SEP = r"(?:(?!" + _S + r").)*?"


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_len = min(len(text), len(tag) - 1)
    for n in range(max_len, 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class KimiK3ToolParser(ToolParser):
    supports_required_and_named = False
    structural_tag_model = "kimi_k3"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        super().__init__(tokenizer, tools)

        self.tools_open = "<|open|>tools<|sep|>"
        self.tools_close = "<|close|>tools<|sep|>"
        self.response_open = "<|open|>response<|sep|>"
        self.response_close = "<|close|>response<|sep|>"

        self._tools_open_re = re.compile(_O + r"\s*tools\s*" + _S)
        self._tools_close_re = re.compile(_C + r"\s*tools\s*" + _S)
        self._response_open_re = re.compile(_O + r"\s*response\s*" + _S)
        self._response_close_re = re.compile(_C + r"\s*response\s*" + _S)
        self._message_close_re = re.compile(_C + r"\s*message\s*" + _S)
        self._call_re = re.compile(
            _O + r"\s*call\s+(?P<attrs>" + _TEXT_UNTIL_SEP + r")" + _S + r"(?P<body>.*?)" + _C + r"\s*call\s*" + _S,
            re.DOTALL,
        )
        self._arg_re = re.compile(
            _O
            + r"\s*argument\s+(?P<attrs>"
            + _TEXT_UNTIL_SEP
            + r")"
            + _S
            + r"(?P<val>.*?)"
            + _C
            + r"\s*argument\s*"
            + _S,
            re.DOTALL,
        )
        self._attr_re = re.compile(r'(?P<k>\w+)="(?P<v>[^"]*)"')
        self._response_re = re.compile(
            _O + r"\s*response\s*" + _S + r"(?P<c>.*?)" + _C + r"\s*response\s*" + _S,
            re.DOTALL,
        )

        self._sent_content_idx = 0
        self._sent_tool_call_count = 0

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ToolParser constructor during construction.")

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        named = isinstance(
            request.tool_choice,
            (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
        )
        structured_outputs = getattr(request, "structured_outputs", None)
        has_structural_tag = structured_outputs is not None and structured_outputs.structural_tag is not None
        if named and not has_structural_tag:
            raise VLLMValidationError(
                "Named tool choice for Kimi K3 requires strict tool calling "
                "(VLLM_ENFORCE_STRICT_TOOL_CALLING) so the XTML structural "
                "tag can force the call. Otherwise use `tool_choice` set to "
                '"auto", "required", or "none".',
                parameter="tool_choice",
                value=request.tool_choice,
            )

        if request.tools and (request.tool_choice == "required" or named):
            request.skip_special_tokens = False
            if hasattr(request, "spaces_between_special_tokens"):
                request.spaces_between_special_tokens = False
            return request

        request = super().adjust_request(request)
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return request

    def _attrs(self, value: str) -> dict[str, str]:
        return {
            match["k"]: match["v"].replace("&quot;", '"').replace("&amp;", "&")
            for match in self._attr_re.finditer(value)
        }

    def _decode_call(self, attrs: str, body: str) -> ToolCall | None:
        call_attrs = self._attrs(attrs)
        tool_name = call_attrs.get("tool", "")
        tool_index = call_attrs.get("index", "")
        arguments: dict = {}
        for arg_match in self._arg_re.finditer(body):
            arg_attrs = self._attrs(arg_match["attrs"])
            key = arg_attrs.get("key", "")
            arg_type = arg_attrs.get("type", "string")
            raw_value = arg_match["val"]
            if arg_type == "string":
                arguments[key] = raw_value
            else:
                try:
                    arguments[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    arguments[key] = raw_value
        if not tool_name:
            return None
        tool_call_id = tool_name
        if tool_index:
            try:
                tool_call_id = f"{tool_name}:{int(tool_index) - 1}"
            except ValueError:
                tool_call_id = f"{tool_name}:{tool_index}"
        return ToolCall(
            id=tool_call_id,
            type="function",
            function=FunctionCall(
                name=tool_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def _strip_response_content(self, text: str) -> str | None:
        m_open = self._response_open_re.search(text)
        if m_open is not None:
            m_close = self._response_close_re.search(text, m_open.end())
            if m_close is not None:
                text = text[m_open.end() : m_close.start()]
            else:
                text = text[m_open.end() :]
        else:
            text = self._response_close_re.sub("", text)
        text = self._message_close_re.sub("", text)
        return text or None

    def _content(self, model_output: str, before: str) -> str | None:
        match = self._response_re.search(model_output)
        if match is not None:
            return match["c"] or None
        return self._strip_response_content(before)

    def _extract_response_content(self, current_text: str) -> str | None:
        m_open = self._response_open_re.search(current_text)
        body_start = m_open.end() if m_open is not None else 0
        m_tools = self._tools_open_re.search(current_text, body_start)
        m_rclose = self._response_close_re.search(current_text, body_start)
        tools_start = m_tools.start() if m_tools else -1
        response_end = m_rclose.start() if m_rclose else -1

        candidates = [i for i in (tools_start, response_end) if i != -1]
        if candidates:
            sendable_idx = min(candidates)
        else:
            overlap = max(
                _partial_tag_overlap(current_text, self.response_open),
                _partial_tag_overlap(current_text, self.response_close),
                _partial_tag_overlap(current_text, self.tools_open),
            )
            sendable_idx = len(current_text) - overlap

        if sendable_idx <= body_start:
            return None
        if self._sent_content_idx < body_start:
            self._sent_content_idx = body_start
        if sendable_idx <= self._sent_content_idx:
            return None

        content = current_text[self._sent_content_idx : sendable_idx]
        self._sent_content_idx = sendable_idx
        return content or None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        m_open = self._tools_open_re.search(model_output)
        if m_open is None:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self._content(model_output, model_output),
            )
        try:
            before = model_output[: m_open.start()]
            start = m_open.end()
            m_close = self._tools_close_re.search(model_output, start)
            section = model_output[start:] if m_close is None else model_output[start : m_close.start()]

            tool_calls = [
                tool_call
                for match in self._call_re.finditer(section)
                if (
                    tool_call := self._decode_call(
                        match["attrs"],
                        match["body"],
                    )
                )
                is not None
            ]
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=self._content(model_output, before),
                )
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=self._content(model_output, before),
            )
        except Exception:
            logger.exception("Error extracting K3 tool calls.")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

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
        content = self._extract_response_content(current_text)

        m_tools = self._tools_open_re.search(current_text)
        if m_tools is None:
            return DeltaMessage(content=content) if content else None

        section = current_text[m_tools.end() :]
        calls = [
            tool_call
            for match in self._call_re.finditer(section)
            if (
                tool_call := self._decode_call(
                    match["attrs"],
                    match["body"],
                )
            )
            is not None
        ]
        if len(calls) <= self._sent_tool_call_count:
            return DeltaMessage(content=content) if content else None
        new_calls = calls[self._sent_tool_call_count :]

        deltas = [
            DeltaToolCall(
                index=self._sent_tool_call_count + index,
                id=tool_call.id,
                type="function",
                function=DeltaFunctionCall(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ).model_dump(exclude_none=True),
            )
            for index, tool_call in enumerate(new_calls)
        ]
        self._sent_tool_call_count = len(calls)
        return DeltaMessage(content=content, tool_calls=deltas)
