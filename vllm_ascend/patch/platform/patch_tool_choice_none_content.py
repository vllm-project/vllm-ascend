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
# OpenAI forced tool choice: tolerate None content after reasoning extraction.
#

from __future__ import annotations

from openai.types.responses import ToolChoiceFunction
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.parser.abstract_parser import DelegatingParser


def _normalize_tool_choice_content(
    request,
    content: str | None,
) -> str | None:
    if content is not None:
        return content

    tool_choice = getattr(request, "tool_choice", None)
    if isinstance(tool_choice, ToolChoiceFunction | ChatCompletionNamedToolChoiceParam):
        return ""
    return content


_original_parse_tool_calls_from_content = OpenAIServing._parse_tool_calls_from_content


def _patched_parse_tool_calls_from_content(
    request,
    tokenizer,
    enable_auto_tools: bool,
    tool_parser_cls,
    content: str | None = None,
):
    content = _normalize_tool_choice_content(request, content)
    return _original_parse_tool_calls_from_content(
        request=request,
        tokenizer=tokenizer,
        enable_auto_tools=enable_auto_tools,
        tool_parser_cls=tool_parser_cls,
        content=content,
    )


OpenAIServing._parse_tool_calls_from_content = staticmethod(
    _patched_parse_tool_calls_from_content
)

_original_delegating_parse_tool_calls = DelegatingParser._parse_tool_calls


def _patched_delegating_parse_tool_calls(
    self,
    request,
    content: str | None,
    enable_auto_tools: bool,
):
    content = _normalize_tool_choice_content(request, content)
    return _original_delegating_parse_tool_calls(
        self,
        request,
        content,
        enable_auto_tools,
    )


DelegatingParser._parse_tool_calls = _patched_delegating_parse_tool_calls
