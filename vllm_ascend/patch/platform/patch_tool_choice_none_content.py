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

_NO_FORCED_TOOL_CALL = "_vllm_ascend_no_forced_tool_call"


def _is_forced_tool_choice(request) -> bool:
    tool_choice = getattr(request, "tool_choice", None)
    return isinstance(
        tool_choice,
        (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam),
    )


def _set_no_forced_tool_call(request, value: bool) -> None:
    tool_choice = getattr(request, "tool_choice", None)
    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        setattr(tool_choice, _NO_FORCED_TOOL_CALL, value)


def _patch_named_tool_choice_bool() -> None:
    if getattr(ChatCompletionNamedToolChoiceParam, "_vllm_ascend_bool_patched", False):
        return

    original_bool = getattr(ChatCompletionNamedToolChoiceParam, "__bool__", None)

    def _patched_named_tool_choice_bool(self) -> bool:
        if getattr(self, _NO_FORCED_TOOL_CALL, False):
            return False
        if original_bool is not None:
            return original_bool(self)
        return True

    ChatCompletionNamedToolChoiceParam.__bool__ = _patched_named_tool_choice_bool
    ChatCompletionNamedToolChoiceParam._vllm_ascend_bool_patched = True


_patch_named_tool_choice_bool()


_original_parse_tool_calls_from_content = OpenAIServing._parse_tool_calls_from_content


def _patched_parse_tool_calls_from_content(
    request,
    tokenizer,
    enable_auto_tools: bool,
    tool_parser_cls,
    content: str | None = None,
):
    if content is None and _is_forced_tool_choice(request):
        _set_no_forced_tool_call(request, True)
        return [], None

    _set_no_forced_tool_call(request, False)
    return _original_parse_tool_calls_from_content(
        request=request,
        tokenizer=tokenizer,
        enable_auto_tools=enable_auto_tools,
        tool_parser_cls=tool_parser_cls,
        content=content,
    )


OpenAIServing._parse_tool_calls_from_content = staticmethod(_patched_parse_tool_calls_from_content)

_original_delegating_parse_tool_calls = DelegatingParser._parse_tool_calls


def _patched_delegating_parse_tool_calls(
    self,
    request,
    content: str | None,
    enable_auto_tools: bool,
):
    if content is None and _is_forced_tool_choice(request):
        return [], None

    return _original_delegating_parse_tool_calls(
        self,
        request,
        content,
        enable_auto_tools,
    )


DelegatingParser._parse_tool_calls = _patched_delegating_parse_tool_calls
