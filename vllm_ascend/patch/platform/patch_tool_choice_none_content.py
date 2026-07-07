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
# OpenAI chat completions: omit empty tool_calls in serialized payloads.
#

from __future__ import annotations

import json
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)

_original_chat_completion_response_model_dump = ChatCompletionResponse.model_dump
_original_chat_completion_stream_response_model_dump = ChatCompletionStreamResponse.model_dump


def _omit_empty_tool_calls(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for field_name in ("message", "delta"):
            message = choice.get(field_name)
            if isinstance(message, dict) and message.get("tool_calls") == []:
                message.pop("tool_calls")

    return payload


def _patched_chat_completion_response_model_dump(self, *args, **kwargs):
    return _omit_empty_tool_calls(_original_chat_completion_response_model_dump(self, *args, **kwargs))


def _dump_json(payload: Any, indent: int | None, ensure_ascii: bool) -> str:
    separators = None if indent is not None else (",", ":")
    return json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent, separators=separators)


def _patched_chat_completion_response_model_dump_json(self, *args, **kwargs):
    dump_kwargs = dict(kwargs)
    indent = dump_kwargs.pop("indent", None)
    ensure_ascii = dump_kwargs.pop("ensure_ascii", False)
    dump_kwargs.setdefault("mode", "json")
    payload = _patched_chat_completion_response_model_dump(self, *args, **dump_kwargs)
    return _dump_json(payload, indent, ensure_ascii)


def _patched_chat_completion_stream_response_model_dump(self, *args, **kwargs):
    return _omit_empty_tool_calls(_original_chat_completion_stream_response_model_dump(self, *args, **kwargs))


def _patched_chat_completion_stream_response_model_dump_json(self, *args, **kwargs):
    dump_kwargs = dict(kwargs)
    indent = dump_kwargs.pop("indent", None)
    ensure_ascii = dump_kwargs.pop("ensure_ascii", False)
    dump_kwargs.setdefault("mode", "json")
    payload = _patched_chat_completion_stream_response_model_dump(self, *args, **dump_kwargs)
    return _dump_json(payload, indent, ensure_ascii)


ChatCompletionResponse.model_dump = _patched_chat_completion_response_model_dump
ChatCompletionResponse.model_dump_json = _patched_chat_completion_response_model_dump_json
ChatCompletionStreamResponse.model_dump = _patched_chat_completion_stream_response_model_dump
ChatCompletionStreamResponse.model_dump_json = _patched_chat_completion_stream_response_model_dump_json
