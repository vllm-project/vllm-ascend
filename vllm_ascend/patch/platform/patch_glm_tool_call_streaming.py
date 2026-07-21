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
# OpenAI chat streaming: backport GLM tool-call parser fixes.
#

from __future__ import annotations

import copy
import json
from typing import Any

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser
from vllm.tool_parsers.utils import partial_tag_overlap


def _schema_allows_string(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return True
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return "string" in schema_type
    if schema_type is not None:
        return schema_type == "string"

    for choice_field in ("anyOf", "oneOf"):
        choices = schema.get(choice_field)
        if isinstance(choices, list) and choices:
            return any(not isinstance(choice, dict) or _schema_allows_string(choice) for choice in choices)
    return True


def _patched_is_string_type(
    tool_name: str,
    arg_name: str,
    tools: list[Any] | None,
) -> bool:
    if tools is None:
        return True
    for tool in tools:
        if tool.function.name != tool_name:
            continue
        if tool.function.parameters is None:
            return True
        schema = tool.function.parameters.get("properties", {}).get(arg_name, {})
        return _schema_allows_string(schema)
    return True


def _patched_build_args_json_so_far(
    self: Glm4MoeModelToolParser,
    tool_name: str,
    inner_text: str,
    is_complete: bool,
) -> str:
    args_so_far = self._ascend_original_build_args_json_so_far(tool_name, inner_text, is_complete)
    if is_complete:
        return args_so_far

    last_val_start = inner_text.rfind(self.arg_val_start)
    last_val_end = inner_text.rfind(self.arg_val_end)
    if last_val_start == -1 or last_val_end > last_val_start:
        return args_so_far

    last_key_match = None
    for match in self._arg_key_pattern.finditer(inner_text[:last_val_start]):
        last_key_match = match
    if last_key_match is None:
        return args_so_far

    partial_key = last_key_match.group(1).strip()
    if self._is_string_type(tool_name, partial_key, self.tools):
        return args_so_far

    partial_content = inner_text[last_val_start + len(self.arg_val_start) :]
    overlap = partial_tag_overlap(partial_content, self.arg_val_end)
    if overlap:
        partial_content = partial_content[:-overlap]
    if partial_content and args_so_far.endswith(partial_content):
        return args_so_far[: -len(partial_content)]
    return args_so_far


def _create_remaining_args_delta(
    delta_message: DeltaMessage,
    remaining_call: str,
    index: int,
    fallback_tool_call_id: str | None = None,
    fallback_tool_call_type: str | None = None,
    fallback_tool_call_name: str | None = None,
) -> DeltaMessage:
    function_kwargs: dict[str, str] = {"arguments": remaining_call}
    if fallback_tool_call_name is not None:
        function_kwargs["name"] = fallback_tool_call_name

    tool_call_kwargs: dict[str, Any] = {
        "index": index,
        "function": DeltaFunctionCall(**function_kwargs),
    }
    if fallback_tool_call_id is not None:
        tool_call_kwargs["id"] = fallback_tool_call_id
    if fallback_tool_call_type is not None:
        tool_call_kwargs["type"] = fallback_tool_call_type

    return DeltaMessage(tool_calls=[DeltaToolCall(**tool_call_kwargs)])


def _terminal_tool_arg_choice(choice: dict[str, Any]) -> bool:
    if choice.get("finish_reason") != "tool_calls":
        return False
    delta = choice.get("delta") or {}
    for tool_call in delta.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        if function.get("arguments"):
            return True
    return False


def _split_terminal_tool_arg_chunk(data: str) -> list[str]:
    prefix = "data: "
    suffix = "\n\n"
    if not data.startswith(prefix):
        return [data]

    payload = data[len(prefix) :]
    if payload.endswith(suffix):
        payload = payload[: -len(suffix)]
    if payload == "[DONE]":
        return [data]

    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return [data]

    choices = chunk.get("choices") or []
    if len(choices) != 1 or not _terminal_tool_arg_choice(choices[0]):
        return [data]

    arg_chunk = copy.deepcopy(chunk)
    arg_choice = arg_chunk["choices"][0]
    arg_choice["finish_reason"] = None
    arg_choice["stop_reason"] = None

    finish_chunk = copy.deepcopy(chunk)
    finish_choice = finish_chunk["choices"][0]
    finish_choice["delta"] = {}

    return [
        f"{prefix}{json.dumps(arg_chunk, ensure_ascii=False)}{suffix}",
        f"{prefix}{json.dumps(finish_chunk, ensure_ascii=False)}{suffix}",
    ]


if not hasattr(OpenAIServingChat, "_ascend_glm_original_chat_completion_stream_generator"):
    OpenAIServingChat._ascend_glm_original_chat_completion_stream_generator = (
        OpenAIServingChat.chat_completion_stream_generator
    )


async def _wrapped_chat_completion_stream_generator(
    self,
    *args,
    **kwargs,
):
    original_stream_generator = self._ascend_glm_original_chat_completion_stream_generator
    async for data in original_stream_generator(*args, **kwargs):
        for chunk in _split_terminal_tool_arg_chunk(data):
            yield chunk


if not hasattr(Glm4MoeModelToolParser, "_ascend_original_build_args_json_so_far"):
    Glm4MoeModelToolParser._ascend_original_build_args_json_so_far = Glm4MoeModelToolParser._build_args_json_so_far


Glm4MoeModelToolParser._is_string_type = staticmethod(_patched_is_string_type)
Glm4MoeModelToolParser._build_args_json_so_far = _patched_build_args_json_so_far
OpenAIServingChat._create_remaining_args_delta = staticmethod(_create_remaining_args_delta)
_wrapped_chat_completion_stream_generator.__module__ = OpenAIServingChat.__module__
_wrapped_chat_completion_stream_generator.__qualname__ = (
    f"{OpenAIServingChat.__qualname__}.chat_completion_stream_generator"
)
OpenAIServingChat.chat_completion_stream_generator = _wrapped_chat_completion_stream_generator
