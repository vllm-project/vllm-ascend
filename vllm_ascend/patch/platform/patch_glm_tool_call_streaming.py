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
# OpenAI chat streaming: backport GLM tool-call final chunk fixes.
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


def _create_remaining_args_delta(
    delta_message: DeltaMessage,
    remaining_call: str,
    index: int,
    fallback_tool_call_id: str | None = None,
    fallback_tool_call_type: str | None = None,
    fallback_tool_call_name: str | None = None,
) -> DeltaMessage:
    if remaining_call == "":
        return delta_message

    original_tool_call = next(
        (tool_call for tool_call in delta_message.tool_calls if tool_call.index == index),
        None,
    )
    original_function = original_tool_call.function if original_tool_call else None

    function_kwargs: dict[str, str] = {"arguments": remaining_call}
    function_name = original_function.name if original_function else None
    if function_name is None:
        function_name = fallback_tool_call_name
    if function_name is not None:
        function_kwargs["name"] = function_name

    tool_call_kwargs: dict[str, Any] = {
        "index": index,
        "function": DeltaFunctionCall(**function_kwargs),
    }
    tool_call_id = original_tool_call.id if original_tool_call else None
    if tool_call_id is None:
        tool_call_id = fallback_tool_call_id
    if tool_call_id is not None:
        tool_call_kwargs["id"] = tool_call_id
    tool_call_type = original_tool_call.type if original_tool_call else None
    if tool_call_type is None:
        tool_call_type = fallback_tool_call_type
    if tool_call_type is not None:
        tool_call_kwargs["type"] = tool_call_type

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


OpenAIServingChat._create_remaining_args_delta = staticmethod(_create_remaining_args_delta)
_wrapped_chat_completion_stream_generator.__module__ = OpenAIServingChat.__module__
_wrapped_chat_completion_stream_generator.__qualname__ = (
    f"{OpenAIServingChat.__qualname__}.chat_completion_stream_generator"
)
OpenAIServingChat.chat_completion_stream_generator = _wrapped_chat_completion_stream_generator
