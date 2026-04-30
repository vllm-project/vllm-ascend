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
# OpenAI chat streaming: backport GLM tool-call parser and finish-chunk fixes.
#

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion import serving as chat_serving
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers import glm4_moe_tool_parser as glm4_parser
from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser

logger = chat_serving.logger


def _ensure_streaming_attrs(self: Glm4MoeModelToolParser) -> None:
    if not hasattr(self, "_buffer"):
        self._buffer = ""
    if not hasattr(self, "_in_tool_call"):
        self._in_tool_call = False
    if not hasattr(self, "_current_tool_name"):
        self._current_tool_name = None
    if not hasattr(self, "_pending_key"):
        self._pending_key = None
    if not hasattr(self, "_streaming_string_value"):
        self._streaming_string_value = False
    if not hasattr(self, "_args_started"):
        self._args_started = []
    if not hasattr(self, "_args_closed"):
        self._args_closed = []
    if not hasattr(self, "_seen_keys"):
        self._seen_keys = []


def _ensure_tool_state(self: Glm4MoeModelToolParser) -> None:
    while len(self._tool_call_ids) <= self.current_tool_id:
        self._tool_call_ids.append(make_tool_call_id(id_type="random", func_name=None, idx=None))
    while len(self.streamed_args_for_tool) <= self.current_tool_id:
        self.streamed_args_for_tool.append("")
    while len(self.prev_tool_call_arr) <= self.current_tool_id:
        self.prev_tool_call_arr.append({})
    while len(self._args_started) <= self.current_tool_id:
        self._args_started.append(False)
    while len(self._args_closed) <= self.current_tool_id:
        self._args_closed.append(False)
    while len(self._seen_keys) <= self.current_tool_id:
        self._seen_keys.append(set())


def _begin_tool_call(self: Glm4MoeModelToolParser) -> None:
    _ensure_streaming_attrs(self)
    if self.current_tool_id == -1:
        self.current_tool_id = 0
    else:
        self.current_tool_id += 1
    self._ensure_tool_state()
    self.current_tool_name_sent = False
    self._current_tool_name = None
    self._pending_key = None
    self._streaming_string_value = False
    self._in_tool_call = True


def _finish_tool_call(self: Glm4MoeModelToolParser) -> None:
    self._in_tool_call = False
    self._current_tool_name = None
    self._pending_key = None
    self._streaming_string_value = False


def _revert_last_tool_call_state(self: Glm4MoeModelToolParser) -> None:
    if self.current_tool_id < 0:
        return
    self._tool_call_ids.pop()
    self.streamed_args_for_tool.pop()
    self.prev_tool_call_arr.pop()
    self._args_started.pop()
    self._args_closed.pop()
    self._seen_keys.pop()
    self.current_tool_id -= 1


def _emit_tool_name_delta(
    self: Glm4MoeModelToolParser,
    tool_name: str,
) -> DeltaMessage:
    self.prev_tool_call_arr[self.current_tool_id] = {
        "name": self._current_tool_name,
        "arguments": {},
    }
    return DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=self.current_tool_id,
                id=self._tool_call_ids[self.current_tool_id],
                type="function",
                function=DeltaFunctionCall(
                    name=tool_name,
                    arguments="",
                ).model_dump(exclude_none=True),
            )
        ]
    )


def _emit_tool_args_delta(
    self: Glm4MoeModelToolParser,
    fragment: str,
) -> DeltaMessage:
    return DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=self.current_tool_id,
                function=DeltaFunctionCall(arguments=fragment).model_dump(exclude_none=True),
            )
        ]
    )


def _append_arg_fragment(
    self: Glm4MoeModelToolParser,
    *,
    key: str,
    raw_val: str,
) -> str | None:
    key = key.strip()
    if not key:
        return None
    if key in self._seen_keys[self.current_tool_id]:
        return None

    val_obj: Any = self._deserialize(raw_val)
    key_json = json.dumps(key, ensure_ascii=False)
    val_json = json.dumps(val_obj, ensure_ascii=False)

    if not self._args_started[self.current_tool_id]:
        fragment = "{" + key_json + ":" + val_json
        self._args_started[self.current_tool_id] = True
    else:
        fragment = "," + key_json + ":" + val_json

    self._seen_keys[self.current_tool_id].add(key)
    self.streamed_args_for_tool[self.current_tool_id] += fragment
    return fragment


def _close_args_if_needed(self: Glm4MoeModelToolParser) -> str | None:
    if self._args_closed[self.current_tool_id]:
        return None
    self._args_closed[self.current_tool_id] = True
    if not self._args_started[self.current_tool_id]:
        fragment = "{}"
        self.streamed_args_for_tool[self.current_tool_id] = fragment
    else:
        fragment = "}"
        self.streamed_args_for_tool[self.current_tool_id] += fragment
    return fragment


def _create_remaining_args_delta(
    delta_message: DeltaMessage,
    remaining_call: str,
    index: int,
    fallback_tool_call_id: str | None = None,
    fallback_tool_call_type: str | None = None,
    fallback_tool_call_name: str | None = None,
) -> DeltaMessage:
    include_header = any(
        v is not None
        for v in (
            fallback_tool_call_id,
            fallback_tool_call_type,
            fallback_tool_call_name,
        )
    )
    if not include_header:
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    function=DeltaFunctionCall(arguments=remaining_call),
                )
            ]
        )

    return DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=index,
                id=fallback_tool_call_id,
                type=fallback_tool_call_type,
                function=DeltaFunctionCall(
                    name=fallback_tool_call_name,
                    arguments=remaining_call,
                ),
            )
        ]
    )


def _merge_delta_messages(
    pending_delta: DeltaMessage | None,
    message: DeltaMessage | None,
) -> DeltaMessage | None:
    if message is None:
        return pending_delta
    if pending_delta is None:
        return message.model_copy(deep=True)

    if message.content:
        pending_delta.content = (pending_delta.content or "") + message.content
    if message.reasoning:
        pending_delta.reasoning = (pending_delta.reasoning or "") + message.reasoning

    for tool_call in message.tool_calls:
        for idx, existing_tool_call in enumerate(pending_delta.tool_calls):
            if existing_tool_call.index != tool_call.index:
                continue
            pending_delta.tool_calls[idx] = _merge_tool_call(
                existing_tool_call,
                tool_call,
            )
            break
        else:
            pending_delta.tool_calls = [
                *pending_delta.tool_calls,
                tool_call.model_copy(deep=True),
            ]
    return pending_delta


def _normalize_function_call(
    function: DeltaFunctionCall | dict[str, Any] | None,
) -> DeltaFunctionCall | None:
    if function is None:
        return None
    if isinstance(function, DeltaFunctionCall):
        return function.model_copy(deep=True)
    return DeltaFunctionCall.model_validate(function)


def _merge_tool_call(
    existing_tool_call: DeltaToolCall,
    tool_call: DeltaToolCall,
) -> DeltaToolCall:
    existing_function = _normalize_function_call(existing_tool_call.function)
    incoming_function = _normalize_function_call(tool_call.function)
    merged_name = (
        incoming_function.name
        if incoming_function and incoming_function.name is not None
        else existing_function.name
        if existing_function
        else None
    )
    merged_arguments = None
    if (existing_function and existing_function.arguments is not None) or (
        incoming_function and incoming_function.arguments is not None
    ):
        merged_arguments = ((existing_function.arguments or "") if existing_function else "") + (
            (incoming_function.arguments or "") if incoming_function else ""
        )

    merged_function = None
    if merged_name is not None or merged_arguments is not None:
        merged_function = DeltaFunctionCall(
            name=merged_name,
            arguments=merged_arguments,
        ).model_dump(exclude_none=True)

    return DeltaToolCall(
        index=existing_tool_call.index,
        id=tool_call.id if tool_call.id is not None else existing_tool_call.id,
        type=tool_call.type if tool_call.type is not None else existing_tool_call.type,
        function=merged_function,
    )


def _flush_pending_delta(pending_delta: DeltaMessage | None) -> DeltaMessage | None:
    if pending_delta is None:
        return None
    if pending_delta.content is None and pending_delta.reasoning is None and not pending_delta.tool_calls:
        return None
    return pending_delta


def _patched_extract_tool_calls_streaming(
    self: Glm4MoeModelToolParser,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
    delta_token_ids: Sequence[int],
    request: ChatCompletionRequest,
) -> DeltaMessage | None:
    if not self._tools_enabled(request):
        return DeltaMessage(content=delta_text) if delta_text else None

    _ensure_streaming_attrs(self)
    self._buffer += delta_text
    pending_delta: DeltaMessage | None = None

    while True:
        if not self._in_tool_call:
            start_idx = self._buffer.find(self.tool_call_start_token)
            if start_idx == -1:
                for i in range(1, len(self.tool_call_start_token)):
                    if self._buffer.endswith(self.tool_call_start_token[:i]):
                        out = self._buffer[:-i]
                        self._buffer = self._buffer[-i:]
                        pending_delta = _merge_delta_messages(
                            pending_delta,
                            DeltaMessage(content=out) if out else None,
                        )
                        return _flush_pending_delta(pending_delta)
                out = self._buffer
                self._buffer = ""
                pending_delta = _merge_delta_messages(
                    pending_delta,
                    DeltaMessage(content=out) if out else None,
                )
                return _flush_pending_delta(pending_delta)

            if start_idx > 0:
                out = self._buffer[:start_idx]
                self._buffer = self._buffer[start_idx:]
                pending_delta = _merge_delta_messages(
                    pending_delta,
                    DeltaMessage(content=out) if out else None,
                )
                continue

            self._buffer = self._buffer[len(self.tool_call_start_token) :]
            self._begin_tool_call()
            continue

        if not self.current_tool_name_sent:
            nl = self._buffer.find("\n")
            ak = self._buffer.find(self.arg_key_start)
            end = self._buffer.find(self.tool_call_end_token)
            candidates = [i for i in [nl, ak, end] if i != -1]
            if not candidates:
                return _flush_pending_delta(pending_delta)
            cut = min(candidates)
            tool_name = self._buffer[:cut].strip()
            if tool_name == "" and cut == end:
                self._buffer = self._buffer[end + len(self.tool_call_end_token) :]
                self._finish_tool_call()
                self._revert_last_tool_call_state()
                continue

            if cut == nl:
                self._buffer = self._buffer[nl + 1 :]
            else:
                self._buffer = self._buffer[cut:]

            self._current_tool_name = tool_name
            self.current_tool_name_sent = True
            pending_delta = _merge_delta_messages(
                pending_delta,
                self._emit_tool_name_delta(tool_name),
            )
            continue

        assert self._current_tool_name is not None

        if self._streaming_string_value:
            val_end = self._buffer.find(self.arg_val_end)
            if val_end != -1:
                raw_content = self._buffer[:val_end]
                self._buffer = self._buffer[val_end + len(self.arg_val_end) :]
                self._streaming_string_value = False
                self._pending_key = None

                escaped = self._json_escape_string_content(raw_content)
                frag = escaped + '"'
                self.streamed_args_for_tool[self.current_tool_id] += frag
                pending_delta = _merge_delta_messages(
                    pending_delta,
                    self._emit_tool_args_delta(frag),
                )
                continue

            safe_len = len(self._buffer)
            for i in range(1, len(self.arg_val_end)):
                if self._buffer.endswith(self.arg_val_end[:i]):
                    safe_len = len(self._buffer) - i
                    break

            if safe_len > 0:
                to_emit = self._buffer[:safe_len]
                self._buffer = self._buffer[safe_len:]
                escaped = self._json_escape_string_content(to_emit)
                if escaped:
                    self.streamed_args_for_tool[self.current_tool_id] += escaped
                    pending_delta = _merge_delta_messages(
                        pending_delta,
                        self._emit_tool_args_delta(escaped),
                    )
            return _flush_pending_delta(pending_delta)

        if self._pending_key is not None:
            val_pos = self._buffer.find(self.arg_val_start)
            if val_pos == -1:
                return _flush_pending_delta(pending_delta)
            if val_pos > 0:
                self._buffer = self._buffer[val_pos:]

            key = (self._pending_key or "").strip()
            is_string = self._is_string_type(self._current_tool_name, key, request.tools)

            if is_string:
                self._buffer = self._buffer[len(self.arg_val_start) :]

                if key in self._seen_keys[self.current_tool_id]:
                    self._pending_key = None
                    continue

                self._seen_keys[self.current_tool_id].add(key)
                key_json = json.dumps(key, ensure_ascii=False)

                if not self._args_started[self.current_tool_id]:
                    frag = "{" + key_json + ':"'
                    self._args_started[self.current_tool_id] = True
                else:
                    frag = "," + key_json + ':"'

                self.streamed_args_for_tool[self.current_tool_id] += frag
                self._streaming_string_value = True
                pending_delta = _merge_delta_messages(
                    pending_delta,
                    self._emit_tool_args_delta(frag),
                )
                continue

            val_end = self._buffer.find(self.arg_val_end)
            if val_end == -1:
                return _flush_pending_delta(pending_delta)

            raw_val = self._buffer[len(self.arg_val_start) : val_end].strip()
            self._buffer = self._buffer[val_end + len(self.arg_val_end) :]
            self._pending_key = None

            frag = self._append_arg_fragment(key=key, raw_val=raw_val)
            if frag:
                pending_delta = _merge_delta_messages(
                    pending_delta,
                    self._emit_tool_args_delta(frag),
                )
            continue

        end_pos = self._buffer.find(self.tool_call_end_token)
        key_pos = self._buffer.find(self.arg_key_start)
        if end_pos != -1 and (key_pos == -1 or end_pos < key_pos):
            self._buffer = self._buffer[end_pos + len(self.tool_call_end_token) :]
            frag = self._close_args_if_needed()
            if self._current_tool_name:
                try:
                    full_args_str = self.streamed_args_for_tool[self.current_tool_id]
                    args_dict = json.loads(full_args_str)
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": self._current_tool_name,
                        "arguments": args_dict,
                    }
                except (json.JSONDecodeError, IndexError) as e:
                    glm4_parser.logger.warning(
                        "Failed to finalize tool call state for tool %d: %s",
                        self.current_tool_id,
                        e,
                    )
            self._finish_tool_call()
            pending_delta = _merge_delta_messages(
                pending_delta,
                self._emit_tool_args_delta(frag) if frag else None,
            )
            continue

        if key_pos == -1:
            return _flush_pending_delta(pending_delta)
        if key_pos > 0:
            self._buffer = self._buffer[key_pos:]
        key_end = self._buffer.find(self.arg_key_end)
        if key_end == -1:
            return _flush_pending_delta(pending_delta)
        key = self._buffer[len(self.arg_key_start) : key_end]
        self._buffer = self._buffer[key_end + len(self.arg_key_end) :]
        self._pending_key = key


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
Glm4MoeModelToolParser._ensure_tool_state = _ensure_tool_state
Glm4MoeModelToolParser._begin_tool_call = _begin_tool_call
Glm4MoeModelToolParser._finish_tool_call = _finish_tool_call
Glm4MoeModelToolParser._revert_last_tool_call_state = _revert_last_tool_call_state
Glm4MoeModelToolParser._emit_tool_name_delta = _emit_tool_name_delta
Glm4MoeModelToolParser._emit_tool_args_delta = _emit_tool_args_delta
Glm4MoeModelToolParser._append_arg_fragment = _append_arg_fragment
Glm4MoeModelToolParser._close_args_if_needed = _close_args_if_needed
Glm4MoeModelToolParser.extract_tool_calls_streaming = _patched_extract_tool_calls_streaming
