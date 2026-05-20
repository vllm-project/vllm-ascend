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
# DeepSeek V4 tool-call streaming parser compatibility patch.
#

from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

import regex as re
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

ESCAPED_ARGUMENTS_PARAM_NAME = "__vllm_param_arguments__"


def _partial_tag_overlap(text: str, tag: str) -> int:
    max_overlap = min(len(text), len(tag) - 1)
    for overlap in range(max_overlap, 0, -1):
        if text.endswith(tag[:overlap]):
            return overlap
    return 0


def _ensure_streaming_attrs(self: DeepSeekV4ToolParser) -> None:
    if not hasattr(self, "_buffer"):
        self._buffer = ""
    if not hasattr(self, "_in_tool_calls"):
        self._in_tool_calls = False
    if not hasattr(self, "_active_tool_index"):
        self._active_tool_index = None
    if not hasattr(self, "_active_tool_name"):
        self._active_tool_name = None
    if not hasattr(self, "_streaming_param_mode"):
        self._streaming_param_mode = None
    if not hasattr(self, "_streaming_param_key"):
        self._streaming_param_key = None
    if not hasattr(self, "_streaming_param_raw_parts"):
        self._streaming_param_raw_parts = []
    if not hasattr(self, "_args_started"):
        self._args_started = []
    if not hasattr(self, "_pending_delta_messages"):
        self._pending_delta_messages = deque()

    if not hasattr(self, "tool_call_complete_regex"):
        self.tool_call_complete_regex = re.compile(
            re.escape(self.tool_call_start_token) + r"(.*?)" + re.escape(self.tool_call_end_token),
            re.DOTALL,
        )
    if not hasattr(self, "invoke_complete_regex"):
        self.invoke_complete_regex = re.compile(
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>',
            re.DOTALL,
        )
    if not hasattr(self, "parameter_complete_regex"):
        self.parameter_complete_regex = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>',
            re.DOTALL,
        )
    if not hasattr(self, "parameter_start_regex"):
        self.parameter_start_regex = re.compile(r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>')
    if not hasattr(self, "invoke_start_regex"):
        self.invoke_start_regex = re.compile(r'<｜DSML｜invoke\s+name="([^"]+)"\s*>')

    if not hasattr(self, "current_tool_index"):
        self.current_tool_index = 0
    if not hasattr(self, "prev_tool_call_arr"):
        self.prev_tool_call_arr = []
    if not hasattr(self, "streamed_args_for_tool"):
        self.streamed_args_for_tool = []


def _function_name(tool) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return function.get("name")
        return getattr(function, "name", None)
    return getattr(getattr(tool, "function", None), "name", None)


def _function_parameters(tool):
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return function.get("parameters")
        return getattr(function, "parameters", None)
    return getattr(getattr(tool, "function", None), "parameters", None)


def _convert_param_value_checked(value: str, param_type: str) -> Any:
    if value.lower() == "null":
        return None

    param_type = param_type.lower()
    if param_type in ["string", "str", "text"]:
        return value
    if param_type in ["integer", "int"]:
        return int(value)
    if param_type in ["number", "float"]:
        val = float(value)
        return val if val != int(val) else int(val)
    if param_type in ["boolean", "bool"]:
        value = value.strip()
        if value.lower() not in ["false", "0", "true", "1"]:
            raise ValueError("Invalid boolean value")
        return value.lower() in ["true", "1"]
    if param_type in ["object", "array"]:
        return json.loads(value)
    return json.loads(value)


def _convert_param_value(self: DeepSeekV4ToolParser, value: str, param_type) -> Any:
    if not isinstance(param_type, list):
        param_type = [param_type]
    for current_type in param_type:
        try:
            return _convert_param_value_checked(value, current_type)
        except Exception:
            continue
    return value


def _extract_param_name(param_name: str) -> str:
    if param_name == ESCAPED_ARGUMENTS_PARAM_NAME:
        return "arguments"
    return param_name


def _get_param_config(self: DeepSeekV4ToolParser, request, function_name):
    if not request or not request.tools or not function_name:
        return {}
    for tool in request.tools:
        if _function_name(tool) != function_name:
            continue
        params = _function_parameters(tool)
        if isinstance(params, dict):
            properties = params.get("properties")
            if isinstance(properties, dict):
                return properties
        return {}
    return {}


def _coerce_param_value(
    self: DeepSeekV4ToolParser,
    value: str,
    *,
    string_attr: str,
    param_type,
):
    if string_attr == "true":
        return value
    if param_type:
        return _convert_param_value(self, value, param_type)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _repair_param_dict(
    param_dict: dict,
    param_config: dict[str, dict],
) -> dict:
    allowed = set(param_config.keys())
    for wrapper in ("arguments", "input"):
        if set(param_dict.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = param_dict[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return param_dict
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return inner
    return param_dict


def _parse_invoke_params(
    self: DeepSeekV4ToolParser,
    invoke_str: str,
    request: ChatCompletionRequest | None = None,
    function_name: str | None = None,
) -> dict:
    param_config = _get_param_config(self, request, function_name)
    param_dict = {}
    for param_name, string_attr, param_val in self.parameter_complete_regex.findall(invoke_str):
        original_param_name = param_name
        param_name = _extract_param_name(param_name)
        param_type = None
        if original_param_name == ESCAPED_ARGUMENTS_PARAM_NAME and "arguments" in param_config:
            param_type = param_config["arguments"].get("type")
        elif param_name in param_config and isinstance(param_config[param_name], dict):
            param_type = param_config[param_name].get("type")

        param_dict[param_name] = _coerce_param_value(
            self,
            param_val,
            string_attr=string_attr,
            param_type=param_type,
        )

    return _repair_param_dict(param_dict, param_config)


def _reset_streaming_state(self: DeepSeekV4ToolParser) -> None:
    _ensure_streaming_attrs(self)
    self.current_tool_index = 0
    self._buffer = ""
    self._in_tool_calls = False
    self._active_tool_index = None
    self._active_tool_name = None
    self._streaming_param_mode = None
    self._streaming_param_key = None
    self._streaming_param_raw_parts.clear()
    self.prev_tool_call_arr.clear()
    self.streamed_args_for_tool.clear()
    self._pending_delta_messages.clear()
    self._args_started.clear()


def _json_escape_string_content(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


def _drain_pending_tool_call_deltas(self: DeepSeekV4ToolParser):
    while self._pending_delta_messages:
        yield self._pending_delta_messages.popleft()


def _pop_pending_delta_message(self: DeepSeekV4ToolParser) -> DeltaMessage | None:
    if not self._pending_delta_messages:
        return None

    content_parts = []
    merged_tool_calls: dict[int, DeltaToolCall] = {}
    while self._pending_delta_messages:
        message = self._pending_delta_messages.popleft()
        if message.content:
            content_parts.append(message.content)
        for tool_call in message.tool_calls or []:
            index = tool_call.index
            function = tool_call.function
            if index not in merged_tool_calls:
                merged_tool_calls[index] = DeltaToolCall(
                    index=index,
                    id=tool_call.id,
                    type=tool_call.type,
                    function=DeltaFunctionCall(
                        name=function.name if function else None,
                        arguments=function.arguments if function else None,
                    ),
                )
                continue

            merged = merged_tool_calls[index]
            if tool_call.id is not None:
                merged.id = tool_call.id
            if tool_call.type is not None:
                merged.type = tool_call.type
            if function is None:
                continue
            if merged.function is None:
                merged.function = DeltaFunctionCall()
            if function.name is not None:
                merged.function.name = function.name
            if function.arguments is not None:
                merged.function.arguments = (merged.function.arguments or "") + function.arguments

    content = "".join(content_parts) or None
    return DeltaMessage(content=content, tool_calls=list(merged_tool_calls.values()))


def _queue_delta_message(self: DeepSeekV4ToolParser, message: DeltaMessage | None) -> None:
    if message is not None:
        self._pending_delta_messages.append(message)


def _emit_tool_name_delta(self: DeepSeekV4ToolParser, index: int, name: str) -> DeltaMessage:
    return DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=index,
                id=self._generate_tool_call_id(),
                function=DeltaFunctionCall(name=name, arguments=""),
                type="function",
            )
        ]
    )


def _emit_tool_args_delta(self: DeepSeekV4ToolParser, index: int, arguments: str) -> DeltaMessage | None:
    if not arguments:
        return None
    self.streamed_args_for_tool[index] += arguments
    return DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=index,
                function=DeltaFunctionCall(arguments=arguments),
            )
        ]
    )


def _begin_streaming_tool_call(self: DeepSeekV4ToolParser, name: str) -> None:
    self._active_tool_index = self.current_tool_index
    self._active_tool_name = name
    self.current_tool_index += 1
    self.prev_tool_call_arr.append({"name": name, "arguments": {}})
    self.streamed_args_for_tool.append("")
    self._args_started.append(False)
    self._queue_delta_message(self._emit_tool_name_delta(self._active_tool_index, name))


def _append_param_prefix(self: DeepSeekV4ToolParser, index: int, key: str, *, is_string: bool) -> None:
    key_json = json.dumps(key, ensure_ascii=False)
    prefix = "{" if not self._args_started[index] else ","
    frag = prefix + key_json + ":"
    if is_string:
        frag += '"'
    self._args_started[index] = True
    self._queue_delta_message(self._emit_tool_args_delta(index, frag))


def _append_json_param_value(self: DeepSeekV4ToolParser, index: int, key: str, value: Any) -> None:
    key_json = json.dumps(key, ensure_ascii=False)
    value_json = json.dumps(value, ensure_ascii=False)
    prefix = "{" if not self._args_started[index] else ","
    self._args_started[index] = True
    self._queue_delta_message(self._emit_tool_args_delta(index, prefix + key_json + ":" + value_json))


def _append_raw_param_value(
    self: DeepSeekV4ToolParser,
    index: int,
    key: str,
    raw_value: str,
    *,
    is_string: bool,
) -> None:
    _append_param_prefix(self, index, key, is_string=is_string)
    if is_string:
        frag = _json_escape_string_content(raw_value) + '"'
    else:
        frag = raw_value
    self._queue_delta_message(self._emit_tool_args_delta(index, frag))


def _should_buffer_wrapper_param(self: DeepSeekV4ToolParser, key: str, request: ChatCompletionRequest | None) -> bool:
    if self._args_started[self._active_tool_index]:
        return False
    param_config = _get_param_config(self, request, self._active_tool_name)
    return bool(param_config and key in ("arguments", "input") and key not in param_config)


def _finish_buffered_wrapper_param(
    self: DeepSeekV4ToolParser,
    index: int,
    request: ChatCompletionRequest | None,
) -> None:
    key = self._streaming_param_key
    if key is None:
        return

    raw_value = "".join(self._streaming_param_raw_parts)
    is_string = self._streaming_param_mode == "wrapper_string"
    value: Any = raw_value
    if not is_string:
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value

    param_dict = {key: value}
    param_config = _get_param_config(self, request, self._active_tool_name)
    repaired = _repair_param_dict(param_dict, param_config)
    if isinstance(repaired, dict) and repaired is not param_dict:
        for repaired_key, repaired_value in repaired.items():
            _append_json_param_value(self, index, repaired_key, repaired_value)
    else:
        _append_raw_param_value(self, index, key, raw_value, is_string=is_string)

    self._streaming_param_key = None
    self._streaming_param_raw_parts.clear()


def _close_streaming_tool_call(self: DeepSeekV4ToolParser) -> None:
    index = self._active_tool_index
    if index is None:
        return

    suffix = "}" if self._args_started[index] else "{}"
    self._queue_delta_message(self._emit_tool_args_delta(index, suffix))
    with suppress(json.JSONDecodeError, IndexError):
        self.prev_tool_call_arr[index] = {
            "name": self._active_tool_name,
            "arguments": json.loads(self.streamed_args_for_tool[index]),
        }

    self._active_tool_index = None
    self._active_tool_name = None
    self._streaming_param_mode = None
    self._streaming_param_key = None
    self._streaming_param_raw_parts.clear()


def _safe_content_len_before_tag_end(self: DeepSeekV4ToolParser) -> int:
    safe_len = len(self._buffer)
    parameter_end_token = "</｜DSML｜parameter>"
    for overlap in range(1, len(parameter_end_token)):
        if self._buffer.endswith(parameter_end_token[:overlap]):
            safe_len = len(self._buffer) - overlap
            break
    return safe_len


def _process_streaming_buffer(self: DeepSeekV4ToolParser, request: ChatCompletionRequest | None) -> None:
    parameter_end_token = "</｜DSML｜parameter>"
    invoke_end_token = "</｜DSML｜invoke>"

    while True:
        if not self._in_tool_calls:
            start_idx = self._buffer.find(self.tool_call_start_token)
            if start_idx == -1:
                overlap = _partial_tag_overlap(self._buffer, self.tool_call_start_token)
                sendable_idx = len(self._buffer) - overlap
                if sendable_idx > 0:
                    content = self._buffer[:sendable_idx]
                    self._buffer = self._buffer[sendable_idx:]
                    self._queue_delta_message(DeltaMessage(content=content))
                return

            if start_idx > 0:
                content = self._buffer[:start_idx]
                self._buffer = self._buffer[start_idx:]
                self._queue_delta_message(DeltaMessage(content=content))
                continue

            self._buffer = self._buffer[len(self.tool_call_start_token) :]
            self._in_tool_calls = True
            continue

        if self._active_tool_index is None:
            stripped_len = len(self._buffer) - len(self._buffer.lstrip())
            if stripped_len:
                self._buffer = self._buffer[stripped_len:]
                continue

            if self._buffer.startswith(self.tool_call_end_token):
                self._buffer = self._buffer[len(self.tool_call_end_token) :]
                self._in_tool_calls = False
                continue

            match = self.invoke_start_regex.match(self._buffer)
            if match is None:
                return

            self._buffer = self._buffer[match.end() :]
            self._begin_streaming_tool_call(match.group(1))
            continue

        index = self._active_tool_index

        if self._streaming_param_mode is not None:
            end_pos = self._buffer.find(parameter_end_token)
            if end_pos != -1:
                raw_content = self._buffer[:end_pos]
                self._buffer = self._buffer[end_pos + len(parameter_end_token) :]
                if self._streaming_param_mode.startswith("wrapper_"):
                    self._streaming_param_raw_parts.append(raw_content)
                    _finish_buffered_wrapper_param(self, index, request)
                elif self._streaming_param_mode == "string":
                    frag = _json_escape_string_content(raw_content) + '"'
                    self._queue_delta_message(self._emit_tool_args_delta(index, frag))
                else:
                    frag = raw_content
                    self._queue_delta_message(self._emit_tool_args_delta(index, frag))

                self._streaming_param_mode = None
                continue

            safe_len = _safe_content_len_before_tag_end(self)
            if safe_len > 0:
                raw_content = self._buffer[:safe_len]
                self._buffer = self._buffer[safe_len:]
                if self._streaming_param_mode.startswith("wrapper_"):
                    self._streaming_param_raw_parts.append(raw_content)
                elif self._streaming_param_mode == "string":
                    frag = _json_escape_string_content(raw_content)
                    self._queue_delta_message(self._emit_tool_args_delta(index, frag))
                else:
                    frag = raw_content
                    self._queue_delta_message(self._emit_tool_args_delta(index, frag))
            return

        stripped_len = len(self._buffer) - len(self._buffer.lstrip())
        if stripped_len:
            self._buffer = self._buffer[stripped_len:]
            continue

        if self._buffer.startswith(invoke_end_token):
            self._buffer = self._buffer[len(invoke_end_token) :]
            _close_streaming_tool_call(self)
            continue

        match = self.parameter_start_regex.match(self._buffer)
        if match is None:
            return

        self._buffer = self._buffer[match.end() :]
        key = _extract_param_name(match.group(1))
        string_attr = match.group(2)
        is_string = string_attr == "true"
        if _should_buffer_wrapper_param(self, key, request):
            self._streaming_param_key = key
            self._streaming_param_raw_parts.clear()
            self._streaming_param_mode = "wrapper_string" if is_string else "wrapper_json"
            continue

        _append_param_prefix(self, index, key, is_string=is_string)
        self._streaming_param_mode = "string" if is_string else "json"


def _patched_extract_tool_calls_streaming(
    self: DeepSeekV4ToolParser,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],
    current_token_ids: Sequence[int],
    delta_token_ids: Sequence[int],
    request: ChatCompletionRequest,
) -> DeltaMessage | None:
    _ensure_streaming_attrs(self)
    if not previous_text:
        self._reset_streaming_state()

    self._buffer += delta_text
    _process_streaming_buffer(self, request)

    pending_delta = _pop_pending_delta_message(self)
    if pending_delta is not None:
        return pending_delta

    if not delta_text and delta_token_ids and self.prev_tool_call_arr:
        return DeltaMessage(content="")

    return None


# Backward-compatible monkey patches.
DeepSeekV4ToolParser._ensure_streaming_attrs = _ensure_streaming_attrs
DeepSeekV4ToolParser._function_name = _function_name
DeepSeekV4ToolParser._function_parameters = _function_parameters
DeepSeekV4ToolParser._convert_param_value = _convert_param_value
DeepSeekV4ToolParser._extract_param_name = _extract_param_name
DeepSeekV4ToolParser._get_param_config = _get_param_config
DeepSeekV4ToolParser._coerce_param_value = _coerce_param_value
DeepSeekV4ToolParser._repair_param_dict = _repair_param_dict
DeepSeekV4ToolParser._parse_invoke_params = _parse_invoke_params
DeepSeekV4ToolParser._reset_streaming_state = _reset_streaming_state
DeepSeekV4ToolParser._json_escape_string_content = _json_escape_string_content
DeepSeekV4ToolParser.drain_pending_tool_call_deltas = _drain_pending_tool_call_deltas
DeepSeekV4ToolParser._pop_pending_delta_message = _pop_pending_delta_message
DeepSeekV4ToolParser._queue_delta_message = _queue_delta_message
DeepSeekV4ToolParser._emit_tool_name_delta = _emit_tool_name_delta
DeepSeekV4ToolParser._emit_tool_args_delta = _emit_tool_args_delta
DeepSeekV4ToolParser._begin_streaming_tool_call = _begin_streaming_tool_call
DeepSeekV4ToolParser._append_param_prefix = _append_param_prefix
DeepSeekV4ToolParser._append_json_param_value = _append_json_param_value
DeepSeekV4ToolParser._append_raw_param_value = _append_raw_param_value
DeepSeekV4ToolParser._should_buffer_wrapper_param = _should_buffer_wrapper_param
DeepSeekV4ToolParser._finish_buffered_wrapper_param = _finish_buffered_wrapper_param
DeepSeekV4ToolParser._close_streaming_tool_call = _close_streaming_tool_call
DeepSeekV4ToolParser._safe_content_len_before_tag_end = _safe_content_len_before_tag_end
DeepSeekV4ToolParser._process_streaming_buffer = _process_streaming_buffer
DeepSeekV4ToolParser.extract_tool_calls_streaming = _patched_extract_tool_calls_streaming
