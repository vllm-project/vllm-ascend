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
# MiniMax M2 tool parser: backport incremental tool-call argument streaming.
#

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import regex as re
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import utils as tool_parser_utils
from vllm.tool_parsers.abstract_tool_parser import Tool
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser
from vllm.tool_parsers.utils import (
    extract_intermediate_diff,
    find_tool_properties,
)

_original_init = MinimaxM2ToolParser.__init__
# vLLM main moved schema helpers from this parser class into tool_parsers.utils.
_extract_types_from_schema = getattr(tool_parser_utils, "extract_types_from_schema", None)
_coerce_to_schema_type = getattr(tool_parser_utils, "coerce_to_schema_type", None)


def _patched_init(
    self: MinimaxM2ToolParser,
    tokenizer: TokenizerLike,
    tools: list[Tool] | None = None,
) -> None:
    _original_init(self, tokenizer, tools)
    tool_call_ids: list[str] = []
    tool_name_sent: list[bool] = []
    self._tool_call_ids = tool_call_ids
    self._tool_name_sent = tool_name_sent
    self._tool_call_started_from_token_id = False


def _extract_types_from_schema_fallback(schema: Any) -> list[str]:
    if not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()
    type_value = schema.get("type")
    if isinstance(type_value, str):
        types.add(type_value)
    elif isinstance(type_value, list):
        types.update(t for t in type_value if isinstance(t, str))

    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        for value in enum_values:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    for choice_field in ("anyOf", "oneOf", "allOf"):
        choices = schema.get(choice_field)
        if isinstance(choices, list):
            for choice in choices:
                types.update(_extract_types_from_schema_fallback(choice))

    return list(types) if types else ["string"]


def _extract_param_types_from_schema(schema: Any) -> list[str]:
    if callable(_extract_types_from_schema):
        return _extract_types_from_schema(schema)
    return _extract_types_from_schema_fallback(schema)


def _coerce_param_value_fallback(value: str, param_types: list[str]) -> Any:
    type_aliases = {
        "str": "string",
        "text": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "dict": "object",
        "list": "array",
    }
    normalized_types = {type_aliases.get(t.lower(), t.lower()) for t in param_types}

    for candidate_type in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if candidate_type not in normalized_types:
            continue

        if candidate_type == "null":
            if value.lower() == "null":
                return None
            continue
        if candidate_type == "string":
            return value
        if candidate_type == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        if candidate_type == "number":
            try:
                val = float(value)
                return val if val != int(val) else int(val)
            except (ValueError, TypeError):
                continue
        if candidate_type == "boolean":
            lower_val = value.lower().strip()
            if lower_val in ("true", "1"):
                return True
            if lower_val in ("false", "0"):
                return False
            continue
        if candidate_type in ("object", "array"):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _coerce_param_value(value: str, param_types: list[str]) -> Any:
    if callable(_coerce_to_schema_type):
        return _coerce_to_schema_type(value, param_types)
    return _coerce_param_value_fallback(value, param_types)


def _get_param_types_from_config(
    param_name: str,
    param_config: dict[str, Any],
) -> list[str]:
    param_schema = param_config.get(param_name)
    if not isinstance(param_schema, dict):
        return ["string"]
    return _extract_param_types_from_schema(param_schema)


def _patched_parse_single_invoke(
    self: MinimaxM2ToolParser,
    invoke_str: str,
    tools: list[Tool] | None,
) -> ToolCall | None:
    name_match = re.search(r"^([^>]+)", invoke_str)
    if not name_match:
        return None

    function_name = self._extract_name(name_match.group(1))
    param_config = find_tool_properties(tools, function_name)

    param_dict = {}
    for match in self.parameter_complete_regex.findall(invoke_str):
        param_match = re.search(r"^([^>]+)>(.*)", match, re.DOTALL)
        if param_match:
            param_name = self._extract_name(param_match.group(1))
            param_value = param_match.group(2).strip()
            param_type = _get_param_types_from_config(param_name, param_config)
            param_dict[param_name] = _coerce_param_value(param_value, param_type)

    return ToolCall(
        type="function",
        function=FunctionCall(
            name=function_name,
            arguments=json.dumps(param_dict, ensure_ascii=False),
        ),
    )


def _reset_streaming_state(
    self: MinimaxM2ToolParser,
    tool_call_started: bool = False,
) -> None:
    self.current_tool_index = 0
    self.prev_tool_call_arr.clear()
    self.streamed_args_for_tool.clear()
    self._tool_call_ids.clear()
    self._tool_name_sent.clear()
    self._tool_call_started_from_token_id = False
    self.is_tool_call_started = tool_call_started


def _ensure_streaming_slots(self: MinimaxM2ToolParser, tool_count: int) -> None:
    while len(self.streamed_args_for_tool) < tool_count:
        self.streamed_args_for_tool.append("")
    while len(self._tool_call_ids) < tool_count:
        self._tool_call_ids.append(self._generate_tool_call_id())
    while len(self._tool_name_sent) < tool_count:
        self._tool_name_sent.append(False)


def _get_param_config(
    self: MinimaxM2ToolParser,
    function_name: str,
) -> dict[str, Any]:
    return find_tool_properties(self.tools, function_name)


def _serialize_partial_param_value(
    self: MinimaxM2ToolParser,
    value: str,
    param_types: list[str],
    *,
    is_complete: bool,
) -> str:
    value = value.strip()
    if is_complete:
        converted = _coerce_param_value(value, param_types)
        return json.dumps(converted, ensure_ascii=False)

    if not value:
        return ""

    normalized_types = {t.lower() for t in param_types}
    string_types = {"string", "str", "text"}

    if "null" in normalized_types and not (normalized_types & string_types) and "null".startswith(value.lower()):
        return value.lower()

    if {"boolean", "bool"} & normalized_types:
        lower_value = value.lower()
        if any(candidate.startswith(lower_value) for candidate in ("true", "false")):
            return lower_value

    if {"integer", "int", "number", "float"} & normalized_types:
        return value

    if {"object", "array"} & normalized_types and value[:1] in "{[":
        return value

    return json.dumps(value, ensure_ascii=False)[:-1]


def _build_partial_arguments(
    self: MinimaxM2ToolParser,
    invoke_body: str,
    *,
    invoke_complete: bool,
    param_config: dict[str, Any],
) -> str:
    args_parts: list[str] = []
    search_pos = 0

    while True:
        param_start = invoke_body.find("<parameter name=", search_pos)
        if param_start == -1:
            break

        name_start = param_start + len("<parameter name=")
        name_end = invoke_body.find(">", name_start)
        if name_end == -1:
            break

        param_name = self._extract_name(invoke_body[name_start:name_end])
        value_start = name_end + 1
        value_end = invoke_body.find("</parameter>", value_start)
        param_complete = value_end != -1
        if param_complete:
            param_value = invoke_body[value_start:value_end]
            search_pos = value_end + len("</parameter>")
        else:
            param_value = invoke_body[value_start:]
            search_pos = len(invoke_body)

        if not param_complete and not param_value.strip():
            break

        param_types = _get_param_types_from_config(param_name, param_config)
        serialized_value = self._serialize_partial_param_value(
            param_value,
            param_types,
            is_complete=param_complete,
        )
        if not serialized_value:
            break

        args_parts.append(f"{json.dumps(param_name, ensure_ascii=False)}:{serialized_value}")

        if not param_complete:
            break

    if not args_parts:
        return "{}" if invoke_complete else ""

    args_json = "{" + ",".join(args_parts)
    if invoke_complete:
        args_json += "}"
    return args_json


def _get_invoke_states(
    self: MinimaxM2ToolParser,
    current_text: str,
) -> list[dict[str, Any]]:
    tool_start = current_text.find(self.tool_call_start_token)
    if tool_start == -1:
        if not self.is_tool_call_started:
            return []
        tool_payload = current_text
    else:
        tool_payload = current_text[tool_start + len(self.tool_call_start_token) :]

    tool_end = tool_payload.find(self.tool_call_end_token)
    if tool_end != -1:
        tool_payload = tool_payload[:tool_end]

    invoke_states: list[dict[str, Any]] = []
    search_pos = 0
    while True:
        invoke_start = tool_payload.find("<invoke name=", search_pos)
        if invoke_start == -1:
            break

        invoke_content_start = invoke_start + len("<invoke name=")
        invoke_end = tool_payload.find("</invoke>", invoke_content_start)
        invoke_complete = invoke_end != -1

        if invoke_complete:
            invoke_str = tool_payload[invoke_content_start:invoke_end]
            search_pos = invoke_end + len("</invoke>")
        else:
            invoke_str = tool_payload[invoke_content_start:]
            search_pos = len(tool_payload)

        name_end = invoke_str.find(">")
        if name_end == -1:
            break

        function_name = self._extract_name(invoke_str[:name_end])
        param_config = self._get_param_config(function_name)
        invoke_body = invoke_str[name_end + 1 :]
        partial_args = self._build_partial_arguments(
            invoke_body,
            invoke_complete=invoke_complete,
            param_config=param_config,
        )

        tool_call = self._parse_single_invoke(invoke_str, self.tools) if invoke_complete else None
        invoke_states.append(
            {
                "name": function_name,
                "arguments": partial_args,
                "complete": invoke_complete,
                "tool_call": tool_call,
            }
        )

        if not invoke_complete:
            break

    return invoke_states


def _finalize_completed_tool_call(
    self: MinimaxM2ToolParser,
    idx: int,
    invoke_state: dict[str, Any],
) -> None:
    if not invoke_state["complete"] or len(self.prev_tool_call_arr) > idx:
        return

    tool_call = invoke_state["tool_call"]
    if tool_call is None:
        return

    self.prev_tool_call_arr.append(
        {
            "name": tool_call.function.name,
            "arguments": json.loads(tool_call.function.arguments),
        }
    )


def _extract_delta_tool_call(
    self: MinimaxM2ToolParser,
    current_text: str,
) -> DeltaToolCall | None:
    invoke_states = self._get_invoke_states(current_text)
    if not invoke_states:
        return None

    self._ensure_streaming_slots(len(invoke_states))

    for idx, invoke_state in enumerate(invoke_states):
        args_json = invoke_state["arguments"]
        sent_args = self.streamed_args_for_tool[idx]
        name_sent = self._tool_name_sent[idx]

        if not name_sent:
            self._tool_name_sent[idx] = True
            self.current_tool_index = idx
            if args_json:
                self.streamed_args_for_tool[idx] = args_json
            self._finalize_completed_tool_call(idx, invoke_state)
            return DeltaToolCall(
                index=idx,
                id=self._tool_call_ids[idx],
                type="function",
                function=DeltaFunctionCall(
                    name=invoke_state["name"],
                    arguments=args_json or None,
                ),
            )

        if args_json and args_json != sent_args:
            if sent_args and args_json.startswith(sent_args):
                args_delta = args_json[len(sent_args) :]
            else:
                args_delta = extract_intermediate_diff(args_json, sent_args)

            if args_delta:
                self.streamed_args_for_tool[idx] = args_json
                self.current_tool_index = idx
                self._finalize_completed_tool_call(idx, invoke_state)
                return DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=args_delta),
                )

        self._finalize_completed_tool_call(idx, invoke_state)

    return None


def _patched_extract_tool_calls_streaming(
    self: MinimaxM2ToolParser,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: Sequence[int],  # pylint: disable=unused-argument
    current_token_ids: Sequence[int],  # pylint: disable=unused-argument
    delta_token_ids: Sequence[int],
    request: ChatCompletionRequest,  # pylint: disable=unused-argument
) -> DeltaMessage | None:
    start_in_text = self.tool_call_start_token in delta_text
    start_in_ids = self.tool_call_start_token_id in delta_token_ids
    tool_call_starting = start_in_text or start_in_ids
    if tool_call_starting:
        self._reset_streaming_state(tool_call_started=tool_call_starting)
        self._tool_call_started_from_token_id = start_in_ids and not start_in_text
    elif not previous_text:
        if self._tool_call_started_from_token_id:
            if current_text:
                self._tool_call_started_from_token_id = False
        else:
            self._reset_streaming_state(tool_call_started=False)

    if not self.is_tool_call_started:
        return DeltaMessage(content=delta_text) if delta_text else None

    content_before = None
    if start_in_text:
        before = delta_text[: delta_text.index(self.tool_call_start_token)]
        content_before = before or None

    delta_tool_call = self._extract_delta_tool_call(current_text)

    if delta_tool_call or content_before:
        return DeltaMessage(
            content=content_before,
            tool_calls=[delta_tool_call] if delta_tool_call else None,
        )

    if (
        not delta_text
        and delta_token_ids
        and self.prev_tool_call_arr
        and self.tool_call_end_token_id not in delta_token_ids
    ):
        return DeltaMessage(content="")

    return None


MinimaxM2ToolParser.__init__ = _patched_init
MinimaxM2ToolParser._parse_single_invoke = _patched_parse_single_invoke
MinimaxM2ToolParser._reset_streaming_state = _reset_streaming_state
MinimaxM2ToolParser._ensure_streaming_slots = _ensure_streaming_slots
MinimaxM2ToolParser._get_param_config = _get_param_config
MinimaxM2ToolParser._serialize_partial_param_value = _serialize_partial_param_value
MinimaxM2ToolParser._build_partial_arguments = _build_partial_arguments
MinimaxM2ToolParser._get_invoke_states = _get_invoke_states
MinimaxM2ToolParser._finalize_completed_tool_call = _finalize_completed_tool_call
MinimaxM2ToolParser._extract_delta_tool_call = _extract_delta_tool_call
MinimaxM2ToolParser.extract_tool_calls_streaming = _patched_extract_tool_calls_streaming
