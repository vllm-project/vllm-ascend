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
# MiniMax M3 tool-call parser: backport of Rust implementation from
# https://github.com/vllm-project/vllm/pull/45381
#
# MiniMax M3 tool-call format uses namespace-delimited XML tags:
#   ]<]minimax[>[<tool_call>
#   ]<]minimax[>[<invoke name="function_name">
#   ]<]minimax[>[<param>value]<]minimax[>[</param>
#   ]<]minimax[>[</invoke>
#   ]<]minimax[>[</tool_call>
#
# This parser adds the missing streaming incremental argument emission
# and supports recursive parameter parsing with schema-aware type conversion.
#


import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import Tool

logger = init_logger(__name__)

# MiniMax M3 namespace constants
_NAMESPACE = "]<]minimax[>["
_TOOL_CALL_START = "]<]minimax[>[<tool_call>"
_TOOL_CALL_END = "]<]minimax[>[</tool_call>"
_INVOKE_START = "]<]minimax[>[<invoke"
_INVOKE_END = "]<]minimax[>[</invoke>"
_ELEMENT_START = "]<]minimax[>[<"
_ELEMENT_END_START = "]<]minimax[>[</"
_MIXED_TEXT_FIELD = "$text"


def _extract_types_from_schema(schema: Any) -> list[str]:
    """Extract all possible types from a JSON schema definition.

    Handles anyOf, oneOf, allOf, type arrays, and enum fields.
    """
    if schema is None or not isinstance(schema, dict):
        return ["string"]

    types: set[str] = set()

    # Handle direct "type" field
    type_value = schema.get("type")
    if isinstance(type_value, str):
        types.add(type_value)
    elif isinstance(type_value, list):
        types.update(t for t in type_value if isinstance(t, str))

    # Handle enum - infer types from enum values
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
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

    # Handle anyOf, oneOf, allOf - recursively extract types
    for choice_field in ("anyOf", "oneOf", "allOf"):
        choices = schema.get(choice_field)
        if isinstance(choices, list):
            for choice in choices:
                types.update(_extract_types_from_schema(choice))

    # Handle additionalProperties
    if schema.get("additionalProperties") and isinstance(schema["additionalProperties"], dict):
        types.add("object")

    return list(types) if types else ["string"]


def _coerce_value(value: str, param_types: list[str]) -> Any:
    """Convert a string parameter value to the correct type based on schema types."""
    if not value:
        return value

    type_aliases = {
        "str": "string", "text": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "dict": "object", "list": "array",
    }
    normalized = {type_aliases.get(t.lower(), t.lower()) for t in param_types}

    # null check
    if "null" in normalized and value.lower() in ("null", "none", "nil"):
        return None

    type_priority = ["integer", "number", "boolean", "object", "array", "string"]
    for candidate_type in type_priority:
        if candidate_type not in normalized:
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
            if lower_val in ("true", "1", "yes", "on"):
                return True
            if lower_val in ("false", "0", "no", "off"):
                return False
            continue
        if candidate_type in ("object", "array"):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    # Fallback: try JSON parse
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _find_tool_properties(tools: list[Any] | None, function_name: str) -> dict[str, Any]:
    """Find the parameter properties for a given function name from the tool list."""
    if not tools:
        return {}
    for tool in tools:
        tool_function = getattr(tool, "function", None)
        if tool_function is None:
            continue
        name = getattr(tool_function, "name", None)
        if name != function_name:
            continue
        params = getattr(tool_function, "parameters", None)
        if isinstance(params, dict):
            return params.get("properties", {})
    return {}


def _normalize_to_json(
    parent_schema: dict[str, Any],
    elements: list[tuple[str, Any]],
    tool_schemas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Convert parsed elements into a JSON-compatible dict.

    Schema-aware type coercion is applied at each level.  For nested
    objects the child's schema is taken from the parent's declared
    ``properties``, and for top-level invokes the parent schema is the
    function's own properties map.
    """
    result: dict[str, Any] = {}
    seen_keys: dict[str, list[Any]] = {}

    for elem_name, elem_value in elements:
        child_config = parent_schema.get(elem_name, {})
        child_type = child_config.get("type", "")
        child_properties = child_config.get("properties", {})

        # Resolve nested elements
        if isinstance(elem_value, list):
            # elem_value is a list of child element tuples
            if child_properties:
                resolved = _normalize_to_json(child_properties, elem_value, tool_schemas)
            elif child_type == "array":
                # Array of objects — try to resolve items schema
                items_schema = child_config.get("items", {})
                items_properties = items_schema.get("properties", {})
                if items_properties:
                    resolved = _normalize_to_json(items_properties, elem_value, tool_schemas)
                else:
                    # Fallback: treat as a flat list of values
                    resolved = [
                        _coerce_value(v[1], _extract_types_from_schema(items_schema))
                        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], str)
                        else v
                        for v in elem_value
                    ]
            else:
                # Unknown object — resolve without schema
                resolved = _normalize_to_json({}, elem_value, tool_schemas)
        elif isinstance(elem_value, str):
            param_types = _extract_types_from_schema(child_config)
            resolved = _coerce_value(elem_value, param_types)
        else:
            resolved = elem_value

        if elem_name not in seen_keys:
            seen_keys[elem_name] = []
        seen_keys[elem_name].append(resolved)

    for key, values in seen_keys.items():
        child_config = parent_schema.get(key, {})
        child_type = child_config.get("type", "")

        if child_type == "array" or (len(values) > 1 and child_type != "object"):
            # Consolidate repeated elements into an array
            # For known object types with single value, keep as object
            if child_type == "array" and len(values) == 1 and isinstance(values[0], list):
                result[key] = values[0]
            elif child_type == "array":
                # Array items might be wrapped in a named element
                items_schema = child_config.get("items", {})
                if isinstance(items_schema, dict) and items_schema.get("type") == "object":
                    result[key] = values
                elif child_type == "array":
                    result[key] = values
                else:
                    result[key] = values
            else:
                result[key] = values
        elif child_type == "object" and len(values) == 1 and isinstance(values[0], dict):
            result[key] = values[0]
        elif isinstance(child_config, dict) and "properties" in child_config and len(values) == 1 and isinstance(values[0], dict):
            result[key] = values[0]
        else:
            # For unknown structs or scalar values
            if len(values) == 1:
                result[key] = values[0]
            else:
                result[key] = values

    return result


def _find_tool_properties_from_schemas(
    tool_schemas: dict[str, dict[str, Any]],
    function_name: str,
) -> dict[str, Any]:
    """Find parameter properties from tool schemas dict."""
    return tool_schemas.get(function_name, {})


class MinimaxM3ToolParser(ToolParser):
    """Tool parser for MiniMax M3 namespace-delimited XML-style tool calls.

    MiniMax M3 uses the namespace marker ``]<]minimax[>[`` before each
    structural tag. Arguments are emitted only after a complete
    ``<invoke>...</invoke>`` block is parsed.

    Tool call format::

        ]<]minimax[>[<tool_call>
        ]<]minimax[>[<invoke name="create_order">
        ]<]minimax[>[<user_id>42]<]minimax[>[</user_id>
        ]<]minimax[>[<shipping>
        ]<]minimax[>[<city>Singapore]<]minimax[>[</city>
        ]<]minimax[>[<zip>018956]<]minimax[>[</zip>
        ]<]minimax[>[</shipping>
        ]<]minimax[>[</invoke>
        ]<]minimax[>[</tool_call>
    """

    supports_required_and_named: bool = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        # Streaming state
        self._buffer: str = ""
        self._emitted_tool_count: int = 0
        self._in_tool_block: bool = False
        self._tool_block_done: bool = False

        # Current tool block content
        self._invoke_names: list[str] = []
        self._tool_call_ids: list[str] = []

        # Tool schemas indexed by function name
        self._tool_schemas: dict[str, dict[str, Any]] = {}
        if self.tools:
            for tool_def in self.tools:
                tool_function = getattr(tool_def, "function", None)
                if tool_function is None:
                    continue
                name = getattr(tool_function, "name", None)
                if name is None:
                    continue
                params = getattr(tool_function, "parameters", None)
                if isinstance(params, dict):
                    self._tool_schemas[name] = params.get("properties", {})

        logger.debug("vLLM-Ascend Successfully import tool parser %s !", self.__class__.__name__)

    # ------------------------------------------------------------------
    # Non-streaming extraction
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
        if _TOOL_CALL_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls, content = self._parse_complete_output(model_output)
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            # Sync state
            self._emitted_tool_count = len(tool_calls)
            self.prev_tool_call_arr.clear()
            for tc in tool_calls:
                self.prev_tool_call_arr.append({
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )
        except Exception:
            logger.exception("Error extracting MiniMax M3 tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # ------------------------------------------------------------------
    # Streaming extraction
    # ------------------------------------------------------------------

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
        """Extract tool calls from streaming model output.

        Uses a buffer-until-complete-invoke strategy: tokens are buffered
        until a complete ``<invoke>...</invoke>`` block is available,
        then parsed and emitted as a DeltaToolCall. Argument streaming
        within an invoke block is supported for incremental updates.
        """
        # Detect new tool-call block start
        tool_call_starting = (
            _TOOL_CALL_START in delta_text
            or self._starts_new_tool_block(previous_text, current_text)
        )

        if tool_call_starting:
            self._reset_streaming_state()

        if not self._in_tool_block and not tool_call_starting:
            # Before any tool call: pass through as normal text
            return DeltaMessage(content=delta_text) if delta_text else None

        # Update buffer
        if tool_call_starting and not previous_text:
            self._buffer = current_text
        else:
            self._buffer = current_text

        # Extract content before tool block starts
        content_before = None
        if tool_call_starting:
            before_idx = current_text.find(_TOOL_CALL_START)
            if before_idx > 0:
                content_before = current_text[:before_idx]

        if not self._in_tool_block and _TOOL_CALL_START in self._buffer:
            self._in_tool_block = True

        if not self._in_tool_block:
            return DeltaMessage(content=content_before)

        # Check if tool block has ended
        if _TOOL_CALL_END in self._buffer and not self._tool_block_done:
            self._tool_block_done = True

        # Extract completed invoke blocks
        delta_tool_calls = self._extract_completed_invokes(self._buffer)

        if delta_tool_calls or content_before:
            return DeltaMessage(
                content=content_before,
                tool_calls=delta_tool_calls if delta_tool_calls else None,
            )

        return None

    # ------------------------------------------------------------------
    # Internal: complete parsing
    # ------------------------------------------------------------------

    def _parse_complete_output(
        self, model_output: str
    ) -> tuple[list[ToolCall], str | None]:
        """Parse complete model output and return (tool_calls, content)."""
        tool_calls: list[ToolCall] = []

        # Find text before first tool block
        first_tool_idx = model_output.find(_TOOL_CALL_START)
        content = model_output[:first_tool_idx] if first_tool_idx > 0 else None

        # Find all tool_call blocks
        tool_block_re = re.compile(
            re.escape(_TOOL_CALL_START) + r"(.*?)" + re.escape(_TOOL_CALL_END),
            re.DOTALL,
        )
        for block_match in tool_block_re.finditer(model_output):
            block_content = block_match.group(1)
            # Find all invoke blocks within this tool_call
            invoke_re = re.compile(
                re.escape(_INVOKE_START) + r'\s+name="([^"]*)"\s*>(.*?)' + re.escape(_INVOKE_END),
                re.DOTALL,
            )
            for invoke_match in invoke_re.finditer(block_content):
                function_name = invoke_match.group(1).strip()
                invoke_body = invoke_match.group(2)
                params = self._parse_invoke_params(function_name, invoke_body)
                arguments = json.dumps(params, ensure_ascii=False)
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_name,
                            arguments=arguments,
                        ),
                    )
                )

        return tool_calls, content

    def _parse_invoke_params(
        self, function_name: str, invoke_body: str
    ) -> dict[str, Any]:
        """Parse parameter elements from a complete invoke body string.

        Supports recursive nested elements (objects within objects) and
        repeated same-name elements (arrays). Schema-aware type coercion
        is applied to leaf text values.
        """
        elements = self._parse_element_children(invoke_body)
        if not elements:
            return {}

        parent_schema = _find_tool_properties_from_schemas(
            self._tool_schemas, function_name
        )
        return _normalize_to_json(parent_schema, elements, self._tool_schemas)

    def _parse_element_children(
        self, text: str
    ) -> list[tuple[str, Any]]:
        """Parse all child elements from a text buffer.

        Returns a list of (element_name, value) tuples. The value is either
        a string (leaf) or a list of tuples (nested child elements).
        """
        elements: list[tuple[str, Any]] = []
        pos = 0

        while pos < len(text):
            # Skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break

            # Check if this looks like the start of an element
            if text[pos:].startswith(_ELEMENT_START):
                # Check this is an opening tag (not closing)
                rest = text[pos + len(_ELEMENT_START):]
                if rest.startswith("/"):
                    # It's a closing tag we don't expect at this level
                    break

                # Find opening tag name
                tag_end = rest.find(">")
                if tag_end == -1:
                    break
                tag_name = rest[:tag_end]

                body_start = pos + len(_ELEMENT_START) + tag_end + 1
                # Find matching closing tag
                close_tag = f"{_ELEMENT_END_START}{tag_name}>"
                body_end = text.find(close_tag, body_start)
                if body_end == -1:
                    break

                body = text[body_start:body_end]
                next_pos = body_end + len(close_tag)

                # Check if body contains child elements
                if _ELEMENT_START in body:
                    child_elements = self._parse_element_children(body)
                    elements.append((tag_name, child_elements))
                else:
                    elements.append((tag_name, body))

                pos = next_pos
                continue

            if text[pos:].startswith(_NAMESPACE):
                # Unexpected namespace marker at this level
                break

            # Handle stray text (junk between elements) - be tolerant,
            # keep parsed elements and stop
            break

        return elements

    # ------------------------------------------------------------------
    # Internal: streaming helpers
    # ------------------------------------------------------------------

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new request or new tool block."""
        self._buffer = ""
        self._in_tool_block = False
        self._tool_block_done = False
        self._emitted_tool_count = 0
        self._invoke_names.clear()
        self._tool_call_ids.clear()
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        self.current_tool_id = -1
        self.current_tool_name_sent = False

    def _starts_new_tool_block(
        self, previous_text: str, current_text: str
    ) -> bool:
        """Detect if a new tool-call block started since previous_text."""
        if not previous_text:
            return _TOOL_CALL_START in current_text
        if _TOOL_CALL_START not in previous_text and _TOOL_CALL_START in current_text:
            return True
        return False

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _find_invoke_blocks(self, text: str) -> list[dict[str, Any]]:
        """Find invoke blocks in text, both complete and incomplete.

        Returns list of dicts with keys: name, body, complete, start_pos.
        """
        invokes: list[dict[str, Any]] = []
        pos = 0

        while pos < len(text):
            invoke_start = text.find(_INVOKE_START, pos)
            if invoke_start == -1:
                break

            # Parse name attribute
            attr_start = invoke_start + len(_INVOKE_START)
            name_match = re.match(r'\s+name="([^"]*)"\s*>', text[attr_start:])
            if not name_match:
                pos = attr_start
                continue

            function_name = name_match.group(1).strip()
            body_start = attr_start + name_match.end()
            invoke_end = text.find(_INVOKE_END, body_start)
            invoke_complete = invoke_end != -1

            if invoke_complete:
                body = text[body_start:invoke_end]
                next_pos = invoke_end + len(_INVOKE_END)
            else:
                body = text[body_start:]
                next_pos = len(text)

            invokes.append({
                "name": function_name,
                "body": body,
                "complete": invoke_complete,
                "start_pos": invoke_start,
            })

            pos = max(next_pos, pos + 1)

        return invokes

    def _extract_completed_invokes(self, current_text: str) -> list[DeltaToolCall]:
        """Extract DeltaToolCalls from newly completed invoke blocks.

        Only emits invokes that haven't been emitted yet, tracking via
        ``_emitted_tool_count``. Supports streaming argument updates.
        """
        invoke_blocks = self._find_invoke_blocks(current_text)
        delta_tool_calls: list[DeltaToolCall] = []

        for idx, invoke in enumerate(invoke_blocks):
            if idx < self._emitted_tool_count:
                # Already emitted this invoke - check for argument updates
                if invoke["complete"]:
                    continue
                # Still incomplete, could be streaming more args
                continue

            if not invoke["complete"]:
                # For partially received invoke, stream argument fragments
                partial_result = self._try_stream_partial_args(
                    idx, invoke["name"], invoke["body"]
                )
                if partial_result:
                    delta_tool_calls.append(partial_result)
                break  # Don't process beyond the first incomplete invoke

            # Complete invoke: parse and emit
            params = self._parse_invoke_params(invoke["name"], invoke["body"])
            arguments = json.dumps(params, ensure_ascii=False)

            # Ensure slots exist
            while len(self._tool_call_ids) <= idx:
                self._tool_call_ids.append(self._generate_tool_call_id())
            while len(self.streamed_args_for_tool) <= idx:
                self.streamed_args_for_tool.append("")

            # Emit the complete tool call
            self._emitted_tool_count = idx + 1
            self.streamed_args_for_tool[idx] = arguments

            self.prev_tool_call_arr.append({
                "name": invoke["name"],
                "arguments": params,
            })

            delta_tool_calls.append(
                DeltaToolCall(
                    index=idx,
                    id=self._tool_call_ids[idx],
                    type="function",
                    function=DeltaFunctionCall(
                        name=invoke["name"],
                        arguments=arguments,
                    ),
                )
            )

        return delta_tool_calls

    def _try_stream_partial_args(
        self, idx: int, function_name: str, body: str
    ) -> DeltaToolCall | None:
        """Try to emit partial arguments for a still-incomplete invoke block.

        Only emits the function name on first sight, and incremental
        argument fragments as they become available.
        """
        # Ensure slots exist
        while len(self._tool_call_ids) <= idx:
            self._tool_call_ids.append(self._generate_tool_call_id())
        while len(self.streamed_args_for_tool) <= idx:
            self.streamed_args_for_tool.append("")

        # First time seeing this invoke - emit name
        if len(self._invoke_names) <= idx:
            self._invoke_names.append(function_name)
            return DeltaToolCall(
                index=idx,
                id=self._tool_call_ids[idx],
                type="function",
                function=DeltaFunctionCall(
                    name=function_name,
                    arguments="",
                ),
            )

        # Try to extract partial arguments from the body so far
        partial_args = self._build_partial_arguments(function_name, body)
        if not partial_args:
            return None

        sent_args = self.streamed_args_for_tool[idx]
        if partial_args != sent_args:
            self.streamed_args_for_tool[idx] = partial_args
            # Compute delta
            if sent_args and partial_args.startswith(sent_args):
                args_delta = partial_args[len(sent_args):]
            else:
                args_delta = partial_args
            return DeltaToolCall(
                index=idx,
                function=DeltaFunctionCall(arguments=args_delta),
            )

        return None

    def _build_partial_arguments(
        self, function_name: str, body: str
    ) -> str:
        """Build a partial JSON arguments string from the invoke body so far.

        Handles both complete and incomplete parameter elements.
        """
        # Quick check: find elements in the body
        param_config = _find_tool_properties_from_schemas(
            self._tool_schemas, function_name
        )

        args_parts: list[str] = []
        pos = 0

        while pos < len(body):
            # Skip whitespace
            while pos < len(body) and body[pos].isspace():
                pos += 1
            if pos >= len(body):
                break

            if body[pos:].startswith(_ELEMENT_START):
                rest = body[pos + len(_ELEMENT_START):]
                if rest.startswith("/"):
                    pos += len(_ELEMENT_START)
                    continue

                tag_end = rest.find(">")
                if tag_end == -1:
                    break
                tag_name = rest[:tag_end]

                value_start = pos + len(_ELEMENT_START) + tag_end + 1
                close_tag = f"{_ELEMENT_END_START}{tag_name}>"
                value_end = body.find(close_tag, value_start)
                param_complete = value_end != -1

                if param_complete:
                    param_value = body[value_start:value_end]
                    pos = value_end + len(close_tag)
                else:
                    param_value = body[value_start:]
                    pos = len(body)

                # Coerce and serialize
                param_types = _extract_types_from_schema(
                    param_config.get(tag_name, {})
                )
                serialized = self._serialize_param_value(
                    param_value, param_types, is_complete=param_complete
                )
                if not serialized and not param_complete:
                    break

                args_parts.append(
                    f"{json.dumps(tag_name, ensure_ascii=False)}:{serialized}"
                )

                if not param_complete:
                    break
            else:
                # Non-element text: skip
                break

        if not args_parts:
            return ""

        result = "{" + ",".join(args_parts)
        return result

    @staticmethod
    def _serialize_param_value(
        value: str, param_types: list[str], *, is_complete: bool
    ) -> str:
        """Serialize a parameter value for JSON output.

        If the value is complete, coerce and serialize.  If incomplete,
        produce a best-effort partial string suitable for streaming.
        """
        value = value.strip()
        if is_complete:
            converted = _coerce_value(value, param_types)
            return json.dumps(converted, ensure_ascii=False)

        if not value:
            return ""

        normalized = {t.lower() for t in param_types}
        string_types = {"string", "str", "text"}

        if "null" in normalized and not (normalized & string_types):
            if "null".startswith(value.lower()):
                return value.lower()

        if {"boolean", "bool"} & normalized:
            lower = value.lower()
            if any(c.startswith(lower) for c in ("true", "false")):
                return lower

        if {"integer", "int", "number", "float"} & normalized:
            return value

        if {"object", "array"} & normalized and value[:1] in "{[":
            return value

        return json.dumps(value, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Register the MiniMax M3 tool parser with the vLLM ToolParserManager
# ---------------------------------------------------------------------------

def _register_minimax_m3_tool_parser() -> None:
    """Register MiniMax M3 tool parser in vLLM's ToolParserManager.

    Installs the parser under the name ``"minimax_m3"`` so that
    ``--tool-parser=minimax_m3`` resolves correctly.
    """
    try:
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

        # Check if already registered
        if "minimax_m3" in ToolParserManager.tool_parsers or "minimax_m3" in ToolParserManager.lazy_parsers:
            logger.debug("MiniMax M3 tool parser already registered.")
            return

        # Register the parser class lazily
        ToolParserManager.register_lazy_module(
            name="minimax_m3",
            module_path="vllm_ascend.patch.platform.patch_minimax_m3_tool_call_parser",
            class_name="MinimaxM3ToolParser",
        )
        logger.info("Registered MiniMax M3 tool parser (minimax_m3).")
    except Exception:
        logger.exception("Failed to register MiniMax M3 tool parser.")


_register_minimax_m3_tool_parser()
