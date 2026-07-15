# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for MiniMax M3 tool-call parser.

Covers: vllm_ascend/patch/platform/patch_minimax_m3_tool_call_parser.py
"""

from __future__ import annotations

import json
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
)
from vllm_ascend.patch.platform.patch_minimax_m3_tool_call_parser import (
    _ELEMENT_END_START,
    _ELEMENT_START,
    _INVOKE_END,
    _INVOKE_START,
    _NAMESPACE,
    _TOOL_CALL_END,
    _TOOL_CALL_START,
    MinimaxM3ToolParser,
    _coerce_value,
    _extract_types_from_schema,
    _find_tool_properties,
    _find_tool_properties_from_schemas,
    _normalize_to_json,
)

# ===========================================================================
# Helpers
# ===========================================================================

TC_START_ID = 1000
TC_END_ID = 1001


class FakeTokenizer:
    model_tokenizer = True

    def get_vocab(self) -> dict[str, int]:
        return {
            _TOOL_CALL_START: TC_START_ID,
            _TOOL_CALL_END: TC_END_ID,
        }


def _element(name: str, body: str) -> str:
    return f"{_ELEMENT_START}{name}>{body}{_ELEMENT_END_START}{name}>"


def _invoke(function_name: str, body: str) -> str:
    return f'{_INVOKE_START} name="{function_name}">{body}{_INVOKE_END}'


def _tool_block(invokes: str) -> str:
    return f"{_TOOL_CALL_START}\n{invokes}\n{_TOOL_CALL_END}"


def _get_weather_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            )
        ),
    ]


def _create_order_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="create_order",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer"},
                        "urgent": {"type": "boolean"},
                        "note": {"type": "string"},
                        "shipping": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "zip": {"type": "integer"},
                            },
                        },
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sku": {"type": "string"},
                                    "qty": {"type": "integer"},
                                },
                            },
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                },
            )
        ),
    ]


def _make_parser(tools=None) -> MinimaxM3ToolParser:
    return MinimaxM3ToolParser(FakeTokenizer(), tools)


def _feed(parser: MinimaxM3ToolParser, chunks, token_ids=None):
    """Stream chunks through extract_tool_calls_streaming."""
    previous = ""
    results: list[DeltaMessage] = []
    for i, chunk in enumerate(chunks):
        delta_ids = token_ids[i] if token_ids and i < len(token_ids) else []
        current = previous + chunk
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=delta_ids,
            request=None,
        )
        if result is not None:
            results.append(result)
        previous = current
    return results


def _collect_content(results: list[DeltaMessage]) -> str:
    return "".join(r.content for r in results if r.content)


def _collect_tool_calls(results: list[DeltaMessage]) -> dict[int, dict[str, Any]]:
    tc_map: dict[int, dict[str, Any]] = {}
    for r in results:
        for tc in r.tool_calls or []:
            tc_map.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            if tc.id:
                tc_map[tc.index]["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    tc_map[tc.index]["name"] += tc.function.name
                if tc.function.arguments:
                    tc_map[tc.index]["arguments"] += tc.function.arguments
    return tc_map


# ===========================================================================
# Module-level function: _extract_types_from_schema
# ===========================================================================


class TestExtractTypesFromSchema:
    def test_none_returns_string(self):
        assert _extract_types_from_schema(None) == ["string"]

    def test_non_dict_returns_string(self):
        assert _extract_types_from_schema("foo") == ["string"]
        assert _extract_types_from_schema(42) == ["string"]

    def test_str_type(self):
        assert _extract_types_from_schema({"type": "string"}) == ["string"]

    def test_list_type(self):
        result = _extract_types_from_schema({"type": ["string", "null"]})
        assert set(result) == {"string", "null"}

    def test_enum_infers_types(self):
        schema = {"enum": ["hello", 1, 3.14, True, None, [1, 2], {"a": 1}]}
        result = _extract_types_from_schema(schema)
        assert set(result) == {"string", "integer", "number", "boolean", "null", "array", "object"}

    def test_anyof(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        result = _extract_types_from_schema(schema)
        assert set(result) == {"string", "integer"}

    def test_oneof(self):
        schema = {"oneOf": [{"type": "boolean"}, {"type": "null"}]}
        result = _extract_types_from_schema(schema)
        assert set(result) == {"boolean", "null"}

    def test_allof(self):
        schema = {"allOf": [{"type": "object"}, {"type": "array"}]}
        result = _extract_types_from_schema(schema)
        assert set(result) == {"object", "array"}

    def test_empty_schema_returns_string(self):
        assert _extract_types_from_schema({}) == ["string"]

    def test_enum_empty_list(self):
        assert _extract_types_from_schema({"enum": []}) == ["string"]

    def test_additional_properties(self):
        schema = {"additionalProperties": {"type": "integer"}}
        result = _extract_types_from_schema(schema)
        assert "object" in result


# ===========================================================================
# Module-level function: _coerce_value
# ===========================================================================


class TestCoerceValue:
    def test_empty_string(self):
        assert _coerce_value("", ["integer"]) == ""

    def test_null_coercion(self):
        assert _coerce_value("null", ["null"]) is None
        assert _coerce_value("None", ["null"]) is None

    def test_integer(self):
        assert _coerce_value("42", ["integer"]) == 42
        assert _coerce_value("-10", ["integer"]) == -10

    def test_integer_failure_falls_back(self):
        # Not a valid integer, should fall through to string
        assert _coerce_value("abc", ["integer"]) == "abc"

    def test_number(self):
        result = _coerce_value("3.14", ["number"])
        assert result == 3.14
        result = _coerce_value("5.0", ["number"])
        assert result == 5  # int representation

    def test_boolean_true(self):
        assert _coerce_value("true", ["boolean"]) is True
        assert _coerce_value("TRUE", ["boolean"]) is True
        assert _coerce_value("1", ["boolean"]) is True
        assert _coerce_value("yes", ["boolean"]) is True

    def test_boolean_false(self):
        assert _coerce_value("false", ["boolean"]) is False
        assert _coerce_value("FALSE", ["boolean"]) is False
        assert _coerce_value("0", ["boolean"]) is False

    def test_object_json(self):
        result = _coerce_value('{"key": "val"}', ["object"])
        assert result == {"key": "val"}

    def test_array_json(self):
        result = _coerce_value("[1, 2, 3]", ["array"])
        assert result == [1, 2, 3]

    def test_object_failure_falls_to_string(self):
        result = _coerce_value("{bad json", ["object"])
        assert result == "{bad json"

    def test_string_type(self):
        assert _coerce_value("hello", ["string"]) == "hello"

    def test_fallback_json(self):
        # No matching type in priority list, try JSON fallback
        result = _coerce_value("42", ["custom_type"])
        # Should try JSON parse first (42 succeeds) -> 42
        assert result == 42

    def test_fallback_string(self):
        result = _coerce_value("plain text", ["custom_type"])
        assert result == "plain text"

    def test_type_alias(self):
        assert _coerce_value("42", ["int"]) == 42
        assert _coerce_value("3.14", ["float"]) == 3.14
        assert _coerce_value("true", ["bool"]) is True


# ===========================================================================
# Module-level function: _find_tool_properties
# ===========================================================================


class TestFindToolProperties:
    def test_none_tools(self):
        assert _find_tool_properties(None, "fn") == {}

    def test_empty_tools(self):
        assert _find_tool_properties([], "fn") == {}

    def test_matching_function_name(self):
        tools = _get_weather_tools()
        result = _find_tool_properties(tools, "get_weather")
        assert "city" in result
        assert "days" in result
        assert result["city"]["type"] == "string"
        assert result["days"]["type"] == "integer"

    def test_non_matching_function_name(self):
        tools = _get_weather_tools()
        result = _find_tool_properties(tools, "nonexistent")
        assert result == {}

    def test_tool_without_function_attr(self):
        class NoFunctionTool:
            pass

        assert _find_tool_properties([NoFunctionTool()], "fn") == {}

    def test_tool_without_name_attr(self):
        class NoNameFunction:
            pass

        class ToolLike:
            function = NoNameFunction()

        assert _find_tool_properties([ToolLike()], "fn") == {}

    def test_tool_without_parameters(self):
        class PlainFunction:
            pass

        class ToolLike:
            function = PlainFunction()
            function.name = "test_func"

        assert _find_tool_properties([ToolLike()], "test_func") == {}

    def test_parameters_not_dict(self):
        class ToolLike:
            function = type("F", (), {"name": "test_func", "parameters": "not_a_dict"})()

        assert _find_tool_properties([ToolLike()], "test_func") == {}


# ===========================================================================
# Module-level function: _find_tool_properties_from_schemas
# ===========================================================================


class TestFindToolPropertiesFromSchemas:
    def test_existing_key(self):
        schemas = {"fn": {"city": {"type": "string"}}}
        result = _find_tool_properties_from_schemas(schemas, "fn")
        assert result == {"city": {"type": "string"}}

    def test_missing_key(self):
        assert _find_tool_properties_from_schemas({}, "fn") == {}


# ===========================================================================
# Module-level function: _normalize_to_json
# ===========================================================================


class TestNormalizeToJson:
    def test_flat_elements_no_schema(self):
        result = _normalize_to_json({}, [("a", "1"), ("b", "2")], {})
        assert result == {"a": "1", "b": "2"}

    def test_flat_elements_with_schema(self):
        schema = {"count": {"type": "integer"}, "name": {"type": "string"}}
        result = _normalize_to_json(schema, [("count", "5"), ("name", "test")], {})
        assert result == {"count": 5, "name": "test"}

    def test_nested_object_with_properties(self):
        parent_schema = {
            "payload": {
                "type": "object",
                "properties": {"inner": {"type": "integer"}},
            }
        }
        elements = [("payload", [("inner", "42")])]
        result = _normalize_to_json(parent_schema, elements, {})
        assert result == {"payload": {"inner": 42}}

    def test_array_type_with_items_schema(self):
        parent_schema = {
            "items": {
                "type": "array",
                "items": {"properties": {"name": {"type": "string"}}},
            }
        }
        elements = [("items", [("item", [("name", "sku-1")])])]
        result = _normalize_to_json(parent_schema, elements, {})
        assert "items" in result

    def test_unknown_object_no_schema(self):
        elements = [("outer", [("inner", "value")])]
        result = _normalize_to_json({}, elements, {})
        assert result == {"outer": {"inner": "value"}}

    def test_repeated_elements_become_array(self):
        result = _normalize_to_json({}, [("x", "1"), ("x", "2"), ("x", "3")], {})
        assert result == {"x": ["1", "2", "3"]}

    def test_single_element_object_type(self):
        schema = {"data": {"type": "object", "properties": {"k": {"type": "string"}}}}
        elements = [("data", [("k", "v")])]
        result = _normalize_to_json(schema, elements, {})
        assert result == {"data": {"k": "v"}}

    def test_array_type_single_value(self):
        schema = {"tags": {"type": "array", "items": {"type": "string"}}}
        result = _normalize_to_json(schema, [("tags", "hello")], {})
        # Single string value with array type: becomes a list
        assert "tags" in result
        assert result["tags"] == ["hello"]

    def test_empty_elements(self):
        assert _normalize_to_json({}, [], {}) == {}


# ===========================================================================
# MinimaxM3ToolParser: __init__
# ===========================================================================


class TestParserInit:
    def test_init_without_tools(self):
        parser = _make_parser()
        assert parser._tool_schemas == {}
        assert parser._buffer == ""
        assert parser._emitted_tool_count == 0
        assert parser._in_tool_block is False
        assert parser._tool_block_done is False

    def test_init_with_tools(self):
        tools = _get_weather_tools()
        parser = _make_parser(tools)
        assert "get_weather" in parser._tool_schemas
        assert parser._tool_schemas["get_weather"]["city"]["type"] == "string"

    def test_init_with_multiple_tools(self):
        tools = _get_weather_tools() + _create_order_tools()
        parser = _make_parser(tools)
        assert "get_weather" in parser._tool_schemas
        assert "create_order" in parser._tool_schemas

    def test_init_with_tool_missing_function_attr(self):
        class BadTool:
            pass

        parser = _make_parser([BadTool()])
        assert parser._tool_schemas == {}

    def test_init_with_tool_missing_name(self):
        class NoNameFunc:
            pass

        class ToolLike:
            function = NoNameFunc()

        parser = _make_parser([ToolLike()])
        assert parser._tool_schemas == {}

    def test_init_with_tool_params_not_dict(self):
        class ToolLike:
            function = type("F", (), {"name": "fn", "parameters": "bad"})()

        parser = _make_parser([ToolLike()])
        assert parser._tool_schemas == {}

    def test_supports_required_and_named(self):
        parser = _make_parser()
        assert parser.supports_required_and_named is False


# ===========================================================================
# MinimaxM3ToolParser: extract_tool_calls (non-streaming)
# ===========================================================================


class TestExtractToolCalls:
    def test_plain_text_no_tool_call(self):
        parser = _make_parser()
        output = parser.extract_tool_calls("Hello, world!", None)
        assert output.tools_called is False
        assert output.content == "Hello, world!"

    def test_single_tool_call(self):
        parser = _make_parser(_get_weather_tools())
        tc = _tool_block(_invoke("get_weather", _element("city", "Seattle")))
        output = parser.extract_tool_calls(f"Let me check. {tc}", None)
        assert output.tools_called is True
        assert output.content == "Let me check. "
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].function.name == "get_weather"
        args = json.loads(output.tool_calls[0].function.arguments)
        assert args == {"city": "Seattle"}

    def test_multiple_invokes_in_one_block(self):
        parser = _make_parser(_get_weather_tools())
        invokes = (
            _invoke("get_weather", _element("city", "Seattle")) + "\n" + _invoke("get_weather", _element("city", "NYC"))
        )
        tc = _tool_block(invokes)
        output = parser.extract_tool_calls(tc, None)
        assert len(output.tool_calls) == 2
        args0 = json.loads(output.tool_calls[0].function.arguments)
        args1 = json.loads(output.tool_calls[1].function.arguments)
        assert args0 == {"city": "Seattle"}
        assert args1 == {"city": "NYC"}

    def test_nested_parameters_with_schema(self):
        parser = _make_parser(_create_order_tools())
        shipping_elem = _element(
            "shipping",
            _element("city", "Singapore") + _element("zip", "018956"),
        )
        body = _element("user_id", "42") + _element("urgent", "true") + shipping_elem
        tc = _tool_block(_invoke("create_order", body))
        output = parser.extract_tool_calls(tc, None)
        args = json.loads(output.tool_calls[0].function.arguments)
        assert args["user_id"] == 42
        assert args["urgent"] is True
        assert args["shipping"]["city"] == "Singapore"
        assert args["shipping"]["zip"] == 18956

    def test_no_tools_called_when_parse_fails(self):
        parser = _make_parser()
        result = parser.extract_tool_calls(_TOOL_CALL_START + "garbage", None)
        assert result.tools_called is False

    def test_tool_calls_cache_prev(self):
        parser = _make_parser(_get_weather_tools())
        tc = _tool_block(_invoke("get_weather", _element("city", "Paris")))
        parser.extract_tool_calls(tc, None)
        assert len(parser.prev_tool_call_arr) == 1
        assert parser.prev_tool_call_arr[0]["name"] == "get_weather"
        assert parser.prev_tool_call_arr[0]["arguments"] == {"city": "Paris"}

    def test_repeated_tags_become_array(self):
        parser = _make_parser(_create_order_tools())
        body = _element("tags", "a") + _element("tags", "b") + _element("tags", "c")
        tc = _tool_block(_invoke("create_order", body))
        output = parser.extract_tool_calls(tc, None)
        args = json.loads(output.tool_calls[0].function.arguments)
        assert args["tags"] == ["a", "b", "c"]

    def test_content_before_tool_call_preserved(self):
        parser = _make_parser(_get_weather_tools())
        tc = _tool_block(_invoke("get_weather", _element("city", "Seattle")))
        output = parser.extract_tool_calls(f"Before text{tc}After text", None)
        assert output.content == "Before text"

    def test_tool_call_without_start_token(self):
        parser = _make_parser()
        output = parser.extract_tool_calls("No tool blocks here", None)
        assert output.tools_called is False
        assert output.content == "No tool blocks here"


# ===========================================================================
# MinimaxM3ToolParser: _parse_complete_output
# ===========================================================================


class TestParseCompleteOutput:
    def test_returns_content_before_tool_block(self):
        parser = _make_parser(_get_weather_tools())
        tc = _tool_block(_invoke("get_weather", _element("city", "Seattle")))
        calls, content = parser._parse_complete_output(f"prefix {tc}")
        assert content == "prefix "
        assert len(calls) == 1

    def test_no_content_when_tool_at_start(self):
        parser = _make_parser(_get_weather_tools())
        tc = _tool_block(_invoke("get_weather", _element("city", "Seattle")))
        calls, content = parser._parse_complete_output(tc)
        assert content is None
        assert len(calls) == 1

    def test_no_tool_blocks(self):
        parser = _make_parser()
        calls, content = parser._parse_complete_output("plain text")
        # No tool blocks -> content is None (no prefix to extract)
        assert content is None
        assert calls == []

    def test_multiple_tool_blocks(self):
        parser = _make_parser(_get_weather_tools())
        tc1 = _tool_block(_invoke("get_weather", _element("city", "A")))
        tc2 = _tool_block(_invoke("get_weather", _element("city", "B")))
        calls, _ = parser._parse_complete_output(f"{tc1}{tc2}")
        assert len(calls) == 2


# ===========================================================================
# MinimaxM3ToolParser: _parse_invoke_params
# ===========================================================================


class TestParseInvokeParams:
    def test_empty_body(self):
        parser = _make_parser(_get_weather_tools())
        result = parser._parse_invoke_params("get_weather", "")
        assert result == {}

    def test_single_param(self):
        parser = _make_parser(_get_weather_tools())
        body = _element("city", "Seattle")
        result = parser._parse_invoke_params("get_weather", body)
        assert result == {"city": "Seattle"}

    def test_multiple_params(self):
        parser = _make_parser(_get_weather_tools())
        body = _element("city", "Seattle") + _element("days", "5")
        result = parser._parse_invoke_params("get_weather", body)
        assert result == {"city": "Seattle", "days": 5}

    def test_nested_params(self):
        parser = _make_parser(_create_order_tools())
        body = _element("shipping", _element("city", "SG") + _element("zip", "12345"))
        result = parser._parse_invoke_params("create_order", body)
        assert result["shipping"] == {"city": "SG", "zip": 12345}


# ===========================================================================
# MinimaxM3ToolParser: _parse_element_children
# ===========================================================================


class TestParseElementChildren:
    def test_empty_text(self):
        parser = _make_parser()
        result = parser._parse_element_children("")
        assert result == []

    def test_single_leaf_element(self):
        parser = _make_parser()
        result = parser._parse_element_children(_element("name", "value"))
        assert len(result) == 1
        assert result[0] == ("name", "value")

    def test_multiple_leaf_elements(self):
        parser = _make_parser()
        body = _element("a", "1") + _element("b", "2")
        result = parser._parse_element_children(body)
        assert len(result) == 2
        assert result[0] == ("a", "1")
        assert result[1] == ("b", "2")

    def test_nested_elements(self):
        parser = _make_parser()
        body = _element("outer", _element("inner", "val"))
        result = parser._parse_element_children(body)
        assert len(result) == 1
        assert result[0][0] == "outer"
        assert isinstance(result[0][1], list)
        assert result[0][1][0] == ("inner", "val")

    def test_closing_tag_at_top_level_stops(self):
        parser = _make_parser()
        body = _element("a", "1") + f"{_ELEMENT_END_START}b>"
        result = parser._parse_element_children(body)
        assert len(result) == 1  # only 'a' parsed

    def test_incomplete_element_missing_close(self):
        parser = _make_parser()
        body = f"{_ELEMENT_START}x>content"  # no closing tag
        result = parser._parse_element_children(body)
        assert result == []

    def test_stray_text_between_elements(self):
        parser = _make_parser()
        body = _element("a", "1") + "junk text"
        result = parser._parse_element_children(body)
        assert len(result) == 1  # junk text stops parsing
        assert result[0] == ("a", "1")

    def test_unexpected_namespace_stops(self):
        parser = _make_parser()
        body = _element("a", "1") + _NAMESPACE
        result = parser._parse_element_children(body)
        assert len(result) == 1  # only 'a'

    def test_skip_whitespace_between_elements(self):
        parser = _make_parser()
        body = _element("a", "1") + "  \n  " + _element("b", "2")
        result = parser._parse_element_children(body)
        assert len(result) == 2


# ===========================================================================
# MinimaxM3ToolParser: streaming state management
# ===========================================================================


class TestResetStreamingState:
    def test_all_fields_reset(self):
        parser = _make_parser()
        parser._buffer = "data"
        parser._in_tool_block = True
        parser._tool_block_done = True
        parser._emitted_tool_count = 5
        parser._invoke_names.append("fn")
        parser._tool_call_ids.append("id")
        parser.prev_tool_call_arr.append({"x": 1})
        parser.streamed_args_for_tool.append("args")
        parser.current_tool_id = 3
        parser.current_tool_name_sent = True

        parser._reset_streaming_state()

        assert parser._buffer == ""
        assert parser._in_tool_block is False
        assert parser._tool_block_done is False
        assert parser._emitted_tool_count == 0
        assert parser._invoke_names == []
        assert parser._tool_call_ids == []
        assert parser.prev_tool_call_arr == []
        assert parser.streamed_args_for_tool == []
        assert parser.current_tool_id == -1
        assert parser.current_tool_name_sent is False


class TestStartsNewToolBlock:
    def test_no_previous_has_start(self):
        parser = _make_parser()
        assert parser._starts_new_tool_block("", _TOOL_CALL_START) is True

    def test_no_previous_no_start(self):
        parser = _make_parser()
        assert parser._starts_new_tool_block("", "hello") is False

    def test_previous_has_no_start_current_has_start(self):
        parser = _make_parser()
        assert parser._starts_new_tool_block("hello", "hello" + _TOOL_CALL_START) is True

    def test_previous_has_start(self):
        parser = _make_parser()
        assert parser._starts_new_tool_block(_TOOL_CALL_START, _TOOL_CALL_START + "more") is False


class TestGenerateToolCallId:
    def test_format(self):
        parser = _make_parser()
        tid = parser._generate_tool_call_id()
        assert tid.startswith("call_")
        assert len(tid) == 5 + 24  # "call_" + 24 hex chars


# ===========================================================================
# MinimaxM3ToolParser: _find_invoke_blocks
# ===========================================================================


class TestFindInvokeBlocks:
    def test_no_invoke(self):
        parser = _make_parser()
        result = parser._find_invoke_blocks("plain text")
        assert result == []

    def test_complete_invoke(self):
        parser = _make_parser()
        inv = _invoke("get_weather", _element("city", "Seattle"))
        result = parser._find_invoke_blocks(inv)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["complete"] is True
        assert "city" in result[0]["body"]

    def test_incomplete_invoke_no_end(self):
        parser = _make_parser()
        inv_partial = f'{_INVOKE_START} name="get_weather">{_element("city", "Seattle")}'
        result = parser._find_invoke_blocks(inv_partial)
        assert len(result) == 1
        assert result[0]["complete"] is False

    def test_multiple_invokes(self):
        parser = _make_parser()
        invokes = _invoke("get_weather", _element("city", "A")) + _invoke("get_weather", _element("city", "B"))
        result = parser._find_invoke_blocks(invokes)
        assert len(result) == 2
        assert result[0]["complete"] is True
        assert result[1]["complete"] is True

    def test_invoke_with_bad_name_format(self):
        parser = _make_parser()
        bad = f"{_INVOKE_START} no_name_attr>body{_INVOKE_END}"
        result = parser._find_invoke_blocks(bad)
        assert result == []


# ===========================================================================
# MinimaxM3ToolParser: _extract_completed_invokes
# ===========================================================================


class TestExtractCompletedInvokes:
    def test_single_complete_invoke(self):
        parser = _make_parser(_get_weather_tools())
        inv = _invoke("get_weather", _element("city", "Seattle"))
        result = parser._extract_completed_invokes(inv)
        assert len(result) == 1
        assert result[0].function.name == "get_weather"
        args = json.loads(result[0].function.arguments)
        assert args == {"city": "Seattle"}

    def test_already_emitted_skipped(self):
        parser = _make_parser(_get_weather_tools())
        inv = _invoke("get_weather", _element("city", "Seattle"))
        parser._emitted_tool_count = 1
        result = parser._extract_completed_invokes(inv)
        assert result == []

    def test_incomplete_invoke_emits_partial(self):
        parser = _make_parser(_get_weather_tools())
        inv_partial = f'{_INVOKE_START} name="get_weather">{_element("city", "Seattle")}'
        result = parser._extract_completed_invokes(inv_partial)
        # First sight should emit name
        assert len(result) >= 1

    def test_mixed_complete_and_incomplete(self):
        parser = _make_parser(_get_weather_tools())
        inv1 = _invoke("get_weather", _element("city", "Seattle"))
        inv2 = f'{_INVOKE_START} name="get_weather">{_element("days", "5")}'
        result = parser._extract_completed_invokes(inv1 + inv2)
        # First invoke is complete -> emitted
        assert len(result) >= 1


# ===========================================================================
# MinimaxM3ToolParser: _try_stream_partial_args
# ===========================================================================


class TestTryStreamPartialArgs:
    def test_first_sight_emits_name(self):
        parser = _make_parser(_get_weather_tools())
        result = parser._try_stream_partial_args(0, "get_weather", "")
        assert result is not None
        assert result.function.name == "get_weather"
        assert result.function.arguments == ""

    def test_same_args_no_new_emission(self):
        parser = _make_parser(_get_weather_tools())
        # First call: emits name
        parser._try_stream_partial_args(0, "get_weather", "")
        # Now set the streamed args to match what will be computed
        body = _element("city", "Seattle")
        partial = parser._build_partial_arguments("get_weather", body)
        parser.streamed_args_for_tool[0] = partial
        # Second call: same body, already streamed -> should return None
        result = parser._try_stream_partial_args(0, "get_weather", body)
        assert result is None

    def test_different_args_emits_delta(self):
        parser = _make_parser(_get_weather_tools())
        parser._invoke_names.append("get_weather")
        parser.streamed_args_for_tool.append("{")
        parser._tool_call_ids.append("call_test123")
        result = parser._try_stream_partial_args(0, "get_weather", _element("city", "Seattle"))
        assert result is not None


# ===========================================================================
# MinimaxM3ToolParser: _build_partial_arguments
# ===========================================================================


class TestBuildPartialArguments:
    def test_complete_params(self):
        parser = _make_parser(_get_weather_tools())
        body = _element("city", "Seattle")
        result = parser._build_partial_arguments("get_weather", body)
        assert '"city":"Seattle"' in result

    def test_partial_params_no_close(self):
        parser = _make_parser(_get_weather_tools())
        body = f"{_ELEMENT_START}city>Seattle"  # no closing tag
        result = parser._build_partial_arguments("get_weather", body)
        # Partial: should have key and partial value
        assert '"city"' in result or result == ""

    def test_empty_body(self):
        parser = _make_parser(_get_weather_tools())
        result = parser._build_partial_arguments("get_weather", "")
        assert result == ""

    def test_non_element_text_skipped(self):
        parser = _make_parser(_get_weather_tools())
        result = parser._build_partial_arguments("get_weather", "plain text")
        assert result == ""

    def test_multiple_params(self):
        parser = _make_parser(_get_weather_tools())
        body = _element("city", "Seattle") + _element("days", "5")
        result = parser._build_partial_arguments("get_weather", body)
        assert '"city":"Seattle"' in result
        assert '"days":5' in result


# ===========================================================================
# MinimaxM3ToolParser: _serialize_param_value (static)
# ===========================================================================


class TestSerializeParamValue:
    def test_complete_string(self):
        result = MinimaxM3ToolParser._serialize_param_value("hello", ["string"], is_complete=True)
        assert "hello" in result

    def test_complete_integer(self):
        result = MinimaxM3ToolParser._serialize_param_value("42", ["integer"], is_complete=True)
        assert "42" in result

    def test_complete_boolean(self):
        result = MinimaxM3ToolParser._serialize_param_value("true", ["boolean"], is_complete=True)
        assert result == "true"

    def test_incomplete_empty_value(self):
        result = MinimaxM3ToolParser._serialize_param_value("", ["string"], is_complete=False)
        assert result == ""

    def test_incomplete_null(self):
        result = MinimaxM3ToolParser._serialize_param_value("nu", ["null"], is_complete=False)
        assert result == "nu"

    def test_incomplete_boolean_true(self):
        result = MinimaxM3ToolParser._serialize_param_value("tr", ["boolean"], is_complete=False)
        assert result == "tr"

    def test_incomplete_boolean_false(self):
        result = MinimaxM3ToolParser._serialize_param_value("fal", ["boolean"], is_complete=False)
        assert result == "fal"

    def test_incomplete_numeric(self):
        result = MinimaxM3ToolParser._serialize_param_value("12", ["integer"], is_complete=False)
        assert result == "12"

    def test_incomplete_object(self):
        result = MinimaxM3ToolParser._serialize_param_value('{"key":', ["object"], is_complete=False)
        assert result == '{"key":'

    def test_incomplete_string_quoted(self):
        result = MinimaxM3ToolParser._serialize_param_value("hello", ["string"], is_complete=False)
        assert "hello" in result  # JSON-quoted


# ===========================================================================
# MinimaxM3ToolParser: extract_tool_calls_streaming
# ===========================================================================


class TestExtractToolCallsStreaming:
    def test_plain_text_before_tool_call(self):
        parser = _make_parser()
        results = _feed(parser, ["Hello, ", "world!"])
        content = _collect_content(results)
        assert content == "Hello, world!"

    def test_tool_call_start_streaming(self):
        parser = _make_parser(_get_weather_tools())
        chunks = [
            _TOOL_CALL_START,
            _invoke("get_weather", _element("city", "Seattle")),
            _TOOL_CALL_END,
        ]
        results = _feed(parser, chunks)
        tcs = _collect_tool_calls(results)
        assert len(tcs) >= 1

    def test_prefix_text_then_tool_call(self):
        parser = _make_parser(_get_weather_tools())
        chunks = [
            "Let me check. ",
            _TOOL_CALL_START,
            _invoke("get_weather", _element("city", "Seattle")),
            _TOOL_CALL_END,
        ]
        results = _feed(parser, chunks)
        content = _collect_content(results)
        assert "Let me check." in content

    def test_streaming_with_token_ids(self):
        parser = _make_parser(_get_weather_tools())
        # Split into chunks and simulate
        results = _feed(
            parser,
            [
                _TOOL_CALL_START,
                _invoke("get_weather", _element("city", "Seattle")),
                _TOOL_CALL_END,
            ],
        )
        tcs = _collect_tool_calls(results)
        assert len(tcs) >= 1

    def test_no_tool_call_returns_none(self):
        parser = _make_parser()
        result = parser.extract_tool_calls_streaming(
            previous_text="hello",
            current_text="hello world",
            delta_text=" world",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        assert result is not None  # passes through content

    def test_streaming_resets_on_new_block(self):
        parser = _make_parser(_get_weather_tools())
        parser._in_tool_block = True
        parser._tool_block_done = True
        parser._emitted_tool_count = 3
        parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=_TOOL_CALL_START,
            delta_text=_TOOL_CALL_START,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        assert parser._emitted_tool_count == 0  # reset

    def test_tool_block_detection_with_start_in_current(self):
        parser = _make_parser()
        result = parser.extract_tool_calls_streaming(
            previous_text="no start",
            current_text="no start" + _TOOL_CALL_START,
            delta_text=_TOOL_CALL_START,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=None,
        )
        assert result is not None


# ===========================================================================
# Registration function
# ===========================================================================


class TestRegistration:
    def test_registration_idempotent(self):
        """Calling the registration function twice should not raise."""
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
        from vllm_ascend.patch.platform.patch_minimax_m3_tool_call_parser import (
            _register_minimax_m3_tool_parser,
        )

        # First call
        _register_minimax_m3_tool_parser()
        # Should be registered now
        assert "minimax_m3" in ToolParserManager.lazy_parsers
        # Second call — should not raise or double-register
        _register_minimax_m3_tool_parser()
        assert "minimax_m3" in ToolParserManager.lazy_parsers


# ===========================================================================
# Integration-style: complete tool call parsing roundtrip
# ===========================================================================


class TestRoundtrip:
    """End-to-end tests that exercise the full parser pipeline."""

    def test_complete_tool_call_roundtrip(self):
        parser = _make_parser(_create_order_tools())
        shipping = _element(
            "shipping",
            _element("city", "Singapore") + _element("zip", "018956"),
        )
        # Items array: direct child elements (the actual model output format
        # has the wrapper element whose name is ignored for schema-typed arrays).
        items_body = _element("sku", "book-001") + _element("qty", "2")
        items = _element("items", items_body)
        body = _element("user_id", "42") + _element("urgent", "true") + shipping + items
        tc = _tool_block(_invoke("create_order", body))
        output = parser.extract_tool_calls(tc, None)
        assert output.tools_called is True
        args = json.loads(output.tool_calls[0].function.arguments)
        assert args["user_id"] == 42
        assert args["urgent"] is True
        assert args["shipping"]["city"] == "Singapore"
        assert args["shipping"]["zip"] == 18956
        # items is an array of objects
        assert isinstance(args["items"], list)
        assert len(args["items"]) == 1
        assert args["items"][0]["sku"] == "book-001"
        assert args["items"][0]["qty"] == 2

    def test_multiple_invokes_different_functions(self):
        tools = _get_weather_tools() + _create_order_tools()
        parser = _make_parser(tools)
        inv1 = _invoke("get_weather", _element("city", "Seattle") + _element("days", "3"))
        inv2 = _invoke(
            "create_order",
            _element("user_id", "99") + _element("urgent", "false"),
        )
        tc = _tool_block(inv1 + "\n" + inv2)
        output = parser.extract_tool_calls(tc, None)
        assert len(output.tool_calls) == 2
        assert output.tool_calls[0].function.name == "get_weather"
        assert output.tool_calls[1].function.name == "create_order"
        args0 = json.loads(output.tool_calls[0].function.arguments)
        args1 = json.loads(output.tool_calls[1].function.arguments)
        assert args0 == {"city": "Seattle", "days": 3}
        assert args1 == {"user_id": 99, "urgent": False}
