# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from openai.types.responses.function_tool import FunctionTool
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser

from vllm_ascend.patch.platform import (
    patch_minimax_m2_tool_call_parser as minimax_m2_patch,
)

TC_START_ID = 1
TC_END_ID = 2
EOS_ID = 99


class FakeTokenizer:
    def get_vocab(self):
        return {
            "<minimax:tool_call>": TC_START_ID,
            "</minimax:tool_call>": TC_END_ID,
        }


def _feed(parser: MinimaxM2ToolParser, chunks):
    previous = ""
    results = []
    for chunk in chunks:
        if isinstance(chunk, tuple):
            delta, delta_ids = chunk
        else:
            delta = chunk
            delta_ids = []

        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=delta_ids,
            request=None,
        )
        if result is not None:
            results.append(result)
        previous = current
    return results


def _collect_content(results):
    return "".join(result.content for result in results if result.content)


def _collect_tool_calls(results):
    tool_calls: dict[int, dict[str, Any]] = {}
    for result in results:
        for tool_call in result.tool_calls or []:
            tool_calls.setdefault(
                tool_call.index,
                {
                    "id": None,
                    "name": "",
                    "arguments": "",
                },
            )
            if tool_call.id:
                tool_calls[tool_call.index]["id"] = tool_call.id
            if tool_call.function:
                if tool_call.function.name:
                    tool_calls[tool_call.index]["name"] += tool_call.function.name
                if tool_call.function.arguments:
                    tool_calls[tool_call.index]["arguments"] += tool_call.function.arguments
    return tool_calls


def test_registered_parser_is_patch_loaded():
    assert MinimaxM2ToolParser.extract_tool_calls_streaming is minimax_m2_patch._patched_extract_tool_calls_streaming


def test_plain_content_before_tool_call_is_preserved():
    parser = MinimaxM2ToolParser(FakeTokenizer())
    results = _feed(
        parser,
        [
            "Let me check. ",
            '<minimax:tool_call><invoke name="get_weather">'
            '<parameter name="city">Seattle</parameter>'
            "</invoke></minimax:tool_call>",
        ],
    )

    assert _collect_content(results) == "Let me check. "
    assert len(parser.prev_tool_call_arr) == 1


def test_streaming_emits_tool_name_before_argument_fragments():
    parser = MinimaxM2ToolParser(FakeTokenizer())
    results = _feed(
        parser,
        [
            "Let me check. ",
            "<minimax:tool_call>",
            '<invoke name="get_weather">',
            '<parameter name="city">Sea',
            "ttle</parameter>",
            "</invoke></minimax:tool_call>",
        ],
    )

    tool_deltas = [tc for result in results for tc in (result.tool_calls or [])]
    argument_fragments = [tc.function.arguments for tc in tool_deltas[1:] if tc.function and tc.function.arguments]

    assert _collect_content(results) == "Let me check. "
    assert tool_deltas[0].function.name == "get_weather"
    assert tool_deltas[0].function.arguments is None
    assert argument_fragments == ['{"city":"Sea', 'ttle"', "}"]
    assert "".join(argument_fragments) == '{"city":"Seattle"}'


def test_streaming_partial_arguments_before_invoke_closes():
    parser = MinimaxM2ToolParser(FakeTokenizer())
    results = _feed(
        parser,
        [
            "<minimax:tool_call>",
            '<invoke name="get_weather">',
            '<parameter name="city">Sea',
        ],
    )

    tool_deltas = [tc for result in results for tc in (result.tool_calls or [])]

    assert tool_deltas[0].function.name == "get_weather"
    assert tool_deltas[0].function.arguments is None
    assert tool_deltas[1].function.arguments == '{"city":"Sea'
    assert parser.prev_tool_call_arr == []


def test_complete_single_chunk_still_reconstructs_tool_call():
    parser = MinimaxM2ToolParser(FakeTokenizer())
    results = _feed(
        parser,
        [
            '<minimax:tool_call><invoke name="get_weather">'
            '<parameter name="city">Seattle</parameter>'
            "</invoke></minimax:tool_call>",
            ("", [EOS_ID]),
        ],
    )

    tool_calls = _collect_tool_calls(results)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["arguments"]) == {"city": "Seattle"}
    assert results[-1].content == ""


def test_start_token_can_arrive_as_special_token_id():
    parser = MinimaxM2ToolParser(FakeTokenizer())
    results = _feed(
        parser,
        [
            ("", [TC_START_ID]),
            '<invoke name="get_weather">',
            '<parameter name="city">Seattle</parameter>',
            "</invoke>",
            ("", [TC_END_ID]),
            ("", [EOS_ID]),
        ],
    )

    tool_calls = _collect_tool_calls(results)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["arguments"]) == {"city": "Seattle"}
    assert results[-1].content == ""


def test_start_token_id_survives_empty_chunks_before_invoke_text():
    parser = MinimaxM2ToolParser(FakeTokenizer())
    results = _feed(
        parser,
        [
            ("", [TC_START_ID]),
            ("", []),
            ("", []),
            '<invoke name="get_weather">',
            '<parameter name="city">Seattle</parameter>',
            "</invoke>",
            ("", [TC_END_ID]),
            ("", [EOS_ID]),
        ],
    )

    tool_calls = _collect_tool_calls(results)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["arguments"]) == {"city": "Seattle"}
    assert results[-1].content == ""


def test_chat_tool_schema_drives_type_conversion():
    parser = MinimaxM2ToolParser(
        FakeTokenizer(),
        tools=[
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"days": {"type": "integer"}},
                    },
                ),
            )
        ],
    )
    results = _feed(
        parser,
        [
            '<minimax:tool_call><invoke name="get_weather">'
            '<parameter name="days">5</parameter>'
            "</invoke></minimax:tool_call>",
        ],
    )

    parsed = json.loads(_collect_tool_calls(results)[0]["arguments"])

    assert parsed["days"] == 5
    assert isinstance(parsed["days"], int)


def test_patch_does_not_require_private_v0202_schema_helpers(monkeypatch):
    monkeypatch.delattr(
        MinimaxM2ToolParser,
        "_get_param_types_from_config",
        raising=False,
    )
    monkeypatch.delattr(
        MinimaxM2ToolParser,
        "_convert_param_value_with_types",
        raising=False,
    )
    parser = MinimaxM2ToolParser(
        FakeTokenizer(),
        tools=[
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"days": {"type": "integer"}},
                    },
                ),
            )
        ],
    )
    results = _feed(
        parser,
        [
            '<minimax:tool_call><invoke name="get_weather">'
            '<parameter name="days">5</parameter>'
            "</invoke></minimax:tool_call>",
        ],
    )

    parsed = json.loads(_collect_tool_calls(results)[0]["arguments"])

    assert parsed["days"] == 5
    assert isinstance(parsed["days"], int)


def test_responses_function_tool_schema_drives_type_conversion():
    parser = MinimaxM2ToolParser(
        FakeTokenizer(),
        tools=[
            FunctionTool(
                type="function",
                name="get_weather",
                description="Get weather data",
                parameters={
                    "type": "object",
                    "properties": {"days": {"type": "integer"}},
                },
            )
        ],
    )
    results = _feed(
        parser,
        [
            '<minimax:tool_call><invoke name="get_weather">'
            '<parameter name="days">5</parameter>'
            "</invoke></minimax:tool_call>",
        ],
    )

    parsed = json.loads(_collect_tool_calls(results)[0]["arguments"])

    assert parsed["days"] == 5
    assert isinstance(parsed["days"], int)
