# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.parser_manager import ParserManager
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

from vllm_ascend.patch.platform import patch_deepseek_v4_tool_call_parser

MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {}

TC_START = "<｜DSML｜tool_calls>"
TC_END = "</｜DSML｜tool_calls>"
INV_START = '<｜DSML｜invoke name="'
INV_END = "</｜DSML｜invoke>"
PARAM_START = '<｜DSML｜parameter name="'
PARAM_END = "</｜DSML｜parameter>"


def _build_tool_call(
    function_name: str,
    tool_args: dict[str, str | int | bool | list[str]],
) -> str:
    params = []
    for key, value in tool_args.items():
        if isinstance(value, bool):
            value = "false" if value is False else "true"
            string_attr = "false"
        elif isinstance(value, int):
            value = str(value)
            string_attr = "false"
        elif isinstance(value, list):
            value = json.dumps(value, ensure_ascii=False)
            string_attr = "false"
        else:
            value = str(value)
            string_attr = "true"

        params.append(f'{PARAM_START}{key}" string="{string_attr}">{value}{PARAM_END}\n')

    return f'{TC_START}\n{INV_START}{function_name}">\n' + "".join(params) + f"{INV_END}\n{TC_END}"


def _stream(
    parser: DeepSeekV4ToolParser,
    full_text: str,
    chunk_size: int = 5,
    tools=None,
):
    deltas = []
    previous_text = ""
    for start in range(0, len(full_text), chunk_size):
        delta_text = full_text[start : start + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=ChatCompletionRequest(
                model="deepseek-ai/DeepSeek-V2-Chat",
                messages=[],
                tools=tools or [_tools()],
            ),
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
    assert not parser._pending_delta_messages
    return deltas


def _tools():
    return {
        "type": "function",
        "function": {
            "name": "plan_trip",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer"},
                    "flexible": {"type": "boolean"},
                    "cities": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "required": ["days", "flexible", "cities", "notes"],
            },
        },
    }


def _unified_parser():
    parser_cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v4",
        enable_auto_tools=True,
        model_name="deepseek-v4-flash",
    )
    assert parser_cls is not None
    return parser_cls(MOCK_TOKENIZER, tools=[_tools()])


def _request(api: str, choice: str):
    function = _tools()["function"]
    if api == "chat":
        tool_choice = (
            "required" if choice == "required" else {"type": "function", "function": {"name": function["name"]}}
        )
        return ChatCompletionRequest(model="deepseek-v4-flash", messages=[], tools=[_tools()], tool_choice=tool_choice)
    tool_choice = "required" if choice == "required" else {"type": "function", "name": function["name"]}
    return ResponsesRequest(
        model="deepseek-v4-flash",
        input="Plan a trip",
        tools=[{"type": "function", **function}],
        tool_choice=tool_choice,
    )


def _model_output(api: str, choice: str, strict: bool, arguments: dict):
    if api == "chat" and strict:
        return _build_tool_call("plan_trip", arguments)
    if choice == "required":
        return json.dumps([{"name": "plan_trip", "parameters": arguments}])
    return json.dumps(arguments)


def _stream_unified(parser, request, full_text: str):
    deltas = []
    previous_text = ""
    function_name_returned = False
    for delta_text in full_text:
        current_text = previous_text + delta_text
        delta, function_name_returned = parser._extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=request,
            tool_call_idx=0,
            function_name_returned=function_name_returned,
        )
        if delta is not None:
            deltas.append(delta)
        previous_text = current_text
    if hasattr(parser._tool_parser, "_pending_delta_messages"):
        deltas.extend(parser._tool_parser.drain_pending_tool_call_deltas())
    return deltas


def test_streaming_deepseek_v4_tool_calls_emit_chunked_arguments():
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    full_text = _build_tool_call(
        "plan_trip",
        {
            "days": 3,
            "flexible": False,
            "cities": ["Beijing", "Shanghai", "Tokyo", "New York"],
            "notes": "靠窗座位",
        },
    )

    deltas = _stream(parser, full_text, chunk_size=4)
    tool_chunks = []
    for delta in deltas:
        for tc in delta.tool_calls or []:
            if tc.index == 0 and tc.function and tc.function.arguments is not None:
                tool_chunks.append(tc.function.arguments)

    reconstructed = "".join(tool_chunks)
    assert json.loads(reconstructed) == {
        "days": 3,
        "flexible": False,
        "cities": ["Beijing", "Shanghai", "Tokyo", "New York"],
        "notes": "靠窗座位",
    }

    arg_chunks = [
        tc.function.arguments
        for delta in deltas
        for tc in delta.tool_calls or []
        if tc.index == 0 and tc.function and tc.function.arguments not in (None, "")
    ]
    assert len(arg_chunks) >= 2


def test_streaming_tool_call_metadata_only_first_chunk():
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    full_text = _build_tool_call(
        "plan_trip",
        {
            "days": 3,
            "flexible": False,
            "cities": ["Beijing"],
            "notes": "靠窗座位",
        },
    )

    deltas = _stream(parser, full_text, chunk_size=3)
    header_chunks = [delta for delta in deltas if delta.tool_calls]
    assert len(header_chunks) >= 1
    first = header_chunks[0].tool_calls[0]
    assert first.id is not None
    assert first.type == "function"
    assert first.function and first.function.name == "plan_trip"

    for delta in header_chunks[1:]:
        tc = delta.tool_calls[0]
        assert tc.id is None
        if tc.function:
            assert tc.function.name is None
            assert tc.function.arguments is not None


def test_streaming_wrapper_param_arguments_fragment():
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    full_text = (
        TC_START
        + "\n"
        + f'{INV_START}plan_trip">\n'
        + PARAM_START
        + '__vllm_param_arguments__" string="false">{'
        + '"days":3,"flexible":false,'
        + '"cities":["Beijing","Shanghai","Tokyo","New York"],"notes":"靠窗座位"}</｜DSML｜parameter>\n'
        + INV_END
        + "\n"
        + TC_END
    )

    deltas = _stream(parser, full_text, chunk_size=6)
    arg_chunks = [
        tc.function.arguments
        for delta in deltas
        for tc in delta.tool_calls or []
        if tc.index == 0 and tc.function and tc.function.arguments is not None
    ]

    reconstructed = "".join(arg_chunks)
    assert json.loads(reconstructed) == {
        "days": 3,
        "flexible": False,
        "cities": ["Beijing", "Shanghai", "Tokyo", "New York"],
        "notes": "靠窗座位",
    }
    assert len(reconstructed) > 0


def test_streaming_full_tool_call_single_chunk_drains_all_deltas():
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    full_text = _build_tool_call(
        "plan_trip",
        {
            "days": 3,
            "flexible": False,
            "cities": ["Beijing", "Shanghai"],
            "notes": "靠窗座位",
        },
    )

    delta = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=full_text,
        delta_text=full_text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[1],
        request=ChatCompletionRequest(
            model="deepseek-ai/DeepSeek-V2-Chat",
            messages=[],
            tools=[_tools()],
        ),
    )

    assert delta is not None
    assert not parser._pending_delta_messages
    assert delta.tool_calls
    tool_call = delta.tool_calls[0]
    assert tool_call.id is not None
    assert tool_call.type == "function"
    assert tool_call.function and tool_call.function.name == "plan_trip"
    assert json.loads(tool_call.function.arguments) == {
        "days": 3,
        "flexible": False,
        "cities": ["Beijing", "Shanghai"],
        "notes": "靠窗座位",
    }


def test_streaming_matches_non_streaming_conversion_fallbacks():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "coerce",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "union_value": {"type": ["null", "string"]},
                        "bad_int": {"type": "integer"},
                        "nullable_string": {"type": ["null", "string"]},
                        "null_string": {"type": "string"},
                        "whole_number": {"type": "number"},
                    },
                },
            },
        }
    ]
    full_text = (
        f"{TC_START}\n"
        f'{INV_START}coerce">\n'
        f'{PARAM_START}union_value" string="false">hello{PARAM_END}\n'
        f'{PARAM_START}bad_int" string="false">abc{PARAM_END}\n'
        f'{PARAM_START}nullable_string" string="false">null{PARAM_END}\n'
        f'{PARAM_START}null_string" string="false">null{PARAM_END}\n'
        f'{PARAM_START}whole_number" string="false">3.0{PARAM_END}\n'
        f"{INV_END}\n"
        f"{TC_END}"
    )
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V2-Chat",
        messages=[],
        tools=tools,
    )

    non_streaming = DeepSeekV4ToolParser(MOCK_TOKENIZER).extract_tool_calls(full_text, request)
    deltas = _stream(DeepSeekV4ToolParser(MOCK_TOKENIZER), full_text, chunk_size=4, tools=tools)

    stream_args = json.loads(
        "".join(
            tc.function.arguments
            for delta in deltas
            for tc in delta.tool_calls or []
            if tc.index == 0 and tc.function and tc.function.arguments is not None
        )
    )
    expected = {
        "union_value": "hello",
        "bad_int": "abc",
        "nullable_string": None,
        "null_string": "null",
        "whole_number": 3,
    }
    assert stream_args == expected
    assert json.loads(non_streaming.tool_calls[0].function.arguments) == expected


def test_composed_schema_conversion_in_streaming():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_timer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wait": {
                            "anyOf": [
                                {"type": "object"},
                                {"type": "null"},
                            ],
                        },
                        "patches": {
                            "allOf": [
                                {"type": "array", "items": {"type": "object"}},
                            ],
                        },
                    },
                },
            },
        }
    ]
    full_text = (
        f"{TC_START}\n"
        f'{INV_START}set_timer">\n'
        f'{PARAM_START}wait" string="false">'
        f'{{"type":"for","minutes":2880}}'
        f"{PARAM_END}\n"
        f'{PARAM_START}patches" string="false">'
        f'[{{"op":"replace","path":"/schedule","value":"quiet"}}]'
        f"{PARAM_END}\n"
        f"{INV_END}\n"
        f"{TC_END}"
    )

    deltas = _stream(DeepSeekV4ToolParser(MOCK_TOKENIZER), full_text, chunk_size=5, tools=tools)
    args = json.loads(
        "".join(
            tc.function.arguments
            for delta in deltas
            for tc in delta.tool_calls or []
            if tc.index == 0 and tc.function and tc.function.arguments is not None
        )
    )

    assert args == {
        "wait": {"type": "for", "minutes": 2880},
        "patches": [{"op": "replace", "path": "/schedule", "value": "quiet"}],
    }


def test_registered_parser_is_patch_loaded():
    # Regression check that Ascend patch applies at import-time.
    assert (
        DeepSeekV4ToolParser.extract_tool_calls_streaming
        is patch_deepseek_v4_tool_call_parser._patched_extract_tool_calls_streaming
    )
    assert (
        DelegatingParser._extract_tool_calls
        is patch_deepseek_v4_tool_call_parser._patched_delegating_extract_tool_calls
    )
    assert (
        DelegatingParser._extract_tool_calls_streaming
        is patch_deepseek_v4_tool_call_parser._patched_delegating_extract_tool_calls_streaming
    )
    assert DeepSeekV4ToolParser.supports_required_and_named is True


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("choice", ["required", "named"])
def test_required_and_named_non_streaming_routing(monkeypatch, strict, api, choice):
    monkeypatch.setattr(patch_deepseek_v4_tool_call_parser, "VLLM_ENFORCE_STRICT_TOOL_CALLING", strict)
    request = _request(api, choice)
    original_request = request.model_dump()
    arguments = {"days": 3, "flexible": False, "cities": ["Beijing"], "notes": "window seat"}

    parser = _unified_parser()
    model_output = _model_output(api, choice, strict, arguments)
    if isinstance(request, ResponsesRequest):
        tool_calls, content = parser._parse_tool_calls(request=request, content=model_output, enable_auto_tools=True)
    else:
        tool_calls, content = parser._extract_tool_calls(content=model_output, request=request, enable_auto_tools=True)

    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "plan_trip"
    assert json.loads(tool_calls[0].arguments) == arguments
    assert content is None
    assert request.model_dump() == original_request
    assert DeepSeekV4ToolParser.supports_required_and_named is True


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("choice", ["required", "named"])
def test_required_and_named_streaming_routing(monkeypatch, strict, api, choice):
    monkeypatch.setattr(patch_deepseek_v4_tool_call_parser, "VLLM_ENFORCE_STRICT_TOOL_CALLING", strict)
    request = _request(api, choice)
    original_request = request.model_dump()
    arguments = {"days": 3, "flexible": False, "cities": ["Beijing"], "notes": "window seat"}

    deltas = _stream_unified(_unified_parser(), request, _model_output(api, choice, strict, arguments))
    calls = [tc for delta in deltas for tc in delta.tool_calls or []]

    assert calls
    assert {tc.index for tc in calls} == {0}
    assert [tc.function.name for tc in calls if tc.function and tc.function.name] == ["plan_trip"]
    arguments_text = "".join(tc.function.arguments or "" for tc in calls if tc.function)
    assert json.loads(arguments_text) == arguments
    assert not any(delta.content for delta in deltas)
    assert request.model_dump() == original_request
    assert DeepSeekV4ToolParser.supports_required_and_named is True


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("choice", ["required", "named"])
def test_strict_generation_format_is_request_specific(monkeypatch, api, choice):
    monkeypatch.setattr("vllm.tool_parsers.abstract_tool_parser.VLLM_ENFORCE_STRICT_TOOL_CALLING", True)
    monkeypatch.setattr("vllm.tool_parsers.structural_tag_registry._enable_structured_outputs_in_reasoning", True)
    request = _unified_parser().adjust_request(_request(api, choice))

    if api == "chat":
        assert request.structured_outputs.structural_tag is not None
        assert "DSML" in request.structured_outputs.structural_tag
        assert "</think>" in request.structured_outputs.structural_tag
    else:
        assert request.text.format.type == "json_schema"


def test_strict_chat_routing_preserves_other_tool_parsers(monkeypatch):
    monkeypatch.setattr(patch_deepseek_v4_tool_call_parser, "VLLM_ENFORCE_STRICT_TOOL_CALLING", True)
    parser = _unified_parser()
    parser._tool_parser = MagicMock()
    parser._tool_parser.supports_required_and_named = True
    arguments = {"days": 3}
    tool_calls, content = parser._extract_tool_calls(
        content=json.dumps(arguments), request=_request("chat", "named"), enable_auto_tools=True
    )

    assert tool_calls[0].name == "plan_trip"
    assert json.loads(tool_calls[0].arguments) == arguments
    assert content is None
    parser._tool_parser.extract_tool_calls.assert_not_called()

    model_output = json.dumps(arguments)
    delta, _ = parser._extract_tool_calls_streaming(
        previous_text="",
        current_text=model_output,
        delta_text=model_output,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[1],
        request=_request("chat", "named"),
    )
    assert delta.tool_calls[0].function.name == "plan_trip"
    assert json.loads(delta.tool_calls[0].function.arguments) == arguments
    parser._tool_parser.extract_tool_calls_streaming.assert_not_called()
