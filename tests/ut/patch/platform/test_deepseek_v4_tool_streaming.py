# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.tool_parsers.deepseekv4_engine_tool_parser import (
    DeepSeekV4EngineToolParser,
)

TOOL_START = "<｜DSML｜tool_calls>\n"
INVOKE_START = '<｜DSML｜invoke name="emit_text">\n'
INVOKE_END = "</｜DSML｜invoke>\n"
TOOL_END = "</｜DSML｜tool_calls>"
PARAM_CLOSE = "</｜DSML｜parameter>"


class FakeTokenizer:
    def get_vocab(self):
        return {}

    def decode(self, token_ids):
        return ""


def _param_open(name: str, is_string: bool) -> str:
    string_attr = "true" if is_string else "false"
    return f'<｜DSML｜parameter name="{name}" string="{string_attr}">'


def _make_parser(properties: dict, parser_cls=DeepSeekV4Parser):
    tool = ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition(
            name="emit_text",
            parameters={
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        ),
    )
    request = MagicMock(tools=[tool], tool_choice="auto")
    parser = parser_cls(
        FakeTokenizer(),
        tools=[tool],
        chat_template_kwargs={"enable_thinking": False},
    )
    return parser, request


def _arguments_from_delta(delta) -> list[str]:
    arguments = []
    if delta and delta.tool_calls:
        for tool_call in delta.tool_calls:
            if tool_call.function and tool_call.function.arguments:
                arguments.append(tool_call.function.arguments)
    return arguments


def _stream_chunks(parser, request, chunks: list[str], *, finish: bool = False):
    previous_text = ""
    argument_deltas: list[tuple[int, str]] = []
    for index, delta_text in enumerate(chunks):
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[index + 1],
            request=request,
        )
        previous_text = current_text
        argument_deltas.extend((index, arguments) for arguments in _arguments_from_delta(delta))

    if finish:
        argument_deltas.extend(
            (len(chunks), arguments) for arguments in _arguments_from_delta(parser.finish_streaming())
        )
    return argument_deltas


def _long_string_chunks(body: str, chunk_size: int = 32) -> list[str]:
    chunks = [TOOL_START + INVOKE_START + _param_open("text", True)]
    chunks.extend(body[i : i + chunk_size] for i in range(0, len(body), chunk_size))
    chunks.append(PARAM_CLOSE + "\n" + INVOKE_END + TOOL_END)
    return chunks


def _joined_arguments(argument_deltas: list[tuple[int, str]]) -> str:
    return "".join(value for _, value in argument_deltas)


def test_long_string_streams_before_parameter_close():
    body = "A" * 4096
    parser, request = _make_parser(
        {"text": {"type": "string"}},
        parser_cls=DeepSeekV4EngineToolParser,
    )
    chunks = _long_string_chunks(body)
    argument_deltas = _stream_chunks(parser, request, chunks)

    body_deltas = [value for index, value in argument_deltas if 0 < index < len(chunks) - 1]

    assert len(body_deltas) > 1
    assert json.loads(_joined_arguments(argument_deltas)) == {"text": body}


def test_string_escaping_and_unicode_reconstruct_exactly():
    body = 'quote=" slash=\\ newline=\n tab=\t nul=\x00 angle=> unicode=杭州'
    parser, request = _make_parser({"text": {"type": "string"}})
    argument_deltas = _stream_chunks(
        parser,
        request,
        _long_string_chunks(body, chunk_size=3),
    )

    assert json.loads(_joined_arguments(argument_deltas)) == {"text": body}


def test_split_parameter_close_does_not_leak_into_arguments():
    parser, request = _make_parser({"text": {"type": "string"}})
    chunks = [
        TOOL_START + INVOKE_START + _param_open("text", True),
        "alpha",
        "</｜DSML｜para",
        "meter>\n" + INVOKE_END + TOOL_END,
    ]
    argument_deltas = _stream_chunks(parser, request, chunks)

    assert all("｜DSML｜" not in value for _, value in argument_deltas)
    assert json.loads(_joined_arguments(argument_deltas)) == {"text": "alpha"}


def test_string_numeric_and_second_string_keep_types_and_streaming():
    properties = {
        "text": {"type": "string"},
        "count": {"type": "integer"},
        "suffix": {"type": "string"},
    }
    parser, request = _make_parser(properties)
    chunks = [
        TOOL_START + INVOKE_START + _param_open("text", True),
        "alpha",
        PARAM_CLOSE + "\n" + _param_open("count", False),
        "42",
        PARAM_CLOSE + "\n" + _param_open("suffix", True),
        "omega",
        PARAM_CLOSE + "\n" + INVOKE_END + TOOL_END,
    ]
    argument_deltas = _stream_chunks(parser, request, chunks)

    assert any(index == 1 and "alpha" in value for index, value in argument_deltas)
    assert not any(index == 3 for index, _ in argument_deltas)
    assert any(index == 5 and "omega" in value for index, value in argument_deltas)
    assert json.loads(_joined_arguments(argument_deltas)) == {
        "text": "alpha",
        "count": 42,
        "suffix": "omega",
    }


def test_split_next_parameter_header_does_not_leak_into_previous_string():
    properties = {
        "text": {"type": "string"},
        "suffix": {"type": "string"},
    }
    parser, request = _make_parser(properties)
    chunks = [
        TOOL_START + INVOKE_START + _param_open("text", True),
        "alpha",
        PARAM_CLOSE + '\n<｜DSML｜parameter name="suf',
        'fix" string="true">',
        "omega",
        PARAM_CLOSE + "\n" + INVOKE_END + TOOL_END,
    ]
    argument_deltas = _stream_chunks(parser, request, chunks)

    arguments = _joined_arguments(argument_deltas)
    assert "｜DSML｜" not in arguments
    assert json.loads(arguments) == {"text": "alpha", "suffix": "omega"}


def test_nullable_string_remains_buffered_until_stable():
    body = "not-null" * 128
    parser, request = _make_parser(
        {
            "text": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ]
            }
        }
    )
    chunks = _long_string_chunks(body)
    argument_deltas = _stream_chunks(parser, request, chunks)

    assert not any(0 < index < len(chunks) - 1 for index, _ in argument_deltas)
    assert json.loads(_joined_arguments(argument_deltas)) == {"text": body}


def test_escaped_parameter_name_remains_buffered_until_stable():
    parameter_name = "text\\key"
    body = "value" * 128
    parser, request = _make_parser({parameter_name: {"type": "string"}})
    chunks = [TOOL_START + INVOKE_START + _param_open(parameter_name, True)]
    chunks.extend(body[i : i + 32] for i in range(0, len(body), 32))
    chunks.append(PARAM_CLOSE + "\n" + INVOKE_END + TOOL_END)
    argument_deltas = _stream_chunks(parser, request, chunks)

    assert not any(0 < index < len(chunks) - 1 for index, _ in argument_deltas)
    assert json.loads(_joined_arguments(argument_deltas)) == {parameter_name: body}


def test_truncated_string_remains_invalid_json():
    parser, request = _make_parser({"text": {"type": "string"}})
    chunks = [
        TOOL_START + INVOKE_START + _param_open("text", True),
        "unfinished ",
        "body",
    ]
    argument_deltas = _stream_chunks(parser, request, chunks, finish=True)

    with pytest.raises(json.JSONDecodeError):
        json.loads(_joined_arguments(argument_deltas))


def test_long_body_does_not_repeatedly_call_converter():
    body = "A" * 4096
    parser, request = _make_parser({"text": {"type": "string"}})
    original_converter = parser._arg_converter
    converter_calls = 0

    def counting_converter(raw_args: str, partial: bool) -> str:
        nonlocal converter_calls
        converter_calls += 1
        return original_converter(raw_args, partial)

    parser._arg_converter = counting_converter
    argument_deltas = _stream_chunks(parser, request, _long_string_chunks(body))

    assert converter_calls <= 4
    assert json.loads(_joined_arguments(argument_deltas)) == {"text": body}


def test_parser_reuse_clears_direct_streaming_state():
    parser, request = _make_parser({"text": {"type": "string"}})
    first = _stream_chunks(parser, request, _long_string_chunks("first"))
    parser._reset()
    second = _stream_chunks(parser, request, _long_string_chunks("second"))

    assert json.loads(_joined_arguments(first)) == {"text": "first"}
    assert json.loads(_joined_arguments(second)) == {"text": "second"}
