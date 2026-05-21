# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
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
                tools=[_tools()],
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


def test_registered_parser_is_patch_loaded():
    # Regression check that Ascend patch applies at import-time.
    assert (
        DeepSeekV4ToolParser.extract_tool_calls_streaming
        is patch_deepseek_v4_tool_call_parser._patched_extract_tool_calls_streaming
    )
