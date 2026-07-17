# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest

from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.24.0"):
    pytest.skip(
        "DeepSeekV4ToolParser compatibility patch only applies to vLLM 0.24.0",
        allow_module_level=True,
    )

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser  # type: ignore[import-not-found]

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


def test_streaming_strips_non_dsml_special_token_in_content():
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    deltas = _stream(parser, "Good.<｜begin▁of▁sentence｜>", chunk_size=3)
    visible = "".join(d.content or "" for d in deltas if d.content)
    assert "<｜" not in visible
    assert visible == "Good."


def test_streaming_strips_special_token_before_dsml_start():
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    full = "Let me run<｜end▁of▁sentence｜>" + _build_tool_call(
        "plan_trip",
        {"days": 1, "flexible": False, "cities": ["Beijing"], "notes": "x"},
    )
    deltas = _stream(parser, full, chunk_size=4)
    visible = "".join(d.content or "" for d in deltas if d.content)
    arg_chunks = "".join(
        tc.function.arguments
        for delta in deltas
        for tc in delta.tool_calls or []
        if tc.index == 0 and tc.function and tc.function.arguments is not None
    )
    assert "<｜" not in visible
    assert json.loads(arg_chunks) == {
        "days": 1,
        "flexible": False,
        "cities": ["Beijing"],
        "notes": "x",
    }


def test_streaming_holds_back_partial_special_token_across_chunks():
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    # chunk_size=5 makes first chunk "abc<｜" so the trailing "<｜" must hold back.
    deltas = _stream(parser, "abc<｜begin▁of▁sentence｜>def", chunk_size=5)
    visible = "".join(d.content or "" for d in deltas if d.content)
    assert "<｜" not in visible
    assert visible == "abcdef"


def test_streaming_preserves_dsml_tool_call_through_strip_path():
    # Pins that the strip helper cannot misfire on the real DSML start token.
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    full = "prefix " + _build_tool_call(
        "plan_trip",
        {"days": 2, "flexible": True, "cities": ["A"], "notes": "y"},
    )
    deltas = _stream(parser, full, chunk_size=7)
    visible = "".join(d.content or "" for d in deltas if d.content)
    arg_chunks = "".join(
        tc.function.arguments
        for delta in deltas
        for tc in delta.tool_calls or []
        if tc.index == 0 and tc.function and tc.function.arguments is not None
    )
    assert visible.strip() == "prefix"
    assert json.loads(arg_chunks) == {
        "days": 2,
        "flexible": True,
        "cities": ["A"],
        "notes": "y",
    }


def test_streaming_strips_complete_token_and_holds_partial_in_same_delta():
    # One delta contains both a complete special token and a trailing partial:
    # the complete one must be stripped and the partial held back so it can
    # complete in the next delta.
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    deltas = _stream(
        parser,
        "a<｜end▁of▁sentence｜>b<｜begin▁of▁sentence｜>c",
        chunk_size=200,
    )
    visible = "".join(d.content or "" for d in deltas if d.content)
    assert "<｜" not in visible
    assert visible == "abc"


def test_streaming_preserves_lone_pipe_bracket_before_dsml_start():
    # Regression guard: a bare `<｜` fragment immediately preceding the DSML
    # start token must NOT be silently dropped — the DSML start is already
    # confirmed, so the fragment can't grow into a real special token.
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    full = "x<｜y" + _build_tool_call(
        "plan_trip",
        {"days": 1, "flexible": False, "cities": ["A"], "notes": "n"},
    )
    deltas = _stream(parser, full, chunk_size=200)
    visible = "".join(d.content or "" for d in deltas if d.content)
    assert visible == "x<｜y"
    arg_chunks = "".join(
        tc.function.arguments
        for delta in deltas
        for tc in delta.tool_calls or []
        if tc.index == 0 and tc.function and tc.function.arguments is not None
    )
    assert json.loads(arg_chunks) == {
        "days": 1,
        "flexible": False,
        "cities": ["A"],
        "notes": "n",
    }


def _final_empty_delta(parser, full_text):
    # Mirror vLLM's terminal call with empty delta_text after the last token.
    return parser.extract_tool_calls_streaming(
        previous_text=full_text,
        current_text=full_text,
        delta_text="",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=ChatCompletionRequest(model="deepseek-ai/DeepSeek-V2-Chat", messages=[], tools=[_tools()]),
    )


def test_streaming_flushes_unclosed_pipe_text_at_stream_end():
    # Plain text ending in an unclosed fullwidth "<｜" must NOT be lost: the
    # trailing-partial guard holds it mid-stream, and the terminal (empty) delta
    # flushes it back as visible content.
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    text = "threshold x <｜ y"
    deltas = _stream(parser, text, chunk_size=4)
    final = _final_empty_delta(parser, text)
    if final is not None:
        deltas.append(final)
    visible = "".join(d.content or "" for d in deltas if d.content)
    assert "<｜" in visible  # the literal text is preserved, not dropped
    assert visible == "threshold x <｜ y"


def test_streaming_drops_truncated_dsml_start_prefix_at_stream_end():
    # A response truncated mid-`<｜DSML｜tool_calls>` must not leak the partial
    # start marker as visible content.
    parser = DeepSeekV4ToolParser(MOCK_TOKENIZER)
    text = "ok <｜DSML｜tool"
    deltas = _stream(parser, text, chunk_size=3)
    final = _final_empty_delta(parser, text)
    if final is not None:
        deltas.append(final)
    visible = "".join(d.content or "" for d in deltas if d.content)
    assert visible == "ok "
