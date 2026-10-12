# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.deepseek_v4 import DeepSeekV4Parser, deepseek_v4_config
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import ParserEngineConfig, ParserState
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine

from vllm_ascend.patch.platform import patch_deepseek_v4_text_terminals

TOOL_START = "<｜DSML｜tool_calls>\n"
INVOKE_START = '<｜DSML｜invoke name="emit_text">\n'
INVOKE_END = "</｜DSML｜invoke>\n"
TOOL_END = "</｜DSML｜tool_calls>"
PARAM_CLOSE = "</｜DSML｜parameter>"
EOS = "<｜end▁of▁sentence｜>"
CONTENT_EVENTS = (EventType.TEXT_CHUNK, EventType.REASONING_CHUNK)


class FakeTokenizer:
    """Tokenizer that resolves the DeepSeek V4 wire markers as special tokens.

    The DeepSeek V4 tests that only exercise text-form markers return an empty
    vocabulary, which leaves ``_token_id_terminal_names`` empty and hides the
    defect this file covers.
    """

    _VOCAB = {
        "<think>": 100,
        "</think>": 101,
        "<｜DSML｜tool_calls>": 102,
        "</｜DSML｜tool_calls>": 103,
        EOS: 104,
    }

    def get_vocab(self):
        return dict(self._VOCAB)

    def decode(self, token_ids):
        reverse = {token_id: text for text, token_id in self._VOCAB.items()}
        return "".join(reverse.get(token_id, "") for token_id in token_ids)

    @property
    def all_special_tokens(self):
        return list(self._VOCAB)

    @property
    def all_special_ids(self):
        return list(self._VOCAB.values())


def _tool_call_text() -> str:
    return (
        "你好"
        + TOOL_START
        + INVOKE_START
        + '<｜DSML｜parameter name="text" string="true">alpha'
        + PARAM_CLOSE
        + "\n"
        + INVOKE_END
        + TOOL_END
        + EOS
    )


def _text_form_chunks(text: str, chunk_size: int = 8):
    """Split a stream the way a text-token stream arrives: markers as plain text."""
    return [(text[i : i + chunk_size], [900 + i // chunk_size]) for i in range(0, len(text), chunk_size)]


def _feed_all(engine, chunks):
    events = []
    for delta_text, token_ids in chunks:
        events.extend(engine.feed(delta_text, token_ids))
    events.extend(engine.finish())
    return events


def _content_of(events) -> str:
    return "".join(event.value for event in events if event.type in CONTENT_EVENTS)


def _upstream_process_lex_tokens():
    """The unpatched engine method, whether or not our wrapper is installed."""
    current = StreamingParserEngine._process_lex_tokens
    return getattr(current, "__wrapped__", current)


def _make_engine(config=None):
    return StreamingParserEngine(config or deepseek_v4_config(thinking=False), FakeTokenizer())


def test_text_form_dsml_markers_do_not_leak_into_content():
    engine = _make_engine()
    events = _feed_all(engine, _text_form_chunks(_tool_call_text()))

    assert engine._ever_had_token_ids is True
    assert "｜DSML｜" not in _content_of(events)
    assert EOS not in _content_of(events)
    assert any(event.type == EventType.TOOL_CALL_START for event in events)
    # The engine emits the tool name per text chunk, so join before comparing.
    tool_name = "".join(event.value for event in events if event.type == EventType.TOOL_NAME)
    assert tool_name == "emit_text"


def test_upstream_token_id_rule_leaks_the_same_stream(monkeypatch):
    """The unpatched engine emits the DSML envelope as content and as arguments."""
    monkeypatch.setattr(StreamingParserEngine, "_process_lex_tokens", _upstream_process_lex_tokens())
    engine = _make_engine()
    events = _feed_all(engine, _text_form_chunks(_tool_call_text()))

    assert engine._ever_had_token_ids is True
    assert "｜DSML｜" in _content_of(events)
    assert "｜DSML｜" in "".join(
        event.value for event in events if event.type in (EventType.ARG_VALUE_CHUNK, EventType.TOOL_CALL_END)
    )


def test_special_token_ids_still_parse_the_tool_call():
    engine = _make_engine()
    events = _feed_all(
        engine,
        [
            ("你好", [900]),
            ("<｜DSML｜tool_calls>", [102]),
            (INVOKE_START, [901]),
            ('<｜DSML｜parameter name="text" string="true">alpha', [902]),
            (PARAM_CLOSE, [903]),
            (INVOKE_END, [904]),
            (TOOL_END, [103]),
        ],
    )

    assert "｜DSML｜" not in _content_of(events)
    assert any(event.type == EventType.TOOL_CALL_START for event in events)


def test_other_parsers_keep_the_token_id_rule():
    config = ParserEngineConfig(
        name="text_form_test_parser",
        terminals={"MARKER": "<marker>"},
        token_id_terminals={"MARKER": "<marker>"},
        transitions={},
        content_events={ParserState.CONTENT: EventType.TEXT_CHUNK},
    )
    engine = _make_engine(config)
    events = _feed_all(engine, [("<marker>", [900])])

    assert _content_of(events) == "<marker>"
    assert not any(event.type == EventType.TOOL_CALL_START for event in events)


def test_token_id_terminal_names_are_restored_after_each_call():
    """The patch drops the strict set for the duration of one call only."""
    engine = _make_engine()
    declared = engine._token_id_terminal_names

    assert declared, "the parser config declared no token-id terminals"
    _feed_all(engine, _text_form_chunks(_tool_call_text()))

    assert engine._token_id_terminal_names == declared


def test_patch_is_idempotent():
    installed = StreamingParserEngine._process_lex_tokens

    patch_deepseek_v4_text_terminals._install()

    assert StreamingParserEngine._process_lex_tokens is installed


def test_patch_is_skipped_once_upstream_drops_the_token_id_terminals(monkeypatch):
    """A config without the DSML token-id terminals makes the patch obsolete."""
    original = _upstream_process_lex_tokens()
    logger = Mock()

    monkeypatch.delattr(StreamingParserEngine, patch_deepseek_v4_text_terminals._PATCHED_ATTR, raising=False)
    monkeypatch.setattr(StreamingParserEngine, "_process_lex_tokens", original)
    monkeypatch.setattr(
        patch_deepseek_v4_text_terminals,
        "deepseek_v4_config",
        lambda: SimpleNamespace(name="deepseek_v4", token_id_terminals={}),
    )
    monkeypatch.setattr(patch_deepseek_v4_text_terminals, "logger", logger)

    patch_deepseek_v4_text_terminals._install()

    assert StreamingParserEngine._process_lex_tokens is original
    assert not getattr(StreamingParserEngine, patch_deepseek_v4_text_terminals._PATCHED_ATTR, False)
    logger.warning_once.assert_called_once()


def _make_parser(properties):
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
    parser = DeepSeekV4Parser(
        FakeTokenizer(),
        tools=[tool],
        chat_template_kwargs={"enable_thinking": False},
    )
    return parser, request


def test_streaming_tool_call_reaches_the_client_without_dsml():
    parser, request = _make_parser({"text": {"type": "string"}})
    previous_text = ""
    content: list[str] = []
    arguments: list[str] = []

    def _collect(delta):
        if delta is None:
            return
        if delta.content:
            content.append(delta.content)
        for tool_call in delta.tool_calls or []:
            if tool_call.function and tool_call.function.arguments:
                arguments.append(tool_call.function.arguments)

    for delta_text, token_ids in _text_form_chunks(_tool_call_text()):
        current_text = previous_text + delta_text
        _collect(
            parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=token_ids,
                request=request,
            )
        )
        previous_text = current_text

    _collect(parser.finish_streaming())

    assert "｜DSML｜" not in "".join(content)
    assert arguments, "the stream produced no tool-call arguments"
    assert json.loads("".join(arguments)) == {"text": "alpha"}
