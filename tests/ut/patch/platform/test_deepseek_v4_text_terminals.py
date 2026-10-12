# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

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
    monkeypatch.setattr(
        StreamingParserEngine,
        "_process_lex_tokens",
        patch_deepseek_v4_text_terminals._original_process_lex_tokens,
    )
    engine = _make_engine()
    events = _feed_all(engine, _text_form_chunks(_tool_call_text()))

    assert engine._ever_had_token_ids is True
    assert "｜DSML｜" in _content_of(events)
    assert not any(event.type == EventType.TOOL_CALL_START for event in events)


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


def test_patch_is_gated_to_the_supported_vllm_version(monkeypatch):
    """On any other vLLM version the patch must not install itself."""
    import importlib

    import vllm_ascend.utils as ascend_utils
    from vllm_ascend.patch.platform import patch_deepseek_v4_text_terminals as module

    original = module._original_process_lex_tokens
    installed_method = StreamingParserEngine._process_lex_tokens
    had_marker = hasattr(StreamingParserEngine, module._PATCHED_ATTR)

    monkeypatch.setattr(ascend_utils, "vllm_version_is", lambda version: False)
    monkeypatch.delattr(StreamingParserEngine, module._PATCHED_ATTR, raising=False)
    monkeypatch.setattr(StreamingParserEngine, "_process_lex_tokens", original)

    importlib.reload(module)

    skipped = StreamingParserEngine._process_lex_tokens is original
    gated = not getattr(StreamingParserEngine, module._PATCHED_ATTR, False)

    # Put back exactly the state this test found, including the marker that
    # pre-installs or pre-skips the patch.
    monkeypatch.undo()
    StreamingParserEngine._process_lex_tokens = installed_method
    if had_marker:
        setattr(StreamingParserEngine, module._PATCHED_ATTR, True)
    elif hasattr(StreamingParserEngine, module._PATCHED_ATTR):
        delattr(StreamingParserEngine, module._PATCHED_ATTR)

    assert skipped, "the patch installed itself on an unsupported vLLM version"
    assert gated, "the patch set its install marker on an unsupported vLLM version"


def test_unsupported_version_skips_the_parser_engine_imports(monkeypatch):
    """The module is imported unconditionally, so it must load without the parser engine.

    The patch is imported on every vLLM version, so neither its imports nor its
    definitions may touch the parser engine of a version it does not support.
    """
    import importlib.util
    import sys

    import vllm_ascend.utils as ascend_utils

    monkeypatch.setattr(ascend_utils, "vllm_version_is", lambda version: False)
    for name in (
        "vllm.parser.deepseek_v4",
        "vllm.parser.engine.events",
        "vllm.parser.engine.incremental_lexer",
        "vllm.parser.engine.streaming_parser_engine",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    spec = importlib.util.spec_from_file_location(
        "patch_deepseek_v4_text_terminals_probe",
        patch_deepseek_v4_text_terminals.__file__,
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    assert probe._PATCHED_ATTR == "_ascend_text_form_dsv4_terminals_patched"
    assert not hasattr(probe, "_patched_process_lex_tokens")


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
