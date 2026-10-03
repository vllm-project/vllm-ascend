# SPDX-License-Identifier: Apache-2.0
"""Upstream-drift and behavior guards for the parser output-preserving patch.

``patch_parser_preserve_model_output`` fixes two ways an engine-based tool
parser could drop model output entirely:

* ``tool_choice="none"`` (also the default of a request that declares no
  ``tools``) raised ``_suppress_tool_calls`` and the whole
  ``<tool_call> ... </tool_call>`` span disappeared, so the client got an
  empty answer while ``usage`` still counted the tokens;
* a span whose tool name is not part of ``request.tools`` never became a tool
  call and its raw text was discarded instead of degrading back to content.

Every test builds the ``ParserEngine`` state the patched methods read through
``ParserEngine.__new__``; the real engine needs a tokenizer and hardware and
the patched methods never touch either of them.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine, ToolCallSlot
from vllm.parser.glm47_moe import Glm47MoeParser

import vllm_ascend.patch.platform.patch_parser_preserve_model_output as _patch


def _tool(name):
    return ChatCompletionToolsParam(
        type="function",
        function={"name": name, "parameters": {"type": "object", "properties": {}}},
    )


def _make_engine(tools=None, validate_tool_names=True):
    engine = ParserEngine.__new__(ParserEngine)
    engine.parser_engine_config = SimpleNamespace(validate_tool_names=validate_tool_names)
    engine._tools = tools
    engine._engine = SimpleNamespace(skip_tool_parsing=False, reset=lambda initial_state=None: None)
    engine._stream_state = SimpleNamespace(tool_call_id_type="random", history_tool_call_cnt=0)
    engine._tool_slots = []
    engine._deferred_content = ""
    engine._deferred_reasoning = ""
    engine._content_has_nonws = False
    engine._suppress_tool_calls = False
    engine._reasoning_ended = True
    engine._has_reasoning = False
    engine._prompt_streaming_prepared = False
    engine._drop_ws_only_content_before_tools = False
    engine._strip_content_ws_with_tools = True
    engine._arg_converter = None
    engine._arg_structural_chars = None
    engine._stream_arg_deltas = False
    return engine


def _span_events(name, index=0):
    return [
        SemanticEvent(EventType.TOOL_CALL_START, value="<tool_call>", tool_index=index),
        SemanticEvent(EventType.TOOL_NAME, value=name, tool_index=index),
        SemanticEvent(EventType.ARG_VALUE_CHUNK, value='{"path": "a"}', tool_index=index),
        SemanticEvent(EventType.TOOL_CALL_END, value="</tool_call>", tool_index=index),
    ]


class _FakeTokenizer:
    def get_vocab(self):
        return {}

    def decode(self, token_ids):
        return ""


def _glm52_markup(name):
    return f"<tool_call>{name}<arg_key>path</arg_key><arg_value>/tmp/a</arg_value></tool_call>"


def _stream_once(parser, request, text):
    return parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=text,
        delta_text=text,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[1],
        request=request,
    )


# ---------------------------------------------------------------------------
# 1. the patch is applied at import and keeps patching private hooks upstream
# ---------------------------------------------------------------------------


def test_patch_applied_at_import():
    assert ParserEngine._vllm_ascend_keep_text_on_tool_choice_none is True
    assert ParserEngine._vllm_ascend_tool_span_text_fallback is True
    assert "skip_tool_parsing = True" in inspect.getsource(ParserEngine._check_skip_tool_parsing)


def test_patched_upstream_hooks_still_exist():
    for hook in ("_check_skip_tool_parsing", "_events_to_delta", "_reset", "skip_tool_parsing"):
        assert hasattr(ParserEngine, hook), hook
    engine = _make_engine()
    for state in ("_tool_slots", "_suppress_tool_calls", "_content_has_nonws", "_deferred_content"):
        assert hasattr(engine, state), state


# ---------------------------------------------------------------------------
# 2. tool_choice="none" never suppresses the model output
# ---------------------------------------------------------------------------


def test_tool_choice_none_runs_without_tool_parsing():
    engine = _make_engine(tools=[_tool("write")])
    engine._check_skip_tool_parsing(SimpleNamespace(tool_choice="none", tools=[_tool("write")]))
    assert engine.skip_tool_parsing is True
    assert engine._suppress_tool_calls is False


def test_tool_choice_none_without_tools_also_keeps_the_text():
    # A request that declares no `tools` defaults to tool_choice="none".
    engine = _make_engine(tools=None)
    engine._check_skip_tool_parsing(SimpleNamespace(tool_choice="none", tools=None))
    assert engine.skip_tool_parsing is True
    assert engine._suppress_tool_calls is False


def test_other_tool_choices_are_not_touched():
    engine = _make_engine(tools=[_tool("write")])
    engine._check_skip_tool_parsing(SimpleNamespace(tool_choice="auto", tools=[_tool("write")]))
    assert engine.skip_tool_parsing is False
    assert engine._suppress_tool_calls is False


# ---------------------------------------------------------------------------
# 3. a span that produces no tool call degrades back to content
# ---------------------------------------------------------------------------


def test_span_with_unknown_tool_name_is_kept_as_content():
    engine = _make_engine(tools=[_tool("write")])
    events = [SemanticEvent(EventType.TEXT_CHUNK, value="让我调用工具："), *_span_events("edit")]
    delta = engine._events_to_delta(events)
    assert delta is not None
    assert delta.tool_calls == []
    assert delta.content == '让我调用工具：<tool_call>edit{"path": "a"}</tool_call>'


def test_span_with_known_tool_name_is_untouched():
    engine = _make_engine(tools=[_tool("write")])
    delta = engine._events_to_delta(_span_events("write"))
    assert delta is not None
    assert delta.content is None
    assert [call.function.name for call in delta.tool_calls] == ["write"]


def test_span_without_follow_up_text_is_not_empty():
    engine = _make_engine(tools=[_tool("write")])
    delta = engine._events_to_delta(_span_events("edit"))
    assert delta is not None
    assert delta.tool_calls == []
    assert delta.content == '<tool_call>edit{"path": "a"}</tool_call>'


def test_dropped_tool_span_text_ignores_materialized_spans():
    engine = _make_engine(tools=[_tool("write")])
    materialized = ToolCallSlot()
    materialized.name_sent = True
    engine._tool_slots = [materialized]
    assert _patch._dropped_tool_span_text(engine, [(0, ["<tool_call>", "write"])]) == ""

    pending = ToolCallSlot()
    engine._tool_slots = [pending]
    assert _patch._dropped_tool_span_text(engine, [(0, ["<tool_call>", "edit"])]) == "<tool_call>edit"
    assert _patch._dropped_tool_span_text(engine, [(9, ["lost"])]) == "lost"


# ---------------------------------------------------------------------------
# 4. buffered span text never bleeds across requests
# ---------------------------------------------------------------------------


def test_reset_drops_buffered_span_text():
    engine = _make_engine(tools=[_tool("write")])
    engine._vllm_ascend_raw_tool_text = {0: ["<tool_call>", "edit"]}
    engine._reset()
    assert engine._vllm_ascend_raw_tool_text == {}


def test_new_span_does_not_reuse_a_truncated_span():
    engine = _make_engine(tools=[_tool("write")])
    # A span truncated before its end marker leaves text behind ...
    engine._events_to_delta(_span_events("edit")[:-1])
    assert engine._vllm_ascend_raw_tool_text
    # ... which the next span at the same index must not pick up.
    delta = engine._events_to_delta(_span_events("edit"))
    assert delta.tool_calls == []
    assert delta.content == '<tool_call>edit{"path": "a"}</tool_call>'


# ---------------------------------------------------------------------------
# 5. end to end through the real GLM-4.7 engine (lexer -> events -> delta)
# ---------------------------------------------------------------------------


def _glm47_parser(tools):
    return Glm47MoeParser(
        _FakeTokenizer(),
        tools=tools,
        chat_template_kwargs={"enable_thinking": False},
    )


def test_glm47_tool_choice_none_returns_the_markup_as_content():
    tool = _tool("write")
    parser = _glm47_parser([tool])
    request = MagicMock(tools=[tool], tool_choice="none")
    text = _glm52_markup("write")

    delta = _stream_once(parser, request, text)

    assert delta is not None
    assert not delta.tool_calls
    assert delta.content == text


def test_glm47_unknown_tool_name_falls_back_to_content():
    tool = _tool("write")
    parser = _glm47_parser([tool])
    request = MagicMock(tools=[tool], tool_choice="auto")
    text = "让我调用工具：" + _glm52_markup("edit")

    delta = _stream_once(parser, request, text)

    assert delta is not None
    assert not delta.tool_calls
    assert delta.content == text


def test_glm47_known_tool_name_still_produces_a_tool_call():
    tool = _tool("write")
    parser = _glm47_parser([tool])
    request = MagicMock(tools=[tool], tool_choice="auto")
    text = "让我调用工具：" + _glm52_markup("write")

    delta = _stream_once(parser, request, text)

    assert delta is not None
    assert [call.function.name for call in delta.tool_calls] == ["write"]
    assert delta.content == "让我调用工具："


def test_glm47_streaming_and_non_streaming_agree():
    """Both entry points must ship the same text to the client."""
    tool = _tool("write")
    text = "让我调用工具：" + _glm52_markup("edit")

    streaming = _glm47_parser([tool])
    request = MagicMock(tools=[tool], tool_choice="none")
    delta = _stream_once(streaming, request, text)

    non_streaming = _glm47_parser([tool])
    parsed = non_streaming.extract_tool_calls(text, request)

    assert not parsed.tools_called
    assert parsed.content == delta.content == text
