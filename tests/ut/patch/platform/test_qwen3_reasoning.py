# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.qwen3_engine_reasoning_parser import Qwen3ParserReasoningAdapter
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

from vllm_ascend.patch.platform import _register_qwen3_parsers, patch_qwen3_tool_parser
from vllm_ascend.patch.platform.patch_qwen3_reasoning_parser import AscendQwen3ReasoningParser
from vllm_ascend.patch.platform.patch_qwen3_tool_parser import AscendQwen3ToolParser

UPSTREAM = ("vllm.reasoning.qwen3_engine_reasoning_parser", "Qwen3ParserReasoningAdapter")
UPSTREAM_TOOL = ("vllm.tool_parsers.qwen3_engine_tool_parser", "Qwen3EngineToolParser")


def weather_call(location):
    return f"<tool_call><function=get_weather><parameter=location>{location}</parameter></function></tool_call>"


@pytest.fixture
def parse(monkeypatch):
    monkeypatch.setattr(ReasoningParserManager, "lazy_parsers", {"qwen3": UPSTREAM})
    monkeypatch.setattr(ReasoningParserManager, "reasoning_parsers", {})
    monkeypatch.setattr(ToolParserManager, "lazy_parsers", dict.fromkeys(("qwen3_xml", "qwen3_coder"), UPSTREAM_TOOL))
    monkeypatch.setattr(ToolParserManager, "tool_parsers", {})
    _register_qwen3_parsers()
    markers = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>"]

    def run(
        parts,
        *,
        thinking=True,
        closed_prompt=False,
        adapter=AscendQwen3ReasoningParser,
        tool_parser="qwen3_xml",
        include_content=False,
        declared="get_weather",
    ):
        request = ChatCompletionRequest(
            model="qwen",
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[{"type": "function", "function": {"name": declared, "parameters": {"type": "object"}}}],
            tool_choice="auto",
        )
        if adapter is Qwen3ParserReasoningAdapter:
            monkeypatch.setitem(ReasoningParserManager.reasoning_parsers, "qwen3", adapter)
        vocab = {text: i for i, text in enumerate(dict.fromkeys(markers + parts))}
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = vocab
        tokenizer.all_special_tokens = markers
        tokenizer.all_special_ids = [vocab[text] for text in markers]
        vocabulary_by_id = {i: text for text, i in vocab.items()}
        tokenizer.decode.side_effect = lambda ids: "".join(vocabulary_by_id[i] for i in ids)
        ids = [vocab[text] for text in parts]
        parser_cls = ParserManager.get_parser(tool_parser, "qwen3", True)
        assert parser_cls.reasoning_parser_cls is adapter
        assert parser_cls.tool_parser_cls is AscendQwen3ToolParser
        parser = parser_cls(tokenizer, tools=request.tools, chat_template_kwargs={"enable_thinking": thinking})
        if closed_prompt:
            parser.set_prompt_token_ids([vocab["<|im_start|>"], vocab["<think>"], vocab["</think>"]])
        reasoning, content, calls = parser.parse(
            "".join(parts), request, enable_auto_tools=True, model_output_token_ids=ids
        )
        functions = [(c.name, json.loads(c.arguments)) for c in calls or []]
        if include_content:
            return content, functions
        return reasoning, functions, parser.count_reasoning_tokens(ids)

    return run


def test_registration_preserves_user_parser(monkeypatch):
    class UserParser(Qwen3ParserReasoningAdapter):
        pass

    class UserToolParser(Qwen3EngineToolParser):
        pass

    monkeypatch.setitem(ReasoningParserManager.lazy_parsers, "qwen3", UPSTREAM)
    monkeypatch.setitem(ReasoningParserManager.reasoning_parsers, "qwen3", Qwen3ParserReasoningAdapter)
    monkeypatch.setitem(ToolParserManager.lazy_parsers, "qwen3_xml", UPSTREAM_TOOL)
    monkeypatch.setitem(ToolParserManager.tool_parsers, "qwen3_xml", Qwen3EngineToolParser)
    _register_qwen3_parsers()
    assert ReasoningParserManager.get_reasoning_parser("qwen3") is AscendQwen3ReasoningParser
    assert ToolParserManager.get_tool_parser("qwen3_xml") is AscendQwen3ToolParser
    monkeypatch.setitem(ReasoningParserManager.reasoning_parsers, "qwen3", UserParser)
    monkeypatch.setitem(ToolParserManager.tool_parsers, "qwen3_xml", UserToolParser)
    _register_qwen3_parsers()
    assert ReasoningParserManager.get_reasoning_parser("qwen3") is UserParser
    assert ToolParserManager.get_tool_parser("qwen3_xml") is UserToolParser


@pytest.mark.parametrize("quoted_name", ["example_function_name", "get_weather"])
def test_tool_examples_before_explicit_end_are_reasoning(parse, quoted_name):
    reasoning = ["Example: ", "<tool_call>", f"<function={quoted_name}></function>", "</tool_call>", ". Actual calls:"]
    parts = reasoning + ["</think>", weather_call("北京"), weather_call("西安")]
    text, calls, count = parse(parts)
    assert text == "".join(reasoning)
    assert calls == [("get_weather", {"location": "北京"}), ("get_weather", {"location": "西安"})]
    # The engine counts the three text tokens. The two special XML markers
    # do not contribute to its text-token accounting.
    assert count == 3


def test_implicit_tool_boundary(parse):
    reasoning, calls, _ = parse(["Checking.", weather_call("北京")])
    assert reasoning == "Checking."
    assert calls == [("get_weather", {"location": "北京"})]


def test_think_marker_in_implicit_tool_argument(parse):
    parts = ["Checking.", weather_call("literal </think> marker")]
    assert parse(parts) == parse(parts, adapter=Qwen3ParserReasoningAdapter)


@pytest.mark.parametrize(("thinking", "closed_prompt"), [(False, False), (True, True)])
def test_content_only_prompt(parse, thinking, closed_prompt):
    reasoning, calls, _ = parse([weather_call("北京"), "</think>"], thinking=thinking, closed_prompt=closed_prompt)
    assert reasoning is None
    assert calls == [("get_weather", {"location": "北京"})]


@pytest.mark.parametrize("alias", ["qwen3_xml", "qwen3_coder"])
def test_only_undeclared_call_stays_in_content(parse, alias):
    output = "Example: <tool_call><function=unknown_name></function></tool_call>"
    content, calls = parse(["</think>", output], tool_parser=alias, include_content=True)
    assert calls == []
    assert content == output


def test_example_name_is_callable_if_declared(parse):
    output = "<tool_call><function=example_function_name></function></tool_call>"
    _, calls = parse(["</think>", output], declared="example_function_name", include_content=True)
    assert calls == [("example_function_name", {})]


def test_auto_tool_choice_keeps_declared_calls_in_order_and_prose(parse):
    parts = [
        "</think>",
        "Checking both cities.",
        "<tool_call><function=example_function_name></function></tool_call>",
        weather_call("北京"),
        "<tool_call><function=...></function></tool_call>",
        weather_call("西安"),
    ]
    content, calls = parse(parts, include_content=True)
    assert content == "Checking both cities."
    assert calls == [("get_weather", {"location": "北京"}), ("get_weather", {"location": "西安"})]


def test_tool_parser_treats_missing_tools_as_empty(monkeypatch):
    request = ChatCompletionRequest(model="qwen", messages=[{"role": "user", "content": "Weather?"}], tool_choice=None)
    output = weather_call("北京")
    call = MagicMock()
    call.function.name = "get_weather"
    parsed = MagicMock(tools_called=True, tool_calls=[call])
    monkeypatch.setattr(Qwen3EngineToolParser, "extract_tool_calls", lambda self, text, req: parsed)
    find_tool_name = MagicMock(wraps=patch_qwen3_tool_parser.find_tool_name)
    monkeypatch.setattr(patch_qwen3_tool_parser, "find_tool_name", find_tool_name)

    parser = object.__new__(AscendQwen3ToolParser)
    result = parser.extract_tool_calls(output, request)

    find_tool_name.assert_called_once_with([], "get_weather")
    parsed.model_copy.assert_called_once_with(update={"tools_called": False, "tool_calls": [], "content": output})
    assert result is parsed.model_copy.return_value


@pytest.mark.parametrize(
    ("stream", "choice"),
    [(True, "auto"), (False, "required"), (False, {"type": "function", "function": {"name": "get_weather"}})],
)
def test_tool_parser_leaves_other_modes_to_upstream(monkeypatch, stream, choice):
    original = MagicMock(tools_called=True)
    monkeypatch.setattr(Qwen3EngineToolParser, "extract_tool_calls", lambda self, text, req: original)
    request = MagicMock(stream=stream, tool_choice=choice)
    parser = object.__new__(AscendQwen3ToolParser)
    assert parser.extract_tool_calls("untouched", request) is original
