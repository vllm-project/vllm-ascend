# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import copy
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.renderers.params import ChatParams
from vllm.utils.async_utils import make_async
from xgrammar import Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo

from vllm_ascend.patch.platform.patch_deepseek_v41_frontend import register_frontend
from vllm_ascend.patch.platform.patch_deepseek_v41_frontend.encoding import (
    encode_messages,
    load_cases,
    parse_message_from_completion_text,
)
from vllm_ascend.patch.platform.patch_deepseek_v41_frontend.parser import DeepseekV41Parser, DeepseekV41ToolParser
from vllm_ascend.patch.platform.patch_deepseek_v41_frontend.renderer import DeepseekV41Renderer
from vllm_ascend.patch.platform.patch_deepseek_v41_frontend.tokenizer import get_deepseek_v41_tokenizer

FIXTURES = Path(__file__).parent / "fixtures"
TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query", "limit"],
        },
    },
}


@pytest.fixture
def tokenizer():
    tokens = ["[UNK]", "<think>", "</think>", "<｜end▁of▁sentence｜>", "｜DSML｜"]
    backend = Tokenizer(WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
    return get_deepseek_v41_tokenizer(
        PreTrainedTokenizerFast(tokenizer_object=backend, additional_special_tokens=tokens[1:])
    )


def request(**kwargs):
    return ChatCompletionRequest(model="v41", messages=[{"role": "user", "content": "hi"}], **kwargs)


def call_text(arguments, name="lookup"):
    params = "\n".join(
        f'<｜DSML｜ parameter name="{key}" string="{str(isinstance(value, str)).lower()}">'
        f"{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}</｜DSML｜ parameter>"
        for key, value in arguments.items()
    )
    return f'<｜DSML｜ invoke name="{name}">\n{params}\n</｜DSML｜ invoke>'


def completion(arguments, thinking=True):
    return (
        ("  reason  </think>" if thinking else "")
        + "summary\n\n<｜DSML｜ calls>\n"
        + call_text(arguments)
        + "\n</｜DSML｜ calls>"
    )


@pytest.mark.parametrize("case_id", range(1, 6))
def test_tokenizer_matches_checkpoint_goldens(tokenizer, case_id):
    case = load_cases(str(FIXTURES / f"test_input_{case_id}.json"))[0]
    before = copy.deepcopy(case)
    actual = tokenizer.apply_chat_template(
        case["messages"],
        tokenize=False,
        thinking=case.get("thinking_mode", "chat") == "thinking",
        reasoning_effort=case.get("reasoning_effort"),
        context=case.get("context"),
        drop_thinking=case.get("drop_thinking", True),
    )
    assert actual == (FIXTURES / f"test_output_{case_id}.txt").read_text()
    assert case == before


@pytest.mark.parametrize(
    "effort,budget", [(None, 50), ("low", 25), ("high", 50), ("xhigh", 75), ("max", 100), (1, 1), (42, 42), (100, 100)]
)
def test_tokenizer_preserves_numeric_effort(tokenizer, effort, budget):
    prompt = tokenizer.apply_chat_template(request().messages, reasoning_effort=effort, tokenize=False)
    assert f"<｜System｜>Reasoning Effort: {budget} (range 1-100," in prompt
    assert prompt.endswith("<think>")


@pytest.mark.parametrize("effort", [0, 101, True, 1.5, "medium", "minimal", "invalid"])
def test_tokenizer_rejects_invalid_effort(tokenizer, effort):
    with pytest.raises((AssertionError, ValueError)):
        tokenizer.apply_chat_template(request().messages, reasoning_effort=effort)


@pytest.mark.parametrize("kwargs", [{"thinking": False}, {"enable_thinking": False}, {"reasoning_effort": "none"}])
def test_chat_mode_matches_parser_initial_state(tokenizer, kwargs):
    prompt = tokenizer.apply_chat_template(request().messages, tokenize=False, **kwargs)
    assert prompt.endswith("</think>") and "Reasoning Effort:" not in prompt
    parser = DeepseekV41Parser(tokenizer, chat_template_kwargs=kwargs)
    assert parser.extract_reasoning("answer", request()) == (None, "answer")


def test_tools_follow_existing_system_and_preserve_history(tokenizer):
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning": "old thought"},
        {"role": "user", "content": "next"},
    ]
    before = copy.deepcopy(messages)
    expected = copy.deepcopy(messages)
    expected[0]["tools"] = [TOOL]
    expected[2]["reasoning_content"] = expected[2].pop("reasoning")
    actual = tokenizer.apply_chat_template(messages, tools=[TOOL], tokenize=False)
    assert actual == encode_messages(expected, thinking_mode="thinking")
    assert actual.index("system") < actual.index("## Tools")
    assert "old thought</think>" in actual
    assert messages == before


def test_openai_tool_defaults_do_not_change_reference_prompt(tokenizer):
    req = request(tools=[TOOL], tool_choice="auto")
    tools = [tool.model_dump() for tool in req.tools]
    assert tools[0]["function"]["description"] is None
    actual = tokenizer.apply_chat_template(req.messages, tools=tools, tokenize=False)
    expected = [{"role": "system", "content": "", "tools": [TOOL]}, *req.messages]
    assert actual == encode_messages(expected, thinking_mode="thinking")


@pytest.mark.parametrize("size", [1, 2, 7, 19, 10000])
@pytest.mark.parametrize("thinking", [False, True])
def test_streaming_parallel_calls_and_types(tokenizer, size, thinking):
    arguments = {
        "text": '中文\\"\n<｜DSML｜parameter>literal',
        "number_string": "42",
        "boolean": False,
        "null": None,
        "array": [1, "x"],
        "object": {"a": 2},
    }
    text = completion(arguments, thinking).replace(
        "\n</｜DSML｜ calls>", "\n" + call_text({}, "empty") + "\n</｜DSML｜ calls>"
    )
    req = request(tools=[TOOL], tool_choice="auto")
    parser = DeepseekV41Parser(tokenizer, chat_template_kwargs={"thinking": thinking})
    reasoning, content, calls = parser.parse(text, req, enable_auto_tools=True)
    assert reasoning == ("  reason  " if thinking else None)
    assert content == "summary"
    assert [call.name for call in calls] == ["lookup", "empty"]
    assert json.loads(calls[0].arguments) == arguments
    assert json.loads(calls[1].arguments) == {}
    parser = DeepseekV41Parser(tokenizer, chat_template_kwargs={"thinking": thinking})
    deltas = []
    for offset in range(0, len(text), size):
        chunk = text[offset : offset + size]
        delta = parser.parse_delta(chunk, [], req, finished=offset + size >= len(text))
        if delta:
            deltas.append(delta)
    assert "".join(delta.reasoning or "" for delta in deltas) == ("  reason  " if thinking else "")
    assert "".join(delta.content or "" for delta in deltas) == "summary"
    for index, expected in enumerate([arguments, {}]):
        fragments = [
            tc.function.arguments or ""
            for delta in deltas
            for tc in delta.tool_calls or []
            if tc.index == index and tc.function
        ]
        assert json.loads("".join(fragments)) == expected


def test_registry_composes_reasoning_and_tool_parser(tokenizer):
    cls = ParserManager.get_parser("deepseek_v41", "deepseek_v41", enable_auto_tools=True)
    parser = cls(tokenizer)
    reason, content, calls = parser.parse(
        completion({"query": "value", "limit": 2}), request(tools=[TOOL], tool_choice="auto"), enable_auto_tools=True
    )
    assert reason == "  reason  " and content == "summary"
    assert json.loads(calls[0].arguments) == {"query": "value", "limit": 2}


def test_parser_matches_reference_completion_fields(tokenizer):
    text = completion({"query": "value", "limit": 2}) + "<｜end▁of▁sentence｜>"
    expected = parse_message_from_completion_text(text, "thinking")
    parser = DeepseekV41Parser(tokenizer)
    reasoning, content, calls = parser.parse(text, request(tools=[TOOL], tool_choice="auto"), enable_auto_tools=True)
    assert reasoning == expected["reasoning_content"]
    assert content == expected["content"]
    assert calls[0].name == expected["tool_calls"][0]["function"]["name"]
    assert json.loads(calls[0].arguments) == json.loads(expected["tool_calls"][0]["function"]["arguments"])


def test_dsml_string_flag_overrides_schema(tokenizer):
    parser = DeepseekV41Parser(tokenizer, tools=request(tools=[TOOL]).tools)
    _, _, calls = parser.parse(
        completion({"query": "value", "limit": "42"}), request(tools=[TOOL], tool_choice="auto"), enable_auto_tools=True
    )
    assert json.loads(calls[0].arguments)["limit"] == "42"


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "lookup"}}, "auto"])
def test_structural_tag_accepts_v41_and_rejects_v4(tokenizer, choice):
    tool = copy.deepcopy(TOOL)
    tool["function"]["strict"] = True
    req = request(tools=[tool], tool_choice=choice)
    tag = DeepseekV41ToolParser(tokenizer).get_structural_tag(req)
    compiler = GrammarCompiler(TokenizerInfo.from_huggingface(tokenizer))
    grammar = compiler.compile_grammar(Grammar.from_structural_tag(tag))
    text = "\n\n<｜DSML｜ calls>\n" + call_text({"query": "value", "limit": 2}) + "\n</｜DSML｜ calls>"
    matcher = GrammarMatcher(grammar)
    assert matcher.accept_string(text)
    assert matcher.is_completed()
    if choice != "auto":
        assert not GrammarMatcher(grammar).accept_string(
            text.replace("｜DSML｜ calls", "｜DSML｜tool_calls").replace("｜DSML｜ ", "｜DSML｜")
        )
    assert not GrammarMatcher(grammar).accept_string(text.replace('string="false">2', 'string="false">"wrong"'))


def test_renderer_uses_raw_blocks_and_keeps_media(tokenizer, monkeypatch):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "unused.png"}},
                {"type": "text", "text": "b"},
            ],
        }
    ]
    media = {"image": [object()]}
    uuids = {"image": ["test-image"]}
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_deepseek_v41_frontend.renderer.parse_chat_messages",
        lambda *a, **kw: ([{"role": "user", "content": "flattened"}], media, uuids),
    )
    renderer = object.__new__(DeepseekV41Renderer)
    renderer.model_config = SimpleNamespace()
    renderer.get_tokenizer = lambda: tokenizer
    _, prompt = renderer.render_messages(
        messages, ChatParams(chat_template_kwargs={"thinking": False, "tokenize": False})
    )
    assert prompt["prompt"] == encode_messages(messages, thinking_mode="chat")
    assert prompt["multi_modal_data"] is media and prompt["multi_modal_uuids"] is uuids


def test_renderer_async_and_request_response_format(tokenizer, monkeypatch):
    req = request(
        reasoning_effort="xhigh",
        tools=[TOOL],
        tool_choice="auto",
        response_format={"type": "json_schema", "json_schema": {"name": "result", "schema": {"type": "object"}}},
    )
    conversation = [{"role": "user", "content": "hi"}]

    async def parse(*args, **kwargs):
        return conversation, None, None

    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_deepseek_v41_frontend.renderer.parse_chat_messages_async", parse
    )
    renderer = object.__new__(DeepseekV41Renderer)
    renderer.model_config = SimpleNamespace()
    renderer.get_tokenizer = lambda: tokenizer
    renderer._apply_chat_template_async = make_async(renderer._apply_chat_template)
    params = req.build_chat_params(None, "auto")
    # OnlineRenderer adds normalized tools after building ChatParams.
    params.chat_template_kwargs["tools"] = [TOOL]
    params.chat_template_kwargs["tokenize"] = False
    _, prompt = asyncio.run(renderer.render_messages_async(req.messages, params))
    expected = [
        {"role": "system", "content": "", "tools": [TOOL], "response_format": {"type": "object"}},
        *req.messages,
    ]
    assert prompt["prompt"] == encode_messages(expected, thinking_mode="thinking", reasoning_effort="xhigh")


@pytest.mark.parametrize(
    "schema,value",
    [
        ({"type": "string", "enum": ["red", "blue"]}, "red"),
        ({"type": "object", "properties": {"nested": {"type": "integer"}}, "required": ["nested"]}, {"nested": 2}),
        ({"type": "array", "items": {"type": "integer"}}, [1, 2]),
        ({"type": ["string", "null"]}, None),
        ({"type": "boolean"}, True),
    ],
)
def test_grammar_parameter_schemas(tokenizer, schema, value):
    tool = copy.deepcopy(TOOL)
    tool["function"]["parameters"] = {
        "type": "object",
        "properties": {"value": schema, "optional": {"type": "integer"}},
        "required": ["value"],
    }
    tag = DeepseekV41ToolParser(tokenizer).get_structural_tag(request(tools=[tool], tool_choice="required"))
    compiler = GrammarCompiler(TokenizerInfo.from_huggingface(tokenizer))
    grammar = compiler.compile_grammar(Grammar.from_structural_tag(tag))
    text = "\n\n<｜DSML｜ calls>\n" + call_text({"value": value}) + "\n</｜DSML｜ calls>"
    matcher = GrammarMatcher(grammar)
    assert matcher.accept_string(text) and matcher.is_completed()
    assert not GrammarMatcher(grammar).accept_string(text.replace('name="value"', 'name="unknown"'))
    if schema.get("enum"):
        assert not GrammarMatcher(grammar).accept_string(text.replace(">red<", ">green<"))


def test_grammar_empty_call_and_named_tool_filter(tokenizer):
    tool = {"type": "function", "function": {"name": "empty", "parameters": {"type": "object", "properties": {}}}}
    req = request(tools=[TOOL, tool], tool_choice={"type": "function", "function": {"name": "empty"}})
    tag = DeepseekV41ToolParser(tokenizer).get_structural_tag(req)
    compiler = GrammarCompiler(TokenizerInfo.from_huggingface(tokenizer))
    grammar = compiler.compile_grammar(Grammar.from_structural_tag(tag))
    text = "\n\n<｜DSML｜ calls>\n" + call_text({}, "empty") + "\n</｜DSML｜ calls>"
    matcher = GrammarMatcher(grammar)
    assert matcher.accept_string(text) and matcher.is_completed()
    assert not GrammarMatcher(grammar).accept_string(text.replace('name="empty"', 'name="lookup"'))


def test_v4_registration_is_unchanged():
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    old = (TokenizerRegistry.load_tokenizer_cls("deepseek_v4"), RENDERER_REGISTRY.load_renderer_cls("deepseek_v4"))
    register_frontend()
    assert old == (
        TokenizerRegistry.load_tokenizer_cls("deepseek_v4"),
        RENDERER_REGISTRY.load_renderer_cls("deepseek_v4"),
    )
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v41") is DeepseekV41Renderer


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_media_loader_preserves_interleaved_order(tokenizer, monkeypatch, asynchronous):
    from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker

    # Only the model processor is a stand-in; vLLM loads both actual PNGs.
    processor = SimpleNamespace(info=SimpleNamespace(validate_num_items=lambda *a: None))
    monkeypatch.setattr(BaseMultiModalItemTracker, "mm_processor", processor)
    monkeypatch.setattr(
        BaseMultiModalItemTracker, "model_cls", SimpleNamespace(get_placeholder_str=lambda *a: "<｜deepseek_image｜>")
    )
    blocks = []
    for color in ("red", "blue"):
        buf = io.BytesIO()
        Image.new("RGB", (2, 2), color).save(buf, format="PNG")
        blocks.extend(
            [
                {"type": "text", "text": color},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()},
                },
            ]
        )
    messages = [{"role": "user", "content": blocks}]
    renderer = object.__new__(DeepseekV41Renderer)
    renderer.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        multimodal_config=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
        is_multimodal_model=True,
        enable_prompt_embeds=False,
    )
    renderer.get_tokenizer = lambda: tokenizer
    renderer._apply_chat_template_async = make_async(renderer._apply_chat_template)
    params = ChatParams(chat_template_kwargs={"thinking": False, "tokenize": False})
    if asynchronous:
        _, prompt = asyncio.run(renderer.render_messages_async(messages, params))
    else:
        _, prompt = renderer.render_messages(messages, params)
    assert prompt["prompt"] == encode_messages(messages, thinking_mode="chat")
    images = prompt["multi_modal_data"]["image"]
    assert [image.media.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255)]


def test_global_patch_registers_frontend_in_fresh_process():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import vllm_ascend; "
            "from vllm_ascend.utils import adapt_patch; "
            "adapt_patch(is_global_patch=True); "
            "from vllm.tokenizers.registry import TokenizerRegistry; "
            "from vllm.renderers.registry import RENDERER_REGISTRY; "
            "from vllm.parser.parser_manager import ParserManager; "
            "assert TokenizerRegistry.load_tokenizer_cls('deepseek_v41').__name__ == 'DeepseekV41Tokenizer'; "
            "assert RENDERER_REGISTRY.load_renderer_cls('deepseek_v41').__name__ == 'DeepseekV41Renderer'; "
            "assert ParserManager.get_parser('deepseek_v41', 'deepseek_v41', enable_auto_tools=True)",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
