# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import json
import string
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_minimax_m31_reasoning.py"
ADAPTIVE_TEMPLATE = (
    "<mm:think></mm:think>\n<effort>{{ reasoning_effort }}</effort>"
    "{% for message in messages %}{{ message.content }}{% endfor %}{{ ']~b]ai\\n' }}"
)
LEGACY_TEMPLATE = (
    "]~b]ai\n{% if thinking_mode == 'enabled' %}<mm:think>{% elif thinking_mode == 'disabled' %}</mm:think>{% endif %}"
)


@pytest.fixture(autouse=True)
def patch_module(monkeypatch):
    # Restore the real vLLM methods afterwards; do not stub the parser itself.
    monkeypatch.setattr(MiniMaxM3ReasoningParser, "__init__", MiniMaxM3ReasoningParser.__init__)
    monkeypatch.setattr(MiniMaxM3ReasoningParser, "is_reasoning_end", MiniMaxM3ReasoningParser.is_reasoning_end)
    spec = importlib.util.spec_from_file_location("_test_minimax_m31_patch", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_tokenizer(template):
    vocab = {char: index for index, char in enumerate(string.printable)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=["<mm:think>", "</mm:think>", "]~b]"],
        chat_template=template,
    )


def parse_stream(parser, tokenizer, request, prompt, output):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    reasoning, content, calls = [], [], {}
    for index, char in enumerate(output):
        delta = parser.parse_delta(
            char,
            tokenizer.encode(char, add_special_tokens=False),
            request,
            prompt_token_ids=prompt_ids,
            finished=index == len(output) - 1,
        )
        if delta is None:
            continue
        reasoning.append(delta.reasoning or "")
        content.append(delta.content or "")
        for call in delta.tool_calls or []:
            entry = calls.setdefault(call.index, {"name": "", "arguments": ""})
            if call.function:
                entry["name"] += call.function.name or ""
                entry["arguments"] += call.function.arguments or ""
    return "".join(reasoning) or None, "".join(content) or None, calls


@pytest.mark.parametrize("mode", ["enabled", "disabled", "adaptive"])
@pytest.mark.parametrize("prefix, expected", [("", None), ("</mm:think>", None), ("<mm:think>plan</mm:think>", "plan")])
def test_adaptive_template_and_composed_parser(mode, prefix, expected):
    tokenizer = make_tokenizer(ADAPTIVE_TEMPLATE)
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="high",
        chat_template_kwargs={"thinking_mode": mode},
    )
    kwargs = request.build_chat_params(None, "string").chat_template_kwargs
    prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, **kwargs)
    assert "<effort>high</effort>" in prompt
    assert prompt.endswith("]~b]ai\n")
    parser_cls = ParserManager.get_parser(reasoning_parser_name="minimax_m3")
    parser = parser_cls(tokenizer, chat_template_kwargs=kwargs)
    assert parser.parse(prefix + "answer", request)[:2] == (expected, "answer")
    parser = parser_cls(tokenizer, chat_template_kwargs=kwargs)
    assert parse_stream(parser, tokenizer, request, prompt, prefix + "answer")[:2] == (expected, "answer")
    assert parser.count_reasoning_tokens(tokenizer.encode("answer")) == 0
    assert request.chat_template_kwargs == {"thinking_mode": mode}
    assert kwargs["thinking_mode"] == mode


@pytest.mark.parametrize("mode", ["enabled", "disabled", "adaptive"])
def test_legacy_m3_modes_are_preserved(mode):
    tokenizer = make_tokenizer(LEGACY_TEMPLATE)
    request = ChatCompletionRequest(model="test", messages=[], chat_template_kwargs={"thinking_mode": mode})
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False, thinking_mode=mode)
    output = {"enabled": "plan</mm:think>answer", "disabled": "answer", "adaptive": "<mm:think>plan</mm:think>answer"}[
        mode
    ]
    expected = None if mode == "disabled" else "plan"
    parser_cls = ParserManager.get_parser(reasoning_parser_name="minimax_m3")
    kwargs = request.chat_template_kwargs
    assert parser_cls(tokenizer, chat_template_kwargs=kwargs).parse(output, request)[:2] == (expected, "answer")
    parser = parser_cls(tokenizer, chat_template_kwargs=kwargs)
    assert parse_stream(parser, tokenizer, request, prompt, output)[:2] == (expected, "answer")


def test_reasoning_and_tools_share_the_adaptive_stream():
    tokenizer = make_tokenizer(ADAPTIVE_TEMPLATE)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    request = ChatCompletionRequest(model="test", messages=[], tools=tools, tool_choice="auto")
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name="minimax_m3",
        tool_parser_name="minimax_m3",
        enable_auto_tools=True,
    )
    namespace = "]<]minimax[>["
    output = (
        f'<mm:think>plan</mm:think>{namespace}<tool_call>{namespace}<invoke name="get_weather">'
        f"{namespace}<city>Beijing{namespace}</city>{namespace}</invoke>{namespace}</tool_call>"
    )
    parser = parser_cls(tokenizer, tools=request.tools, chat_template_kwargs={"thinking_mode": "enabled"})
    reasoning, content, calls = parser.parse(output, request, enable_auto_tools=True)
    assert reasoning == "plan"
    assert not content
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Beijing"}
    parser = parser_cls(tokenizer, tools=request.tools, chat_template_kwargs={"thinking_mode": "enabled"})
    reasoning, content, calls = parse_stream(parser, tokenizer, request, "</mm:think>old answer[e~[\n]~b]ai\n", output)
    assert (reasoning, content) == ("plan", None)
    assert calls[0]["name"] == "get_weather"
    assert json.loads(calls[0]["arguments"]) == {"city": "Beijing"}


def test_patch_is_idempotent_and_accepts_keyword_tokenizer(patch_module):
    constructor = MiniMaxM3ReasoningParser.__init__
    end_check = MiniMaxM3ReasoningParser.is_reasoning_end
    patch_module._patch_minimax_m31_reasoning()
    assert MiniMaxM3ReasoningParser.__init__ is constructor
    assert MiniMaxM3ReasoningParser.is_reasoning_end is end_check
    tokenizer = make_tokenizer(ADAPTIVE_TEMPLATE)
    parser = MiniMaxM3ReasoningParser(tokenizer=tokenizer, chat_template_kwargs={"thinking_mode": "enabled"})
    request = ChatCompletionRequest(model="test", messages=[])
    assert parser.extract_reasoning("answer", request) == (None, "answer")
