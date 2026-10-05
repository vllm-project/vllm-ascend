# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend" / "patch" / "platform" / "patch_glm53_reasoning.py"
# The identifying fragments from the template used in vllm-project/vllm#56994.
GLM53_TEMPLATE = "[gMASK]<sop> Reasoning Effort: <tool_call><arg_key><arg_value> <|assistant|><think>"
DISABLE_KWARGS = [
    {"enable_thinking": False},
    {"thinking": False},
    {"enable_thinking": False, "thinking": False},
]


def load_patch():
    spec = importlib.util.spec_from_file_location("_test_glm53_reasoning_patch", PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_patch(monkeypatch):
    """Exercise the wrapper without importing unrelated NPU platform patches."""

    class RecordingParser:
        def __init__(self, tokenizer, tools=None, **kwargs):
            self.tokenizer = tokenizer
            self.tools = tools
            self.kwargs = kwargs

    vllm = ModuleType("vllm")
    parser_module = ModuleType("vllm.parser")
    glm_module = ModuleType("vllm.parser.glm47_moe")
    logger_module = ModuleType("vllm.logger")
    monkeypatch.setattr(glm_module, "Glm47MoeParser", RecordingParser, raising=False)
    logger = Mock()
    monkeypatch.setattr(logger_module, "logger", logger, raising=False)
    monkeypatch.setattr(parser_module, "glm47_moe", glm_module, raising=False)
    monkeypatch.setattr(vllm, "parser", parser_module, raising=False)
    monkeypatch.setattr(vllm, "logger", logger_module, raising=False)
    for module in (vllm, parser_module, glm_module, logger_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    original_init = RecordingParser.__init__
    patch = load_patch()
    return patch, RecordingParser, original_init, logger


@pytest.mark.parametrize("disable_kwargs", DISABLE_KWARGS)
def test_normalizes_disabled_thinking_without_mutating_request(isolated_patch, disable_kwargs):
    _, parser_cls, _, logger = isolated_patch
    tokenizer = SimpleNamespace(chat_template=GLM53_TEMPLATE)
    chat_kwargs = {**disable_kwargs, "reasoning_effort": "low", "clear_thinking": False}
    before = chat_kwargs.copy()
    tools = [object()]
    config = object()

    parser = parser_cls(tokenizer, tools, chat_template_kwargs=chat_kwargs, parser_engine_config=config)

    assert parser.kwargs["chat_template_kwargs"] == {**before, "thinking": None, "enable_thinking": None}
    assert chat_kwargs == before
    assert parser.kwargs["chat_template_kwargs"] is not chat_kwargs
    assert parser.tokenizer is tokenizer
    assert parser.tools is tools
    assert parser.kwargs["parser_engine_config"] is config
    logger.warning_once.assert_called_once()


@pytest.mark.parametrize("chat_kwargs", [None, {}, {"thinking": True}, {"enable_thinking": True}])
def test_normal_requests_are_unchanged(isolated_patch, chat_kwargs):
    _, parser_cls, _, logger = isolated_patch
    parser = parser_cls(SimpleNamespace(chat_template=GLM53_TEMPLATE), chat_template_kwargs=chat_kwargs)
    assert parser.kwargs["chat_template_kwargs"] is chat_kwargs
    logger.warning_once.assert_not_called()


@pytest.mark.parametrize(
    "template",
    [None, {}, "", GLM53_TEMPLATE + " enable_thinking"]
    + [
        GLM53_TEMPLATE.replace(marker, "")
        for marker in ("[gMASK]<sop>", "Reasoning Effort:", "<tool_call>", "<arg_key>", "<arg_value>")
    ],
)
def test_other_templates_keep_their_switch(isolated_patch, template):
    _, parser_cls, _, logger = isolated_patch
    chat_kwargs = {"enable_thinking": False}
    parser = parser_cls(SimpleNamespace(chat_template=template), chat_template_kwargs=chat_kwargs)
    assert parser.kwargs["chat_template_kwargs"] is chat_kwargs
    assert parser.kwargs["chat_template_kwargs"]["enable_thinking"] is False
    logger.warning_once.assert_not_called()


def test_missing_template_is_unchanged(isolated_patch):
    _, parser_cls, _, logger = isolated_patch
    parser = parser_cls(object(), chat_template_kwargs={"thinking": False})
    assert parser.kwargs["chat_template_kwargs"] == {"thinking": False}
    logger.warning_once.assert_not_called()


def test_patch_is_idempotent(isolated_patch):
    patch, parser_cls, _, _ = isolated_patch
    patched_init = parser_cls.__init__
    patch._patch_glm53_reasoning()
    assert parser_cls.__init__ is patched_init


def test_upstream_fix_is_not_wrapped(isolated_patch, monkeypatch):
    patch, parser_cls, original_init, _ = isolated_patch
    monkeypatch.setattr(parser_cls, "__init__", original_init)
    monkeypatch.setattr(patch.glm47_moe, "_glm53_always_thinks", lambda _: True, raising=False)
    patch._patch_glm53_reasoning()
    assert parser_cls.__init__ is original_init


@pytest.fixture
def real_parser(monkeypatch):
    # Runs in the regular CPU UT environment. A standalone Windows run can
    # still execute the wrapper tests when vLLM is not installed.
    pytest.importorskip("vllm")
    from vllm.parser import glm47_moe

    monkeypatch.setattr(glm47_moe.Glm47MoeParser, "__init__", glm47_moe.Glm47MoeParser.__init__)
    load_patch()
    vocab = {token: i for i, token in enumerate(("<think>", "</think>", "<tool_call>", "</tool_call>"))}
    tokenizer = SimpleNamespace(
        chat_template=GLM53_TEMPLATE,
        get_vocab=lambda: vocab,
        all_special_tokens=list(vocab),
        all_special_ids=list(vocab.values()),
    )
    return glm47_moe.Glm47MoeParser, tokenizer


@pytest.mark.parametrize("disable_kwargs", DISABLE_KWARGS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_real_parser_keeps_reasoning_out_of_content(real_parser, disable_kwargs, streaming, closed):
    parser_cls, tokenizer = real_parser
    parser = parser_cls(tokenizer, chat_template_kwargs=disable_kwargs)
    assert parser.thinking_enabled
    assert not parser.is_reasoning_end([])

    chunks = ["Simple", " question."]
    if closed:
        chunks += ["</think>", "4"]
    if not streaming:
        reasoning, content = parser.extract_reasoning("".join(chunks), None)
    else:
        deltas = []
        text = ""
        token_ids: list[int] = []
        for chunk in chunks:
            # Include the special token ID, as the serving detokenizer does.
            delta_ids = [tokenizer.get_vocab()[chunk]] if chunk in tokenizer.get_vocab() else []
            delta = parser.extract_reasoning_streaming(
                text, text + chunk, chunk, token_ids, token_ids + delta_ids, delta_ids
            )
            if delta is not None:
                deltas.append(delta)
            text += chunk
            token_ids += delta_ids
        final_delta = parser.finish_streaming()
        if final_delta is not None:
            deltas.append(final_delta)
        reasoning = "".join(delta.reasoning or "" for delta in deltas) or None
        content = "".join(delta.content or "" for delta in deltas) or None

    assert reasoning == "Simple question."
    assert content == ("4" if closed else None)
    assert parser.is_reasoning_end([tokenizer.get_vocab()["</think>"]])


def test_real_parser_preserves_older_glm_behavior(real_parser):
    parser_cls, tokenizer = real_parser
    tokenizer.chat_template += " enable_thinking"
    parser = parser_cls(tokenizer, chat_template_kwargs={"enable_thinking": False})
    assert not parser.thinking_enabled
    output = "Simple question.</think>4"
    assert parser.extract_reasoning(output, None) == (None, output)


def test_real_parser_normal_request(real_parser):
    parser_cls, tokenizer = real_parser
    parser = parser_cls(tokenizer)
    assert parser.extract_reasoning("Simple question.</think>4", None) == ("Simple question.", "4")
