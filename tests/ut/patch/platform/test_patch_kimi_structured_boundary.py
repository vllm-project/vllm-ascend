# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import builtins
import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2Config, PreTrainedTokenizerFast
from vllm.config import DeviceConfig, StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_kimi_structured_boundary.py"


def _load_module(path):
    spec = importlib.util.spec_from_file_location("kimi_boundary_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def patch(monkeypatch):
    # Install only during a test and restore the exact pre-test method, even
    # when the platform has already installed a wrapper before pytest starts.
    name = (
        "_get_constraint_bounds"
        if hasattr(StructuredOutputManager, "_get_constraint_bounds")
        else "_get_constraint_start"
    )
    monkeypatch.setattr(StructuredOutputManager, name, getattr(StructuredOutputManager, name))
    return _load_module(PATCH_PATH)


def _boundary(manager, request, tokens, committed=False, original=False):
    name = "_get_constraint_bounds" if hasattr(manager, "_get_constraint_bounds") else "_get_constraint_start"
    method = getattr(type(manager), name)
    if original:
        method = method.__wrapped__
    result = method(manager, request, tokens, committed)
    return result.constraint_start if hasattr(result, "constraint_start") else result


@pytest.fixture(scope="module", params=["atomic", "unequal"])
def model(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("kimi-boundary")
    # A byte-level tokenizer and config built entirely offline. No HF cache or
    # weights are needed, including for real XGrammar compilation below.
    vocab = {token: i for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab["<|endoftext|>"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>", model_max_length=1024)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ["<|open|>", "<|close|>", "<|sep|>", "<think>", "</think>"]
            if request.param == "atomic"
            else ["<|open|>", "<think>", "</think>"]
        }
    )
    tokenizer.save_pretrained(path)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_embd=64,
        n_layer=1,
        n_head=2,
        architectures=["GPT2LMHeadModel"],
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    config.save_pretrained(path)
    return str(path), tokenizer


def _manager(model, parser="kimi_k3", enable_in_reasoning=False):
    return StructuredOutputManager(
        VllmConfig(
            model_config=ModelConfig(model=model),
            device_config=DeviceConfig(device="cpu"),
            structured_outputs_config=StructuredOutputsConfig(
                backend="xgrammar", reasoning_parser=parser, enable_in_reasoning=enable_in_reasoning
            ),
        )
    )


def _request(manager, tokens, ended=False):
    params = SamplingParams(structured_outputs=StructuredOutputsParams(json={"type": "object"}))
    # Match the request validation path; older releases do not take a tokenizer.
    from inspect import signature

    kwargs = {"tokenizer": manager.tokenizer} if "tokenizer" in signature(params.verify).parameters else {}
    params.verify(manager.vllm_config.model_config, None, manager.vllm_config.structured_outputs_config, **kwargs)
    return Request(
        request_id="kimi-boundary",
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        reasoning_ended=ended,
    )


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("prior_turns", [0, 4])
def test_close_reopen_and_split(model, committed, prior_turns):
    model_path, tokenizer = model
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    opening, closing = "<|open|>think<|sep|>", "<|close|>think<|sep|>"
    history = enc((opening + "old" + closing + "{}") * prior_turns + opening + "current")
    close, reopen = enc(closing), enc(opening)
    manager = _manager(model_path)
    try:
        for split in range(len(close)):
            request = _request(manager, history, None)
            request.append_output_token_ids(close[:split])
            draft = close[split:] + reopen
            if committed:
                request.append_output_token_ids(draft)
            before = list(request.all_token_ids)
            assert _boundary(manager, request, draft, committed) == len(close) - split
            assert list(request.all_token_ids) == before
            # Fixed draft on exactly the same real Request exposes the old bug.
            assert _boundary(manager, request, draft, committed, original=True) > len(close) - split
    finally:
        manager.clear_backend()


@pytest.mark.parametrize("ended,enabled", [(True, False), (False, True)])
def test_early_exits(model, ended, enabled):
    manager = _manager(model[0], enable_in_reasoning=enabled)
    try:
        request = _request(manager, model[1].encode("<|open|>think<|sep|>"), ended)
        assert _boundary(manager, request, [1, 2]) == 0
        assert _boundary(manager, request, []) == 0
    finally:
        manager.clear_backend()


@pytest.mark.parametrize("enabled", [False, True])
def test_enable_in_reasoning_with_close_reopen(model, enabled):
    model_path, tokenizer = model
    manager = _manager(model_path, enable_in_reasoning=enabled)
    try:
        opening = tokenizer.encode("<|open|>think<|sep|>", add_special_tokens=False)
        closing = tokenizer.encode("<|close|>think<|sep|>", add_special_tokens=False)
        request = _request(manager, opening, None)
        draft = closing + opening
        assert _boundary(manager, request, draft) == (0 if enabled else len(closing))
    finally:
        manager.clear_backend()


def test_other_parser_delegates(model):
    manager = _manager(model[0], parser="deepseek_r1")
    try:
        request = _request(manager, model[1].encode("<think>"))
        for tokens in ([], [1], model[1].encode("</think>")):
            assert _boundary(manager, request, tokens) == _boundary(manager, request, tokens, original=True)
    finally:
        manager.clear_backend()


def test_view_bounds_and_slices(patch):
    expected = list(range(9))
    history, draft = expected[:6], expected[6:]
    view = patch._TokenView(history, draft)
    for _ in range(9):
        assert list(view) == expected
        for index in (slice(-4, None), slice(None, None, -1), slice(2, 8), slice(8, 2)):
            assert view[index] == expected[index]
        with pytest.raises(IndexError):
            view[len(expected)]
        view.pop()
        expected.pop()
    assert list(view) == []
    assert history == list(range(6))
    assert draft == list(range(6, 9))


def test_install_is_idempotent(patch):
    name = (
        "_get_constraint_bounds"
        if hasattr(StructuredOutputManager, "_get_constraint_bounds")
        else "_get_constraint_start"
    )
    first = getattr(StructuredOutputManager, name)
    patch._apply_patch()
    assert getattr(StructuredOutputManager, name) is first


@pytest.mark.parametrize("missing", ["boundary", "reasoner", "bounds_type"])
def test_unsupported_api_leaves_manager_untouched(patch, monkeypatch, missing):
    def original(*args):
        return 7

    manager_cls = type("LegacyManager", (), {"_get_constraint_bounds": original, "_get_reasoner": original})
    if missing == "boundary":
        delattr(manager_cls, "_get_constraint_bounds")
    elif missing == "reasoner":
        delattr(manager_cls, "_get_reasoner")
    else:
        monkeypatch.delattr(patch.structured_output, "ConstraintBounds", raising=False)
    before = dict(vars(manager_cls))
    monkeypatch.setattr(patch.structured_output, "StructuredOutputManager", manager_cls)
    warning = Mock()
    monkeypatch.setattr(patch.logger, "warning", warning)
    # Execute the real import-time installer, as platform startup does.
    _load_module(PATCH_PATH)
    assert dict(vars(manager_cls)) == before
    warning.assert_called_once()
    assert "not installed" in warning.call_args.args[0]


def test_missing_kimi_parser_leaves_manager_untouched(patch, monkeypatch):
    name = (
        "_get_constraint_bounds"
        if hasattr(StructuredOutputManager, "_get_constraint_bounds")
        else "_get_constraint_start"
    )
    original = getattr(StructuredOutputManager, name).__wrapped__
    monkeypatch.setattr(StructuredOutputManager, name, original)
    monkeypatch.setitem(sys.modules, "vllm.reasoning.kimi_k3_reasoning_parser", None)
    warning = Mock()
    monkeypatch.setattr(patch.logger, "warning", warning)
    _load_module(PATCH_PATH)
    assert getattr(StructuredOutputManager, name) is original
    warning.assert_called_once()


def test_collection_does_not_install_patch(monkeypatch):
    name = (
        "_get_constraint_bounds"
        if hasattr(StructuredOutputManager, "_get_constraint_bounds")
        else "_get_constraint_start"
    )
    original = getattr(StructuredOutputManager, name).__wrapped__
    monkeypatch.setattr(StructuredOutputManager, name, original)
    _load_module(Path(__file__))
    assert getattr(StructuredOutputManager, name) is original


def test_parser_dependency_error_is_not_hidden(patch, monkeypatch):
    name = (
        "_get_constraint_bounds"
        if hasattr(StructuredOutputManager, "_get_constraint_bounds")
        else "_get_constraint_start"
    )
    original = getattr(StructuredOutputManager, name).__wrapped__
    monkeypatch.setattr(StructuredOutputManager, name, original)
    real_import = builtins.__import__

    def import_without_dependency(module_name, *args, **kwargs):
        if module_name == "vllm.reasoning.kimi_k3_reasoning_parser":
            raise ModuleNotFoundError("broken parser dependency", name="parser_dependency")
        return real_import(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dependency)
    with pytest.raises(ModuleNotFoundError, match="broken parser dependency"):
        patch._apply_patch()
    assert getattr(StructuredOutputManager, name) is original


@pytest.mark.parametrize("ended", [None, True])
@pytest.mark.parametrize("committed", [False, True])
def test_thinking_disabled_constrains_first_json_token(model, ended, committed):
    manager = _manager(model[0])
    try:
        tokenizer = model[1]
        request = _request(manager, tokenizer.encode("<|open|>think<|sep|>"), ended)
        structured_req = request.structured_output_request
        structured_req.reasoning_parser_kwargs = {"chat_template_kwargs": {"thinking": False}}
        assert manager._get_reasoner(request).is_reasoning_end(request.prompt_token_ids)
        draft = tokenizer.encode("{}")
        if committed:
            request.append_output_token_ids(draft)
        assert _boundary(manager, request, draft, committed) == 0
        assert _boundary(manager, request, draft, committed, original=True) == 0
        manager.grammar_init(request)
        deadline = time.monotonic() + 10
        while not structured_req.is_grammar_ready:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        grammar = structured_req.grammar
        assert grammar is not None
        assert not grammar.validate_tokens(tokenizer.encode("<|open|>"))
        assert manager.accept_tokens(request, draft)
    finally:
        manager.clear_backend()


def test_json_commit_after_first_boundary(model):
    model_path, tokenizer = model
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    manager = _manager(model_path)
    try:
        request = _request(manager, enc("<|open|>think<|sep|>current"))
        manager.grammar_init(request)
        deadline = time.monotonic() + 10
        while not request.structured_output_request.is_grammar_ready:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        grammar = request.structured_output_request.grammar
        close = enc("<|close|>think<|sep|>")
        reopen = enc("<|open|>think<|sep|>")
        assert _boundary(manager, request, close + reopen) == len(close)
        assert not grammar.validate_tokens([reopen[0]])
        accepted = close + enc("{}")
        request.append_output_token_ids(accepted)
        assert manager.accept_tokens(request, accepted)
        assert request.structured_output_request.reasoning_ended is True
    finally:
        manager.clear_backend()
