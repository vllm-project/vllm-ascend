# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
from pathlib import Path

import pytest
from transformers import AutoConfig, AutoTokenizer
from vllm.config import DeviceConfig, StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

# Load this CPU-only patch without importing unrelated NPU platform patches.
PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_kimi_structured_boundary.py"
spec = importlib.util.spec_from_file_location("kimi_boundary_patch", PATCH_PATH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ["<|open|>", "<|close|>", "<|sep|>", "<think>", "</think>"]
            if request.param == "atomic"
            else ["<|open|>", "<think>", "</think>"]
        }
    )
    tokenizer.save_pretrained(path)
    config = AutoConfig.from_pretrained("gpt2")
    config.vocab_size = len(tokenizer)
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


def test_view_bounds_and_slices():
    expected = list(range(9))
    view = patch._TokenView(expected[:6], expected[6:])
    for _ in range(9):
        assert list(view) == expected
        for index in (slice(-4, None), slice(None, None, -1), slice(2, 8), slice(8, 2)):
            assert view[index] == expected[index]
        with pytest.raises(IndexError):
            view[len(expected)]
        view.pop()
        expected.pop()
    assert list(view) == []


def test_install_is_idempotent():
    name = (
        "_get_constraint_bounds"
        if hasattr(StructuredOutputManager, "_get_constraint_bounds")
        else "_get_constraint_start"
    )
    first = getattr(StructuredOutputManager, name)
    patch._apply_patch()
    assert getattr(StructuredOutputManager, name) is first


def test_json_commit_after_first_boundary(model):
    import time

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
