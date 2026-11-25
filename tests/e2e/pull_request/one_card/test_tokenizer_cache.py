# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E and token-id parity tests for the segment-level tokenizer cache."""

import json
import os

import pytest
import requests
from transformers import AutoTokenizer
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from vllm_ascend.patch.platform.patch_tokenizer_cache import IncrementalTokenizerCache

MODEL_NAME = os.getenv("TOKENIZER_CACHE_TEST_MODEL", "Qwen/Qwen3-0.6B")
_BASE_TURNS = [
    {"role": "system", "content": "You are a helpful coding assistant. Be concise."},
    {"role": "user", "content": "Write a Python function that reverses a singly linked list."},
    {
        "role": "assistant",
        "content": "def reverse(h):\n    p = None\n    while h:\n        h.next, p, h = p, h, h.next\n    return p",
    },
]
_APPENDS = [
    {"role": "user", "content": "Now make it iterative only, no recursion."},
    {"role": "assistant", "content": "The version above is already iterative."},
    {"role": "user", "content": "Add a docstring and type hints."},
    {"role": "assistant", "content": "Only a summary this time: reverse a linked list in place."},
]


def _transcripts(tok):
    turns, texts = list(_BASE_TURNS), []
    for turn in [None, *_APPENDS]:
        if turn is not None:
            turns.append(turn)
        texts.append(tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True))
    return texts


@pytest.mark.e2e_model(MODEL_NAME)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="prefix_caching",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
def test_cache_encode_matches_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    cache = IncrementalTokenizerCache(tok, capacity_gb=1)
    assert cache.enabled
    for text in _transcripts(tok):
        exp = list(tok(text, add_special_tokens=False)["input_ids"])
        assert cache.encode(text) == exp and cache.encode(text) == exp
    full = tok.apply_chat_template(
        list(_BASE_TURNS) + _APPENDS + [{"role": "user", "content": "Thanks!"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert cache.encode(full) == list(tok(full, add_special_tokens=False)["input_ids"]) and cache.stat().hits > 0


def _collect_token_ids(server, prompts):
    url = server.url_for("tokenize")
    plain = [
        requests.post(url, json={"model": MODEL_NAME, "prompt": p, "add_special_tokens": False}, timeout=60).json()[
            "tokens"
        ]
        for p in prompts
    ]
    chat = [
        requests.post(url, json={"model": MODEL_NAME, "messages": t, "add_generation_prompt": True}, timeout=60).json()[
            "tokens"
        ]
        for t in (_BASE_TURNS, _BASE_TURNS + _APPENDS)
    ]
    return plain, chat


@pytest.mark.e2e_model(MODEL_NAME)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="prefix_caching",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_cache_does_not_change_served_token_ids():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompts = _transcripts(tok)
    prompts.append(prompts[-1] + "\n\nSummarize the API above in one sentence.")
    results = []
    for cache_gb in (0, 1):
        port = get_open_port()
        args = [
            "--enforce-eager",
            "--max-model-len",
            "4096",
            "--port",
            str(port),
            "--additional-config",
            json.dumps({"tokenizer_cache_gb": cache_gb}),
        ]
        with RemoteOpenAIServer(
            MODEL_NAME, vllm_serve_args=args, server_host="127.0.0.1", server_port=port, auto_port=False
        ) as srv:
            results.append(_collect_token_ids(srv, prompts))
    (off_ids, off_chat), (on_ids, on_chat) = results
    assert on_ids == off_ids and on_chat == off_chat
    assert off_ids == [list(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
