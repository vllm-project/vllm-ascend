#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Coverage for the segment-level tokenizer cache.

``additional_config.tokenizer_cache_gb`` is only allowed to remove redundant
work: every token id it returns must be bit-identical to what the wrapped
tokenizer would have produced. The tests below pin that down at both ends of
the patch - the cache object itself and a served endpoint.

Only *token ids* are compared across servers, never generated text: greedy
decode is not bit-reproducible on Ascend (the batch composition changes the
MoE reduction order), so two servers running the very same configuration
already disagree on the sampled tokens. Comparing text would be flaky.

Why this needs an e2e test on top of the unit tests: the unit tests drive
``IncrementalTokenizerCache`` directly, so they stay green no matter how the
patch is wired in. Only a served endpoint can show that
``BaseRenderer._tokenize_prompt`` and ``safe_apply_chat_template`` are still the
symbols this vLLM revision calls, that the cache actually reaches them (it is
published on the renderer's tokenizer), and that ``make_async`` binds the
patched function rather than the original one. Getting any of that wrong leaves
the feature dead with no failing unit test - and for a tokenizer cache, token
ids are also the only currency in which correctness can be stated at all.
"""

import json
import os

import pytest
import requests
from transformers import AutoTokenizer
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from vllm_ascend.patch.platform.patch_tokenizer_cache import IncrementalTokenizerCache

# Mirrors the other e2e tests: the CI runners serve this from the local HF
# cache, and a local checkout can point it at an on-disk copy instead.
MODEL_NAME = os.getenv("TOKENIZER_CACHE_TEST_MODEL", "Qwen/Qwen3-0.6B")
# A three-turn agent transcript. Every extra turn re-sends the previous ones
# verbatim, which is the workload the cache exists for.
_BASE_TURNS = [
    {"role": "system", "content": "You are a helpful coding assistant. Be concise."},
    {"role": "user", "content": "Write a Python function that reverses a singly linked list."},
    {
        "role": "assistant",
        "content": (
            "def reverse(head):\n"
            "    prev = None\n"
            "    while head:\n"
            "        head.next, prev, head = prev, head, head.next\n"
            "    return prev"
        ),
    },
]

_APPENDS = [
    {"role": "user", "content": "Now make it iterative only, no recursion."},
    {"role": "assistant", "content": "The version above is already iterative."},
    {"role": "user", "content": "Add a docstring and type hints."},
    {"role": "assistant", "content": "Only a summary this time: reverse a linked list in place."},
]


def _render(tokenizer, turns):
    return tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)


def _transcripts(tokenizer):
    """Growing agent transcripts, i.e. long shared prefixes plus a short tail."""
    turns = list(_BASE_TURNS)
    texts = [_render(tokenizer, turns)]
    for turn in _APPENDS:
        turns.append(turn)
        texts.append(_render(tokenizer, turns))
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
    """The cache must be id-exact for growing agent transcripts."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=1)
    assert cache.enabled, "this tokenizer exposes added tokens, so the cache must arm itself"

    for text in _transcripts(tokenizer):
        expected = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        assert cache.encode(text) == expected
        # Second pass is the interesting one: it is served from the segments
        # the first pass stored, so a wrong segment boundary shows up here.
        assert cache.encode(text) == expected

    # A short suffix on top of an already-cached transcript is the real agent
    # step, and the only case where a segment is reused across prompts.
    turns = list(_BASE_TURNS) + _APPENDS
    turns.append({"role": "user", "content": "Thanks!"})
    text = _render(tokenizer, turns)
    assert cache.encode(text) == list(tokenizer(text, add_special_tokens=False)["input_ids"])
    assert cache.stat().hits > 0, "re-encoding a shared prefix must hit the cache"


def _serve(port, cache_gb):
    return RemoteOpenAIServer(
        MODEL_NAME,
        vllm_serve_args=[
            "--enforce-eager",
            "--max-model-len",
            "4096",
            "--port",
            str(port),
            # The cache is opt-in through additional_config; both servers get an
            # explicit value so the only difference between them is the size.
            "--additional-config",
            json.dumps({"tokenizer_cache_gb": cache_gb}),
        ],
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
    )


def _collect_token_ids(server, prompts):
    """Token ids via both renderer entry points the cache hooks into."""
    plain = [
        requests.post(
            server.url_for("tokenize"),
            json={"model": MODEL_NAME, "prompt": prompt, "add_special_tokens": False},
            timeout=60,
        ).json()["tokens"]
        for prompt in prompts
    ]
    chat = [
        requests.post(
            server.url_for("tokenize"),
            json={"model": MODEL_NAME, "messages": turns, "add_generation_prompt": True},
            timeout=60,
        ).json()["tokens"]
        for turns in (_BASE_TURNS, _BASE_TURNS + _APPENDS)
    ]
    return plain, chat


@wait_until_npu_memory_free()
def test_cache_does_not_change_served_token_ids():
    """A server with the cache on must tokenize exactly like one with it off.

    The chat requests are the interesting half: the DeepSeek/HF renderer fast
    path is only reached through ``apply_chat_template``, and the prompt path
    through ``_tokenize_prompt``. A regression in either one shows up as a
    token-id difference here.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompts = _transcripts(tokenizer)
    # The last transcript is the full conversation, so it is the one that can
    # hit cache segments written by the earlier requests.
    prompts.append(prompts[-1] + "\n\nSummarize the API above in one sentence.")

    with _serve(get_open_port(), cache_gb=0) as server:
        off_ids, off_chat = _collect_token_ids(server, prompts)
    with _serve(get_open_port(), cache_gb=1) as server:
        on_ids, on_chat = _collect_token_ids(server, prompts)

    assert on_ids == off_ids, "the cache changed the token ids returned by /tokenize"
    assert on_chat == off_chat, "the cache changed the chat token ids returned by /tokenize"

    # Guard against a self-consistently wrong cache: both servers must also
    # agree with the plain tokenizer on the pre-tokenized prompt.
    expected = [list(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
    assert off_ids == expected
