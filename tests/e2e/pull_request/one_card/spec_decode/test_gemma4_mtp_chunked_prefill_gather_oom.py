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
"""E2E acceptance: Gemma4 MTP chunked prefill 512-dim gather must not OOM.

Validates the fix for the ``_gather_paged_kv_to_dense`` max-padding OOM
(commit a60d57541). Before the fix, a batch mixing a very long sequence
(~100K tokens, e.g. a long-context decode request) with short prompts
triggered NPU OOM in ``_gather_paged_kv_to_dense`` (device/utils.py): the
512-dim global-attention prefill fallback gathered every sequence's paged KV
padded to the batch's ``max(seq_lens)``, materialising a
``(num_seqs, max_seq_len, num_kv_heads, head_size)`` dense tensor (~10 GiB
each for K/V) that blew past the 60 GiB budget.

These tests exercise the three load shapes that reproduced the crash and
assert they now return 200 with valid output. They target an already-running
vLLM OpenAI server (Gemma4 31B MTP, typically TP4). Configure via:

    VLLM_TEST_SERVER_URL  (default http://127.0.0.1:8031)
    VLLM_TEST_MODEL       (default gemma-4-31b-it)

They skip when the server is unreachable, so collecting them in the default
suite is harmless on machines without the deployed model.
"""

from __future__ import annotations

import os
import threading

import pytest
import requests

SERVER_URL = os.environ.get("VLLM_TEST_SERVER_URL", "http://127.0.0.1:8031").rstrip("/")
MODEL = os.environ.get("VLLM_TEST_MODEL", "gemma-4-31b-it")
CHAT_URL = f"{SERVER_URL}/v1/chat/completions"

# ~38 tokens each; repeated to build long prompts that force chunked prefill
# and the 512-dim large-head gather path.
PARA = (
    "In high-performance LLM inference, speculative decoding cuts latency by "
    "having a draft model propose candidate tokens that the target model verifies "
    "in one forward pass, accepting matched tokens to skip decode steps. "
)


def _server_up() -> bool:
    try:
        return requests.get(f"{SERVER_URL}/v1/models", timeout=5).status_code == 200
    except Exception:
        return False


# Skip the whole module if the Gemma4 server is not deployed.
pytestmark = pytest.mark.skipif(
    not _server_up(),
    reason=f"Gemma4 server not reachable at {SERVER_URL} (set VLLM_TEST_SERVER_URL)",
)


def _post(content: str, max_tokens: int = 8, timeout: int = 300) -> requests.Response:
    return requests.post(
        CHAT_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=timeout,
    )


def _concurrent(jobs: list[tuple[str, str, int]]) -> list[tuple[str, int, int, str]]:
    """Run (name, content, max_tokens) jobs concurrently; return results.

    Concurrent issue is what lets a long and several short requests land in the
    same scheduler batch — the exact long+short mix that max-padded the gather.
    """
    out: list[tuple[str, int, int, str]] = []
    lock = threading.Lock()

    def run(name: str, content: str, max_tok: int) -> None:
        r = _post(content, max_tok)
        j = r.json() if r.status_code == 200 else {}
        with lock:
            out.append(
                (
                    name,
                    r.status_code,
                    j.get("usage", {}).get("prompt_tokens", 0),
                    j.get("choices", [{}])[0].get("message", {}).get("content", ""),
                )
            )

    ts = [threading.Thread(target=run, args=j) for j in jobs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def test_long_prompt_chunked_prefill_no_oom():
    """A single ~7K-token prompt exercises chunked prefill + the 512-dim
    large-head gather path. Must not OOM and must reply correctly."""
    r = _post(PARA * 180 + "\n\nReply with exactly: OK", max_tokens=10)
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    assert j["usage"]["prompt_tokens"] > 1000, f"expected a long prompt, got {j['usage']['prompt_tokens']}"
    assert "OK" in j["choices"][0]["message"]["content"]


def test_mixed_long_short_batch_no_oom():
    """Long+short mixed batch — the OOM trigger: padding blows up when
    ``max(seq_lens)`` >> the short sequences' lengths. All requests must
    succeed (the pre-fix crash returned 500 here)."""
    jobs = [
        ("L25k", PARA * 660 + "\n\nReply OK", 5),
        ("L15k", PARA * 400 + "\n\nReply OK", 5),
        ("L8k", PARA * 210 + "\n\nReply OK", 5),
    ] + [(f"s{i}", f"Say the number {i}.", 5) for i in range(10)]
    out = _concurrent(jobs)
    failed = [(n, s) for (n, s, _, _) in out if s != 200]
    assert not failed, f"requests failed (OOM?): {failed}"
    # the long prompts must actually be long, exercising the gather path
    longs = {n: pt for (n, _, pt, _) in out if n.startswith("L")}
    assert longs["L25k"] > 20000, f"expected ~25k prompt, got {longs}"


def test_100k_long_short_mixed_no_oom():
    """Closest to the original 109K-token crash: one ~100K-token request plus
    short prompts issued into the same batch window. This is the shape that
    OOMed before the fix (allocated 2.52 GiB with only 470 MiB free)."""
    jobs = [("L100k", PARA * 2630 + "\n\nReply OK", 5)] + [(f"s{i}", f"Say the number {i}.", 5) for i in range(8)]
    out = _concurrent(jobs)
    failed = [(n, s) for (n, s, _, _) in out if s != 200]
    assert not failed, f"requests failed (OOM?): {failed}"
    long_pt = next(pt for (n, _, pt, _) in out if n == "L100k")
    assert long_pt > 90000, f"expected ~100k prompt, got {long_pt}"
    assert "OK" in next(c for (n, _, _, c) in out if n == "L100k")
