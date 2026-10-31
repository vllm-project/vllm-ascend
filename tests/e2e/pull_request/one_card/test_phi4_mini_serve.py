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
"""T4 (issue #10677): Phi-4-mini-instruct online (serve) e2e.

- harness follows the one_card precedent (Popen + /health polling).
- serialization discipline: the offline reference engine runs FIRST and is
  released before the server starts (single-card exclusive); the served
  fixture blocks on wait_until_npu_memory_free so the server does not race
  the engine's asynchronous HBM teardown. After the run, kill the server
  and confirm HBM release with npu-smi info. Server stdout/stderr go to a
  log file (never an undrained PIPE, which deadlocks once full).
- assertion policy (issue #10677, expert-adjudicated): serve and offline
  batch assembly and float reduction order differ, so bit-level equality is
  NOT asserted; online greedy vs offline greedy requires the first 16
  tokens to be >= half equal and output length within 4. Online
  non-greedy is recorded with a fixed seed (coverage note in the PR).
- port 8123 avoids the local 8000 (reader-facing tutorial examples use
  8000; the two do not conflict).

Run: PHI4_MODEL_PATH=/path/to/Phi-4-mini-instruct pytest -sv
tests/e2e/pull_request/one_card/test_phi4_mini_serve.py
"""

import os
import subprocess
import tempfile
import time

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free

# Default ModelScope id (pre-downloaded in CI via model_dataset_list.json);
# override with PHI4_MODEL_PATH for local runs.
MODEL = os.environ.get("PHI4_MODEL_PATH", "LLM-Research/Phi-4-mini-instruct")
PORT = 8123
SERVED_NAME = "phi4-mini-e2e"  # explicit served name; stable across path/id inputs
PREFIX_N = 16
LEN_TOL = 4

ONLINE_CASES = [
    ("zh", "用两句话介绍长城的历史。"),
    ("en", "Introduce the Great Wall in two sentences."),
]


def _wait_healthy(timeout_s: int = 900) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5):
                return
        except Exception:
            time.sleep(5)
    raise TimeoutError("vllm serve not healthy in time")


@pytest.fixture(scope="module")
def offline_refs():
    """Offline greedy reference, computed BEFORE the server starts.

    Returns ({lang: (text, n_tokens)}, tok): the tokenizer resolved by the
    vLLM engine is reused for the serve-vs-offline comparison, so the test
    does not make a second direct HF-hub resolution of the ModelScope id.
    """
    import gc

    from vllm import LLM, SamplingParams

    # In a single pytest session the offline test module's engine is torn
    # down right before this module starts; wait out its HBM release.
    _wait_hbm_free()
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()
    try:
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for _, p in ONLINE_CASES
        ]
        outs = llm.generate(texts, SamplingParams(temperature=0, max_tokens=64))
        refs = {lang: (o.outputs[0].text, len(o.outputs[0].token_ids)) for (lang, _), o in zip(ONLINE_CASES, outs)}
    finally:
        del llm
        gc.collect()
    return refs, tok


def _wait_hbm_free():
    """Block until >= 50% NPU HBM is free (engine teardown is asynchronous).

    wait_until_npu_memory_free is a decorator in tests/e2e/conftest.py;
    applying it to a no-op and calling the wrapper gives the wait alone.
    """
    wait_until_npu_memory_free(target_free_percentage=0.5, max_wait_seconds=120)(lambda: None)()


@pytest.fixture(scope="module")
def served(offline_refs):
    """Start vllm serve only after the offline engine's HBM is released."""
    _wait_hbm_free()
    # Logs go to a file: an undrained stdout PIPE deadlocks the server once
    # the OS pipe buffer fills, and the file doubles as a run artifact.
    log_path = os.path.join(tempfile.gettempdir(), f"phi4_mini_serve_{os.getpid()}.log")
    print(f"server log: {log_path}")
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            [
                "vllm",
                "serve",
                MODEL,
                "--port",
                str(PORT),
                "--served-model-name",
                SERVED_NAME,
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "4096",
                "--gpu-memory-utilization",
                "0.8",
                "--enforce-eager",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_healthy()
            yield f"http://127.0.0.1:{PORT}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)


def _online_chat(base_url: str, prompt: str, temperature: float, seed=None, max_tokens=64):
    from openai import OpenAI

    client = OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY")
    kwargs = dict(
        model=SERVED_NAME,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if seed is not None:
        kwargs["seed"] = seed
    resp = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], **kwargs)
    return resp.choices[0].message.content, resp.usage.completion_tokens


def test_online_greedy_matches_offline_prefix(served, offline_refs):
    """Online greedy vs offline greedy: first 16 tokens >= half equal +
    output length within 4 (not bit-level; batch assembly and float
    reduction order differ)."""
    refs, tok = offline_refs

    for lang, prompt in ONLINE_CASES:
        on_text, on_len = _online_chat(served, prompt, temperature=0)
        off_text, off_len = refs[lang]
        eq = 0
        for a, b in zip(tok(off_text).input_ids[:PREFIX_N], tok(on_text).input_ids[:PREFIX_N]):
            if a != b:
                break
            eq += 1
        print(f"{lang}: prefix_eq={eq}/{PREFIX_N} off_len={off_len} on_len={on_len}")
        print(f"  offline: {off_text[:160]!r}")
        print(f"  online : {on_text[:160]!r}")
        assert eq >= PREFIX_N // 2, f"{lang}: prefix {eq}/{PREFIX_N}"
        assert abs(off_len - on_len) <= LEN_TOL, f"{lang}: len {off_len} vs {on_len}"


def test_online_sampling_recorded(served):
    """Online non-greedy: fixed seed, two requests recorded (coverage note:
    no hard cross-run equality assertion; NPU scheduling is non-deterministic)."""
    prompt = ONLINE_CASES[0][1]
    _, n1 = _online_chat(served, prompt, temperature=0.7, seed=42)
    _, n2 = _online_chat(served, prompt, temperature=0.7, seed=42)
    print(f"online sampling same-seed completion_tokens: {n1}, {n2}")
    assert n1 > 0 and n2 > 0


def test_online_empty_prompt_no_crash(served):
    """Empty user content must not crash or hang the server.

    Measured behavior (vllm 0.26.0): the chat template renders the empty
    message into special tokens and the API answers with HTTP 200, so a
    400 rejection is NOT asserted (upstream semantics). The engine must
    stay healthy and keep serving afterwards.
    """
    from openai import BadRequestError, OpenAI

    client = OpenAI(base_url=f"{served}/v1", api_key="EMPTY")
    try:
        resp = client.chat.completions.create(
            model=SERVED_NAME,
            messages=[{"role": "user", "content": ""}],
            max_tokens=8,
            temperature=0,
        )
        assert resp.choices, "empty prompt answered with an empty choice list"
    except BadRequestError:
        pass  # rejection is also acceptable engine behavior

    # the engine must still be healthy after the boundary case
    follow = client.chat.completions.create(
        model=SERVED_NAME,
        messages=[{"role": "user", "content": "Reply with the word: ok."}],
        max_tokens=8,
        temperature=0,
    )
    assert follow.choices[0].message.content
