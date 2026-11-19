#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Kimi-K2.6 (W4A8 + DFlash) Model Runner V1 vs V2 parity on one 16-die A3 node.

Reproducible configuration (see REPRODUCE-kimi-k26-mrv.md for the full record):

    image            quay.io/ascend/vllm-ascend:v0.30.0-a3
                     (content digest sha256:e6c22b94f4ab3634ad20ae4ad28b73b79ef59a1e9eb0372454affd8c186e36a6)
    hardware         Atlas 800T A3, 16 x Ascend910 (d803), 64 GiB HBM per die
    target model     /models/Kimi-K2.6-w4a8          (W4A8, 126 shards, 535.9 GB)
    draft model      /models/Kimi-K2.5-DFlash        (DFlash, 2 shards, 6.96 GB)
    parallelism      TP8 / DP2 / EP16, FULL_DECODE_ONLY cudagraph
    context          65536 tokens, max-num-seqs 8, max-num-batched-tokens 8192
    speculative      dflash, num_speculative_tokens=7
    only difference  VLLM_USE_V2_MODEL_RUNNER=0 (MRV1) vs 1 (MRV2)

Run:

    KIMI_MODEL=/models/Kimi-K2.6-w4a8 \
    KIMI_DRAFT_MODEL=/models/Kimi-K2.5-DFlash \
    HCCL_SOCKET_IFNAME=eth0 \
    pytest -sv tests/e2e/nightly/single_node/model_runner_v2/test_kimi_k26_mrv1_mrv2_parity.py

Exit criteria
  * greedy parity: identical token ids, |dlogprob| <= 0.0689 on the first token
    and <= 2 x 0.0689 afterwards (same rule as the DeepSeek V3.2 parity test).
  * API contract: Kimi control markers must never leak into `content`; the
    reasoning channel and tool calls must be parsed into structured fields.
  * performance: recorded for both runners; a > 10 % gap is only asserted when
    KIMI_STRICT_PERF=1, because a single unpaired run cannot separate the runner
    effect from same-runner drift (use the four-round ABBA harness for that).
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, cleanup_dist_env_and_memory, wait_until_npu_memory_free

MODEL = os.getenv("KIMI_MODEL", "Eco-Tech/Kimi-K2.6-w4a8")
DRAFT_MODEL = os.getenv("KIMI_DRAFT_MODEL", "z-lab/Kimi-K2.5-DFlash")
MODEL_REVISION = os.getenv("KIMI_MODEL_REVISION") or None
RESULT_PATH = Path(os.getenv("KIMI_MRV_RESULT", "/tmp/kimi_k26_mrv1_mrv2_parity.json"))

SEED = 1234
ACCURACY_MAX_TOKENS = 32
PERFORMANCE_MAX_TOKENS = 32
MAX_MODEL_LEN = int(os.getenv("KIMI_MAX_MODEL_LEN", "65536"))
MAX_NUM_SEQS = int(os.getenv("KIMI_MAX_NUM_SEQS", "8"))
MAX_NUM_BATCHED_TOKENS = int(os.getenv("KIMI_MAX_NUM_BATCHED_TOKENS", "8192"))
GPU_MEM_UTIL = float(os.getenv("KIMI_GPU_MEM_UTIL", "0.85"))
TP = int(os.getenv("KIMI_TP", "8"))
DP = int(os.getenv("KIMI_DP", "2"))
SPEC_TOKENS = int(os.getenv("KIMI_SPEC_TOKENS", "7"))
SAMPLED_REPEATS = int(os.getenv("KIMI_SAMPLED_REPEATS", "2"))
PERFORMANCE_REPEATS = int(os.getenv("KIMI_REPEATS", "3"))
PERFORMANCE_WARMUP_REPEATS = int(os.getenv("KIMI_WARMUPS", "1"))
PERFORMANCE_BATCH_SIZES = tuple(int(v) for v in os.getenv("KIMI_BATCH_SIZES", "1,4,8").split(","))
PERFORMANCE_SEQ_LENGTHS = tuple(int(v) for v in os.getenv("KIMI_SEQ_LENGTHS", "128,2048,32768").split(","))
PERFORMANCE_TOLERANCE = 0.10
PREFILL_LOGPROB_ATOL = 0.0689
DECODE_LOGPROB_ATOL = 2 * PREFILL_LOGPROB_ATOL
TOP_LOGPROBS = 20
REQUEST_TIMEOUT_SECONDS = (30, 3600)
HBM_SAMPLE_INTERVAL_SECONDS = 0.5
RUNNER_CLEANUP_SECONDS = int(os.getenv("KIMI_RUNNER_CLEANUP_SECONDS", "45"))
STRICT_PERF = os.getenv("KIMI_STRICT_PERF", "0") == "1"

# Short English, short Chinese, code, long English, long Chinese.
PROMPTS = [
    "The capital of France is",
    "用三句话解释什么是张量并行。",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Summarize this English passage: " + "the quick brown fox jumps over the lazy dog " * 160,
    "请总结下面这段重复文本的主题：" + "人工智能需要可靠的软硬件协同。" * 240,
]

# Markers the Kimi chat template emits; they must be consumed by the parser.
_KIMI_CONTROL_MARKERS = (
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|channel|>",
    "<|constrain|>",
)

_HBM_USAGE_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_NPU_ROW_RE = re.compile(r"^\|\s*(\d+)\s+(\d+)\s+\|")
_METRIC_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9eE.+-]+)\s*$")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]


class _NpuMemoryMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_total_mb = 0
        self.peak_per_npu_mb: dict[str, int] = {}
        self.sampling_error = None

    @staticmethod
    def _sample() -> dict[str, int]:
        output = subprocess.check_output(["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL)
        usage: dict[str, int] = {}
        for line in output.splitlines():
            npu_row = _NPU_ROW_RE.match(line)
            values = _HBM_USAGE_RE.findall(line)
            if npu_row and values and int(values[-1][1]) > 0:
                usage[f"{npu_row.group(1)}:{npu_row.group(2)}"] = int(values[-1][0])
        return usage

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._sample()
            except (OSError, subprocess.SubprocessError) as exc:
                self.sampling_error = str(exc)
                self._stop.wait(HBM_SAMPLE_INTERVAL_SECONDS)
                continue
            if sample:
                self.peak_total_mb = max(self.peak_total_mb, sum(sample.values()))
                for npu, used_mb in sample.items():
                    self.peak_per_npu_mb[npu] = max(self.peak_per_npu_mb.get(npu, 0), used_mb)
                self.sampling_error = None
            self._stop.wait(HBM_SAMPLE_INTERVAL_SECONDS)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._stop.set()
        self._thread.join(timeout=15)


def _software_versions() -> dict[str, str]:
    def safe(pkg: str) -> str:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "absent"

    return {
        "vllm": safe("vllm"),
        "vllm_ascend": safe("vllm-ascend"),
        "torch": torch.__version__,
        "torch_npu": safe("torch-npu"),
        "cann": os.getenv("CANN_VERSION", "unreported"),
        "platform": platform.platform(),
    }


def _serve_args(port: int) -> list[str]:
    args = [
        "--host", "0.0.0.0",
        "--port", str(port),
        "--quantization", "ascend",
        "--tensor-parallel-size", str(TP),
        "--data-parallel-size", str(DP),
        "--enable-expert-parallel",
        "--trust-remote-code",
        "--safetensors-load-strategy", "prefetch",
        "--no-enable-prefix-caching",
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--mm-processor-cache-gb", "0",
        "--mm-encoder-tp-mode", "data",
        "--compilation-config", json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"}),
        "--tool-call-parser", "kimi_k2",
        "--reasoning-parser", "kimi_k2",
        # required: without it a request carrying `tools` is rejected with 400
        "--enable-auto-tool-choice",
    ]
    if SPEC_TOKENS > 0:
        args.extend(
            [
                "--speculative-config",
                json.dumps({"method": "dflash", "model": DRAFT_MODEL, "num_speculative_tokens": SPEC_TOKENS}),
            ]
        )
    if MODEL_REVISION:
        args.extend(["--revision", MODEL_REVISION])
    return args


def _server_environment(model_runner: str) -> dict[str, str]:
    ifname = os.getenv("HCCL_SOCKET_IFNAME", "eth0")
    return {
        "VLLM_USE_V2_MODEL_RUNNER": model_runner,
        "VLLM_USE_V1": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": os.getenv("KIMI_HCCL_BUFFSIZE", "800"),
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": os.getenv("KIMI_OMP_THREADS", "10"),
        "TASK_QUEUE_ENABLE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HCCL_SOCKET_IFNAME": ifname,
        "GLOO_SOCKET_IFNAME": ifname,
        "TP_SOCKET_IFNAME": ifname,
    }


# --------------------------------------------------------------------- accuracy
def _completion(server: RemoteOpenAIServer, prompt: str, *, temperature: float) -> dict[str, Any]:
    response = requests.post(
        server.url_for("v1", "completions"),
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": ACCURACY_MAX_TOKENS,
            "temperature": temperature,
            "top_p": 0.95 if temperature else 1.0,
            "top_k": 20 if temperature else 0,
            "seed": SEED,
            "logprobs": TOP_LOGPROBS,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _accuracy_run(server: RemoteOpenAIServer) -> dict[str, Any]:
    greedy = [_completion(server, prompt, temperature=0.0) for prompt in PROMPTS]
    sampled = [
        [_completion(server, prompt, temperature=0.8) for _ in range(SAMPLED_REPEATS)] for prompt in PROMPTS
    ]
    return {"greedy": greedy, "sampled": sampled}


def _choice_tokens(response: dict[str, Any]) -> list[str]:
    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return logprobs.get("tokens") or [choice["text"]]


def _choice_logprobs(response: dict[str, Any]) -> list[float | None]:
    logprobs = response["choices"][0].get("logprobs")
    if logprobs is None:
        return []
    return logprobs.get("token_logprobs") or []


def _choice_top_logprobs(response: dict[str, Any]) -> list[dict[str, float]]:
    logprobs = response["choices"][0].get("logprobs") or {}
    return logprobs.get("top_logprobs") or []


def _response_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_tokens = _choice_tokens(left)
    right_tokens = _choice_tokens(right)
    left_logprobs = _choice_logprobs(left)
    right_logprobs = _choice_logprobs(right)
    right_top_logprobs = _choice_top_logprobs(right)
    deltas: list[float] = []
    violations: list[dict[str, Any]] = []
    first_divergence = None
    for index, (left_token, right_token) in enumerate(zip(left_tokens, right_tokens)):
        tolerance = PREFILL_LOGPROB_ATOL if index == 0 else DECODE_LOGPROB_ATOL
        left_value = left_logprobs[index] if index < len(left_logprobs) else None
        if left_token != right_token:
            first_divergence = index
            candidate_value = right_top_logprobs[index].get(left_token) if index < len(right_top_logprobs) else None
            if index == 0:
                violations.append({"token_index": index, "kind": "prefill_token_mismatch"})
            elif left_value is None or candidate_value is None:
                violations.append({"token_index": index, "kind": "missing_cross_lookup_logprob"})
            else:
                delta = abs(left_value - candidate_value)
                deltas.append(delta)
                if delta > tolerance:
                    violations.append({"token_index": index, "kind": "decode_distribution_delta", "delta": delta})
            break
        right_value = right_logprobs[index] if index < len(right_logprobs) else None
        if left_value is None or right_value is None:
            violations.append({"token_index": index, "kind": "missing_selected_token_logprob"})
            continue
        delta = abs(left_value - right_value)
        deltas.append(delta)
        if delta > tolerance:
            violations.append({"token_index": index, "kind": "logprob_delta", "delta": delta})
    if len(left_tokens) != len(right_tokens):
        violations.append({"kind": "token_count_mismatch"})
    return {
        "text_equal": left["choices"][0]["text"] == right["choices"][0]["text"],
        "token_ids_equal": left_tokens == right_tokens,
        "first_divergence": first_divergence,
        "logprob_deltas": deltas,
        "max_abs_delta": max(deltas, default=0.0),
        "violations": violations,
    }


def _sampled_match_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    hits = total = 0
    for left_pair, right_pair in zip(left["sampled"], right["sampled"]):
        left_tokens = _choice_tokens(left_pair[0])
        right_tokens = _choice_tokens(right_pair[0])
        hits += sum(1 for a, b in zip(left_tokens, right_tokens) if a == b)
        total += max(len(left_tokens), len(right_tokens))
    return hits / total if total else 0.0


# ---------------------------------------------------------------- API contract
def _post_chat(server: RemoteOpenAIServer, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    response = requests.post(
        server.url_for("v1", "chat", "completions"),
        json={"model": MODEL, "seed": SEED, **payload},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text[:500]}
    return response.status_code, body


def _contract_run(server: RemoteOpenAIServer) -> dict[str, Any]:
    """Model-specific output parsing + API boundary cases.

    The invariant is that no Kimi control marker may survive into `content`:
    if the template emitted one, the server-side parser has to turn it into a
    structured field (`tool_calls` / `reasoning_content`).
    """
    checks: dict[str, Any] = {}

    # 1. tool call: the model must either call the tool with parsed arguments,
    #    or answer in plain text - but never leak the raw section markers.
    status, body = _post_chat(
        server,
        {
            "messages": [{"role": "user", "content": "What is the weather in Beijing right now? Use the tool."}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 256,
        },
    )
    message = (body.get("choices") or [{}])[0].get("message") or {}
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    leaked = [marker for marker in _KIMI_CONTROL_MARKERS if marker in content]
    parsed_arguments = True
    for call in tool_calls:
        arguments = (call.get("function") or {}).get("arguments")
        try:
            json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            parsed_arguments = False
    checks["tool_call"] = {
        "status": status,
        "tool_calls": len(tool_calls),
        "function_names": [(call.get("function") or {}).get("name") for call in tool_calls],
        "arguments_json_valid": parsed_arguments,
        "leaked_markers": leaked,
        "ok": status == 200 and not leaked and parsed_arguments,
    }

    # 2. thinking / reasoning channel separation.
    status, body = _post_chat(
        server,
        {
            "messages": [{"role": "user", "content": "用两步说明 17 乘 23 怎么算。"}],
            "max_tokens": 512,
            "temperature": 0.0,
        },
    )
    message = (body.get("choices") or [{}])[0].get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    leaked = [marker for marker in _KIMI_CONTROL_MARKERS if marker in content]
    checks["thinking"] = {
        "status": status,
        "has_reasoning_content": bool(reasoning),
        "leaked_markers": leaked,
        "content_head": content[:120],
        "ok": status == 200 and not leaked,
    }

    # 3. boundaries: a single-token request, a stop sequence, n=2, an
    #    over-long prompt and an empty prompt.  Only 5xx counts as failure.
    status, body = _post_chat(server, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
    choices = body.get("choices") or []
    checks["max_tokens_1"] = {
        "status": status,
        "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
        "ok": status == 200 and len(choices) == 1,
    }

    status, body = _post_chat(
        server,
        {"messages": [{"role": "user", "content": "List three colors, one per line."}],
         "max_tokens": 64, "stop": ["\n"]},
    )
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    checks["stop_sequence"] = {"status": status, "contains_stop": "\n" in text,
                               "ok": status == 200 and "\n" not in text}

    status, body = _post_chat(server, {"messages": [{"role": "user", "content": "hi"}],
                                       "max_tokens": 4, "n": 2})
    checks["n_equals_2"] = {"status": status, "choices": len(body.get("choices") or []),
                            "ok": status < 500}

    over_long = "alpha beta gamma delta " * ((MAX_MODEL_LEN + 100) // 4)
    status, body = _post_chat(server, {"messages": [{"role": "user", "content": over_long}], "max_tokens": 4})
    checks["prompt_too_long"] = {"status": status, "ok": status == 400}

    status, body = _completion_status(server, "")
    checks["empty_prompt"] = {"status": status, "ok": status < 500}

    checks["ok"] = all(item.get("ok", False) for item in checks.values() if isinstance(item, dict))
    return checks


def _completion_status(server: RemoteOpenAIServer, prompt: str) -> tuple[int, dict[str, Any]]:
    response = requests.post(
        server.url_for("v1", "completions"),
        json={"model": MODEL, "prompt": prompt, "max_tokens": 4, "temperature": 0.0},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"raw": response.text[:200]}


# ----------------------------------------------------------------- performance
def _prompt_for_seq_len(seq_len: int, request_index: int) -> str:
    repetitions = (seq_len + 3) // 4
    return "alpha beta gamma delta " * repetitions + f"\nRequest {request_index}: summarize the pattern."


def _streaming_completion(
    server: RemoteOpenAIServer, prompt: str, *, case: dict[str, int], repeat: int
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at = None
    text_parts = []
    usage = None
    chunk_count = 0
    with requests.post(
        server.url_for("v1", "completions"),
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": PERFORMANCE_MAX_TOKENS,
            "temperature": 0.0,
            "seed": SEED,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        stream=True,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = raw_line[len("data: "):]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            chunk_count += 1
            for choice in event.get("choices") or []:
                text = choice.get("text") or ""
                if text and first_token_at is None:
                    first_token_at = time.perf_counter()
                text_parts.append(text)
            if event.get("usage") is not None:
                usage = event["usage"]
    ended = time.perf_counter()
    if first_token_at is None or usage is None:
        raise RuntimeError("Streaming response did not contain generated text and final usage")
    ttft = first_token_at - started
    decode_time = ended - first_token_at
    decode_tokens = max(usage["completion_tokens"] - 1, 0)
    return {
        "case": case,
        "repeat": repeat,
        "started_s": started,
        "ended_s": ended,
        "ttft_s": ttft,
        "tpot_s": decode_time / decode_tokens if decode_tokens else None,
        "end_to_end_s": ended - started,
        "prefill_tokens_per_s": usage["prompt_tokens"] / ttft,
        "decode_tokens_per_s": decode_tokens / decode_time if decode_tokens and decode_time else None,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "chunk_count": chunk_count,
    }


def _spec_metrics(server: RemoteOpenAIServer) -> dict[str, float]:
    try:
        text = requests.get(server.url_for("metrics"), timeout=(10, 120)).text
    except requests.RequestException:
        return {"num_drafts": 0.0, "num_accepted_tokens": 0.0}
    totals = {"num_drafts": 0.0, "num_accepted_tokens": 0.0}
    wanted = {
        "vllm:spec_decode_num_drafts": "num_drafts",
        "vllm:spec_decode_num_drafts_total": "num_drafts",
        "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "num_accepted_tokens",
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line.strip())
        if not match:
            continue
        key = wanted.get(match.group(1))
        if key:
            totals[key] += float(match.group(2))
    return totals


def _performance_run(server: RemoteOpenAIServer) -> list[dict[str, Any]]:
    measurements = []
    for seq_len in PERFORMANCE_SEQ_LENGTHS:
        for batch_size in PERFORMANCE_BATCH_SIZES:
            case = {"batch_size": batch_size, "requested_seq_len": seq_len}
            for repeat in range(-PERFORMANCE_WARMUP_REPEATS, PERFORMANCE_REPEATS):
                before = _spec_metrics(server)
                prompts = [_prompt_for_seq_len(seq_len, index) for index in range(batch_size)]
                with ThreadPoolExecutor(max_workers=batch_size) as pool:
                    batch = list(
                        pool.map(
                            lambda prompt, case=case, repeat=repeat: _streaming_completion(
                                server, prompt, case=case, repeat=repeat
                            ),
                            prompts,
                        )
                    )
                after = _spec_metrics(server)
                if repeat < 0:
                    continue
                batch_elapsed = max(item["ended_s"] for item in batch) - min(item["started_s"] for item in batch)
                for item in batch:
                    item["batch_elapsed_s"] = batch_elapsed
                    item["batch_output_tokens_per_s"] = (
                        sum(other["completion_tokens"] for other in batch) / batch_elapsed
                    )
                    drafts = after["num_drafts"] - before["num_drafts"]
                    accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]
                    item["spec_decode"] = {
                        "num_drafts": drafts,
                        "num_accepted_tokens": accepted,
                        "accepted_per_draft": (accepted / drafts) if drafts else None,
                    }
                measurements.extend(batch)
    return measurements


def _median(values):
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def _performance_summary(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    cases = {(item["case"]["batch_size"], item["case"]["requested_seq_len"]) for item in measurements}
    for batch_size, seq_len in sorted(cases):
        group = [
            item for item in measurements if item["case"] == {"batch_size": batch_size, "requested_seq_len": seq_len}
        ]
        summary.append(
            {
                "case": {"batch_size": batch_size, "requested_seq_len": seq_len},
                "ttft_s": _median([item["ttft_s"] for item in group]),
                "tpot_s": _median([item["tpot_s"] for item in group]),
                "end_to_end_s": _median([item["end_to_end_s"] for item in group]),
                "prefill_tokens_per_s": _median([item["prefill_tokens_per_s"] for item in group]),
                "decode_tokens_per_s": _median([item["decode_tokens_per_s"] for item in group]),
                "batch_output_tokens_per_s": _median([item["batch_output_tokens_per_s"] for item in group]),
                "prompt_tokens": _median([float(item["prompt_tokens"]) for item in group]),
            }
        )
    return summary


def _performance_regressions(mrv1: list[dict[str, Any]], mrv2: list[dict[str, Any]]) -> list[str]:
    regressions = []
    lower_is_better = ("ttft_s", "tpot_s", "end_to_end_s")
    higher_is_better = ("prefill_tokens_per_s", "decode_tokens_per_s", "batch_output_tokens_per_s")
    for baseline, candidate in zip(mrv1, mrv2):
        assert baseline["case"] == candidate["case"]
        for metric in lower_is_better + higher_is_better:
            baseline_value = baseline[metric]
            candidate_value = candidate[metric]
            if baseline_value is None or candidate_value is None:
                continue
            if metric in lower_is_better:
                allowed = baseline_value * (1 + PERFORMANCE_TOLERANCE)
                failed = candidate_value > allowed
            else:
                allowed = baseline_value * (1 - PERFORMANCE_TOLERANCE)
                failed = candidate_value < allowed
            if failed:
                regressions.append(
                    f"{candidate['case']} {metric}: MRV1={baseline_value:.6g}, "
                    f"MRV2={candidate_value:.6g}, allowed={allowed:.6g}"
                )
    return regressions


def _write_results(results: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.e2e_model("Kimi-K2.6-W4A8")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="spec_decode,logprobs,long_sequence",
    parallel="DP,TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free(target_free_percentage=0.8, max_wait_seconds=900)
def test_kimi_k26_mrv1_mrv2_parity_and_performance():
    results: dict[str, Any] = {
        "model": MODEL,
        "model_revision": MODEL_REVISION or "unpinned",
        "draft_model": DRAFT_MODEL,
        "software": _software_versions(),
        "hardware": {"name": "Atlas 800T A3", "npu_count": 16, "hbm_per_npu_gb": 64},
        "parallel_config": {
            "tensor_parallel_size": TP,
            "data_parallel_size": DP,
            "enable_expert_parallel": True,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "gpu_memory_utilization": GPU_MEM_UTIL,
        },
        "speculative_config": {"method": "dflash", "model": DRAFT_MODEL,
                              "num_speculative_tokens": SPEC_TOKENS} if SPEC_TOKENS else None,
        "seed": SEED,
        "accuracy_prompts": PROMPTS,
        "tolerances": {"prefill_logprob_atol": PREFILL_LOGPROB_ATOL,
                       "decode_logprob_atol": DECODE_LOGPROB_ATOL, "top_logprobs": TOP_LOGPROBS},
        "runners": {},
    }
    _write_results(results)

    for runner_name, runner_flag in (("mrv1", "0"), ("mrv2", "1")):
        port = get_open_port()
        serve_args = _serve_args(port)
        arm: dict[str, Any] = {
            "serve_command": shlex.join(["vllm", "serve", MODEL, *serve_args, "--seed", str(SEED)]),
            "environment": _server_environment(runner_flag),
        }
        results["runners"][runner_name] = arm
        memory_monitor = _NpuMemoryMonitor()
        try:
            with (
                memory_monitor,
                RemoteOpenAIServer(
                    MODEL,
                    serve_args,
                    server_host="127.0.0.1",
                    server_port=port,
                    auto_port=False,
                    seed=SEED,
                    env_dict=_server_environment(runner_flag),
                    max_wait_seconds=3600,
                ) as server,
            ):
                arm["accuracy"] = _accuracy_run(server)
                arm["contract"] = _contract_run(server)
                arm["performance_raw"] = _performance_run(server)
                arm["performance_summary"] = _performance_summary(arm["performance_raw"])
        finally:
            cleanup_dist_env_and_memory()
            arm["hbm"] = {
                "peak_total_mb": memory_monitor.peak_total_mb,
                "peak_per_npu_mb": memory_monitor.peak_per_npu_mb,
                "sampling_error": memory_monitor.sampling_error,
            }
            _write_results(results)
        time.sleep(RUNNER_CLEANUP_SECONDS)

    mrv1 = results["runners"]["mrv1"]
    mrv2 = results["runners"]["mrv2"]
    comparisons = [
        _response_comparison(left, right)
        for left, right in zip(mrv1["accuracy"]["greedy"], mrv2["accuracy"]["greedy"])
    ]
    deltas = [delta for item in comparisons for delta in item["logprob_deltas"]]
    results["accuracy_comparison"] = {
        "greedy_pairs": comparisons,
        "greedy_logprob_max_abs_delta": max(deltas, default=0.0),
        "greedy_logprob_mean_abs_delta": statistics.mean(deltas) if deltas else 0.0,
        "greedy_tokens_equal": all(item["token_ids_equal"] for item in comparisons),
        "greedy_tolerance_pass": all(not item["violations"] for item in comparisons),
        "sampled_token_match_ratio": _sampled_match_ratio(mrv1["accuracy"], mrv2["accuracy"]),
        "sampled_note": (
            "MRV1 and MRV2 do not share an RNG stream; sampled runs are compared by token "
            "match ratio only. Per-request seeds were measured not to reproduce bit-exactly "
            "on this stack, so no sampled equality assertion is made."
        ),
    }
    results["contract"] = {
        "mrv1": mrv1["contract"],
        "mrv2": mrv2["contract"],
        "both_ok": bool(mrv1["contract"].get("ok")) and bool(mrv2["contract"].get("ok")),
    }
    results["performance_regressions"] = _performance_regressions(
        mrv1["performance_summary"], mrv2["performance_summary"]
    )
    results["performance_regressions_enforced"] = STRICT_PERF
    _write_results(results)

    for left, right in zip(mrv1["accuracy"]["greedy"], mrv2["accuracy"]["greedy"]):
        assert left["choices"][0]["text"], "MRV1 returned an empty greedy response"
        assert right["choices"][0]["text"], "MRV2 returned an empty greedy response"
    assert deltas, "No comparable greedy logprobs were returned"
    assert results["accuracy_comparison"]["greedy_tolerance_pass"], "greedy logprob parity failed"
    assert results["contract"]["both_ok"], "Kimi API contract checks failed"
    for runner_name in ("mrv1", "mrv2"):
        hbm = results["runners"][runner_name]["hbm"]
        assert hbm["peak_total_mb"] > 0 and hbm["sampling_error"] is None, f"{runner_name} HBM sampling failed"
    if STRICT_PERF:
        assert not results["performance_regressions"], "\n".join(results["performance_regressions"])
