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

"""Compare DeepSeek V3.2 on MRV1 and MRV2 using one 16-card A3 node.

Run with an immutable model revision for acceptance testing:

    DSV32_MODEL=/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8 \
    DSV32_MODEL_REVISION=<commit> CANN_VERSION=<version> \
    pytest -sv tests/e2e/nightly/single_node/model_runner_v2/test_deepseek_v3_2_mrv1_v2_parity.py

The JSON artifact retains raw accuracy responses and per-request performance
measurements. Sampled outputs are checked for reproducibility within each
runner, not equality across runners: MRV1 and MRV2 do not share an RNG stream.
"""

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
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, cleanup_dist_env_and_memory, wait_until_npu_memory_free

MODEL = os.getenv("DSV32_MODEL", "vllm-ascend/DeepSeek-V3.2-W8A8")
MODEL_REVISION = os.getenv("DSV32_MODEL_REVISION")
RESULT_PATH = Path(os.getenv("DSV32_MRV_RESULT", "/tmp/deepseek_v3_2_mrv1_v2_parity.json"))
SEED = 1234
ACCURACY_MAX_TOKENS = 32
PERFORMANCE_MAX_TOKENS = 32
MAX_MODEL_LEN = 65536
PERFORMANCE_REPEATS = int(os.getenv("DSV32_MRV_PERFORMANCE_REPEATS", "3"))
PERFORMANCE_WARMUP_REPEATS = int(os.getenv("DSV32_MRV_PERFORMANCE_WARMUPS", "1"))
PERFORMANCE_BATCH_SIZES = tuple(int(value) for value in os.getenv("DSV32_MRV_BATCH_SIZES", "1,4").split(","))
PERFORMANCE_SEQ_LENGTHS = tuple(int(value) for value in os.getenv("DSV32_MRV_SEQ_LENGTHS", "128,2048,32768").split(","))
PERFORMANCE_TOLERANCE = 0.10
LOGPROB_TOLERANCE = 1e-4
REQUEST_TIMEOUT_SECONDS = (30, 1800)
HBM_SAMPLE_INTERVAL_SECONDS = 0.5
RUNNER_CLEANUP_SECONDS = 30

PROMPTS = [
    "The capital of France is",
    "用三句话解释什么是张量并行。",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Summarize this English passage: " + "the quick brown fox jumps over the lazy dog " * 160,
    "请总结下面这段重复文本的主题：" + "人工智能需要可靠的软硬件协同。" * 240,
]

_HBM_USAGE_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_NPU_ROW_RE = re.compile(r"^\|\s*(\d+)\s+(\d+)\s+\|")


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
            self._stop.wait(HBM_SAMPLE_INTERVAL_SECONDS)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join(timeout=2)


def _serve_args(port: int) -> list[str]:
    args = [
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--data-parallel-size",
        "2",
        "--tensor-parallel-size",
        "8",
        "--quantization",
        "ascend",
        "--enable-expert-parallel",
        "--max-num-seqs",
        "16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        "4096",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.92",
        "--compilation-config",
        json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"}),
        "--speculative-config",
        json.dumps({"num_speculative_tokens": 3, "method": "deepseek_mtp"}),
    ]
    if MODEL_REVISION:
        args.extend(["--revision", MODEL_REVISION])
    return args


def _server_environment(model_runner: str) -> dict[str, str]:
    return {
        "VLLM_USE_V2_MODEL_RUNNER": model_runner,
        "VLLM_USE_V1": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "200",
        "VLLM_ASCEND_ENABLE_MLAPO": "1",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "10",
    }


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
            "logprobs": 5,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _accuracy_run(server: RemoteOpenAIServer) -> dict[str, Any]:
    greedy = [_completion(server, prompt, temperature=0.0) for prompt in PROMPTS]
    sampled = [[_completion(server, prompt, temperature=0.8) for _ in range(2)] for prompt in PROMPTS]
    return {"greedy": greedy, "sampled": sampled}


def _choice_tokens(response: dict[str, Any]) -> list[str]:
    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return logprobs.get("tokens") or [choice["text"]]


def _choice_logprobs(response: dict[str, Any]) -> list[float | None]:
    return response["choices"][0]["logprobs"]["token_logprobs"]


def _greedy_logprob_deltas(mrv1: dict[str, Any], mrv2: dict[str, Any]) -> list[float]:
    deltas = []
    for left, right in zip(mrv1["greedy"], mrv2["greedy"], strict=True):
        if _choice_tokens(left) != _choice_tokens(right):
            continue
        for left_value, right_value in zip(_choice_logprobs(left), _choice_logprobs(right), strict=True):
            if left_value is not None and right_value is not None:
                deltas.append(abs(left_value - right_value))
    return deltas


def _sampled_reproducibility(accuracy: dict[str, Any]) -> bool:
    return all(
        _choice_tokens(repeat) == _choice_tokens(first) and repeat["choices"][0]["text"] == first["choices"][0]["text"]
        for first, repeat in accuracy["sampled"]
    )


def _sampled_match_ratio(mrv1: dict[str, Any], mrv2: dict[str, Any]) -> float:
    matches = 0
    total = 0
    for left_pair, right_pair in zip(mrv1["sampled"], mrv2["sampled"], strict=True):
        left_tokens = _choice_tokens(left_pair[0])
        right_tokens = _choice_tokens(right_pair[0])
        matches += sum(left == right for left, right in zip(left_tokens, right_tokens))
        total += max(len(left_tokens), len(right_tokens))
    return matches / total if total else 0.0


def _prompt_for_seq_len(seq_len: int, request_index: int) -> str:
    repetitions = (seq_len + 3) // 4
    return "alpha beta gamma delta " * repetitions + f"\nRequest {request_index}: summarize the pattern."


def _streaming_completion(
    server: RemoteOpenAIServer, prompt: str, *, case: dict[str, int], repeat: int
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at = None
    ended = started
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
            payload = raw_line[len("data: ") :]
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
    return _latency_metrics(
        started=started,
        first_token_at=first_token_at,
        ended=ended,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        case=case,
        repeat=repeat,
        text="".join(text_parts),
        chunk_count=chunk_count,
    )


def _latency_metrics(
    *,
    started: float,
    first_token_at: float,
    ended: float,
    prompt_tokens: int,
    completion_tokens: int,
    case: dict[str, int],
    repeat: int,
    text: str,
    chunk_count: int,
) -> dict[str, Any]:
    ttft = first_token_at - started
    decode_time = ended - first_token_at
    decode_tokens = max(completion_tokens - 1, 0)
    return {
        "case": case,
        "repeat": repeat,
        "started_s": started,
        "ended_s": ended,
        "ttft_s": ttft,
        "tpot_s": decode_time / decode_tokens if decode_tokens else None,
        "end_to_end_s": ended - started,
        "prefill_tokens_per_s": prompt_tokens / ttft,
        "decode_tokens_per_s": decode_tokens / decode_time if decode_tokens and decode_time else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_time_s": decode_time,
        "text": text,
        "chunk_count": chunk_count,
    }


def _performance_run(server: RemoteOpenAIServer) -> list[dict[str, Any]]:
    measurements = []
    for seq_len in PERFORMANCE_SEQ_LENGTHS:
        for batch_size in PERFORMANCE_BATCH_SIZES:
            case = {"batch_size": batch_size, "requested_seq_len": seq_len}
            for repeat in range(-PERFORMANCE_WARMUP_REPEATS, PERFORMANCE_REPEATS):
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
                if repeat < 0:
                    continue
                batch_elapsed = max(item["ended_s"] for item in batch) - min(item["started_s"] for item in batch)
                for item in batch:
                    item["batch_elapsed_s"] = batch_elapsed
                    item["batch_output_tokens_per_s"] = (
                        sum(other["completion_tokens"] for other in batch) / batch_elapsed
                    )
                measurements.extend(batch)
    return measurements


def _median(values: list[float | None]) -> float | None:
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
            }
        )
    return summary


def _performance_regressions(mrv1: list[dict[str, Any]], mrv2: list[dict[str, Any]]) -> list[str]:
    regressions = []
    lower_is_better = ("ttft_s", "tpot_s", "end_to_end_s")
    higher_is_better = ("prefill_tokens_per_s", "decode_tokens_per_s", "batch_output_tokens_per_s")
    for baseline, candidate in zip(mrv1, mrv2, strict=True):
        assert baseline["case"] == candidate["case"]
        for metric in lower_is_better + higher_is_better:
            baseline_value = baseline[metric]
            candidate_value = candidate[metric]
            if baseline_value is None or candidate_value is None:
                regressions.append(f"{candidate['case']} {metric}: missing MRV1 or MRV2 measurement")
                continue
            if metric in lower_is_better:
                allowed = baseline_value * (1 + PERFORMANCE_TOLERANCE)
            else:
                allowed = baseline_value * (1 - PERFORMANCE_TOLERANCE)
            failed = candidate_value > allowed if metric in lower_is_better else candidate_value < allowed
            if failed:
                regressions.append(
                    f"{candidate['case']} {metric}: MRV1={baseline_value:.6g}, "
                    f"MRV2={candidate_value:.6g}, allowed={allowed:.6g}"
                )
    return regressions


def _software_versions() -> dict[str, str]:
    return {
        "vllm": version("vllm"),
        "vllm_ascend": version("vllm-ascend"),
        "torch": torch.__version__,
        "torch_npu": version("torch-npu"),
        "cann": os.getenv("CANN_VERSION", "unreported"),
        "platform": platform.platform(),
    }


def _write_results(results: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,logprobs,long_sequence",
    parallel="DP,TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free(target_free_percentage=0.8, max_wait_seconds=300)
def test_deepseek_v3_2_mrv1_mrv2_parity_and_performance():
    results: dict[str, Any] = {
        "model": MODEL,
        "model_revision": MODEL_REVISION or "unpinned",
        "software": _software_versions(),
        "hardware": {"name": "Atlas 800I A3", "npu_count": 16, "hbm_per_npu_gb": 64},
        "seed": SEED,
        "accuracy_prompts": PROMPTS,
        "runners": {},
    }
    _write_results(results)
    for runner_name, runner_flag in (("mrv1", "0"), ("mrv2", "1")):
        port = get_open_port()
        serve_args = _serve_args(port)
        results["runners"][runner_name] = {
            "serve_command": shlex.join(["vllm", "serve", MODEL, *serve_args, "--seed", str(SEED)]),
            "environment": _server_environment(runner_flag),
        }
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
                results["runners"][runner_name]["accuracy"] = _accuracy_run(server)
                results["runners"][runner_name]["performance_raw"] = _performance_run(server)
        finally:
            cleanup_dist_env_and_memory()
            results["runners"][runner_name]["hbm"] = {
                "peak_total_mb": memory_monitor.peak_total_mb,
                "peak_per_npu_mb": memory_monitor.peak_per_npu_mb,
                "sampling_error": memory_monitor.sampling_error,
            }
            _write_results(results)
        time.sleep(RUNNER_CLEANUP_SECONDS)

    mrv1 = results["runners"]["mrv1"]
    mrv2 = results["runners"]["mrv2"]
    greedy_logprob_deltas = _greedy_logprob_deltas(mrv1["accuracy"], mrv2["accuracy"])
    results["accuracy_comparison"] = {
        "greedy_logprob_max_abs_delta": max(greedy_logprob_deltas, default=0.0),
        "greedy_logprob_mean_abs_delta": (statistics.mean(greedy_logprob_deltas) if greedy_logprob_deltas else 0.0),
        "sampled_token_match_ratio": _sampled_match_ratio(mrv1["accuracy"], mrv2["accuracy"]),
        "mrv1_sampled_reproducible": _sampled_reproducibility(mrv1["accuracy"]),
        "mrv2_sampled_reproducible": _sampled_reproducibility(mrv2["accuracy"]),
    }
    for runner_name in ("mrv1", "mrv2"):
        results["runners"][runner_name]["performance_summary"] = _performance_summary(
            results["runners"][runner_name]["performance_raw"]
        )
    results["performance_regressions"] = _performance_regressions(
        mrv1["performance_summary"], mrv2["performance_summary"]
    )
    _write_results(results)

    for left, right in zip(mrv1["accuracy"]["greedy"], mrv2["accuracy"]["greedy"], strict=True):
        assert left["choices"][0]["text"], "MRV1 returned an empty greedy response"
        assert right["choices"][0]["text"], "MRV2 returned an empty greedy response"
        assert _choice_tokens(left) == _choice_tokens(right), "MRV1/MRV2 greedy tokens diverged"
        assert left["choices"][0]["text"] == right["choices"][0]["text"], "MRV1/MRV2 greedy text diverged"
    assert greedy_logprob_deltas, "No comparable greedy logprobs were returned"
    assert results["accuracy_comparison"]["greedy_logprob_max_abs_delta"] <= LOGPROB_TOLERANCE
    assert results["accuracy_comparison"]["mrv1_sampled_reproducible"]
    assert results["accuracy_comparison"]["mrv2_sampled_reproducible"]
    for runner_name in ("mrv1", "mrv2"):
        hbm = results["runners"][runner_name]["hbm"]
        assert hbm["peak_total_mb"] > 0 and hbm["sampling_error"] is None, f"{runner_name} HBM sampling failed"
    assert not results["performance_regressions"], "\n".join(results["performance_regressions"])
