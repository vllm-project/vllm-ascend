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
"""Shared helpers for the Qwen3-32B V1 vs V2 PR performance guards (LC/LL).

Both guards run the same scenario once with the V1 model runner
(``VLLM_USE_V2_MODEL_RUNNER=0``) and once with the V2 model runner
(``VLLM_USE_V2_MODEL_RUNNER=1``) and compare the metrics inside the test.

The benchmarks use vLLM's built-in ``vllm bench serve`` CLI with its
synthetic datasets (``random`` for the short-context LL scenario and
``prefix_repetition`` for the long-context LC scenario), so no external
aisbench/benchmark configuration is required. This follows the PR E2E
toolchain used by the existing PR performance cases (``tools/vllm_bench.py``
and the MiniMax-M2.7 four-card case); aisbench resources are only available
in nightly/weekly workflows.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

QWEN3_32B_W8A8_MODEL = os.environ.get("QWEN3_32B_W8A8_MODEL_PATH", "vllm-ascend/Qwen3-32B-W8A8")
QWEN3_32B_BF16_MODEL = os.environ.get("QWEN3_32B_MODEL_PATH", "Qwen/Qwen3-32B")

# LC: V2/V1 total token throughput ratio must stay within +-3%.
LC_THROUGHPUT_RATIO_LOWER = 0.97
LC_THROUGHPUT_RATIO_UPPER = 1.03

# LL: V2 TTFT/TPOT must be at most 3% worse than V1.
LL_LATENCY_REGRESSION_RATIO = 1.03

_COMMON_ENV = {
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

_LC_ENV = {**_COMMON_ENV, "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}

_LC_YARN_HF_OVERRIDES = {
    "rope_parameters": {
        "rope_type": "yarn",
        "rope_theta": 1000000,
        "factor": 4,
        "original_max_position_embeddings": 131072,
    }
}

# LC: 128k input / 1k output, ~90% shared prefix (16 prompts, concurrency 4).
BENCH_LC_ARGS = [
    "--dataset-name",
    "prefix_repetition",
    "--num-prompts",
    "16",
    "--max-concurrency",
    "4",
    "--prefix-repetition-prefix-len",
    "117965",
    "--prefix-repetition-suffix-len",
    "13107",
    "--prefix-repetition-num-prefixes",
    "1",
    "--prefix-repetition-output-len",
    "1024",
]

# LL: 16k input / 1k output, 0% shared prefix (50 prompts, concurrency 2).
BENCH_LL_ARGS = [
    "--dataset-name",
    "random",
    "--num-prompts",
    "50",
    "--max-concurrency",
    "2",
    "--random-input-len",
    "16384",
    "--random-output-len",
    "1024",
]


def _lc_server_args(model: str) -> list[str]:
    return [
        "--served-model-name",
        model,
        "--trust-remote-code",
        "--seed",
        "1024",
        "--max-model-len",
        "135000",
        "--max-num-batched-tokens",
        "40960",
        "--tensor-parallel-size",
        "4",
        "--distributed-executor-backend",
        "mp",
        "--enable-prefix-caching",
        "--async-scheduling",
        "--compilation-config",
        '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        "--hf-overrides",
        json.dumps(_LC_YARN_HF_OVERRIDES),
        "--gpu-memory-utilization",
        "0.9",
        "--quantization",
        "ascend",
    ]


def _ll_server_args(model: str) -> list[str]:
    return [
        "--served-model-name",
        model,
        "--trust-remote-code",
        "--max-model-len",
        "18000",
        "--max-num-batched-tokens",
        "40960",
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "2",
        "--data-parallel-start-rank",
        "0",
        "--distributed-executor-backend",
        "mp",
        "--async-scheduling",
        "--no-enable-prefix-caching",
        "--compilation-config",
        '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        "--gpu-memory-utilization",
        "0.9",
    ]


def _bench_common_args(model: str) -> list[str]:
    return [
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--served-model-name",
        model,
        "--model",
        model,
        "--tokenizer",
        model,
        "--metric-percentiles",
        "50,90,99",
        "--request-rate",
        "inf",
        "--num-warmups",
        "5",
        "--temperature",
        "0",
        "--ignore-eos",
        "--seed",
        "0",
        "--disable-tqdm",
        "--save-result",
        "--save-detailed",
        "--trust-remote-code",
    ]


def _run_bench(port: int, model: str, bench_args: list[str]) -> dict[str, Any]:
    """Run ``vllm bench serve`` against the already-started server and return
    the parsed result JSON."""
    with tempfile.TemporaryDirectory() as result_dir:
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--result-filename",
            "result.json",
            "--result-dir",
            result_dir,
            *_bench_common_args(model),
            *bench_args,
        ]
        print(f"Running vllm bench: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"vllm bench serve failed (rc={proc.returncode}):\n"
                f"stdout tail:\n{proc.stdout[-4000:]}\n"
                f"stderr tail:\n{proc.stderr[-4000:]}"
            )
        result_file = Path(result_dir) / "result.json"
        with result_file.open(encoding="utf-8") as f:
            return json.load(f)


# Wait for the NPU driver to reclaim memory after the previous server exits,
# so the next V1/V2 server does not OOM on a busy device.
@wait_until_npu_memory_free()
def _run_one_side(
    model: str,
    server_args_builder: Callable[[str], list[str]],
    env: dict[str, str],
    bench_args: list[str],
    use_v2: bool,
    case: str,
) -> dict[str, Any]:
    """Start one server (V1 or V2) and run the benchmark once on it."""
    port = get_open_port()
    env_dict = {**env, "VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0"}
    runner = "V2" if use_v2 else "V1"
    with RemoteOpenAIServer(
        model,
        server_args_builder(model) + ["--port", str(port)],
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
    ):
        result = _run_bench(port, model, bench_args)
    failed = int(result.get("failed", 0))
    assert failed == 0, f"[{case}] {runner} had {failed} failed request(s)"
    return result


def _benchmark_pair(
    model: str,
    server_args_builder: Callable[[str], list[str]],
    env: dict[str, str],
    bench_args: list[str],
    case: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the same scenario on V1 then V2 and return both result dicts."""
    v1_result = _run_one_side(model, server_args_builder, env, bench_args, use_v2=False, case=case)
    v2_result = _run_one_side(model, server_args_builder, env, bench_args, use_v2=True, case=case)
    return v1_result, v2_result


def _total_token_throughput(result: dict[str, Any], case: str, label: str) -> float:
    value = float(result["total_token_throughput"])
    print(f"[{case}] {label} total token throughput: {value:.2f} tok/s")
    return value


def _mean_ttft_ms(result: dict[str, Any], case: str, label: str) -> float:
    ttfts = [float(v) for v in result["ttfts"]]
    value = sum(ttfts) / len(ttfts) * 1000.0
    print(f"[{case}] {label} TTFT mean: {value:.2f} ms")
    return value


def _mean_tpot_ms(result: dict[str, Any], case: str, label: str) -> float:
    # ``itls`` is a list of per-request inter-token latency values; each value
    # is a list (per-token latencies of that request) in current vLLM versions,
    # so average each request's list first, then average across requests.
    tpots = []
    for itl in result["itls"]:
        if isinstance(itl, (list, tuple)):
            values = [float(v) for v in itl]
            if not values:
                continue
            tpots.append(sum(values) / len(values))
        else:
            tpots.append(float(itl))
    value = sum(tpots) / len(tpots) * 1000.0
    print(f"[{case}] {label} TPOT mean: {value:.2f} ms")
    return value


def _assert_lc_throughput_ratio(
    v1_result: dict[str, Any],
    v2_result: dict[str, Any],
    case: str = "LC",
) -> None:
    """LC: V2/V1 total token throughput ratio must stay within +-3%."""
    v1_throughput = _total_token_throughput(v1_result, case, "V1")
    v2_throughput = _total_token_throughput(v2_result, case, "V2")
    ratio = v2_throughput / v1_throughput
    print(f"[{case}] V2/V1 total token throughput ratio: {ratio:.4f}")
    assert LC_THROUGHPUT_RATIO_LOWER <= ratio <= LC_THROUGHPUT_RATIO_UPPER, (
        f"[{case}] LC 128k/1k performance regression: V2/V1 total token "
        f"throughput ratio {ratio:.4f} (V1={v1_throughput:.2f}, "
        f"V2={v2_throughput:.2f}) is outside "
        f"[{LC_THROUGHPUT_RATIO_LOWER}, {LC_THROUGHPUT_RATIO_UPPER}]."
    )


def _assert_ll_latency(
    v1_result: dict[str, Any],
    v2_result: dict[str, Any],
    case: str = "LL",
) -> None:
    """LL: V2 TTFT/TPOT must be at most 3% worse than V1."""
    v1_ttft = _mean_ttft_ms(v1_result, case, "V1")
    v2_ttft = _mean_ttft_ms(v2_result, case, "V2")
    v1_tpot = _mean_tpot_ms(v1_result, case, "V1")
    v2_tpot = _mean_tpot_ms(v2_result, case, "V2")
    for metric, v1_value, v2_value in (("TTFT", v1_ttft, v2_ttft), ("TPOT", v1_tpot, v2_tpot)):
        print(f"[{case}] {metric}: V1={v1_value:.2f} ms, V2={v2_value:.2f} ms")
        assert v2_value <= v1_value * LL_LATENCY_REGRESSION_RATIO, (
            f"[{case}] LL 16k/1k latency regression: V2 {metric} {v2_value:.2f} ms "
            f"exceeds V1 {v1_value:.2f} ms * {LL_LATENCY_REGRESSION_RATIO}."
        )
