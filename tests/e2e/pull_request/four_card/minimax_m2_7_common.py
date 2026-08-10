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
"""Shared helpers for the MiniMax-M2.7 4-card V1 vs V2 benchmark E2E cases.

The scenario is validated on the internal 90-net A3 machine (2026-08-06):

* 16k1k: 80 requests, 20 concurrent, 0% prefix hit (streaming, low latency).
* 128k1k: 32 requests, 8 concurrent, ~90% shared prefix (high throughput).

The full MiniMax-M2.7-w8a8-QuaRot checkpoint (62 layers) does not fit on
4x64GB A3 cards, so the model is loaded with a 16-layer config via
``hf_overrides``. ``vllm_ascend/patch/worker/patch_minimax_m2.py`` skips the
surplus ``layers.{16..61}`` weights during loading.

Both scenarios run ModelRunner V1 and V2 on the same machine and assert
that V2 mean output throughput is not worse than V1 by more than 3%
(``V2 >= V1 * 0.97``). Each side runs several benchmark rounds and the
first round is discarded; the assertion compares the mean of the remaining
rounds to reduce single-run throughput noise.

Benchmarks use vLLM's built-in ``vllm bench serve`` CLI with its synthetic
datasets (``random`` for 16k1k, ``prefix_repetition`` for 128k1k), so no
external dataset publication is required. Each round is preceded by 5
warm-up requests that are excluded from the metrics, and the first round
itself is discarded, mirroring the internal methodology of discarding the
first complete round. (Nightly single-node cases use aisbench instead;
this PR E2E case follows the PR E2E toolchain, whose only performance
precedent is ``tools/vllm_bench.py``.)

The 16k1k and 128k1k cases live in separate test files so each runs as its
own CI job; the 16k case runs 5 rounds per side and the 128k case 3 rounds.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemotePDServer, wait_npu_memory_free

# ModelScope repo published by the model owner; override locally with
# MINIMAX_M2_7_MODEL_PATH for private/internal weight paths.
MINIMAX_M2_7_MODEL = os.environ.get("MINIMAX_M2_7_MODEL_PATH", "vllm-ascend/MiniMax-M2.7-w8a8-QuaRot")

# 16-layer reduced config validated on the internal machine. The full
# checkpoint keeps `num_hidden_layers_orig=62` so the load_weights patch
# knows which layers to skip.
HF_OVERRIDES = {
    "num_hidden_layers": 16,
    "num_hidden_layers_orig": 62,
}

# V2 is considered not slower than V1 when V2 >= V1 * THROUGHPUT_THRESHOLD.
THROUGHPUT_THRESHOLD = 0.97

SERVER_ENV = {
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "PYTHONHASHSEED": "0",
}


# Server launch parameters identical to the validated internal setup.
def _server_args(port: int) -> list[str]:
    return [
        "--model",
        MINIMAX_M2_7_MODEL,
        "--served-model-name",
        MINIMAX_M2_7_MODEL,
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "2",
        "--enable-expert-parallel",
        "--max-model-len",
        "196608",
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.9",
        "--quantization",
        "ascend",
        "--trust-remote-code",
        "--enforce-eager",
        "--hf-overrides",
        json.dumps(HF_OVERRIDES),
        "--port",
        str(port),
    ]


BENCH_COMMON_ARGS = [
    "--backend",
    "openai-chat",
    "--endpoint",
    "/v1/chat/completions",
    "--served-model-name",
    MINIMAX_M2_7_MODEL,
    "--model",
    MINIMAX_M2_7_MODEL,
    "--tokenizer",
    MINIMAX_M2_7_MODEL,
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

BENCH_16K_ARGS = [
    "--dataset-name",
    "random",
    "--num-prompts",
    "80",
    "--max-concurrency",
    "20",
    "--random-input-len",
    "16410",
    "--random-output-len",
    "1024",
]

BENCH_128K_ARGS = [
    "--dataset-name",
    "prefix_repetition",
    "--num-prompts",
    "32",
    "--max-concurrency",
    "8",
    "--prefix-repetition-prefix-len",
    "117900",
    "--prefix-repetition-suffix-len",
    "13100",
    "--prefix-repetition-num-prefixes",
    "1",
    "--prefix-repetition-output-len",
    "1024",
]


def _run_bench(port: int, bench_args: list[str]) -> dict[str, Any]:
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
            *BENCH_COMMON_ARGS,
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


def _run_server_and_bench(
    use_v2: bool,
    bench_args: list[str],
    num_repeats: int,
) -> list[dict[str, Any]]:
    """Start one server and run the benchmark *num_repeats* times on it."""
    port = get_open_port()
    env_dict = {
        **SERVER_ENV,
        "VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0",
    }
    runner = "V2" if use_v2 else "V1"
    results: list[dict[str, Any]] = []
    with RemotePDServer(_server_args(port), env_dict=env_dict, max_wait_seconds=1800) as server:
        for round_index in range(1, num_repeats + 1):
            print(f"[{runner}] bench round {round_index}/{num_repeats}")
            results.append(_run_bench(server.port, bench_args))
    return results


def _mean_output_throughput(
    results: list[dict[str, Any]],
    case: str,
    label: str,
) -> float:
    """Return the mean output throughput over all rounds except the first."""
    for round_index, result in enumerate(results, 1):
        assert result["failed"] == 0, f"[{case}] {label} round {round_index} had {result['failed']} failed request(s)"
    kept = results[1:]
    values = [float(result["output_throughput"]) for result in kept]
    mean = sum(values) / len(values)
    print(f"[{case}] {label} output_throughput per kept round: {[f'{v:.2f}' for v in values]} tok/s")
    print(f"[{case}] {label} output_throughput mean: {mean:.2f} tok/s")
    return mean


def _assert_v2_not_slower(
    v1_results: list[dict[str, Any]],
    v2_results: list[dict[str, Any]],
    case: str,
) -> None:
    v1_throughput = _mean_output_throughput(v1_results, case, "V1")
    v2_throughput = _mean_output_throughput(v2_results, case, "V2")
    assert v2_throughput >= v1_throughput * THROUGHPUT_THRESHOLD, (
        f"[{case}] V2 mean output throughput {v2_throughput:.2f} tok/s is below "
        f"V1 mean * {THROUGHPUT_THRESHOLD} = {v1_throughput * THROUGHPUT_THRESHOLD:.2f} tok/s"
    )


def _benchmark_pair(
    bench_args: list[str],
    case: str,
    num_repeats: int,
) -> dict[str, Any]:
    """Run the same scenario on V1 then V2 and assert V2 >= V1 * 0.97.

    Each side runs *num_repeats* rounds on a single server; the first round
    is discarded and the assertion compares mean throughput.
    """
    v1_results = _run_server_and_bench(
        use_v2=False,
        bench_args=bench_args,
        num_repeats=num_repeats,
    )
    wait_npu_memory_free(max_wait_seconds=120)
    v2_results = _run_server_and_bench(
        use_v2=True,
        bench_args=bench_args,
        num_repeats=num_repeats,
    )
    _assert_v2_not_slower(v1_results, v2_results, case)
    return v2_results[-1]
