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
"""MiniMax-M2.7 4-card e2e: 16-layer reduced model, TP2+DP2+EP, V1 vs V2.

The scenario is validated on the internal 90-net A3 machine (2026-08-06):

* 16k1k: 80 requests, 20 concurrent, 0% prefix hit (streaming, low latency).
* 128k1k: 32 requests, 8 concurrent, ~90% shared prefix (high throughput).

The full MiniMax-M2.7-w8a8-QuaRot checkpoint (62 layers) does not fit on
4x64GB A3 cards, so the model is loaded with a 16-layer config via
``hf_overrides``. ``vllm_ascend/patch/worker/patch_minimax_m2.py`` skips the
surplus ``layers.{16..61}`` weights during loading.

Both scenarios run ModelRunner V1 and V2 on the same machine and assert
that V2 output throughput is not worse than V1 by more than 3%
(``V2 >= V1 * 0.97``).

Benchmarks use vLLM's built-in ``vllm bench serve`` CLI with its synthetic
datasets (``random`` for 16k1k, ``prefix_repetition`` for 128k1k), so no
external dataset publication is required. (Nightly single-node cases use
aisbench instead; this PR E2E case follows the PR E2E toolchain, whose only
performance precedent is ``tools/vllm_bench.py``.)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import (
    RemotePDServer,
    cleanup_dist_env_and_memory,
    wait_until_npu_memory_free,
)
from vllm.utils.network_utils import get_open_port

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


def _run_server_and_bench(use_v2: bool, bench_args: list[str]) -> dict[str, Any]:
    port = get_open_port()
    env_dict = {
        **SERVER_ENV,
        "VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0",
    }
    with RemotePDServer(_server_args(port), env_dict=env_dict, max_wait_seconds=1800) as server:
        result = _run_bench(server.port, bench_args)
    return result


def _assert_v2_not_slower(v1: dict[str, Any], v2: dict[str, Any], case: str) -> None:
    assert v1["failed"] == 0, f"[{case}] V1 had {v1['failed']} failed request(s)"
    assert v2["failed"] == 0, f"[{case}] V2 had {v2['failed']} failed request(s)"

    v1_throughput = float(v1["output_throughput"])
    v2_throughput = float(v2["output_throughput"])
    print(f"[{case}] V1: output_throughput={v1_throughput:.2f} tok/s")
    print(f"[{case}] V2: output_throughput={v2_throughput:.2f} tok/s")
    assert v2_throughput >= v1_throughput * THROUGHPUT_THRESHOLD, (
        f"[{case}] V2 output throughput {v2_throughput:.2f} tok/s is below "
        f"V1 * {THROUGHPUT_THRESHOLD} = {v1_throughput * THROUGHPUT_THRESHOLD:.2f} tok/s"
    )


def _benchmark_pair(bench_args: list[str], case: str) -> dict[str, Any]:
    """Run the same scenario on V1 then V2 and assert V2 >= V1 * 0.97."""
    v1 = _run_server_and_bench(use_v2=False, bench_args=bench_args)
    cleanup_dist_env_and_memory()
    time.sleep(10)  # allow the previous NPU processes to fully release memory
    v2 = _run_server_and_bench(use_v2=True, bench_args=bench_args)
    _assert_v2_not_slower(v1, v2, case)
    return v2


@pytest.mark.e2e_model(MINIMAX_M2_7_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="long_sequence",
    parallel="TP,EP,DP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_minimax_m2_7_16k1k_v2_vs_v1() -> None:
    """16k1k: 80 requests, 20 concurrent, 0% prefix hit, V2 >= V1 * 0.97."""
    _benchmark_pair(bench_args=BENCH_16K_ARGS, case="16k1k")


@pytest.mark.e2e_model(MINIMAX_M2_7_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="long_sequence,prefix_caching",
    parallel="TP,EP,DP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_minimax_m2_7_128k1k_v2_vs_v1() -> None:
    """128k1k: 32 requests, 8 concurrent, ~90% shared prefix, V2 >= V1 * 0.97."""
    _benchmark_pair(bench_args=BENCH_128K_ARGS, case="128k1k")
