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
# This file is a part of the vllm-ascend project.

import json
from typing import Any

import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from tools.vllm_bench import run_vllm_bench_case

BF16_MODEL = "Qwen/Qwen3-30B-A3B"
W8A8_MODEL = "Eco-Tech/Qwen3-30B-A3B-w8a8"

# Baselines are the average of the successful vLLM v0.26.0 and pinned-main A2
# PR E2E measurements. Keep the repository-standard 3% throughput tolerance,
# and use the corresponding 3% regression allowance for TPOT.
THROUGHPUT_THRESHOLD = 0.97
PERFORMANCE_BASELINES = {
    BF16_MODEL: {
        "output_throughput": 845.8,
        "mean_tpot_ms": 17.5,
    },
    W8A8_MODEL: {
        "output_throughput": 801.2,
        "mean_tpot_ms": 19.0,
    },
}

SERVER_ENVS = {
    "OMP_PROC_BIND": "false",
    "OMP_NUM_THREADS": "1",
    "HCCL_BUFFSIZE": "512",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "NPU_MEMORY_FRACTION": "0.95",
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
    "VLLM_USE_V1": "1",
    "TASK_QUEUE_ENABLE": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "VLLM_ASCEND_ENABLE_NZ": "2",
    "LD_PRELOAD": "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2",
}

CUDAGRAPH_CAPTURE_SIZES = [
    1,
    2,
    4,
    8,
    16,
    32,
    48,
    64,
    96,
    128,
    160,
    192,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
]

VLLM_BENCH_CASE = {
    "dataset_name": "random",
    "num_prompts": 180,
    # vLLM Bench uses infinity for the all-at-once traffic mode represented by
    # request_rate=0 in the Nightly AISBench configuration.
    "request_rate": float("inf"),
    "max_concurrency": 45,
    "random_input_len": 3500,
    "random_output_len": 1500,
    "temperature": 0.0,
}

BASELINE_METRICS = (
    "output_throughput",
    "mean_tpot_ms",
    "mean_ttft_ms",
)


def _server_args(port: int, quantization: str | None = None) -> list[str]:
    args = [
        "--async-scheduling",
        "--tensor-parallel-size",
        "4",
        "--port",
        str(port),
        "--max-num-seqs",
        "16",
        "--max-model-len",
        "16384",
        "--max-num-batched-tokens",
        "16384",
        "--gpu-memory-utilization",
        "0.9",
        "--trust-remote-code",
        "--additional-config",
        json.dumps({"enable_cpu_binding": True}),
        "--compilation-config",
        json.dumps(
            {
                "cudagraph_mode": "FULL",
                "cudagraph_capture_sizes": CUDAGRAPH_CAPTURE_SIZES,
            }
        ),
    ]
    if quantization is not None:
        args.extend(["--quantization", quantization])
    return args


def _run_vllm_bench_performance_guard(model: str, quantization: str | None = None) -> dict[str, Any]:
    port = get_open_port()
    server_args = _server_args(port, quantization)
    baseline = PERFORMANCE_BASELINES[model]
    with RemoteOpenAIServer(
        model,
        server_args,
        server_port=port,
        env_dict=SERVER_ENVS,
        auto_port=False,
    ):
        result = run_vllm_bench_case(
            model,
            port,
            VLLM_BENCH_CASE.copy(),
            baseline["output_throughput"],
            threshold=THROUGHPUT_THRESHOLD,
        )

    missing = [metric for metric in BASELINE_METRICS if metric not in result]
    assert not missing, f"vLLM Bench result is missing baseline metrics: {missing}"
    metrics = {metric: float(result[metric]) for metric in BASELINE_METRICS}
    assert all(value > 0 for value in metrics.values()), f"Invalid baseline metrics: {metrics}"
    assert metrics["mean_tpot_ms"] <= baseline["mean_tpot_ms"], (
        f"TPOT regression for {model}: {metrics['mean_tpot_ms']:.3f} ms exceeds "
        f"the {baseline['mean_tpot_ms']:.3f} ms upper bound."
    )
    print(f"Qwen3-30B-A3B A2 performance ({model}): {json.dumps(metrics, sort_keys=True)}")
    return result


@pytest.mark.e2e_model(BF16_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="aclgraph",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="full_graph",
)
@wait_until_npu_memory_free()
def test_qwen3_30b_a3b_bf16_performance() -> None:
    _run_vllm_bench_performance_guard(BF16_MODEL)


@pytest.mark.e2e_model(W8A8_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="aclgraph",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="W8A8",
    graph_mode="full_graph",
)
@wait_until_npu_memory_free()
def test_qwen3_30b_a3b_w8a8_performance() -> None:
    _run_vllm_bench_performance_guard(W8A8_MODEL, quantization="ascend")
