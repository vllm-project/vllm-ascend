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
"""PR performance guard for the Qwen3-32B W8A8 short-context exit scenario.

Short-context low-latency (LL): 16k input / 1k output.

The same benchmark is run once with the V1 model runner
(``VLLM_USE_V2_MODEL_RUNNER=0``) and once with the V2 model runner
(``VLLM_USE_V2_MODEL_RUNNER=1``); V2 TTFT/TPOT must not regress more than +3%
vs V1. Being faster is allowed: the 2026-08-06 measurement showed V2 TTFT is
~2.6% faster than V1 at bs=2, so a two-sided +-3% gate would be fragile.

Scenario (2026-08-09 revision, eagle3 excluded until its accuracy/perf fix
lands upstream):
  TP2 x DP2 + FULL_DECODE_ONLY + quantization (W8A8), max-model-len=18000,
  num_prompts=50, max_out_len=1024, batch_size (concurrency)=2.
  Prefix caching is not enabled in this low-latency scenario.

The 2026-08-06 V1 baseline (TTFT 2454.5 ms / TPOT 12.2 ms at bs=2) was
measured with eagle3; re-measure without eagle3 on NPU before finalizing the
guard. Any change to the scenario parameters or tolerances must be approved by
the team.
"""

import os

from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tools.aisbench import run_aisbench_cases

MODEL = os.getenv("QWEN3_32B_PDMIX_PATH", "/mnt/a800_weight/qwen3-32b-pdmix")

DATASET_PATH = "vllm-ascend/GSM8K-in16384-bs50"

# V2 latency (TTFT/TPOT) must be at most 3% worse than V1.
LATENCY_REGRESSION_RATIO = 1.03

_BENCH_CASE = {
    "case_type": "performance",
    "dataset_path": DATASET_PATH,
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "num_prompts": 50,
    "max_out_len": 1024,
    "batch_size": 2,
    "request_rate": 0,
    "baseline": 1,
    "threshold": 0,
}

_COMMON_ENV = {
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

_V1_ENV = {**_COMMON_ENV, "VLLM_USE_V2_MODEL_RUNNER": "0"}
_V2_ENV = {**_COMMON_ENV, "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_VERSION": "0.27.0"}

_SERVER_ARGS = [
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
    "--no-enable-prefix-caching",
    "--compilation-config",
    '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
    "--gpu-memory-utilization",
    "0.9",
    "--quantization",
    "ascend",
]


def _extract_latency_ms(result_csv, metric):
    return float(str(result_csv.loc[metric, "Average"]).replace("ms", "").strip())


def _run_ll_benchmark(env):
    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL, _SERVER_ARGS + ["--port", str(port)], server_port=port, env_dict=env, auto_port=False
    ):
        results = run_aisbench_cases(MODEL, port, [_BENCH_CASE])
    result_csv = results[0][0]
    ttft = _extract_latency_ms(result_csv, "TTFT")
    tpot = _extract_latency_ms(result_csv, "TPOT")
    print(f"[LL perf] TTFT={ttft} ms, TPOT={tpot} ms")
    return ttft, tpot


def test_qwen3_32b_w8a8_ll_16k_v1_v2_latency_within_3pct():
    v1_ttft, v1_tpot = _run_ll_benchmark(_V1_ENV)
    v2_ttft, v2_tpot = _run_ll_benchmark(_V2_ENV)
    for metric, v1_value, v2_value in (("TTFT", v1_ttft, v2_ttft), ("TPOT", v1_tpot, v2_tpot)):
        print(f"[LL perf] {metric}: V1={v1_value} ms, V2={v2_value} ms")
        assert v2_value <= v1_value * LATENCY_REGRESSION_RATIO, (
            f"LL 16k/1k latency regression: V2 {metric} {v2_value} ms exceeds "
            f"V1 {v1_value} ms * {LATENCY_REGRESSION_RATIO}."
        )
