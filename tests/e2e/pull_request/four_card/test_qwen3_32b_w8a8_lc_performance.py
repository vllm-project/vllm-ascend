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
"""PR performance guard for the Qwen3-32B W8A8 long-context exit scenario.

Long-context high-throughput (LC): 128k input / 1k output.

The same benchmark is run once with the V1 model runner
(``VLLM_USE_V2_MODEL_RUNNER=0``) and once with the V2 model runner
(``VLLM_USE_V2_MODEL_RUNNER=1``); the V2 Total Token Throughput must stay
within +-3% of V1, so stacking/regression issues are exposed at PR time.

Scenario (2026-08-09 revision):
  TP4 + YaRN + prefix caching (90% shared prefix) + FULL_DECODE_ONLY +
  quantization (W8A8), max-model-len=135000, num_prompts=16,
  max_out_len=1024, batch_size (concurrency)=4.

Scenario revised 2026-08-09: eagle3 is excluded until its accuracy/perf fix
lands upstream, and prefix caching is stacked in (the dataset repeats 90% of
its prefix). V1/V2 baselines for this scenario are to be re-measured on NPU
before finalizing the guard. Any change to the scenario parameters or
tolerances must be approved by the team.
"""

import os

from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from tools.aisbench import run_aisbench_cases

MODEL = os.getenv("QWEN3_32B_PDMIX_PATH", "/mnt/a800_weight/qwen3-32b-pdmix")

# The 128k-in / 1k-out dataset repeats 90% of its prefix so prefix caching is
# exercised; it is self-constructed by the team and this is a mock placeholder
# id. Publish the dataset under this id on ModelScope (or switch to a local
# path via `dataset_path_local`) before enabling the guard.
DATASET_PATH = "vllm-ascend/GSM8K-in131072-prefix90-bs16"

# V2/V1 Total Token Throughput ratio must stay within +-3%.
THROUGHPUT_RATIO_LOWER = 0.97
THROUGHPUT_RATIO_UPPER = 1.03

_BENCH_CASE = {
    "case_type": "performance",
    "dataset_path": DATASET_PATH,
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "num_prompts": 16,
    "max_out_len": 1024,
    "batch_size": 4,
    "request_rate": 0,
    # Disable the built-in absolute-baseline gate; the V1 vs V2 comparison is
    # done below in this test.
    "baseline": 1,
    "threshold": 0,
}

_COMMON_ENV = {
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
}

_V1_ENV = {**_COMMON_ENV, "VLLM_USE_V2_MODEL_RUNNER": "0"}
_V2_ENV = {**_COMMON_ENV, "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_VERSION": "0.27.0"}

_SERVER_ARGS = [
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
    (
        '{"rope_parameters": {"rope_type": "yarn", "rope_theta": 1000000, '
        '"factor": 4, "original_max_position_embeddings": 131072}}'
    ),
    "--gpu-memory-utilization",
    "0.9",
    "--quantization",
    "ascend",
]


# Wait for the NPU driver to reclaim memory after the previous server exits,
# so the next V1/V2 server does not OOM on a busy device.
@wait_until_npu_memory_free()
def _run_lc_benchmark(env):
    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL, _SERVER_ARGS + ["--port", str(port)], server_port=port, env_dict=env, auto_port=False
    ):
        results = run_aisbench_cases(MODEL, port, [_BENCH_CASE])
    result_json = results[0][1]
    throughput = float(result_json["Total Token Throughput"]["total"].replace("token/s", "").strip())
    print(f"[LC perf] Total Token Throughput: {throughput} token/s")
    return throughput


def test_qwen3_32b_w8a8_lc_128k_v1_v2_throughput_within_3pct():
    v1_throughput = _run_lc_benchmark(_V1_ENV)
    v2_throughput = _run_lc_benchmark(_V2_ENV)
    ratio = v2_throughput / v1_throughput
    print(f"[LC perf] V1={v1_throughput:.2f} V2={v2_throughput:.2f} ratio={ratio:.4f}")
    assert THROUGHPUT_RATIO_LOWER <= ratio <= THROUGHPUT_RATIO_UPPER, (
        f"LC 128k/1k performance regression: V2/V1 Total Token Throughput ratio "
        f"{ratio:.4f} (V1={v1_throughput:.2f}, V2={v2_throughput:.2f}) is outside "
        f"[{THROUGHPUT_RATIO_LOWER}, {THROUGHPUT_RATIO_UPPER}]."
    )
