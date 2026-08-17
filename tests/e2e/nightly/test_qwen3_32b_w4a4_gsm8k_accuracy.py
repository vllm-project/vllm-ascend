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
"""Nightly accuracy guard for the Qwen3-32B W4A4 exit scenario (GSM8K).

The same GSM8K accuracy benchmark is run once with the V1 model runner
(``VLLM_USE_V2_MODEL_RUNNER=0``) and once with the V2 model runner
(``VLLM_USE_V2_MODEL_RUNNER=1``); the absolute V2-V1 accuracy difference must
stay within the configured tolerance.

Scenario (2026-08-17):
  TP2 + quantization (W4A4), max-model-len=40960, max-num-batched-tokens=16384,
  max-num-seqs=64, batch_size=32, cudagraph_capture_sizes=[64].
  Sampling is temporarily greedy (temperature=0, top_k/top_p disabled) until
  the eagle3 accuracy fix lands upstream; restore temperature=0.6, top_k=20,
  top_p=0.95 afterwards.

The dataset is ``vllm-ascend/gsm8k`` (full, 1319 questions).
|V2-V1| <= 2.00pp (about 26 questions of 1319).
"""

import os

from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from tools.aisbench import run_aisbench_cases

MODEL = os.environ.get("QWEN3_32B_W4A4_MODEL_PATH", "vllm-ascend/Qwen3-32B-W4A4")

# GSM8K full: 1319 questions; team-confirmed tolerance |V2-V1| <= 2.00pp
# (about 26 questions).
MAX_ACCURACY_DELTA_PP = 2.0

_BENCH_CASE = {
    "case_type": "accuracy",
    "dataset_path": "vllm-ascend/gsm8k",
    "request_conf": "vllm_api_general_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_chat_prompt",
    "max_out_len": 32768,
    "batch_size": 32,
    # Temporarily force greedy decoding until the eagle3 accuracy fix lands
    # upstream. When the eagle3 PR is merged, restore the original sampling
    # parameters: temperature=0.6, top_k=20, top_p=0.95.
    "temperature": 0,
    # "top_k": 20,
    # "top_p": 0.95,
    "baseline": 100,
    "threshold": 100,
}

_COMMON_ENV = {
    "ASCEND_RT_VISIBLE_DEVICES": "0,1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    "HCCL_BUFFSIZE": "1024",
}

_V1_ENV = {**_COMMON_ENV, "VLLM_USE_V2_MODEL_RUNNER": "0"}
_V2_ENV = {**_COMMON_ENV, "VLLM_USE_V2_MODEL_RUNNER": "1"}

_SERVER_ARGS = [
    "--trust-remote-code",
    "--max-model-len",
    "40960",
    "--max-num-batched-tokens",
    "16384",
    "--max-num-seqs",
    "64",
    "--data-parallel-size",
    "1",
    "--tensor-parallel-size",
    "2",
    "--distributed-executor-backend",
    "mp",
    "--quantization",
    "ascend",
    "--compilation-config",
    '{"cudagraph_capture_sizes": [64]}',
    "--gpu-memory-utilization",
    "0.9",
]


# Wait for the NPU driver to reclaim memory after the previous server exits,
# so the next V1/V2 server does not OOM on a busy device.
@wait_until_npu_memory_free()
def _run_gsm8k_accuracy(env):
    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL, _SERVER_ARGS + ["--port", str(port)], server_port=port, env_dict=env, auto_port=False
    ):
        results = run_aisbench_cases(MODEL, port, [_BENCH_CASE])
    accuracy = float(results[0])
    print(f"[GSM8K acc] accuracy: {accuracy}%")
    return accuracy


def test_qwen3_32b_w4a4_gsm8k_v1_v2_accuracy_within_tolerance():
    v1_accuracy = _run_gsm8k_accuracy(_V1_ENV)
    v2_accuracy = _run_gsm8k_accuracy(_V2_ENV)
    delta = abs(v2_accuracy - v1_accuracy)
    print(f"[GSM8K acc] V1={v1_accuracy:.2f}% V2={v2_accuracy:.2f}% delta={delta:.2f}pp")
    assert delta <= MAX_ACCURACY_DELTA_PP, (
        f"GSM8K accuracy regression: |V2-V1|={delta:.2f}pp exceeds the "
        f"{MAX_ACCURACY_DELTA_PP}pp limit "
        f"(V1={v1_accuracy:.2f}%, V2={v2_accuracy:.2f}%)."
    )
