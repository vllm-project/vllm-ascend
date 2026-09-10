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
"""Nightly accuracy guard for the Qwen3-32B W8A8 exit scenario (AIME2024).

The same AIME2024 accuracy benchmark is run once with the V1 model runner
(``VLLM_USE_V2_MODEL_RUNNER=0``) and once with the V2 model runner
(``VLLM_USE_V2_MODEL_RUNNER=1``); the absolute V2-V1 accuracy difference must
stay within 3 questions. AIME2024 has 30 questions, so 3 questions == 10.0pp.

Scenario (2026-08-17 revision, eagle3 excluded until its accuracy fix lands
upstream):
  TP4 + async-scheduling + FULL_DECODE_ONLY + quantization (W8A8),
  max-model-len=36864, max_out_len=32768, batch_size=64.
  Sampling is temporarily greedy (temperature=0, top_k/top_p disabled) until
  the eagle3 accuracy fix lands upstream; restore temperature=0.6, top_k=20,
  top_p=0.95 afterwards.

Measured V1 mean accuracy (3 rounds, with eagle3): 83.33%; re-measure without
eagle3 on NPU before finalizing. Any change to the scenario parameters or
tolerances must be approved by the team.
"""

import os

from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from tools.aisbench import run_aisbench_cases

MODEL = os.environ.get("QWEN3_32B_W8A8_MODEL_PATH", "vllm-ascend/Qwen3-32B-W8A8")

# AIME2024: 30 questions, 3 questions == 100/30*3 == 10.0pp.
MAX_ACCURACY_DELTA_PP = 10.0

_BENCH_CASE = {
    "case_type": "accuracy",
    "dataset_path": "vllm-ascend/aime2024",
    "request_conf": "vllm_api_general_chat",
    "dataset_conf": "aime2024_gen_0_shot_chat_prompt",
    "num_prompts": 30,
    "max_out_len": 32768,
    "batch_size": 64,
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
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_OP_EXPANSION_MODE": "AIV",
}

_V1_ENV = {**_COMMON_ENV, "VLLM_USE_V2_MODEL_RUNNER": "0"}
_V2_ENV = {**_COMMON_ENV, "VLLM_USE_V2_MODEL_RUNNER": "1"}

_SERVER_ARGS = [
    "--trust-remote-code",
    "--max-model-len",
    "36864",
    "--max-num-batched-tokens",
    "40960",
    "--tensor-parallel-size",
    "4",
    "--distributed-executor-backend",
    "mp",
    "--async-scheduling",
    "--quantization",
    "ascend",
    "--compilation-config",
    '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
    "--gpu-memory-utilization",
    "0.9",
]


# Wait for the NPU driver to reclaim memory after the previous server exits,
# so the next V1/V2 server does not OOM on a busy device.
@wait_until_npu_memory_free()
def _run_aime2024_accuracy(env):
    port = get_open_port()
    with RemoteOpenAIServer(
        MODEL, _SERVER_ARGS + ["--port", str(port)], server_port=port, env_dict=env, auto_port=False
    ):
        results = run_aisbench_cases(MODEL, port, [_BENCH_CASE])
    accuracy = float(results[0])
    print(f"[AIME2024 acc] accuracy: {accuracy}%")
    return accuracy


def test_qwen3_32b_w8a8_aime2024_v1_v2_accuracy_within_3_questions():
    v1_accuracy = _run_aime2024_accuracy(_V1_ENV)
    v2_accuracy = _run_aime2024_accuracy(_V2_ENV)
    delta = abs(v2_accuracy - v1_accuracy)
    print(f"[AIME2024 acc] V1={v1_accuracy:.2f}% V2={v2_accuracy:.2f}% delta={delta:.2f}pp")
    assert delta <= MAX_ACCURACY_DELTA_PP, (
        f"AIME2024 accuracy regression: |V2-V1|={delta:.2f}pp exceeds the "
        f"{MAX_ACCURACY_DELTA_PP}pp (3-question) limit "
        f"(V1={v1_accuracy:.2f}%, V2={v2_accuracy:.2f}%)."
    )
