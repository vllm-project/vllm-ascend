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

"""PR-level smoke test for MiniMax-M2.5 w8a8 QuaRot on a single A2 node.

This test serves the MiniMax-M2.5 w8a8 QuaRot model on an 8-NPU A2 node
(TP=8, expert parallel, eagle3 speculative decoding) and issues a single
chat completion request to verify the serving stack end to end.

The serving arguments mirror the nightly case
``tests/e2e/nightly/single_node/models/configs/MiniMax-M2.5-w8a8-QuaRot-A2.yaml``,
with ``max_model_len`` reduced to keep the PR-level runtime bounded.
"""

from __future__ import annotations

import json
import os

import pytest
import requests

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

MINIMAX_M2_5_MODEL_PATH = os.environ.get(
    "MINIMAX_M2_5_MODEL_PATH", "Eco-Tech/MiniMax-M2.5-w8a8-QuaRot"
)
EAGLE_MODEL_PATH = os.environ.get(
    "MINIMAX_M2_5_EAGLE_MODEL_PATH", "vllm-ascend/MiniMax-M2.5-eagle-model-0318"
)


@pytest.mark.e2e_model(MINIMAX_M2_5_MODEL_PATH)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="quarot,eagle3",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A2",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_minimax_m2_5_w8a8_quarot_single_request() -> None:
    """Serve MiniMax-M2.5 w8a8 QuaRot on a single A2 node and verify one response."""
    env_dict = {
        "HCCL_BUFFSIZE": "1200",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
        "OMP_NUM_THREADS": "1",
        "TASK_QUEUE_ENABLE": "1",
        "VLLM_ASCEND_ENABLE_FUSED_MC2": "1",
        "VLLM_ASCEND_ENABLE_NZ": "1",
        "VLLM_USE_MODELSCOPE": "true",
    }
    server_args = [
        "--tensor-parallel-size",
        "8",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.9",
        "--quantization",
        "ascend",
        "--speculative-config",
        json.dumps(
            {
                "method": "eagle3",
                "model": EAGLE_MODEL_PATH,
                "num_speculative_tokens": 3,
            }
        ),
        "--enable-expert-parallel",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--max-num-seqs",
        "128",
        "--max-model-len",
        "10240",
        "--max-num-batched-tokens",
        "16384",
        "--seed",
        "1024",
        "--compilation-config",
        json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"}),
        "--additional-config",
        json.dumps(
            {
                "enable_cpu_binding": True,
                "enable_npugraph_ex": True,
                "enable_static_kernel": True,
            }
        ),
    ]

    with RemoteOpenAIServer(
        MINIMAX_M2_5_MODEL_PATH, server_args, env_dict=env_dict
    ) as server:
        response = requests.post(
            server.url_for("v1", "chat/completions"),
            json={
                "model": MINIMAX_M2_5_MODEL_PATH,
                "messages": [{"role": "user", "content": "What is deep learning?"}],
                "max_tokens": 128,
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
            },
            timeout=600,
        )
        response.raise_for_status()
        output = response.json()

        assert output["choices"][0]["message"]["content"]
