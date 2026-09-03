#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import json

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free


@pytest.mark.e2e_model("Qwen/Qwen3-30B-A3B")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="eplb,dynamic_eplb",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_moe_tp_ep_eplb_full_decode_only():
    """Verify MoE serving with TP, EP, EPLB, and full decode only."""
    model = "Qwen/Qwen3-30B-A3B"
    port = get_open_port()
    env_dict = {
        "DYNAMIC_EPLB": "true",
        "HCCL_BUFFSIZE": "1024",
    }
    server_args = [
        "--max_model_len",
        "8192",
        "--tensor_parallel_size",
        "2",
        "--enable_expert_parallel",
        "--port",
        str(port),
        "--compilation-config",
        json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [8]}),
        "--additional-config",
        json.dumps(
            {
                "eplb_config": {
                    "dynamic_eplb": True,
                    "expert_heat_collection_interval": 100,
                    "algorithm_execution_interval": 20,
                    "num_redundant_experts": 2,
                }
            }
        ),
    ]

    with RemoteOpenAIServer(model, server_args, server_port=port, auto_port=False, env_dict=env_dict) as server:
        response = requests.post(
            server.url_for("v1", "completions"),
            json={
                "model": model,
                "prompt": "What is deeplearning?",
                "max_tokens": 400,
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
            },
            timeout=600,
        )
        response.raise_for_status()
        output = response.json()

        assert output["choices"][0]["text"]
