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

import openai
import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer


@pytest.mark.asyncio
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="eplb,dynamic_eplb",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A2",
    quantization="W8A8",
    graph_mode="aclgraph",
)
async def test_qwen3_moe_w8a8_distributed_tp2_ep_dynamic_eplb():
    """Dynamic EPLB with redundant experts must serve on A2 (ALLGATHER MoE).

    This is the #14080 repro: EPLB with ``num_redundant_experts > 0`` on A2 +
    MRv1 previously crashed at the first forward (ALLGATHER OOB), so the
    acceptance criterion is that the engine comes up with dynamic EPLB and
    serves a request, mirroring the existing A3 EPLB case
    (``two_card/test_qwen3_30b_a3b.py::test_moe_tp_ep_eplb_full_decode_only``).

    The case deliberately does **not** compare the generated text against a
    non-EPLB baseline. Redundant experts change the expert partition
    (``(num_experts + redundant) / ep_size`` per rank instead of
    ``num_experts / ep_size``), and therefore the MoE all-reduce grouping, so
    the two runs differ by ~0.1 nat in the next-token logprobs (measured on A2:
    max |delta| = 0.130 over the top-20, 19/20 identical tokens, with the
    expert placement itself verified valid). On an ambiguous prompt whose top-1
    is a 0.000-margin tie (``What is deeplearning?`` -> `` How`` and `` What``
    both at -2.069) that difference flips the first greedy token and the whole
    continuation diverges, which makes a strict text comparison a test of
    numerical luck rather than of correctness.

    The invariant that dynamic EPLB must preserve on the ALLGATHER path --
    every logical expert is executed by exactly one rank -- is pinned at unit
    level by
    ``tests/ut/eplb/core/test_eplb_utils.py::TestAscendConfig::test_log2phy_rank_independent_keeps_exactly_once_execution``.
    """
    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    port = get_open_port()
    compilation_config = json.dumps({"cudagraph_capture_sizes": [8]})
    additional_config = {
        "eplb_config": {
            "dynamic_eplb": True,
            "expert_heat_collection_interval": 100,
            "algorithm_execution_interval": 20,
            "num_redundant_experts": 2,
            "eplb_policy_type": 2,  # SwiftBalance
        }
    }
    server_args = [
        "--max_model_len",
        "8192",
        "--tensor_parallel_size",
        "2",
        "--enable_expert_parallel",
        "--quantization",
        "ascend",
        "--port",
        str(port),
        "--compilation-config",
        compilation_config,
        "--additional-config",
        json.dumps(additional_config),
    ]
    env_dict = {"HCCL_BUFFSIZE": "1024", "DYNAMIC_EPLB": "true"}

    with RemoteOpenAIServer(model, server_args, server_port=port, auto_port=False, env_dict=env_dict) as server:
        client = server.get_async_client()
        batch = await client.completions.create(
            model=model, prompt="What is deeplearning?", max_tokens=400, temperature=0, top_p=1.0, n=1
        )
        choices: list[openai.types.CompletionChoice] = batch.choices

    assert choices[0].text
