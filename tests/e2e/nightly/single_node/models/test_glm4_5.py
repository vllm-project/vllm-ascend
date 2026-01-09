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
from typing import Any

import openai
import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tools.aisbench import run_aisbench_cases

MODELS = [
    "ZhipuAI/GLM-4.5",
]

TENSOR_PARALLELS = [8]
DATA_PARALLELS = [2]
FULL_GRAPH = [True, False]

prompts = [
    "San Francisco is a",
]

api_keyword_args = {
    "max_tokens": 10,
}

aisbench_cases = [{
    "case_type": "accuracy",
    "dataset_path": "vllm-ascend/gsm8k-lite",
    "request_conf": "vllm_api_general_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_chat_prompt",
    "max_out_len": 4096,
    "batch_size": 8,
    "baseline": 95,
    "threshold": 5
}, {
    "case_type": "performance",
    "dataset_path": "vllm-ascend/GSM8K-in3500-bs400",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "num_prompts": 16,
    "max_out_len": 1500,
    "batch_size": 8,
    "request_rate": 0,
    "baseline": 1,
    "threshold": 0.97
}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dp_size", DATA_PARALLELS)
@pytest.mark.parametrize("full_graph", FULL_GRAPH)
async def test_models(model: str, tp_size: int, dp_size: int,
                      full_graph: bool) -> None:
    port = get_open_port()
    env_dict = {"HCCL_BUFFSIZE": "1024"}
    server_args = [
        "--no-enable-prefix-caching",
        "--enable-expert-parallel",
        "--tensor-parallel-size",
        str(tp_size),
        "--data-parallel-size",
        str(dp_size),
        "--port",
        str(port),
        "--max-model-len",
        "8192",
        "--max-num-batched-tokens",
        "8192",
        "--block-size",
        "16",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.9",
    ]
    if full_graph:
        server_args += [
            "--compilation-config",
            '{"cudagraph_capture": [1,2,4,8,16], "cudagraph_model":"FULL_DECODE_ONLY"}'
        ]
    request_keyword_args: dict[str, Any] = {
        **api_keyword_args,
    }
    with RemoteOpenAIServer(model,
                            server_args,
                            server_port=port,
                            env_dict=env_dict,
                            auto_port=False) as server:
        client = server.get_async_client()
        batch = await client.completions.create(
            model=model,
            prompt=prompts,
            **request_keyword_args,
        )
        choices: list[openai.types.CompletionChoice] = batch.choices
        assert choices[0].text, "empty response"
        # aisbench test
        run_aisbench_cases(model, port, aisbench_cases)
