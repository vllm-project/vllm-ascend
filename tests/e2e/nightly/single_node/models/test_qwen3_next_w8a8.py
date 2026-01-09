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
    "vllm-ascend/Qwen3-Next-80B-A3B-Instruct-W8A8",
]

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
    "max_out_len": 32768,
    "batch_size": 32,
    "baseline": 95,
    "threshold": 5
}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_models(model: str) -> None:
    port = get_open_port()
    env_dict = {
        "OMP_NUM_THREADS": "10",
        "OMP_PROC_BIND": "false",
        "HCCL_BUFFSIZE": "1024",
    }
    server_args = [
        "--quantization",
        "ascend",
        "--async-scheduling",
        "--no-enable-prefix-caching",
        "--data-parallel-size",
        "1",
        "--tensor-parallel-size",
        "4",
        "--enable-expert-parallel",
        "--port",
        str(port),
        "--max-model-len",
        "40960",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "32",
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.65",
        "--compilation-config",
        '{"cudagraph_capture_sizes": [32]}',
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
        print(choices)
        # aisbench test
        run_aisbench_cases(model,
                           port,
                           aisbench_cases,
                           server_args=server_args)
