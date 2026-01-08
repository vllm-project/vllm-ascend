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
import asyncio
from typing import Any

import openai
import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

MODELS = ["Qwen/Qwen3-0.6B", "vllm-ascend/DeepSeek-V2-Lite-W8A8"]

DATA_PARALLELS = [2]

prompts = [
    "San Francisco is a",
]

api_keyword_args = {
    "max_tokens": 10,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dp_size", DATA_PARALLELS)
async def test_models_single_request_aclgraph_dp2(model: str,
                                                  dp_size: int) -> None:
    port = get_open_port()
    env_dict = {
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
    }
    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        server_args = [
            "--no-enable-prefix-caching", "--tensor-parallel-size", "1",
            "--data-parallel-size",
            str(dp_size), "--quantization", "ascend", "--max-model-len",
            "1024", "--port",
            str(port), "--trust-remote-code", "--gpu-memory-utilization", "0.9"
        ]
    else:
        server_args = [
            "--no-enable-prefix-caching", "--tensor-parallel-size", "1",
            "--data-parallel-size",
            str(dp_size), "--port",
            str(port), "--trust-remote-code", "--gpu-memory-utilization", "0.9"
        ]
    request_keyword_args: dict[str, Any] = {
        **api_keyword_args,
    }
    with RemoteOpenAIServer(model,
                            vllm_serve_args=server_args,
                            server_port=port,
                            env_dict=env_dict,
                            auto_port=False) as server:
        client = server.get_async_client()

        try:
            batch = await asyncio.wait_for(client.completions.create(
                model=model,
                prompt=prompts,
                **request_keyword_args,
            ),
                                           timeout=10.0)
        except asyncio.TimeoutError:
            pytest.fail("Model did not return response within 10 seconds")

        choices: list[openai.types.CompletionChoice] = batch.choices
        assert choices[0].text, "empty response"
