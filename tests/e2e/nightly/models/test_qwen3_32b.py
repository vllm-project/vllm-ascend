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

from tests.e2e.conftest import RemoteOpenAIServer

MODELS = [
    "Qwen/Qwen3-32B",
]

TENSOR_PARALLELS = [4]

prompts = [
    "San Francisco is a",
]

api_keyword_args = {
    "max_tokens": 10,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_models(model: str, tp_size: int) -> None:
    env_dict = {
        "TASK_QUEUE_ENABLE": "1",
        "OMP_PROC_BIND": "false",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "PAGED_ATTENTION_MASK_LEN": "5500"
    }
    server_args = [
        "--no-enable-prefix-caching", "--tensor-parallel-size",
        str(tp_size), "--port", "20002", "--max-model-len", "36864",
        "--max-num-batched-tokens", "36864", "--block-size", "128",
        "--trust-remote-code", "--gpu-memory-utilization", "0.9",
        "--additional-config", '{"enable_weight_nz_layout":true}'
    ]
    request_keyword_args: dict[str, Any] = {
        **api_keyword_args,
    }
    with RemoteOpenAIServer(model,
                            server_args,
                            server_port=20002,
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
