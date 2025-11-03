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
from vllm.utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tools.aisbench import run_aisbench_cases
from tools.send_mm_request import send_image_request

MODELS = [
    "Qwen/Qwen2.5-VL-32B-Instruct",
]

TENSOR_PARALLELS = [4]

prompts = [
    "San Francisco is a",
]

api_keyword_args = {
    "max_tokens": 10,
}

aisbench_cases = [{
    "case_type": "accuracy",
    "dataset_path": "vllm-ascend/textvqa-lite",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "textvqa/textvqa_gen_base64",
    "max_out_len": 2048,
    "batch_size": 128,
    "baseline": 76,
    "temperature": 0,
    "top_k": -1,
    "top_p": 1,
    "repetition_penalty": 1,
    "threshold": 5
}, {
    "case_type": "performance",
    "dataset_path": "vllm-ascend/textvqa-perf-1080p",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "textvqa/textvqa_gen_base64",
    "num_prompts": 512,
    "max_out_len": 256,
    "batch_size": 128,
    "temperature": 0,
    "top_k": -1,
    "top_p": 1,
    "repetition_penalty": 1,
    "request_rate": 0,
    "baseline": 1,
    "threshold": 0.97
}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_models(model: str, tp_size: int) -> None:
    port = get_open_port()
    env_dict = {
        "TASK_QUEUE_ENABLE": "1",
        "VLLM_ASCEND_ENABLE_NZ": "0",
        "HCCL_OP_EXPANSION_MODE": "AIV"
    }
    server_args = [
        "--no-enable-prefix-caching", "--disable-mm-preprocessor-cache",
        "--tensor-parallel-size",
        str(tp_size), "--port",
        str(port), "--max-model-len", "30000", "--max-num-batched-tokens",
        "40000", "--max-num-seqs", "400", "--trust-remote-code",
        "--gpu-memory-utilization", "0.8", "--additional-config",
        '{"ascend_scheduler_config":{"enabled":false}}'
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
        send_image_request(model, server)
        # aisbench test
        run_aisbench_cases(model, port, aisbench_cases)
