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

MODELS = [
    "Qwen/QwQ-32B",
]

MODES = [
    "aclgraph",
    "single",
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
    "dataset_path": "vllm-ascend/gsm8k-lite",
    "request_conf": "vllm_api_general_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_chat_prompt",
    "max_out_len": 32768,
    "batch_size": 32,
    "baseline": 95,
    "threshold": 5
}, {
    "case_type": "performance",
    "dataset_path": "vllm-ascend/GSM8K-in3500-bs400",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "num_prompts": 240,
    "max_out_len": 1500,
    "batch_size": 60,
    "baseline": 1,
    "threshold": 0.97
}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_models(model: str, mode: str, tp_size: int) -> None:
    port = get_open_port()
    env_dict = {
        "TASK_QUEUE_ENABLE": "1",
        "OMP_PROC_BIND": "false",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "VLLM_ASCEND_ENABLE_FLASHCOMM": "1",
        "VLLM_ASCEND_ENABLE_DEBSE_OPTIMIZE": "1",
        "VLLM_ASCEND_ENABLE_PREFETCH_MLP": "1"
    }
    server_args = [
        "--tensor-parallel-size",
        str(tp_size), "--port",
        str(port), "--max-model-len", "36864", "--max-num-batched-tokens",
        "36864", "--block-size", "128", "--trust-remote-code",
        "--gpu-memory-utilization", "0.9", "--compilation_config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 8, 24, 48, 60]}',
        "--reasoning-parser", "deepseek_r1", "--distributed_executor_backend",
        "mp"
    ]
    if mode == "single":
        server_args.remove("--compilation_config")
        server_args.remove(
            '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 8, 24, 48, 60]}'
        )
        server_args.append("--enforce-eager")
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
        if mode == "single":
            return
        # aisbench test
        run_aisbench_cases(model, port, aisbench_cases)
