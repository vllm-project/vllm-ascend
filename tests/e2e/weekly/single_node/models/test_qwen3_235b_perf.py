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
from typing import Any

import openai
import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tools.aisbench import run_aisbench_cases

MODELS = [
    "vllm-ascend/Qwen3-235B-A22B-W8A8",
]

MODES = ["low"]

prompts = [
    "San Francisco is a",
]

api_keyword_args = {
    "max_tokens": 10,
}

aisbench_cases = [{
    "case_type": "performance",
    "dataset_path": "vllm-ascend/GSM8K-in3500-bs400",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "max_out_len": 1500,
    "batch_size": [1,2,4,8,16],
    "baseline": 1,
    "threshold": 0.97
}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
async def test_models(model: str, mode: str) -> None:
    port = get_open_port()
    env_dict = {
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "1024",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "1",
        "LD_PRELOAD": "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2",
        "TASK_QUEUE_ENABLE": "1",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
        "VLLM_ASCEND_ENABLE_FUSED_MC2": "1",
    }
    compilation_config = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    additional_config = {"enable_cpu_binding": True}
    server_args = [
    "--port", str(port),
    "--async-scheduling",
    "--tensor-parallel-size", "16",
    "--data-parallel-size", "1",
    "--data-parallel-size-local", "1",
    "--data-parallel-start-rank", "0",
    "--enable-expert-parallel",
    "--max-num-seqs", "128",
    "--max-model-len", "32768",
    "--max-num-batched-tokens", "16384",
    "--gpu-memory-utilization", "0.9",
    "--trust-remote-code",
    "--quantization", "ascend",
    "--no-enable-prefix-caching"
    ]
    server_args.extend(
        ["--compilation-config",
         json.dumps(compilation_config), "--additional-config", json.dumps(additional_config)])
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
