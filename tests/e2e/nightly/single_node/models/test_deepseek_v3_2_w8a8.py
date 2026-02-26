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

MODELS = ["vllm-ascend/DeepSeek-V3.2-W8A8"]

TENSOR_PARALLELS = [8]
DATA_PARALLELS = [2]

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
    "num_prompts": 1,
    "max_out_len": 1500,
    "batch_size": 1,
    "request_rate": 11.2,
    "baseline": 134,
    "threshold": 0.97
}, {
    "case_type": "performance",
    "dataset_path": "vllm-ascend/GSM8K-in3500-bs400",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "num_prompts": 100,
    "max_out_len": 1500,
    "batch_size": 4,
    "request_rate": 11.2,
    "baseline": 134,
    "threshold": 0.97
}
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
@pytest.mark.parametrize("dp_size", DATA_PARALLELS)
async def test_models(model: str, tp_size: int, dp_size: int) -> None:
    port = get_open_port()
    env_dict = {
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "1",
        "HCCL_BUFFSIZE": "1024",
        "VLLM_ASCEND_ENABLE_MLAPO": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
    }

    server_args = [
        "--enable-expert-parallel", "--tensor-parallel-size",
        str(tp_size), "--data-parallel-size",
        str(dp_size), "--port",
        str(port), "--max-model-len", "8192", "--max-num-batched-tokens",
        "8192", "--max-num-seqs", "4", "--trust-remote-code", "--quantization",
        "ascend", "--gpu-memory-utilization", "0.98", "--compilation-config",
        '{"cudagraph_capture_sizes":[8, 16, 24, 32, 40, 48], "cudagraph_mode":"FULL_DECODE_ONLY"}',
        "--speculative-config",
        '{"num_speculative_tokens": 3, "method":"deepseek_mtp"}',
        "--additional-config",
        '{"layer_sharding": ["q_b_proj", "o_proj"]}',
        "--reasoning-parser", "deepseek_v3", "--tokenizer_mode", "deepseek_v32"
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
