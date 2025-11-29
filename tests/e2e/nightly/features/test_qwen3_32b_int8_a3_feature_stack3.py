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

import pytest
from vllm.utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tools.aisbench import run_aisbench_cases
from tools.send_request import send_text_request

MODELS = [
    "vllm-ascend/Qwen3-32B-W8A8",
]

TENSOR_PARALLELS = [4]

prompts = [
    "9.11 and 9.8, which is greater?",
]

api_keyword_args = {
    "chat_template_kwargs": {
        "enable_thinking": True
    },
}

aisbench_cases = [{
    "case_type": "accuracy",
    "dataset_path": "vllm-ascend/gsm8k-lite",
    "request_conf": "vllm_api_general_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_noncot_chat_prompt",
    "max_out_len": 10240,
    "batch_size": 32,
    "baseline": 96,
    "threshold": 4
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
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_models(model: str, tp_size: int) -> None:
    port = get_open_port()
    env_dict = {
        "VLLM_USE": "1",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "OMP_PROC_BIND": "false",
        "VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE": "1",
        "VLLM_ASCEND_ENABLE_FLASHCOMM": "1",
        "VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE": "1",
        "VLLM_ASCEND_ENABLE_PREFETCH_MLP": "1"
    }
    server_args = [
        "--quantization", "ascend", "--tensor-parallel-size",
        str(tp_size), "--port",
        str(port), "--trust-remote-code", "--reasoning-parser", "qwen3",
        "--distributed_executor_backend", "mp", "--gpu-memory-utilization",
        "0.9", "--block-size", "128", "--max-num-seqs", "256",
        "--enforce-eager", "--max-model-len", "35840",
        "--max-num-batched-tokens", "35840", "--additional-config",
        '{"enable_weight_nz_layout":true}', "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,8,24,48,60]}'
    ]
    with RemoteOpenAIServer(model,
                            server_args,
                            server_port=port,
                            env_dict=env_dict,
                            auto_port=False) as server:
        send_text_request(prompts[0],
                          model,
                          server,
                          request_args=api_keyword_args)
        # aisbench test
        run_aisbench_cases(model, port, aisbench_cases)
