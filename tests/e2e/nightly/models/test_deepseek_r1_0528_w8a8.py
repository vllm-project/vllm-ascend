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
from vllm.utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tools.aisbench import run_aisbench_cases

MODELS = [
    "vllm-ascend/DeepSeek-R1-0528-W8A8",
]

MODES = [
    "torchair",
    "single",
    "aclgraph",
    "no_chunkprefill",
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
}, {
    "case_type": "performance",
    "dataset_path": "vllm-ascend/GSM8K-in3500-bs400",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "num_prompts": 400,
    "max_out_len": 1500,
    "batch_size": 1000,
    "baseline": 1,
    "threshold": 0.97
}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mode", MODES)
async def test_models(model: str, mode: str) -> None:
    port = get_open_port()
    env_dict = {
        "OMP_NUM_THREADS": "10",
        "OMP_PROC_BIND": "false",
        "HCCL_BUFFSIZE": "1024",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"
    }
    speculative_config = {
        "num_speculative_tokens": 1,
        "method": "deepseek_mtp"
    }
    additional_config = {
        "ascend_scheduler_config": {
            "enabled": False
        },
        "torchair_graph_config": {
            "enabled": True,
            "enable_multistream_moe": False,
            "enable_multistream_mla": True,
            "graph_batch_sizes": [16],
            "use_cached_graph": True
        },
        "chunked_prefill_for_mla": True,
        "enable_weight_nz_layout": True
    }
    server_args = [
        "--quantization", "ascend", "--data-parallel-size", "2",
        "--tensor-parallel-size", "8", "--enable-expert-parallel", "--port",
        str(port), "--seed", "1024", "--max-model-len", "36864",
        "--max-num-batched-tokens", "4096", "--max-num-seqs", "16",
        "--trust-remote-code", "--gpu-memory-utilization", "0.9",
        "--speculative-config",
        json.dumps(speculative_config)
    ]
    if mode == "single":
        server_args.append("--enforce-eager")
        additional_config["torchair_graph_config"] = {"enabled": False}
    if mode == "aclgraph":
        additional_config["torchair_graph_config"] = {"enabled": False}
    if mode == "no_chunkprefill":
        additional_config["ascend_scheduler_config"] = {"enabled": True}
        i = server_args.index("--max-num-batched-tokens") + 1
        server_args[i] = "36864"
    server_args.extend(["--additional-config", json.dumps(additional_config)])
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
        if mode in ["single", "no_chunkprefill"]:
            return
        # aisbench test
        run_aisbench_cases(model,
                           port,
                           aisbench_cases,
                           server_args=server_args)
