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

import json
import openai
import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, MooncakeLauncher
from tools.aisbench import run_aisbench_cases, maybe_download_from_modelscope

MODELS = [
    "vllm-ascend/Qwen3-30B-A3B-W8A8",
]

eagle_model = maybe_download_from_modelscope("vllm-ascend/Qwen3-a3B_eagle3")

TENSOR_PARALLELS = [1, 4]

prompts = [
    "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
]

api_keyword_args = {
    "max_tokens": 10,
}

mooncake_json = {
    "local_hostname": "localhost",
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "use_ascend_direct": True,
    "master_server_address": "",
    "global_segment_size": 30000000000
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
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_models(model: str, tp_size: int) -> None:
    port = get_open_port()
    mooncake_port = get_open_port()
    mooncake_metrics_port = get_open_port()
    mooncake_json["master_server_address"] = f"127.0.0.1:{mooncake_port}"
    with open("mooncake.json", "w") as f:
        json.dump(mooncake_json, f)
    env_dict = {
        "PYTHONHASHSEED": "0",
        "ASCEND_CONNECT_TIMEOUT": "10000",
        "ASCEND_TRANSFER_TIMEOUT": "10000",
        "ASCEND_BUFFER_POOL": "4:8",
        "VLLM_USE_V1": "1",
        "OMP_PROC_BIND": "false",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "1024",
        "OMP_NUM_THREADS": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ASCEND_ENABLE_NZ": "2",
        "MOONCAKE_CONFIG_PATH": "mooncake.json"
    }
    if tp_size != 1:
        env_dict["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "1"
    kv_transfer_config = {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "register_buffer": True,
            "use_layerwise": False,
            "mooncake_rpc_port": "0"
        }
    }
    speculative_config = {
        "method": "eagle3",
        "model": eagle_model,
        "num_speculative_tokens": 3
    }
    server_args = [
        "--trust-remote-code", "--max-num-seqs", "100", "--max-model-len",
        "37364", "--max-num-batched-tokens", "16384", "--tensor-parallel-size",
        str(tp_size), "--enable-expert-parallel", "--port",
        str(port), "--distributed_executor_backend", "mp",
        "--async-scheduling", "--quantization", "ascend",
        "--compilation-config", '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        "--gpu-memory-utilization", "0.95", "--speculative-config",
        json.dumps(speculative_config), "--kv-transfer-config",
        json.dumps(kv_transfer_config)
    ]
    request_keyword_args: dict[str, Any] = {
        **api_keyword_args,
    }
    with MooncakeLauncher(mooncake_port,
                          mooncake_metrics_port) as mooncake_server:
        with RemoteOpenAIServer(model,
                                server_args,
                                server_port=port,
                                env_dict=env_dict,
                                auto_port=False) as server:
            client = server.get_async_client()
            for _ in range(2):
                batch = await client.completions.create(
                    model=model,
                    prompt=prompts,
                    **request_keyword_args,
                )
                choices: list[openai.types.CompletionChoice] = batch.choices
                assert choices[0].text, "empty response"
            # aisbench test
            run_aisbench_cases(model, port, aisbench_cases)
            run_aisbench_cases(model, port, aisbench_cases)
