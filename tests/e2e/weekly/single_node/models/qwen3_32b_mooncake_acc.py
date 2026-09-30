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

from tests.e2e.conftest import MooncakeLauncher, RemoteOpenAIServer
from tools.aisbench import maybe_download_from_modelscope, run_aisbench_cases

MODELS = [
    "vllm-ascend/Qwen3-32B-W8A8",
]


TENSOR_PARALLELS = [4]

prompts = [
    "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her "
    "friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?",
]

api_keyword_args = {
    "max_tokens": 10,
}

mooncake_json = {
    "local_hostname": "localhost",
    "metadata_server": "P2PHANDSHAKE",
    "protocol": "ascend",
    "device_name": "",
    "master_server_address": "",
    "global_segment_size": 30000000000,
}

aisbench_cases = [
    {
        "case_type": "accuracy",
        "dataset_path": "vllm-ascend/aime2025",
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "aime2024/aime2024_gen_0_shot_chat_prompt",
        "max_out_len": 32768,
        "batch_size": 32,
        "temperature": 0.6,
        "top_p": 0.95,
        "baseline": 83.3,
        "threshold": 7
    }
]


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
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "VLLM_ASCEND_ENABLE_PREFETCH_MLP": "1",
        "LD_PRELOAD": "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD",
        "MOONCAKE_CONFIG_PATH": "mooncake.json"
    }
    kv_transfer_config = {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"register_buffer": True, "use_layerwise": False, "mooncake_rpc_port": "0"},
    }

    server_args = [
        "--trust-remote-code",
        "--max-model-len",
        "40960",
        "--max-num-batched-tokens",
        "16384",
        "--tensor-parallel-size",
        str(tp_size),
        "--port",
        str(port),
        "--block-size",
        "128",
        "--distributed_executor_backend",
        "mp",
        "--async-scheduling",
        "--quantization",
        "ascend",
        "--gpu-memory-utilization",
        "0.9",
        "--compilation-config",
        '{"cudagraph_mode": "FULL_DECODE_ONLY"}',
        "--kv-transfer-config",
        json.dumps(kv_transfer_config),
    ]
    request_keyword_args: dict[str, Any] = {
        **api_keyword_args,
    }
    with (
        MooncakeLauncher(mooncake_port, mooncake_metrics_port),
        RemoteOpenAIServer(model, server_args, server_port=port, env_dict=env_dict, auto_port=False) as server,
    ):
        client = server.get_async_client()
        for _ in range(1):
            batch = await client.completions.create(
                model=model,
                prompt=prompts,
                **request_keyword_args,
            )
            choices: list[openai.types.CompletionChoice] = batch.choices
            assert choices[0].text, "empty response"
        # aisbench test
        run_aisbench_cases(model, port, aisbench_cases)
