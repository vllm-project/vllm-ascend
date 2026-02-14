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
from .test_qwen3_235b_w8a8 import *


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_models_eplb(model: str) -> None:
    port, aisbench_cases, env_dict, compilation_config, server_args = config()
    env_dict.update(
        {
            "DYNAMIC_EPLB": "true",
        }
    )
    additional_config: dict[str, Any] = {}
    additional_config["eplb_config"] = {
        "dynamic_eplb": "true",
        "expert_heat_collection_interval": 600,
        "algorithm_execution_interval": 50,
        "num_redundant_experts": 16,
        "eplb_policy_type": 2,
    }
    server_args.extend(["--additional-config", json.dumps(additional_config)])
    server_args.extend(
        ["--compilation-config",
         json.dumps(compilation_config)])
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
