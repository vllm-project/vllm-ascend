#
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
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
#
import json
import os
from unittest.mock import patch

import openai
import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, VllmRunner


@pytest.mark.asyncio
async def test_mm_mrv2():
    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    port = get_open_port()
    compilation_config = json.dumps({"cudagraph_capture_sizes": [8]})
    server_args = [
        "--max_model_len",
        "8192",
        "--tensor_parallel_size",
        "2",
        "--enable_expert_parallel",
        "--port",
        str(port),
        "--compilation-config",
        compilation_config,
    ]
    env_dict = {"HCCL_BUFFSIZE": "1024",}
    env_dict.update({"VLLM_USE_V2_MODEL_RUNNER": "1"})
    additional_config = {
      "enable_cpu_binding": true
    }
    server_args.extend(["--additional-config", json.dumps(additional_config)])
    with RemoteOpenAIServer(model, server_args, server_port=port, auto_port=False, env_dict=env_dict) as server:
        client = server.get_async_client()
        batch = await client.completions.create(
            model=model, prompt="What is deeplearning?", max_tokens=400, temperature=0, top_p=1.0, n=1
        )
        choices: list[openai.types.CompletionChoice] = batch.choices
        assert choices[0].text, "empty response"