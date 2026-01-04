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
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/e2e/multicard/test_qwen3_moe.py`.
"""

import json
import os
from unittest.mock import patch

import openai
import pytest
from modelscope import snapshot_download  # type: ignore
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, VllmRunner


@patch.dict(os.environ, {"HCCL_BUFFSIZE": "1024"})
def test_qwen3_moe_distributed_mp_tp2_ep():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_qwen3_moe_w8a8_distributed_tp2():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            snapshot_download("vllm-ascend/Qwen3-30B-A3B-W8A8"),
            max_model_len=8192,
            tensor_parallel_size=2,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_qwen3_moe_distributed_aiv_tp2():
    os.environ['HCCL_OP_EXPANSION_MODE'] = 'AIV'
    example_prompts = [
        "Hello, my name is",
    ]
    dtype = "auto"
    max_tokens = 5
    with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            dtype=dtype,
            tensor_parallel_size=2,
            cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.asyncio
async def test_qwen3_moe_w8a8_distributed_tp2_ep_dynamic_eplb():
    model = "vllm-ascend/Qwen3-30B-A3B-W8A8"
    port = get_open_port()
    server_args = [
        "--max_model_len", "8192", "--tensor_parallel_size", "2",
        "--enable_expert_parallel", "--quantization", "ascend", "--port",
        str(port), "--enforce_eager"
    ]
    env_dict = {"HCCL_BUFFSIZE": "1024"}
    with RemoteOpenAIServer(model,
                            server_args,
                            server_port=port,
                            auto_port=False,
                            env_dict=env_dict) as server:
        client = server.get_async_client()
        batch = await client.completions.create(model=model,
                                                prompt="What is deeplearning?",
                                                max_tokens=300,
                                                temperature=0,
                                                top_p=1.0,
                                                n=1)
        gt_choices: list[openai.types.CompletionChoice] = batch.choices

    # dynamic eplb test
    # Since pytest runs as a daemon, it conflicts with the dynamic eplb manager
    # during initialization in offline mode, so the online mode is used instead.
    env_dict.update({"DYNAMIC_EPLB": "true"})
    additional_config = {
        "dynamic_eplb": True,
        "num_iterations_eplb_update": 100,
        "num_wait_worker_iterations": 20
    }
    server_args.extend(["--additional-config", json.dumps(additional_config)])
    with RemoteOpenAIServer(model,
                            server_args,
                            server_port=port,
                            auto_port=False,
                            env_dict=env_dict) as server:
        client = server.get_async_client()
        batch = await client.completions.create(model=model,
                                                prompt="What is deeplearning?",
                                                max_tokens=300,
                                                temperature=0,
                                                top_p=1.0,
                                                n=1)
        eplb_choices: list[openai.types.CompletionChoice] = batch.choices
    assert gt_choices[0].text == eplb_choices[
        0].text, f"{gt_choices[0].text=} \n {eplb_choices[0].text=}"
