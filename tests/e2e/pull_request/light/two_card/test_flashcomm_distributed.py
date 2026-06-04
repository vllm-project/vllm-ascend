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

Run `pytest tests/e2e/pull_request/full/two_cards/test_flashcomm_distributed.py`.
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.config import KVTransferConfig

from tests.e2e.conftest import VllmRunner

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

QWEN_DENSE_MODELS = [
    "vllm-ascend/Qwen3-0.6B-W8A8",
]


@pytest.mark.skip(reason="test is broken, fix me")
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
@patch.dict(os.environ, {"VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "1"})
def test_qwen3_moe_fc2_oshard_tp2() -> None:
    example_prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0, top_k=50, top_p=0.9)

    with VllmRunner(
        "Qwen/Qwen3-30B-A3B",
        dtype="auto",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        enable_expert_parallel=True,
        enforce_eager=True,
        additional_config={"layer_sharding": ["o_proj"]},
        kv_transfer_config=KVTransferConfig(kv_role="kv_producer"),
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
def test_deepseek_v2_lite_fc1_tp2() -> None:
    example_prompts = [
        "test" * 1001,
    ]
    sampling_params = SamplingParams(max_tokens=5, temperature=0.0, top_k=50, top_p=0.9)
    with VllmRunner(
        "vllm-ascend/DeepSeek-V2-Lite-W8A8",
        dtype="auto",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        enable_expert_parallel=True,
        enforce_eager=True,
        quantization="ascend",
    ) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@pytest.mark.skip(reason="test is broken, fix me")
@pytest.mark.parametrize("model", QWEN_DENSE_MODELS)
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
def test_qwen3_dense_fc1_tp2(model):
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
        model,
        max_model_len=8192,
        dtype="auto",
        tensor_parallel_size=2,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.skip(reason="test is broken, fix me")
@pytest.mark.parametrize("model", QWEN_DENSE_MODELS)
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"})
def test_qwen3_dense_prefetch_mlp_weight_tp2(model):
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
        model,
        max_model_len=8192,
        dtype="auto",
        tensor_parallel_size=2,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        quantization="ascend",
        additional_config={"weight_prefetch_config": {"enabled": True}},
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
