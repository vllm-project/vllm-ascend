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

Run `pytest tests/test_offline_inference.py`.
"""
import os
from unittest.mock import patch

import pytest
from modelscope import snapshot_download  # type: ignore
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"

QWEN_DENSE_MODELS = [
    "vllm-ascend/Qwen3-8B-W8A8", "vllm-ascend/Qwen2.5-0.5B-Instruct-W8A8"
]


def test_models_distributed_QwQ():
    example_prompts = [
        "Hello, my name is",
    ]
    dtype = "half"
    max_tokens = 5
    with VllmRunner(
            "Qwen/QwQ-32B",
            dtype=dtype,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            enforce_eager=True,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_models_distributed_DeepSeek_multistream_moe():
    example_prompts = [
        "Hello, my name is",
    ]
    dtype = "half"
    max_tokens = 5
    with VllmRunner(
            "vllm-ascend/DeepSeek-V3-Pruning",
            dtype=dtype,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            additional_config={
                "torchair_graph_config": {
                    "enabled": True,
                    "enable_multistream_moe": True,
                },
                "ascend_scheduler_config": {
                    "enabled": True,
                },
                "refresh": True,
            },
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_models_distributed_Qwen3_W8A8():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
            snapshot_download("vllm-ascend/Qwen3-8B-W8A8"),
            max_model_len=8192,
            dtype="auto",
            tensor_parallel_size=2,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_models_distributed_Qwen3_W4A8DYNAMIC():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
            snapshot_download("vllm-ascend/Qwen3-8B-W4A8"),
            max_model_len=8192,
            dtype="auto",
            tensor_parallel_size=2,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@patch.dict(os.environ, {"VLLM_ASCEND_MLA_PA": "1"})
def test_models_distributed_DeepSeek_W4A8DYNAMIC():
    prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            snapshot_download("vllm-ascend/DeepSeek-V3-W4A8-Pruing"),
            dtype="auto",
            tensor_parallel_size=2,
            quantization="ascend",
            enforce_eager=True,
            enable_expert_parallel=True,
            additional_config={
                "torchair_graph_config": {
                    "enabled": False,
                },
                "ascend_scheduler_config": {
                    "enabled": True,
                }
            },
    ) as vllm_model:
        vllm_model.generate_greedy(prompts, max_tokens)


def test_sp_for_qwen3_moe() -> None:
    example_prompts = [
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(max_tokens=5,
                                     temperature=0.0,
                                     top_k=50,
                                     top_p=0.9)

    with VllmRunner(snapshot_download("Qwen/Qwen3-30B-A3B"),
                    dtype="auto",
                    tensor_parallel_size=2,
                    distributed_executor_backend="mp",
                    compilation_config={
                        "pass_config": {
                            "enable_sequence_parallelism": True
                        }
                    },
                    enable_expert_parallel=True,
                    enforce_eager=True) as vllm_model:
        vllm_model.generate(example_prompts, sampling_params)


@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("model", QWEN_DENSE_MODELS)
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE": "1"})
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM": "1"})
def test_models_distributed_Qwen_Dense_with_flashcomm_v1(model, enforce_eager):
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
            snapshot_download(model),
            max_model_len=8192,
            enforce_eager=enforce_eager,
            dtype="auto",
            tensor_parallel_size=2,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("model", QWEN_DENSE_MODELS)
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE": "1"})
@patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_PREFETCH_MLP": "1"})
def test_models_distributed_Qwen_Dense_with_prefetch_mlp_weight(
        model, enforce_eager):
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5

    with VllmRunner(
            snapshot_download(model),
            max_model_len=8192,
            enforce_eager=enforce_eager,
            dtype="auto",
            tensor_parallel_size=2,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
