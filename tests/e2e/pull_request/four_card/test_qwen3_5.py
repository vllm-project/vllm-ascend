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
import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import DPVllmRunner, VllmRunner


# ===================== Fixture 复用模型定义 =====================
@pytest.fixture(scope="module")
def fixture_qwen35_27b_tp4():
    """27B TP4 基础模型复用实例"""
    warm_prompts = ["Hello, my name is"] * 4
    warm_max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3.5-27B",
        tensor_parallel_size=4,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
    ) as runner:
        # 预热：完成权重加载、MP进程初始化、CUDA Graph捕获
        runner.generate_greedy(warm_prompts, warm_max_tokens)
        yield runner


@pytest.fixture(scope="module")
def fixture_qwen35_35b_tp4_base():
    """35B TP4 基础无MTP模型复用实例"""
    warm_prompts = ["Hello, my name is"] * 4
    warm_max_tokens = 5
    with VllmRunner(
        "Qwen/Qwen3.5-35B-A3B",
        tensor_parallel_size=4,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
    ) as runner:
        runner.generate_greedy(warm_prompts, warm_max_tokens)
        yield runner


@pytest.fixture(scope="module")
def fixture_qwen35_35b_tp4_mtp3():
    """35B TP4 + FULL_DECODE_ONLY + MTP3 复用实例"""
    warm_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    warm_max_tokens = 20
    with VllmRunner(
        "Qwen/Qwen3.5-35B-A3B",
        tensor_parallel_size=4,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        distributed_executor_backend="mp",
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [4, 8, 12, 16],
        },
        speculative_config={
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": 3,
        },
    ) as runner:
        runner.generate_greedy(warm_prompts, warm_max_tokens)
        yield runner


@pytest.fixture(scope="module")
def fixture_qwen35_35b_dp_tp_flashcomm_mtp3():
    """DP2+TP2 + FlashComm1 + MTP3 + EP 混合并行实例"""
    test_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    max_tokens = 20
    # patch 全程包裹Runner生命周期，退出自动清理环境变量
    with (
        patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"}),
        DPVllmRunner(
            "Qwen/Qwen3.5-35B-A3B",
            data_parallel_size=2,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            max_model_len=4096,
            gpu_memory_utilization=0.8,
            distributed_executor_backend="mp",
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [4, 8, 12, 16],
            },
            speculative_config={
                "method": "qwen3_5_mtp",
                "num_speculative_tokens": 3,
            },
        ) as runner,
    ):
        runner.generate_greedy(test_prompts, max_tokens)
        yield runner


# ===================== 测试用例（全部复用Fixture，无重复初始化） =====================
def test_qwen3_5_27b_distributed_mp_tp4(fixture_qwen35_27b_tp4):
    example_prompts = ["Hello, my name is"] * 4
    max_tokens = 5
    fixture_qwen35_27b_tp4.generate_greedy(example_prompts, max_tokens)


def test_qwen3_5_35b_distributed_mp_tp4(fixture_qwen35_35b_tp4_base):
    example_prompts = ["Hello, my name is"] * 4
    max_tokens = 5
    fixture_qwen35_35b_tp4_base.generate_greedy(example_prompts, max_tokens)


def test_qwen3_5_35b_distributed_mp_tp4_full_decode_only_mtp3(fixture_qwen35_35b_tp4_mtp3):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    max_tokens = 20
    fixture_qwen35_35b_tp4_mtp3.generate_greedy(example_prompts, max_tokens)


def test_qwen3_5_35b_distributed_mp_tp4_full_decode_only_mtp3_flashcomm(fixture_qwen35_35b_dp_tp_flashcomm_mtp3):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    max_tokens = 20
    fixture_qwen35_35b_dp_tp_flashcomm_mtp3.generate_greedy(example_prompts, max_tokens)
