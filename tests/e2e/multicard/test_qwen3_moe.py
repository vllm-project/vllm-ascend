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

import os

from modelscope import snapshot_download  # type: ignore

from tests.e2e.conftest import VllmRunner


def test_models_distributed_Qwen3_MOE_TP2():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_models_distributed_Qwen3_MOE_TP2_WITH_EP():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
            enforce_eager=False,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_models_distributed_Qwen3_MOE_W8A8():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
            snapshot_download("vllm-ascend/Qwen3-30B-A3B-W8A8"),
            max_model_len=8192,
            tensor_parallel_size=2,
            quantization="ascend",
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_models_distributed_Qwen3_MOE_TP2_WITH_ACLGRAPH_AIV():
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
            enforce_eager=False,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_models_distributed_Qwen3_MOE_TP2_WITH_ACLGRAPH():
    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']
    example_prompts = [
        "Hello, my name is",
    ]
    dtype = "auto"
    max_tokens = 5
    with VllmRunner(
            "Qwen/Qwen3-30B-A3B",
            dtype=dtype,
            tensor_parallel_size=2,
            enforce_eager=False,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


def test_Qwen3_235b_all2allv_mc2_quant(monkeypatch):
    """Test Qwen3-235B with all2all sequence and multi-card configuration."""
    # Set environment variables similar to the startup command
    monkeypatch.setenv('VLLM_USE_V1', '1')
    monkeypatch.setenv('VLLM_VERSION', '0.10.1.1')
    monkeypatch.setenv('VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ', '1')
    monkeypatch.setenv('TASK_QUEUE_ENABLE', '2')
    monkeypatch.setenv('PYTORCH_NPU_ALLOC_CONF', 'expandable_segments:True')
    monkeypatch.setenv('ACL_STREAM_TIMEOUT', '340000')
    monkeypatch.setenv('HCCL_OP_EXPANSION_MODE', 'AIV')
    monkeypatch.setenv('HCCL_OP_BASE_FFTS_MODE_ENABLE', 'true')

    example_prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In the field of artificial intelligence,",
    ]
    max_tokens = 32

    # Additional config matching the startup command
    additional_config = {
        "torchair_graph_config": {
            "enabled": True,
            "use_cached_graph": False,
            "graph_batch_sizes_init": False,
            "graph_batch_sizes": [1, 4, 8, 16, 24]
        },
        "ascend_scheduler_config": {
            "enabled": True
        },
        "refresh": True
    }

    with VllmRunner(
            "vllm-ascend/Qwen3-235B-A22B-W8A8",  # Use quantized model path
            tensor_parallel_size=4,
            data_parallel_size=4,
            enable_expert_parallel=True,
            max_model_len=35840,
            max_num_seqs=24,
            max_num_batched_tokens=35840,
            gpu_memory_utilization=0.95,
            quantization="ascend",
            enforce_eager=False,
            distributed_executor_backend="mp",
            additional_config=additional_config,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
