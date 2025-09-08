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
#
"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/multicard/test_torchair_graph_mode.py`.
"""
import os
from typing import Dict

from tests.e2e.conftest import VllmRunner
from vllm_ascend.ascend_config import clear_ascend_config

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"


def _deepseek_torchair_test_fixture(
    additional_config: Dict,
    *,
    tensor_parallel_size=16,
):
    example_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    kwargs = {
        "refresh": True,
    }
    additional_config.update(**kwargs)

    with VllmRunner("vllm-ascend/DeepSeek-V3-W8A8",
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=0.9,
                    max_model_len=256,
                    max_num_seqs=8,
                    distributed_executor_backend="mp",
                    enable_expert_parallel=True,
                    enforce_eager=False,
                    additional_config=additional_config) as ref_llm:
        ref_outputs = ref_llm.generate_greedy(example_prompts, 20)
    clear_ascend_config()
    with VllmRunner("vllm-ascend/DeepSeek-V3-W8A8",
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=0.9,
                    max_model_len=256,
                    max_num_seqs=8,
                    distributed_executor_backend="mp",
                    enable_expert_parallel=True,
                    enforce_eager=False,
                    speculative_config={
                        "method": "deepseek_mtp",
                        "num_speculative_tokens": 1,
                    },
                    additional_config=additional_config) as spec_llm:
        spec_outputs = spec_llm.generate_greedy(example_prompts, 20)

    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        ref_token_ids = ref_output[0][0]
        spec_token_ids = spec_output[0][0]
        if ref_token_ids == spec_token_ids[:len(ref_token_ids)]:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output[1][0]}")
            print(f"spec_output: {spec_output[1][0]}")

    # Heuristic: expect at least 66% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches > int(0.66 * len(ref_outputs))
    del spec_llm
    clear_ascend_config()


def test_e2e_deepseekv3_mtp_with_torchair():
    additional_config = {
        "torchair_graph_config": {
            "enabled": True,
            "use_cached_graph": False,
            "graph_batch_sizes": [1],
        },
    }
    _deepseek_torchair_test_fixture(additional_config)


def test_e2e_deepseekv3_mtp_with_torchair_ms_mla():
    additional_config = {
        "torchair_graph_config": {
            "enabled": True,
            "enable_multistream_mla": True,
            "use_cached_graph": False,
            "graph_batch_sizes": [1],
        },
    }
    _deepseek_torchair_test_fixture(additional_config)


def test_e2e_deepseekv3_mtp_with_torchair_v1scheduler():
    additional_config = {
        "torchair_graph_config": {
            "enabled": True,
            "use_cached_graph": False,
            "graph_batch_sizes": [1],
        },
    }
    _deepseek_torchair_test_fixture(additional_config)