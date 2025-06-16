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
#
"""
Compare the outputs of vLLM with and without aclgraph.

Run `pytest tests/multicard/test_data_parallel.py`.
"""

import os

import pytest

from tests.conftest import VllmRunner
from tests.model_utils import check_outputs_equal

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]


@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "0",
                    reason="Data parallel only support on v1")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
def test_data_parallel_correctness(
    model: str,
    max_tokens: int,
) -> None:
    example_prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]

    with VllmRunner(model_name=model,
                    max_model_len=1024,
                    max_num_seqs=16,
                    data_parallel_size=2,
                    distributed_executor_backend="mp") as vllm_model:
        vllm_dp_outputs = vllm_model.generate_greedy(example_prompts,
                                                     max_tokens)

    with VllmRunner(
            model_name=model,
            max_model_len=1024,
            max_num_seqs=16,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=vllm_outputs,
        outputs_1_lst=vllm_dp_outputs,
        name_0="vllm_outputs",
        name_1="vllm_dp_outputs",
    )
