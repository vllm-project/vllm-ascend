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

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal


def test_models_distributed_Qwen3_MOE_TP2_WITH_FULL_DECODE_ONLY():
    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]
    model = "Qwen/Qwen3-30B-A3B"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    enforce_eager=False,
                    compilation_config={
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
                    }) as runner:
        vllm_fullgraph_outputs = runner.model.generate(prompts,
                                                       sampling_params)

    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            enforce_eager=False,
    ) as runner:
        vllm_eager_outputs = runner.model.generate(prompts, sampling_params)

    vllm_fullgraph_outputs_list = []
    for output in vllm_fullgraph_outputs:
        vllm_fullgraph_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_fullgraph_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_fullgraph_outputs",
    )


def test_models_distributed_Qwen3_MOE_TP2_WITH_FULL():
    if 'HCCL_OP_EXPANSION_MODE' in os.environ:
        del os.environ['HCCL_OP_EXPANSION_MODE']
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]
    model = "Qwen/Qwen3-30B-A3B"
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
    with VllmRunner(model,
                    max_model_len=1024,
                    tensor_parallel_size=2,
                    enforce_eager=False,
                    compilation_config={
                        "cudagraph_mode": "FULL",
                        "cudagraph_capture_sizes": [4, 8, 24, 48, 60]
                    }) as runner:
        vllm_fullgraph_outputs = runner.model.generate(prompts,
                                                       sampling_params)

    with VllmRunner(
            model,
            max_model_len=1024,
            tensor_parallel_size=2,
            enforce_eager=False,
    ) as runner:
        vllm_eager_outputs = runner.model.generate(prompts, sampling_params)

    vllm_fullgraph_outputs_list = []
    for output in vllm_fullgraph_outputs:
        vllm_fullgraph_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_fullgraph_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_fullgraph_outputs",
    )
