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
Compare the outputs of vLLM with and without context parallel.

Run `pytest tests/compile/test_accuracy.py`.
"""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.model_utils import check_outputs_equal
from vllm_ascend.utils import vllm_version_is

MODELS = [
    "Qwen/Qwen3-8B",
    "vllm-ascend/DeepSeek-V2-Lite-W8A8",
]


@pytest.mark.skipif(vllm_version_is('0.12.0'),
                    reason="0.12.0 is not supported for context sequence.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [10])
def test_output_between_tp_and_cp(
    model: str,
    max_tokens: int,
) -> None:
    prompts = [
        "The president of the United States is", "The capital of France is"
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    if model == "vllm-ascend/DeepSeek-V2-Lite-W8A8":
        with VllmRunner(
                model,
                tensor_parallel_size=2,
                decode_context_parallel_size=2,
                prefill_context_parallel_size=2,
                max_model_len=1024,
                enable_expert_parallel=True,
                enforce_eager=True,
                # quantization="ascend",
        ) as runner:
            vllm_context_parallel_outputs = runner.model.generate(
                prompts, sampling_params)

        with VllmRunner(
                model,
                tensor_parallel_size=4,
                max_model_len=1024,
                enable_expert_parallel=True,
                enforce_eager=True,
                # quantization="ascend",
        ) as runner:
            vllm_eager_outputs = runner.model.generate(prompts,
                                                       sampling_params)
    else:
        with VllmRunner(
                model,
                tensor_parallel_size=1,
                prefill_context_parallel_size=2,
                max_model_len=1024,
                enforce_eager=True,
        ) as runner:
            vllm_context_parallel_outputs = runner.model.generate(
                prompts, sampling_params)

        with VllmRunner(
                model,
                tensor_parallel_size=2,
                prefill_context_parallel_size=1,
                max_model_len=1024,
                enforce_eager=True,
        ) as runner:
            vllm_eager_outputs = runner.model.generate(prompts,
                                                       sampling_params)
    vllm_context_parallel_outputs_list = []
    for output in vllm_context_parallel_outputs:
        vllm_context_parallel_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_context_parallel_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_context_parallel_outputs",
    )
