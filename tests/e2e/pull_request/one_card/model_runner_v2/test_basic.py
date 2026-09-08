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

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.model_runner_v2.utils import (
    DEFAULT_PIECEWISE,
    FULL_DECODE_ONLY,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    PROMPTS,
)

DENSE_EAGER_MODELS = ["Qwen/Qwen3-0.6B"]
DEEPSEEK_W8A8_MODEL = "vllm-ascend/DeepSeek-V2-Lite-W8A8"

# One graph mode per spec method lives in test_spec_decode.py; both modes
# are still covered here on Qwen3-0.6B. DeepSeek-V2-Lite is a single
# graph-mode smoke (no output golden).
DENSE_GRAPH_CASES = [
    pytest.param("Qwen/Qwen3-0.6B", FULL_DECODE_ONLY, id="qwen3-full_decode_only"),
    pytest.param("Qwen/Qwen3-0.6B", DEFAULT_PIECEWISE, id="qwen3-default_full_and_piecewise"),
    pytest.param(DEEPSEEK_W8A8_MODEL, FULL_DECODE_ONLY, id="dsv2-full_decode_only"),
]


@pytest.mark.parametrize("model", DENSE_EAGER_MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_dense_eager_mode(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=0.95,
        top_k=10,
        repetition_penalty=1.03,
        logprobs=2,
        prompt_logprobs=2,
        logit_bias={0: -1.0, 1: 0.5},
        min_p=0.01,
        bad_words=["the", " the"],
    )
    with VllmRunner(
        model,
        max_model_len=1024,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enforce_eager=enforce_eager,
        async_scheduling=True,
    ) as runner:
        runner.generate(PROMPTS, sampling_params)


@pytest.mark.parametrize("model, compilation_config", DENSE_GRAPH_CASES)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [False])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_dense_graph_mode(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
    compilation_config: dict,
) -> None:
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enforce_eager=enforce_eager,
        compilation_config=compilation_config,
    ) as runner:
        outputs = runner.model.generate(PROMPTS, sampling_params)

    if model != "Qwen/Qwen3-0.6B":
        return

    expected_outputs = [
        " Lina. I'm a 22-year-old student from China.",
        " the same as the president of the United Nations. This is because the president",
        " Paris. The capital of France is also the capital of the Republic of France",
        " not just about the technology itself but also about the human aspect-how we",
    ]

    matches = 0
    misses = 0
    for output, expected_output in zip(outputs, expected_outputs):
        if output.outputs[0].text[:10] == expected_output[:10]:
            matches += 1
        else:
            misses += 1
            print(f"output: {output.outputs[0].text}")
            print(f"expected_output: {expected_output}")

    assert misses == 0
