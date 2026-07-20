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
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.model_runner_v2.utils import calculate_acceptance_per_pos

MODELS = ["Qwen/Qwen3-0.6B", "vllm-ascend/DeepSeek-V2-Lite-W8A8"]

MAIN_MODELS = ["LLM-Research/Meta-Llama-3.1-8B-Instruct"]
EGALE_MODELS = ["vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B"]
DFLASH_MAIN_MODEL = ["Qwen/Qwen3-8B"]
DFLASH_MODELS = ["z-lab/Qwen3-8B-DFlash-b16"]
DSPARK_MAIN_MODEL = ["Qwen/Qwen3-8B"]
DSPARK_MODELS = ["deepseek-ai/dspark_qwen3_8b_block7"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_dense_eager_mode(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

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
        enforce_eager=enforce_eager,
        async_scheduling=True,
    ) as runner:
        runner.generate(prompts, sampling_params)


@pytest.mark.parametrize("model", MAIN_MODELS)
@pytest.mark.parametrize("eagle_model", EGALE_MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize(
    "compilation_config",
    [
        pytest.param(
            {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8]},
            id="full_decode_only",
        ),
        pytest.param({}, id="default_full_and_piecewise"),
    ],
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_egale_spec_decoding(
    model: str,
    eagle_model: str,
    max_tokens: int,
    enforce_eager: bool,
    compilation_config: dict,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    num_speculative_tokens = 3
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
            "model": eagle_model,
            "method": "eagle",
            "num_speculative_tokens": num_speculative_tokens,
        },
        compilation_config=compilation_config,
    ) as runner:
        runner.model.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()

    acceptance_per_pos = calculate_acceptance_per_pos(
        metrics,
        num_speculative_tokens,
        Counter,
        Vector,
    )
    golden = [0.43, 0.13, 0.05]
    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"


@pytest.mark.parametrize("model", DFLASH_MAIN_MODEL)
@pytest.mark.parametrize("dflash_model", DFLASH_MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize(
    "compilation_config",
    [
        pytest.param(
            {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8]},
            id="full_decode_only",
        ),
        pytest.param({}, id="default_full_and_piecewise"),
    ],
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_dflash_spec_decoding(
    model: str,
    dflash_model: str,
    max_tokens: int,
    enforce_eager: bool,
    compilation_config: dict,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    num_speculative_tokens = 7
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
            "model": dflash_model,
            "method": "dflash",
            "num_speculative_tokens": num_speculative_tokens,
        },
        compilation_config=compilation_config,
    ) as runner:
        runner.model.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()

    acceptance_per_pos = calculate_acceptance_per_pos(
        metrics,
        num_speculative_tokens,
        Counter,
        Vector,
    )

    golden = [0.51, 0.16, 0.07, 0.07, 0.01, 0.01, 0.0]
    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"


@pytest.mark.parametrize("model", DSPARK_MAIN_MODEL)
@pytest.mark.parametrize("dspark_model", DSPARK_MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_dspark_spec_decoding(
    model: str,
    dspark_model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    num_speculative_tokens = 7
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
            "model": dspark_model,
            "method": "dspark",
            "num_speculative_tokens": num_speculative_tokens,
        },
    ) as runner:
        runner.model.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()

    acceptance_per_pos = calculate_acceptance_per_pos(
        metrics,
        num_speculative_tokens,
        Counter,
        Vector,
    )
    golden = [0.84, 0.48, 0.32, 0.20, 0.09, 0.09, 0.02]
    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize(
    "compilation_config",
    [
        pytest.param({"cudagraph_mode": "FULL_DECODE_ONLY"}, id="full_decode_only"),
        pytest.param({}, id="default_full_and_piecewise"),
    ],
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_qwen3_dense_graph_mode(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
    compilation_config: dict,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        compilation_config=compilation_config,
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params)

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
