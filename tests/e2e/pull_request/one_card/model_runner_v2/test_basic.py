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
from vllm_ascend.utils import vllm_version_is

MODELS = ["Qwen/Qwen3-0.6B", "vllm-ascend/DeepSeek-V2-Lite-W8A8"]

MAIN_MODELS = ["LLM-Research/Meta-Llama-3.1-8B-Instruct"]
EGALE_MODELS = ["vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B"]
DFLASH_MAIN_MODEL = ["Qwen/Qwen3-8B"]
DFLASH_MODELS = ["z-lab/Qwen3-8B-DFlash-b16"]

pytestmark = pytest.mark.skipif(
    vllm_version_is("0.23.0"),
    reason="v2 model runner patches not supported on v0.23.0",
)


@pytest.mark.skipif(True, reason="Fix me, it's broken after CANN and trition-ascend are upgraded.")
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
        runner.model.generate(prompts, sampling_params)


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

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        async_scheduling=True,
        speculative_config={
            "model": eagle_model,
            "method": "eagle",
            "num_speculative_tokens": 3,
        },
        compilation_config=compilation_config,
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@pytest.mark.skipif(True, reason="Fix me, it's broken because of vllm new commit.")
@pytest.mark.parametrize("model", DFLASH_MAIN_MODEL)
@pytest.mark.parametrize("dflash_model", DFLASH_MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_dflash_spec_decoding(
    model: str,
    dflash_model: str,
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
            "model": dflash_model,
            "method": "dflash",
            "num_speculative_tokens": num_speculative_tokens,
        },
    ) as runner:
        runner.model.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()

    num_drafts = 0
    acceptance_counts = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 60)
    for i in range(num_speculative_tokens):
        rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"acceptance at token {i}: {rate:.4f}")
    print("-" * 60)


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
