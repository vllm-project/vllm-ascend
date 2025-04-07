#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/tests/spec_decode/e2e/test_logprobs.py
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

from itertools import cycle

import pytest
from vllm import SamplingParams

from ..utils import maybe_enable_chunked_prefill
from .conftest import run_equality_correctness_test

MAIN_MODEL = "JackFram/llama-160m"
SPEC_MODEL = "JackFram/llama-68m"


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model_name": MAIN_MODEL,

        # Skip cuda graph recording for fast test.
        "enforce_eager": True
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs",
                         [{
                             "speculative_model": SPEC_MODEL,
                             "num_speculative_tokens": 3,
                             "disable_logprobs_during_spec_decoding": False,
                         }, {
                             "speculative_model": SPEC_MODEL,
                             "num_speculative_tokens": 3,
                             "disable_logprobs_during_spec_decoding": True,
                         }])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        7,
    ])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("logprobs", [1, 6])
@pytest.mark.parametrize("prefill_chunk_size", [-1, 4, 12])
def test_logprobs_equality(vllm_runner, common_llm_kwargs,
                           per_test_common_llm_kwargs, baseline_llm_kwargs,
                           test_llm_kwargs, batch_size: int, output_len: int,
                           seed: int, logprobs: int, prefill_chunk_size: int):
    """Verify output logprobs are equal with and without speculative decoding,
        as well as with and without chunked prefill.
    """
    maybe_enable_chunked_prefill(prefill_chunk_size, common_llm_kwargs)
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  output_len,
                                  seed,
                                  temperature=0.0,
                                  logprobs=logprobs,
                                  prompt_logprobs=logprobs,
                                  disable_logprobs=test_llm_kwargs[
                                      'disable_logprobs_during_spec_decoding'])


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model_name": SPEC_MODEL,

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs",
                         [{
                             "speculative_model": MAIN_MODEL,
                             "num_speculative_tokens": 3,
                             "disable_logprobs_during_spec_decoding": False,
                         }, {
                             "speculative_model": MAIN_MODEL,
                             "num_speculative_tokens": 6,
                             "disable_logprobs_during_spec_decoding": False,
                         }])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("logprobs", [1, 6])
def test_logprobs_different_k(vllm_runner, common_llm_kwargs,
                              per_test_common_llm_kwargs, baseline_llm_kwargs,
                              test_llm_kwargs, batch_size: int,
                              output_len: int, seed: int, logprobs: int):
    """Veriy logprob greedy equality with different speculation lens.
    """
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  output_len,
                                  seed,
                                  temperature=0.0,
                                  logprobs=logprobs,
                                  disable_logprobs=test_llm_kwargs[
                                      'disable_logprobs_during_spec_decoding'])


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model_name": SPEC_MODEL,

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [{
        "speculative_model": MAIN_MODEL,
        "num_speculative_tokens": 3,
        "disable_logprobs_during_spec_decoding": False,

        # Artificially limit the draft model max model len; this forces vLLM
        # to skip speculation once the sequences grow beyond 32-k tokens.
        "speculative_max_model_len": 32,
    }])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("logprobs", [1])
def test_logprobs_when_skip_speculation(vllm_runner, common_llm_kwargs,
                                        per_test_common_llm_kwargs,
                                        baseline_llm_kwargs, test_llm_kwargs,
                                        batch_size: int, output_len: int,
                                        seed: int, logprobs: int):
    """Verify logprobs greedy equality when some sequences skip speculation.
    """
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  output_len,
                                  seed,
                                  temperature=0.0,
                                  logprobs=logprobs,
                                  disable_logprobs=test_llm_kwargs[
                                      'disable_logprobs_during_spec_decoding'])


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model_name": SPEC_MODEL,

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs",
                         [{
                             "speculative_model": MAIN_MODEL,
                             "num_speculative_tokens": 3,
                             "disable_logprobs_during_spec_decoding": False,
                         }])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("logprobs", [6])
def test_logprobs_temp_1(vllm_runner, common_llm_kwargs,
                         per_test_common_llm_kwargs, baseline_llm_kwargs,
                         test_llm_kwargs, batch_size: int, output_len: int,
                         seed: int, logprobs: int):
    """Verify at least one logprob result has num_logprobs+1, which tests the
    case where the sampled token is not in top-k logprobs.

    Ideally, this test should validate equality with non-spec by getting
    logprobs. This is left as future improvement.
    """
    temperature = 1.0

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
        logprobs=logprobs,
    )

    sd_args = {
        **common_llm_kwargs,
        **per_test_common_llm_kwargs,
        **test_llm_kwargs,
    }

    with vllm_runner(**sd_args) as vllm_model:
        sd_outputs = vllm_model.generate_w_logprobs(prompts, sampling_params)

    num_returned_logprobs = [
        len(seq_logprobs) for seq_logprobs in sd_outputs[-1]
    ]

    # Assert one of the returned logprobs has > num_logprobs (indicating the
    # sampled token is not in top-k).
    assert any(
        [num_returned > logprobs for num_returned in num_returned_logprobs])


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        "model_name": MAIN_MODEL,
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,
    }])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs",
                         [{
                             "speculative_model": SPEC_MODEL,
                             "num_speculative_tokens": 3,
                             "disable_logprobs_during_spec_decoding": True,
                         }])
@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize(
    "output_len",
    [
        # Use smaller output len for fast test.
        32,
    ])
@pytest.mark.parametrize("logprobs", [0])
def test_logprobs_disabled(vllm_runner, common_llm_kwargs,
                           per_test_common_llm_kwargs, baseline_llm_kwargs,
                           test_llm_kwargs, batch_size: int, output_len: int,
                           seed: int, logprobs: int):
    """Check the behavior when logprobs are disabled.
    Token choices should match with the base model.
    """
    run_equality_correctness_test(vllm_runner,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size,
                                  output_len,
                                  seed,
                                  temperature=0.0,
                                  logprobs=logprobs,
                                  disable_logprobs=test_llm_kwargs[
                                      'disable_logprobs_during_spec_decoding'])
