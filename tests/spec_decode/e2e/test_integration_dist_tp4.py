#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/tests/spec_decode/e2e/test_integration_dist_tp4.py
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
"""Tests which cover integration of the speculative decoding framework with
tensor parallelism.
"""

import openai
import pytest
import torch  # noqa: F401
import torch_npu

from .conftest import run_equality_correctness_test_tp

MAIN_MODEL = "JackFram/llama-68m"
SPEC_MODEL = "JackFram/llama-68m"


@pytest.mark.skipif(torch_npu.npu.device_count() < 4,
                    reason="Need at least 4 NPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[
        # Skip cuda graph recording for fast test.
        "--enforce_eager",
        "--tensor-parallel-size",
        "4",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [
    [
        "--speculative-model",
        f"{SPEC_MODEL}",
        "--num-speculative-tokens",
        "5",
    ],
])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        #TODO(wooyeon): add spec_draft_dp=2 case
        [
            "--speculative-draft-tensor-parallel-size",
            "1",
        ],
    ])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seed", [1])
def test_draft_model_tp_lt_target_model_tp4(common_llm_kwargs,
                                            per_test_common_llm_kwargs,
                                            baseline_llm_kwargs,
                                            test_llm_kwargs, batch_size: int,
                                            seed: int):
    """Verify spec decode works well with smaller tp for draft models.
    """
    run_equality_correctness_test_tp(MAIN_MODEL,
                                     common_llm_kwargs,
                                     per_test_common_llm_kwargs,
                                     baseline_llm_kwargs,
                                     test_llm_kwargs,
                                     batch_size,
                                     max_output_len=32,
                                     seed=seed,
                                     temperature=0.0)


@pytest.mark.skipif(torch_npu.npu.device_count() < 4,
                    reason="Need at least 4 NPUs to run the test.")
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [[

        # Skip cuda graph recording for fast test.
        "--enforce-eager",
        "--tensor-parallel-size",
        "4",
    ]])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [[]])
@pytest.mark.parametrize("baseline_llm_kwargs", [[]])
@pytest.mark.parametrize(
    "test_llm_kwargs",
    [
        [
            "--speculative-model",
            f"{SPEC_MODEL}",
            "--num-speculative-tokens",
            "5",

            # Artificially limit the draft model max model len; this forces vLLM
            # to skip speculation once the sequences grow beyond 32-k tokens.
            "--speculative-max-model-len",
            "32",
        ],
    ])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize(
    "output_len",
    [
        # This must be a good bit larger than speculative_max_model_len so that
        # we can test the case where all seqs are skipped, but still small to
        # ensure fast test.
        64,
    ])
@pytest.mark.parametrize("seed", [1])
def test_skip_speculation(common_llm_kwargs, per_test_common_llm_kwargs,
                          baseline_llm_kwargs, test_llm_kwargs,
                          batch_size: int, output_len: int, seed: int):
    """Verify job failure with RuntimeError when all sequences skip speculation.
    We do this by setting the max model len of the draft model to an
    artificially low value, such that when the sequences grow beyond it, they
    are skipped in speculative decoding.

    TODO: fix it to pass without raising Error. (#5814)
    """
    with pytest.raises(
        (openai.APIConnectionError, openai.InternalServerError)):
        run_equality_correctness_test_tp(MAIN_MODEL,
                                         common_llm_kwargs,
                                         per_test_common_llm_kwargs,
                                         baseline_llm_kwargs,
                                         test_llm_kwargs,
                                         batch_size,
                                         output_len,
                                         seed,
                                         temperature=0.0)
