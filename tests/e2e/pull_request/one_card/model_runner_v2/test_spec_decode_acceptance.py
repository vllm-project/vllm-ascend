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

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.model_runner_v2.utils import (
    DEFAULT_PIECEWISE,
    FULL_DECODE_ONLY,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    PROMPTS,
    calculate_acceptance_per_pos,
)

MAIN_MODELS = ["LLM-Research/Meta-Llama-3.1-8B-Instruct"]
EGALE_MODELS = ["vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B"]
DFLASH_MAIN_MODEL = ["Qwen/Qwen3-8B"]
DFLASH_MODELS = ["z-lab/Qwen3-8B-DFlash-b16"]
DSPARK_MAIN_MODEL = ["Qwen/Qwen3-8B"]
DSPARK_MODELS = ["deepseek-ai/dspark_qwen3_8b_block7"]
MTP_MODELS = ["wemaster/deepseek_mtp_main_random_bf16"]


@pytest.mark.parametrize("model", MAIN_MODELS)
@pytest.mark.parametrize("eagle_model", EGALE_MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [False])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_egale_spec_decoding(
    model: str,
    eagle_model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    num_speculative_tokens = 3
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
            "model": eagle_model,
            "method": "eagle",
            "num_speculative_tokens": num_speculative_tokens,
        },
        compilation_config=FULL_DECODE_ONLY,
    ) as runner:
        runner.model.generate(PROMPTS, sampling_params)
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
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_dflash_spec_decoding(
    model: str,
    dflash_model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    num_speculative_tokens = 7
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
            "model": dflash_model,
            "method": "dflash",
            "num_speculative_tokens": num_speculative_tokens,
        },
        compilation_config=DEFAULT_PIECEWISE,
    ) as runner:
        runner.model.generate(PROMPTS, sampling_params)
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
@pytest.mark.parametrize("enforce_eager", [False])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dspark_spec_decoding(
    model: str,
    dspark_model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    num_speculative_tokens = 7
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
            "model": dspark_model,
            "method": "dspark",
            "num_speculative_tokens": num_speculative_tokens,
        },
        compilation_config=FULL_DECODE_ONLY,
    ) as runner:
        runner.model.generate(PROMPTS, sampling_params)
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


@pytest.mark.parametrize("model", MTP_MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [False])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mtp_spec_decoding(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    # The MTP draft head has random weights, so acceptance is ~0 and there is
    # no trained golden to compare against -- this is a smoke test (assert only
    # that the MTP MLA propose->verify loop produces output).
    num_speculative_tokens = 3
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enforce_eager=enforce_eager,
        async_scheduling=True,
        enable_expert_parallel=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
        compilation_config=DEFAULT_PIECEWISE,
    ) as runner:
        outputs = runner.model.generate(PROMPTS, sampling_params)

    assert len(outputs) == len(PROMPTS)
