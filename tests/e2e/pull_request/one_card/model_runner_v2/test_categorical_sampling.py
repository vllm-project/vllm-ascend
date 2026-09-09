# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="",
    parallel="",
    deploy="pd_mix",
    hardware="",
    quantization="BF16",
    graph_mode="eager",
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_random_sampling() -> None:
    sampling_params = SamplingParams(
        temperature=0.7,
        seed=17,
        max_tokens=8,
        ignore_eos=True,
    )
    prompts = ["The capital of France is", "A short story begins with"]

    with VllmRunner(
        MODEL,
        max_model_len=256,
        max_num_seqs=len(prompts),
        enforce_eager=True,
        use_fp64_gumbel=True,
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params=sampling_params)

    assert len(outputs) == len(prompts)
    assert all(len(output.outputs[0].token_ids) == sampling_params.max_tokens for output in outputs)
