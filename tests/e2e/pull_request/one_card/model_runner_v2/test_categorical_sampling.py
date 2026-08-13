# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter

from tests.e2e.conftest import VllmRunner

MODEL = "Qwen/Qwen3-0.6B"
NUM_SPECULATIVE_TOKENS = 3


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


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="spec_decode",
    parallel="",
    deploy="pd_mix",
    hardware="",
    quantization="BF16",
    graph_mode="eager",
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_model_runner_v2_fp64_rejection_sampling() -> None:
    sampling_params = SamplingParams(
        temperature=0.7,
        seed=23,
        max_tokens=32,
        ignore_eos=True,
    )
    prompts = [
        "The capital of France is Berlin. The capital of France is",
        "One two three four. One two three four. One two three",
    ]

    with VllmRunner(
        MODEL,
        max_model_len=256,
        max_num_seqs=len(prompts),
        enforce_eager=True,
        disable_log_stats=False,
        use_fp64_gumbel=True,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 5,
        },
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params=sampling_params)
        metrics = runner.model.get_metrics()

    assert len(outputs) == len(prompts)
    assert all(len(output.outputs[0].token_ids) == sampling_params.max_tokens for output in outputs)

    counters = {
        metric.name: metric.value
        for metric in metrics
        if isinstance(metric, Counter)
        and metric.name
        in {
            "vllm:spec_decode_num_drafts",
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens",
        }
    }
    num_drafts = counters.get("vllm:spec_decode_num_drafts", 0)
    num_draft_tokens = counters.get("vllm:spec_decode_num_draft_tokens", 0)
    num_accepted_tokens = counters.get("vllm:spec_decode_num_accepted_tokens", 0)
    assert num_drafts > 0, "N-gram speculative decoding did not produce a draft"
    assert num_draft_tokens > 0
    assert num_accepted_tokens < num_draft_tokens, "No draft token was rejected"
