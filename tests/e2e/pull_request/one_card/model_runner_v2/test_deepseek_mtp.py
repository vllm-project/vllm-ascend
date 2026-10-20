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

MTP_MODELS = ["wemaster/deepseek_mtp_main_random_bf16"]


@pytest.mark.parametrize("model", MTP_MODELS)
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
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mtp_spec_decoding(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
    compilation_config: dict,
) -> None:
    # The MTP draft head has random weights, so acceptance is ~0 and there is
    # no trained golden to compare against -- this is a smoke test (assert only
    # that the MTP MLA propose->verify loop produces output).
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
        async_scheduling=True,
        enable_expert_parallel=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
        compilation_config=compilation_config,
    ) as runner:
        outputs = runner.model.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)


@pytest.mark.parametrize("entropy", [False, True])
@pytest.mark.parametrize("enforce_eager,logprobs_mode", [(True, "raw_logprobs"), (False, "processed_logprobs")])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mtp_entropy_verify(entropy, enforce_eager, logprobs_mode):
    """Exercise random verification, ragged completions and reuse of request state.

    Random MTP weights cannot establish acceptance/quality improvements. The
    numerical kernel tests establish the policy; this test checks integration.
    """
    prompts = ["The capital of France is", "Write a short story about a mountain.", "Count from one to ten:"]
    params = [
        SamplingParams(
            max_tokens=length,
            ignore_eos=True,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.05,
            logprobs=2,
            seed=42 + length,
        )
        for length in (8, 16, 24)
    ]
    with VllmRunner(
        MTP_MODELS[0],
        max_model_len=1024,
        enforce_eager=enforce_eager,
        async_scheduling=True,
        enable_expert_parallel=True,
        disable_log_stats=False,
        logprobs_mode=logprobs_mode,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8]},
        additional_config={
            "rejection_sampler_config": {
                "enable_entropy_verify": entropy,
            }
        },
    ) as runner:
        for _ in range(2):
            outputs = runner.model.generate(prompts, params)
            assert len(outputs) == len(prompts)
            for request, param in zip(outputs, params):
                completion = request.outputs[0]
                assert len(completion.token_ids) == param.max_tokens
                assert completion.logprobs is not None
                assert len(completion.logprobs) == len(completion.token_ids)
                assert all(token >= 0 for token in completion.token_ids)
                assert all(token in scores for token, scores in zip(completion.token_ids, completion.logprobs))
        num_drafts = sum(m.value for m in runner.model.get_metrics() if m.name == "vllm:spec_decode_num_drafts")
        assert num_drafts > 0
