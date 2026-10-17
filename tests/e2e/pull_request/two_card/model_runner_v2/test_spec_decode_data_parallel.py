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
"""Speculative decoding under DP with Model Runner V2."""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free

MTP_MODELS = ["wemaster/deepseek_mtp_main_random_bf16"]

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


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
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HCCL_BUFFSIZE": "1024",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.7)
def test_mtp_spec_decoding_dp(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
    compilation_config: dict,
) -> None:
    num_speculative_tokens = 3
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with DPVllmRunner(
        model,
        data_parallel_size=2,
        tensor_parallel_size=1,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        async_scheduling=True,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
        compilation_config=compilation_config,
    ) as vllm_model:
        outputs = vllm_model.generate(PROMPTS, sampling_params=sampling_params)

    assert len(outputs) == len(PROMPTS)
