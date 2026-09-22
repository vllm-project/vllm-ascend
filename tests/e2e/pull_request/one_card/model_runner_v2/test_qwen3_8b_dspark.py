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
from tests.e2e.pull_request.one_card.model_runner_v2.utils import calculate_acceptance_per_pos

DSPARK_MAIN_MODEL = ["Qwen/Qwen3-8B"]
DSPARK_MODELS = ["deepseek-ai/dspark_qwen3_8b_block7"]


@pytest.mark.parametrize("model", DSPARK_MAIN_MODEL)
@pytest.mark.parametrize("dspark_model", DSPARK_MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [False])
@pytest.mark.parametrize(
    ("compilation_config", "enable_adaptive_verification"),
    [
        pytest.param(
            {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8]},
            False,
            id="full_decode_only",
        ),
        pytest.param({}, False, id="default_full_and_piecewise"),
        pytest.param(
            {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8]},
            True,
            id="full_decode_only-adaptive",
        ),
    ],
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dspark_spec_decoding(
    model: str,
    dspark_model: str,
    max_tokens: int,
    enforce_eager: bool,
    enable_adaptive_verification: bool,
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
            "model": dspark_model,
            "method": "dspark",
            "num_speculative_tokens": num_speculative_tokens,
            **({"enable_adaptive_verification": True} if enable_adaptive_verification else {}),
        },
        compilation_config=compilation_config,
    ) as runner:
        runner.model.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()

    if enable_adaptive_verification:
        return

    acceptance_per_pos = calculate_acceptance_per_pos(
        metrics,
        num_speculative_tokens,
        Counter,
        Vector,
    )
    golden = [0.84, 0.48, 0.32, 0.20, 0.09, 0.09, 0.02]
    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"
