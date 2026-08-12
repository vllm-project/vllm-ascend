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

os.environ["HCCL_BUFFSIZE"] = "2048"
DSPARK_MAIN_MODEL = ["UploadWeight/DeepSeek-V4-Flash-DSpark-w4a8-test"]


@pytest.mark.parametrize("model", DSPARK_MAIN_MODEL)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dspark_spec_decoding(
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

    num_speculative_tokens = 5
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
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
    golden = [0.80, 0.74, 0.60, 0.37, 0.11]
    match = all(abs(a - b) < 0.1 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"
