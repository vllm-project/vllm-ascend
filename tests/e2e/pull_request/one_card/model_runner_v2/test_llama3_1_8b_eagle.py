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
from tests.e2e.pull_request.one_card.model_runner_v2.utils import calculate_acceptance_per_pos
from vllm_ascend.utils import vllm_version_is

MAIN_MODELS = ["LLM-Research/Meta-Llama-3.1-8B-Instruct"]
EAGLE_MODELS = ["vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B"]


@pytest.mark.parametrize("model", MAIN_MODELS)
@pytest.mark.parametrize("eagle_model", EAGLE_MODELS)
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
        pytest.param({"cudagraph_mode": "NONE"}, id="eager"),
    ],
)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_eagle_spec_decoding(
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
    num_speculative_tokens = 3
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        disable_log_stats=False,
        async_scheduling=True,
        speculative_config={
            "model": eagle_model,
            "method": "eagle",
            "num_speculative_tokens": num_speculative_tokens,
        },
        compilation_config=compilation_config,
    ) as runner:
        runner.model.generate(prompts, sampling_params)
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


@pytest.mark.parametrize("graph_mode", ["NONE", "FULL_DECODE_ONLY"])
@pytest.mark.skipif(vllm_version_is("0.30.0"), reason="Runtime K requires upstream #57053")
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_dynamic_eagle_matches_target_across_width_changes(graph_mode):
    """Repeated K changes must not expose stale drafts or replay another K's graph."""
    prompts = ["Hello, my name is", "The capital of France is", "The future of AI is", "Once upon a time"]
    sampling = SamplingParams(max_tokens=32, temperature=0.0)
    runner_kwargs = dict(
        max_model_len=1024,
        max_num_seqs=4,
        async_scheduling=True,
        compilation_config={"cudagraph_mode": graph_mode, "cudagraph_capture_sizes": [4, 8, 16]},
    )
    batch_sizes = [1, 2, 3, 4, 2, 1]
    with VllmRunner(MAIN_MODELS[0], **runner_kwargs) as runner:
        expected = [
            [output.outputs[0].token_ids for output in runner.model.generate(prompts[:size], sampling)]
            for size in batch_sizes
        ]
    with VllmRunner(
        MAIN_MODELS[0],
        speculative_config={
            "model": EAGLE_MODELS[0],
            "method": "eagle",
            "num_speculative_tokens": 3,
            "num_speculative_tokens_per_batch_size": [(1, 1, 3), (2, 2, 2), (3, 3, 1), (4, 4, 0)],
        },
        **runner_kwargs,
    ) as runner:
        for size, reference in zip(batch_sizes, expected):
            outputs = runner.model.generate(prompts[:size], sampling)
            assert [output.outputs[0].token_ids for output in outputs] == reference
