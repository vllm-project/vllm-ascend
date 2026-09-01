#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Model Runner V2 SFA DCP accuracy guard.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_accuracy_v2.py`.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import patch

from tests.e2e.conftest import DPVllmRunner, VllmRunner, wait_until_npu_memory_free

MAX_NUM_SEQS = 4

FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [MAX_NUM_SEQS],
}

COMMON_PROMPTS = [
    "The capital of France is",
    "Hello, my name is Tom, I am",
    "The president of United States is",
]

DSV3_2_DCP_GOLDENS = (
    [
        "The capital of France isoint054 Rund compasses",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike挂",
        "The president of United States isoint054 Rund959arki",
    ],
    [
        "The capital of France isoint054 Rund959arki",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike挂",
        "The president of United States isoint054 Rund959arki",
    ],
    [
        "The capital of France isorrionicALLY casmith",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike挂",
        "The president of United States is平行于我 charm与技术oi",
    ],
    [
        "The capital of France isorrionic Tudefeault",
        "Hello, my name is Tom, I am" + "ERIC slicpacelike挂",
        "The president of United States is平行于我 charm与技术oi",
    ],
)

MODEL = "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning"


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    model: str
    prompts: Sequence[str]
    expected_outputs: Sequence[str] | Sequence[Sequence[str]]
    max_tokens: int
    runner_kwargs: dict[str, Any]


def match_outputs_with_goldens(outputs: list[tuple[list[int], str]], goldens: Sequence[str]) -> None:
    """Helper function to compare output with golden output, ignoring whitespace differences."""
    outputs_str: Sequence[str] = [output[1] for output in outputs]
    assert len(outputs_str) == len(goldens)
    for index, (output, golden) in enumerate(zip(outputs_str, goldens)):
        assert isinstance(output, str) and isinstance(golden, str), "Both output and golden must be strings"
        assert output and golden, "Output and golden should not be empty"
        assert output.strip() == golden.strip()


def _run_accuracy_case(case: AccuracyCase) -> None:
    runner_cls = DPVllmRunner if case.runner_kwargs.get("data_parallel_size", 1) > 1 else VllmRunner
    with runner_cls(case.model, **case.runner_kwargs) as runner:
        outputs = runner.generate_greedy(list(case.prompts), case.max_tokens)

    if isinstance(case.expected_outputs[0], str):
        expected_outputs = cast(Sequence[str], case.expected_outputs)
        match_outputs_with_goldens(outputs, expected_outputs)
    else:
        # If multiple expected output sets are provided, the output is considered correct if it matches any of the sets.
        multi_expected_outputs = cast(Sequence[Sequence[str]], case.expected_outputs)
        tries = []
        for expected in multi_expected_outputs:
            try:
                match_outputs_with_goldens(outputs, expected)
            except AssertionError as exc:
                tries.append(f"Output did not match expected set:\n{exc}")
            else:
                break
        if len(tries) == len(multi_expected_outputs):
            failure_details = "\n\n".join(tries)
            raise AssertionError(f"Output did not match any of the expected output sets:\n{failure_details}")


FULL_FEATURE_MODEL_CASES = AccuracyCase(
    name="dsv3_2_sfa_dcp_replicated_indexer_mrv2_tp2_dcp2",
    model=MODEL,
    prompts=COMMON_PROMPTS,
    expected_outputs=DSV3_2_DCP_GOLDENS,
    max_tokens=5,
    runner_kwargs={
        "max_model_len": 1024,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": 1024,
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "enable_expert_parallel": True,
        "gpu_memory_utilization": 0.4,
        "block_size": 128,
        "quantization": "ascend",
        "compilation_config": FULL_DECODE_GRAPH,
        "additional_config": {
            "enable_dsa_cp": False,
            "enable_sparse_li_c8": False,
        },
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
    },
)


@pytest.mark.skip("The use case is unstable and temporarily taken offline.")
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv3_2_sfa_dcp_tp2_dcp2_model_runner_v2_accuracy() -> None:
    """Guard MRV2 accuracy."""
    _run_accuracy_case(FULL_FEATURE_MODEL_CASES)
