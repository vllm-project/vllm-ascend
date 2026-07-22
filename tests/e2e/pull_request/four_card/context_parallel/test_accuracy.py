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
"""PCP/DCP long-sequence accuracy guards.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_accuracy.py`.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from tests.e2e.conftest import DPVllmRunner, VllmRunner, wait_until_npu_memory_free

MAX_NUM_SEQS = 4
E2E_ROOT = Path(__file__).resolve().parents[3]

FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [MAX_NUM_SEQS],
}

COMMON_PROMPTS = [
    "The capital of France is",
    "Hello, my name is Tom, I am",
    "The president of United States is",
]

DSV3_2_DCP_GOLDEN = [
    "The capital of France isoint054 Rund959arki",
    "Hello, my name is Tom, I am" + "ERIC slicpacelikeabra",
    "The president of United States isoint054 Rund959arki",
]

DEEPSEEK_V4_PROMPTS = [
    "Hello, my name is",
    "What is the meaning of life?",
]

DEEPSEEK_V4_GOLDEN = ["Hello, my name is {name} and I", 'What is the meaning of life?",\n    "What is']


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    model: str
    prompts: Sequence[str]
    expected_outputs: Sequence[str] | Sequence[Sequence[str]]
    max_tokens: int
    runner_kwargs: dict[str, Any]


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


def match_outputs_with_goldens(outputs: list[tuple[list[int], str]], goldens: Sequence[str]) -> None:
    """Helper function to compare output with golden output, ignoring whitespace differences."""
    outputs_str: Sequence[str] = [output[1] for output in outputs]
    assert len(outputs_str) == len(goldens)
    for index, (output, golden) in enumerate(zip(outputs_str, goldens)):
        assert isinstance(output, str) and isinstance(golden, str), "Both output and golden must be strings"
        assert output and golden, "Output and golden should not be empty"
        assert output.strip() == golden.strip()


FULL_FEATURE_MODEL_CASES = [
    AccuracyCase(
        name="dsv3_2_sfa_dcp_replicated_indexer",
        model="vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
        prompts=COMMON_PROMPTS,
        expected_outputs=DSV3_2_DCP_GOLDEN,
        max_tokens=5,
        runner_kwargs={
            "max_model_len": 1024,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": 1024,
            "data_parallel_size": 2,
            "tensor_parallel_size": 2,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 2,
            "enable_expert_parallel": True,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.4,
            "cp_kv_cache_interleave_size": 1,
            "block_size": 128,
            "quantization": "ascend",
            "long_prefill_token_threshold": 128,
            "compilation_config": FULL_DECODE_GRAPH,
            "additional_config": {
                "enable_flashcomm1": True,
                "enable_sparse_c8": False,
            },
            "speculative_config": {
                "method": "mtp",
                "num_speculative_tokens": 3,
            },
        },
    ),
    AccuracyCase(
        name="deepseek_v4_w4a8_dsa_cp_full_features",
        model="gdydems/DeepSeek-V4-Flash-w4a8-mtp",
        prompts=DEEPSEEK_V4_PROMPTS,
        expected_outputs=DEEPSEEK_V4_GOLDEN,
        max_tokens=5,
        runner_kwargs={
            "max_model_len": 8192,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 4096,
            "dtype": "auto",
            "tensor_parallel_size": 4,
            "prefill_context_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "enable_expert_parallel": True,
            "gpu_memory_utilization": 0.9,
            "quantization": "ascend",
            "tokenizer_mode": "deepseek_v4",
            "block_size": 128,
            "compilation_config": {
                "cudagraph_mode": "FULL_DECODE_ONLY",
            },
            "additional_config": {
                "enable_flashcomm1": True,
                "enable_dsa_cp": True,
            },
        },
    ),
]


@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
@pytest.mark.parametrize("case", FULL_FEATURE_MODEL_CASES, ids=lambda case: case.name)
def test_models_pcp_dcp_full_feature_accuracy(case: AccuracyCase) -> None:
    _run_accuracy_case(case)
