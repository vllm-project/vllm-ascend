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
"""Model Runner V2 context-parallel accuracy and feature guards.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_accuracy_v2.py`.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner, VllmRunner, wait_until_npu_memory_free

MAX_NUM_SEQS = 4
FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [MAX_NUM_SEQS],
}

PCP_FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [4, 8],
}

DSV3_2_MODEL = "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning"
DSV3_2_PROMPTS = [
    "The capital of France is",
    "Hello, my name is Tom, I am",
    "The president of United States is",
]
DSV3_2_SFA_DCP_GOLDENS = (
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
DSV3_2_SFA_PCP_GOLDENS = [
    "The capital of France isoint054 Rund compasses",
    "Hello, my name is Tom, I am" + "ERIC slicpacelike\u6302",
    "The president of United States isoint054 Rund959arki",
]

DSV4_MODEL = "gdydems/DeepSeek-V4-Flash-w4a8-mtp"
DSV4_PROMPTS = [
    "Hello, my name is",
    "What is the meaning of life?",
]
DSV4_DSA_PCP_GOLDENS = [
    "Hello, my name is {name} and I",
    'What is the meaning of life?",\n    "What is',
]


MTP_PCP_MODEL = "wemaster/deepseek_mtp_main_random_bf16"
EAGLE3_PCP_TARGET_MODEL = "Qwen/Qwen3-8B"
EAGLE3_PCP_DRAFT_MODEL = "RedHatAI/Qwen3-8B-speculator.eagle3"

LONG_CONTEXT = "This is a long context for PCP prefill validation. " * 64
PCP_PROMPTS = [
    LONG_CONTEXT + "Hello, my name is",
    LONG_CONTEXT + "The president of the United States is",
]


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    model: str
    prompts: Sequence[str]
    expected_outputs: Sequence[str] | Sequence[Sequence[str]]
    max_tokens: int
    runner_kwargs: dict[str, Any]


def _match_outputs_with_goldens(outputs: list[tuple[list[int], str]], goldens: Sequence[str]) -> None:
    """Helper function to compare output with golden output, ignoring whitespace differences."""
    outputs_str: Sequence[str] = [output[1] for output in outputs]
    assert len(outputs_str) == len(goldens)
    for output, golden in zip(outputs_str, goldens):
        assert isinstance(output, str) and isinstance(golden, str), "Both output and golden must be strings"
        assert output and golden, "Output and golden should not be empty"
        assert output.strip() == golden.strip()


def _run_accuracy_case(case: AccuracyCase) -> None:
    runner_cls = DPVllmRunner if case.runner_kwargs.get("data_parallel_size", 1) > 1 else VllmRunner
    with runner_cls(case.model, **case.runner_kwargs) as runner:
        outputs = runner.generate_greedy(list(case.prompts), case.max_tokens)

    if isinstance(case.expected_outputs[0], str):
        expected_outputs = cast(Sequence[str], case.expected_outputs)
        _match_outputs_with_goldens(outputs, expected_outputs)
    else:
        # If multiple expected output sets are provided, the output is considered correct if it matches any of the sets.
        multi_expected_outputs = cast(Sequence[Sequence[str]], case.expected_outputs)
        tries = []
        for expected in multi_expected_outputs:
            try:
                _match_outputs_with_goldens(outputs, expected)
            except AssertionError as exc:
                tries.append(f"Output did not match expected set:\n{exc}")
            else:
                break
        if len(tries) == len(multi_expected_outputs):
            failure_details = "\n\n".join(tries)
            raise AssertionError(f"Output did not match any of the expected output sets:\n{failure_details}")


DSV3_2_SFA_DCP_CASE = AccuracyCase(
    name="dsv3_2_sfa_dcp_replicated_indexer_mrv2_tp2_dcp2",
    model=DSV3_2_MODEL,
    prompts=DSV3_2_PROMPTS,
    expected_outputs=DSV3_2_SFA_DCP_GOLDENS,
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
        "attention_config.indexer_kv_dtype": "int8",
        "additional_config": {
            "enable_dsa_cp": False,
        },
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
    },
)

DSV3_2_SFA_PCP_CASE = AccuracyCase(
    name="dsv3_2_sfa_pcp_mrv2_full_decode_only",
    model=DSV3_2_MODEL,
    prompts=DSV3_2_PROMPTS,
    expected_outputs=DSV3_2_SFA_PCP_GOLDENS,
    max_tokens=5,
    runner_kwargs={
        "max_model_len": 1024,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": 1024,
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "enable_expert_parallel": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.8,
        "cp_kv_cache_interleave_size": 128,
        "block_size": 128,
        "quantization": "ascend",
        "compilation_config": FULL_DECODE_GRAPH,
    },
)

DSV4_DSA_PCP_CASE = AccuracyCase(
    name="dsv4_dsa_pcp_mrv2_full_decode_only",
    model=DSV4_MODEL,
    prompts=DSV4_PROMPTS,
    expected_outputs=DSV4_DSA_PCP_GOLDENS,
    max_tokens=5,
    runner_kwargs={
        "max_model_len": 1024,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": 1024,
        "dtype": "auto",
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "enable_expert_parallel": True,
        "gpu_memory_utilization": 0.9,
        "block_size": 128,
        "attention_config.indexer_kv_dtype": "int8"
        "quantization": "ascend",
        "tokenizer_mode": "deepseek_v4",
        "compilation_config": FULL_DECODE_GRAPH,
        "additional_config": {
            "enable_dsa_cp": False,
            "enable_prefill_mc2": True,
        },
    },
)


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
    _run_accuracy_case(DSV3_2_SFA_DCP_CASE)


@pytest.mark.e2e_model(DSV3_2_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_pcp",
    parallel="TP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv3_2_sfa_pcp_model_runner_v2_graph_accuracy() -> None:
    """Guard MRV2 SFA PCP full-decode-only graph accuracy."""
    _run_accuracy_case(DSV3_2_SFA_PCP_CASE)


@pytest.mark.e2e_model(DSV4_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dsa_pcp",
    parallel="TP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "2560",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv4_dsa_pcp_model_runner_v2_graph_accuracy() -> None:
    """Guard MRV2 DSA PCP full-decode-only graph accuracy."""
    _run_accuracy_case(DSV4_DSA_PCP_CASE)


def _run_pcp_spec_decode(
    model: str,
    speculative_config: dict[str, object],
) -> None:
    sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        max_model_len=1024,
        max_num_batched_tokens=64,
        max_num_seqs=MAX_NUM_SEQS,
        disable_log_stats=False,
        distributed_executor_backend="mp",
        enable_chunked_prefill=True,
        compilation_config=PCP_FULL_DECODE_GRAPH,
        speculative_config=speculative_config,
    ) as runner:
        outputs = runner.model.generate(PCP_PROMPTS, sampling_params)
        metrics = runner.model.get_metrics()
        num_drafts = sum(metric.value for metric in metrics if metric.name == "vllm:spec_decode_num_drafts")

    token_ids = [output.outputs[0].token_ids for output in outputs]
    assert len(token_ids) == len(PCP_PROMPTS)
    assert all(token_ids)
    assert num_drafts > 0


@pytest.mark.e2e_model(MTP_PCP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,chunked_prefill",
    parallel="TP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "1024",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mtp_mla_spec_decode_with_pcp() -> None:
    """Guard MRV2 MTP MLA PCP full-decode-only graph execution."""
    _run_pcp_spec_decode(
        MTP_PCP_MODEL,
        {
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
    )


@pytest.mark.e2e_model(EAGLE3_PCP_TARGET_MODEL, EAGLE3_PCP_DRAFT_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="eagle3,chunked_prefill",
    parallel="TP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "1024",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_eagle3_gqa_spec_decode_with_pcp() -> None:
    """Guard MRV2 Eagle3 GQA PCP full-decode-only graph execution."""
    _run_pcp_spec_decode(
        EAGLE3_PCP_TARGET_MODEL,
        {
            "method": "eagle3",
            "model": EAGLE3_PCP_DRAFT_MODEL,
            "num_speculative_tokens": 3,
        },
    )
