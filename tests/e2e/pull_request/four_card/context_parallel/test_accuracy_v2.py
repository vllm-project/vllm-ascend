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
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import DPVllmRunner, VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.one_card.model_runner_v2.utils import calculate_acceptance_per_pos
from tests.e2e.pull_request.utils import run_pd_disaggregation
from vllm_ascend.utils import vllm_version_is

MAX_NUM_SEQS = 4
FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [MAX_NUM_SEQS],
}

PCP_FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [1, 2, 4, 8],
}

DSV3_2_MODEL = os.getenv(
    "DSV3_2_MODEL_PATH",
    "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
)
ACCURACY_PROMPTS = [
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
MTP_PCP_MODEL = "wemaster/deepseek_mtp_main_random_bf16"
EAGLE3_PCP_TARGET_MODEL = "Qwen/Qwen3-8B"
EAGLE3_PCP_DRAFT_MODEL = "RedHatAI/Qwen3-8B-speculator.eagle3"

# The random MTP model provides an output-consistency baseline, not semantic accuracy.
MTP_PCP_GOLDENS = [
    "The capital of France is Salmonella团团 elsewhereッγκ",
    "Hello, my name is Tom, I amEiSlowukt Analysis sprouts",
    "The president of United States is Salmonella团团 elsewhereッγκ",
]
# Five generated tokens from the reference runs; match each three-prompt set as a whole.
EAGLE3_PCP_GOLDENS = (
    [
        "The capital of France is Paris. The capital of",
        "Hello, my name is Tom, I am 25 years old",
        "The president of United States is the head of state and",
    ],
    [
        "The capital of France is Paris. The capital of",
        "Hello, my name is Tom, I am 23 years old",
        "The president of United States is the head of state and",
    ],
)
# Five reference PCP1 runs with the shared prompts and the same generation configuration.
EAGLE3_PCP_MIN_ACCEPTANCE_RATES = [0.54, 0.29, 0.15]
ACCEPTANCE_RATE_TOLERANCE = 0.03


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    model: str
    prompts: Sequence[str]
    expected_outputs: Sequence[str] | Sequence[Sequence[str]]
    max_tokens: int
    runner_kwargs: dict[str, Any]
    minimum_acceptance_rates: Sequence[float] | None = None
    # None checks the full output; a prefix leaves the generation/acceptance workload unchanged.
    output_prefix_tokens: int | None = None


@dataclass(frozen=True)
class InferenceCase:
    model: str
    prompts: Sequence[str]
    max_tokens: int
    runner_kwargs: dict[str, Any]


def _match_outputs_with_goldens(outputs: list[tuple[list[int], str]], goldens: Sequence[str]) -> None:
    """Compare complete outputs with goldens, ignoring only leading and trailing whitespace."""
    outputs_str: Sequence[str] = [output[1] for output in outputs]
    assert len(outputs_str) == len(goldens)
    for output, golden in zip(outputs_str, goldens):
        assert isinstance(output, str) and isinstance(golden, str), "Both output and golden must be strings"
        assert output and golden, "Output and golden should not be empty"
        assert output.strip() == golden.strip()


def _run_accuracy_case(case: AccuracyCase) -> None:
    runner_cls = DPVllmRunner if case.runner_kwargs.get("data_parallel_size", 1) > 1 else VllmRunner
    with runner_cls(case.model, **case.runner_kwargs) as runner:
        # This existing runner path returns completion-only IDs; None disables logprobs collection.
        generated_outputs = runner.generate_greedy_logprobs(list(case.prompts), case.max_tokens, num_logprobs=None)
        outputs = []
        if case.output_prefix_tokens is not None:
            assert 0 < case.output_prefix_tokens <= case.max_tokens
            tokenizer = runner.model.get_tokenizer()
        for prompt, (token_ids, text, _) in zip(case.prompts, generated_outputs, strict=True):
            if case.output_prefix_tokens is not None:
                token_ids = token_ids[: case.output_prefix_tokens]
                text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            outputs.append((token_ids, prompt + text))
        if case.minimum_acceptance_rates is not None:
            metrics = runner.model.get_metrics()

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

    if case.minimum_acceptance_rates is not None:
        acceptance_rates = calculate_acceptance_per_pos(
            metrics, case.runner_kwargs["speculative_config"]["num_speculative_tokens"], Counter, Vector
        )
        print(f"Case: {case.name}, Acceptance rates per draft position: {acceptance_rates}", flush=True)
        assert len(acceptance_rates) == len(case.minimum_acceptance_rates), (
            f"Expected {len(case.minimum_acceptance_rates)} acceptance rates, got {len(acceptance_rates)}"
        )
        for position, (actual, minimum) in enumerate(zip(acceptance_rates, case.minimum_acceptance_rates, strict=True)):
            assert actual >= minimum or minimum - actual < ACCEPTANCE_RATE_TOLERANCE, (
                f"Acceptance rate at draft position {position} is {actual}, below minimum {minimum}"
            )


def _run_inference_case(case: InferenceCase) -> None:
    """Verify that the configured service starts and returns generated tokens."""
    runner_cls = DPVllmRunner if case.runner_kwargs.get("data_parallel_size", 1) > 1 else VllmRunner
    with runner_cls(case.model, **case.runner_kwargs) as runner:
        outputs = runner.generate_greedy(list(case.prompts), case.max_tokens)

    assert len(outputs) == len(case.prompts)
    for token_ids, output_text in outputs:
        assert token_ids, "Each request should return at least one generated token"
        assert isinstance(output_text, str) and output_text, "Each request should return non-empty text"


DSV3_2_SFA_PCP_CASE = InferenceCase(
    model=DSV3_2_MODEL,
    prompts=ACCURACY_PROMPTS,
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

DSV3_2_SFA_PCP_DCP_CASE = AccuracyCase(
    name="dsv3_2_sfa_pcp_dcp_replicated_indexer_mrv2_tp2_pcp2_dcp4",
    model=DSV3_2_MODEL,
    prompts=ACCURACY_PROMPTS,
    expected_outputs=DSV3_2_SFA_DCP_GOLDENS,
    max_tokens=5,
    runner_kwargs={
        "max_model_len": 1024,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": 1024,
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "decode_context_parallel_size": 4,
        "enable_expert_parallel": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.8,
        "cp_kv_cache_interleave_size": 128,
        "block_size": 128,
        "quantization": "ascend",
        "compilation_config": FULL_DECODE_GRAPH,
        "additional_config": {
            "enable_dsa_cp": False,
            "enable_sparse_li_c8": False,
        },
    },
)

DSV3_2_SFA_PCP_DP_CASE = InferenceCase(
    model=DSV3_2_MODEL,
    prompts=ACCURACY_PROMPTS,
    max_tokens=5,
    runner_kwargs={
        **DSV3_2_SFA_PCP_CASE.runner_kwargs,
        "tensor_parallel_size": 1,
        "data_parallel_size": 2,
        "distributed_executor_backend": "mp",
    },
)

DSV3_2_SFA_PCP_PP_MTP_CASE = InferenceCase(
    model=DSV3_2_MODEL,
    prompts=ACCURACY_PROMPTS,
    max_tokens=5,
    runner_kwargs={
        **DSV3_2_SFA_PCP_CASE.runner_kwargs,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "async_scheduling": True,
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
    },
)


@pytest.mark.e2e_model(DSV3_2_MODEL)
@pytest.mark.skipif(
    vllm_version_is("0.28.0"),
    reason="Temporary v0.28.0 SFA PCP accuracy skip; root cause is under separate investigation (PR #16009).",
)
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
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv3_2_sfa_pcp_model_runner_v2_graph() -> None:
    """Guard MRV2 SFA PCP full-decode-only graph execution."""
    _run_inference_case(DSV3_2_SFA_PCP_CASE)


@pytest.mark.e2e_model(DSV3_2_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_pcp,chunked_prefill,prefix_caching",
    parallel="DP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv3_2_sfa_pcp_dp_model_runner_v2_graph() -> None:
    """Guard MRV2 SFA PCP graph execution with two DP replicas."""
    _run_inference_case(DSV3_2_SFA_PCP_DP_CASE)


@pytest.mark.e2e_model(DSV3_2_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_pcp,mtp",
    parallel="EP,PCP,PP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv3_2_sfa_pcp_pp_mtp_model_runner_v2_graph() -> None:
    """Guard MRV2 SFA PCP+PP+MTP graph execution with async scheduling."""
    _run_inference_case(DSV3_2_SFA_PCP_PP_MTP_CASE)


@pytest.mark.e2e_model(DSV3_2_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_pcp",
    parallel="TP,EP,PCP,DCP",
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
def test_dsv3_2_sfa_pcp_dcp_model_runner_v2_graph_accuracy() -> None:
    """Guard MRV2 SFA PCP+DCP full-decode-only graph accuracy."""
    _run_accuracy_case(DSV3_2_SFA_PCP_DCP_CASE)


MTP_PCP_CASE = AccuracyCase(
    name="mtp_mla_pcp",
    model=MTP_PCP_MODEL,
    prompts=ACCURACY_PROMPTS,
    expected_outputs=MTP_PCP_GOLDENS,
    # The random MTP head accepts no drafts; a zero golden cannot guard acceptance regressions.
    max_tokens=32,
    output_prefix_tokens=5,
    runner_kwargs={
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "max_model_len": 1024,
        "max_num_batched_tokens": 64,
        "max_num_seqs": MAX_NUM_SEQS,
        "disable_log_stats": False,
        "distributed_executor_backend": "mp",
        "enable_chunked_prefill": True,
        "seed": 0,
        "compilation_config": PCP_FULL_DECODE_GRAPH,
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
    },
)


EAGLE3_PCP_CASE = AccuracyCase(
    name="eagle3_gqa_pcp",
    model=EAGLE3_PCP_TARGET_MODEL,
    prompts=ACCURACY_PROMPTS,
    expected_outputs=EAGLE3_PCP_GOLDENS,
    minimum_acceptance_rates=EAGLE3_PCP_MIN_ACCEPTANCE_RATES,
    max_tokens=32,
    output_prefix_tokens=5,
    runner_kwargs={
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "max_model_len": 2048,
        "max_num_seqs": 256,
        "distributed_executor_backend": "mp",
        "gpu_memory_utilization": 0.7,
        "disable_log_stats": False,
        "async_scheduling": True,
        "seed": 0,
        "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [12]},
        "speculative_config": {
            "method": "eagle3",
            "model": EAGLE3_PCP_DRAFT_MODEL,
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 2,
            "disable_padded_drafter_batch": False,
        },
    },
)


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
    """Verify MRV2 MTP MLA PCP output consistency against its reference golden."""
    _run_accuracy_case(MTP_PCP_CASE)


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
    """Verify MRV2 Eagle3 PCP output accuracy and acceptance in one generation."""
    _run_accuracy_case(EAGLE3_PCP_CASE)


@pytest.mark.e2e_model(EAGLE3_PCP_TARGET_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="aclgraph",
    parallel="PCP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_gqa_pcp_pd_model_runner_v2_graph() -> None:
    """Verify PCP2 KV transfer and greedy accuracy with an async TP1 GQA decoder."""
    output = run_pd_disaggregation(
        model=EAGLE3_PCP_TARGET_MODEL,
        model_args=[
            "--dtype",
            "bfloat16",
            "--seed",
            "0",
            "--chat-template",
            "{% for message in messages %}{{ message['content'] }}{% endfor %}",
        ],
        prefill_tp_size=1,
        prefill_pcp_size=2,
        decode_tp_size=1,
        async_scheduling=True,
        use_model_runner_v2=True,
    )
    assert output == " Alex, and I am"
