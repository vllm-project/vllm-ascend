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

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner, RemoteOpenAIServer, VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import _run_pd_disaggregation

MAX_NUM_SEQS = 4
FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [MAX_NUM_SEQS],
}

PCP_FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [4, 8],
}

DSV3_2_MODEL = os.getenv(
    "DSV3_2_MODEL_PATH",
    "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
)
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


@dataclass(frozen=True)
class InferenceCase:
    model: str
    prompts: Sequence[str]
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
    prompts=DSV3_2_PROMPTS,
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
    prompts=DSV3_2_PROMPTS,
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
    prompts=DSV3_2_PROMPTS,
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
    prompts=DSV3_2_PROMPTS,
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


PCP_RUNNER_KWARGS = {
    "tensor_parallel_size": 2,
    "prefill_context_parallel_size": 2,
    "max_model_len": 1024,
    "max_num_batched_tokens": 64,
    "max_num_seqs": MAX_NUM_SEQS,
    "distributed_executor_backend": "mp",
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "compilation_config": PCP_FULL_DECODE_GRAPH,
}

MLA_PCP_CASE = InferenceCase(
    model=MTP_PCP_MODEL,
    prompts=PCP_PROMPTS,
    max_tokens=16,
    runner_kwargs={
        **PCP_RUNNER_KWARGS,
        "enable_expert_parallel": True,
    },
)

MLA_PCP_DP_CASE = InferenceCase(
    model=MTP_PCP_MODEL,
    prompts=PCP_PROMPTS,
    max_tokens=16,
    runner_kwargs={
        **MLA_PCP_CASE.runner_kwargs,
        "tensor_parallel_size": 1,
        "data_parallel_size": 2,
    },
)

MLA_PCP_PP_MTP_CASE = InferenceCase(
    model=MTP_PCP_MODEL,
    prompts=PCP_PROMPTS,
    max_tokens=16,
    runner_kwargs={
        **MLA_PCP_CASE.runner_kwargs,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "async_scheduling": True,
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
    },
)

GQA_PCP_CASE = InferenceCase(
    model=EAGLE3_PCP_TARGET_MODEL,
    prompts=PCP_PROMPTS,
    max_tokens=16,
    runner_kwargs={
        **PCP_RUNNER_KWARGS,
    },
)

GQA_PCP_DP_CASE = InferenceCase(
    model=EAGLE3_PCP_TARGET_MODEL,
    prompts=PCP_PROMPTS,
    max_tokens=16,
    runner_kwargs={
        **GQA_PCP_CASE.runner_kwargs,
        "tensor_parallel_size": 1,
        "data_parallel_size": 2,
    },
)

GQA_PCP_PP_EAGLE3_CASE = InferenceCase(
    model=EAGLE3_PCP_TARGET_MODEL,
    prompts=PCP_PROMPTS,
    max_tokens=16,
    runner_kwargs={
        **GQA_PCP_CASE.runner_kwargs,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 2,
        "async_scheduling": True,
        "speculative_config": {
            "method": "eagle3",
            "model": EAGLE3_PCP_DRAFT_MODEL,
            "num_speculative_tokens": 3,
        },
    },
)


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


@pytest.mark.e2e_model(MTP_PCP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,chunked_prefill,prefix_caching",
    parallel="EP,PCP,PP",
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
def test_mtp_mla_spec_decode_with_pcp_pp() -> None:
    """Guard MRV2 MLA PCP+PP+MTP graph execution with async scheduling."""
    _run_inference_case(MLA_PCP_PP_MTP_CASE)


@pytest.mark.e2e_model(EAGLE3_PCP_TARGET_MODEL, EAGLE3_PCP_DRAFT_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="eagle3,chunked_prefill,prefix_caching",
    parallel="PCP,PP",
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
def test_eagle3_gqa_spec_decode_with_pcp_pp() -> None:
    """Guard MRV2 GQA PCP+PP+Eagle3 graph execution with async scheduling."""
    _run_inference_case(GQA_PCP_PP_EAGLE3_CASE)


@pytest.mark.e2e_model(MTP_PCP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="chunked_prefill,prefix_caching",
    parallel="TP,EP,PCP",
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
def test_mla_with_pcp() -> None:
    """Guard MRV2 MLA PCP full-decode-only graph execution."""
    _run_inference_case(MLA_PCP_CASE)


@pytest.mark.e2e_model(MTP_PCP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="chunked_prefill,prefix_caching",
    parallel="DP,EP,PCP",
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
def test_mla_with_pcp_dp() -> None:
    """Guard MRV2 MLA PCP+DP full-decode-only graph execution."""
    _run_inference_case(MLA_PCP_DP_CASE)


@pytest.mark.e2e_model(EAGLE3_PCP_TARGET_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="chunked_prefill,prefix_caching",
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
def test_gqa_with_pcp() -> None:
    """Guard MRV2 GQA PCP full-decode-only graph execution."""
    _run_inference_case(GQA_PCP_CASE)


@pytest.mark.e2e_model(EAGLE3_PCP_TARGET_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="chunked_prefill,prefix_caching",
    parallel="DP,PCP",
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
def test_gqa_with_pcp_dp() -> None:
    """Guard MRV2 GQA PCP+DP full-decode-only graph execution."""
    case = GQA_PCP_DP_CASE
    # Dense models require online DP; DPVllmRunner uses the MoE-only offline path.
    server_args = []
    for name, value in case.runner_kwargs.items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            server_args.append(option if value else "--no-" + name.replace("_", "-"))
        else:
            server_args.extend([option, json.dumps(value) if isinstance(value, dict) else str(value)])

    with RemoteOpenAIServer(case.model, server_args) as server:
        with server.get_client() as client:
            outputs = client.completions.create(
                model=case.model,
                prompt=list(case.prompts),
                max_tokens=case.max_tokens,
                temperature=0,
            )
        assert len(outputs.choices) == len(case.prompts)
        for choice in outputs.choices:
            assert choice.text, "Each request should return non-empty text"


@pytest.mark.e2e_model(MTP_PCP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="",
    parallel="EP,PCP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_mla_pcp_pd_disaggregation() -> None:
    """Verify MRV2 MLA PCP2 prefill KV transfer to an async TP1 decoder."""
    _run_pd_disaggregation(
        MTP_PCP_MODEL,
        model_args=[
            "--dtype",
            "bfloat16",
            "--enable-expert-parallel",
            # These smoke requests only require a plain-text chat template.
            "--chat-template",
            "{% for message in messages %}{{ message['content'] }}{% endfor %}",
        ],
        prefill_tp_size=1,
        prefill_pcp_size=2,
        decode_tp_size=1,
        async_scheduling=True,
        env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"},
    )


@pytest.mark.e2e_model(EAGLE3_PCP_TARGET_MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="",
    parallel="PCP",
    deploy="pd_disaggregation",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_gqa_pcp_pd_disaggregation() -> None:
    """Verify MRV2 GQA PCP2 prefill KV transfer to an async TP1 decoder."""
    _run_pd_disaggregation(
        EAGLE3_PCP_TARGET_MODEL,
        model_args=[
            "--dtype",
            "bfloat16",
            # These smoke requests only require a plain-text chat template.
            "--chat-template",
            "{% for message in messages %}{{ message['content'] }}{% endfor %}",
        ],
        prefill_tp_size=1,
        prefill_pcp_size=2,
        decode_tp_size=1,
        async_scheduling=True,
        env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"},
    )


def _run_pcp_spec_decode(
    model: str,
    speculative_config: dict[str, object],
    prompts: Sequence[str] = PCP_PROMPTS,
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
        outputs = runner.model.generate(prompts, sampling_params)
        metrics = runner.model.get_metrics()
        num_drafts = sum(metric.value for metric in metrics if metric.name == "vllm:spec_decode_num_drafts")

    token_ids = [output.outputs[0].token_ids for output in outputs]
    assert len(token_ids) == len(prompts)
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
@pytest.mark.parametrize("pcp_segment_boundary", [False, True], ids=["chunked_prefill", "pcp_segment_boundary"])
def test_eagle3_gqa_spec_decode_with_pcp(pcp_segment_boundary: bool) -> None:
    """Guard MRV2 Eagle3 GQA PCP full-decode-only graph execution."""
    prompts = PCP_PROMPTS
    env = {}
    if pcp_segment_boundary:
        # Queue all three prefills before scheduling: PCP creates six local
        # block-table rows, while draft decode uses a four-request graph.
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        prompts = [
            "Alice explains how to grow flowers in a garden. Alice explains how to grow flowers",
            "Bob describes a journey across the mountains. Bob describes a journey across the mountains.",
            "Carol teaches students how to solve a math problem. Carol teaches students how to solve",
        ]
    with patch.dict(os.environ, env):
        _run_pcp_spec_decode(
            EAGLE3_PCP_TARGET_MODEL,
            {
                "method": "eagle3",
                "model": EAGLE3_PCP_DRAFT_MODEL,
                "num_speculative_tokens": 3,
            },
            prompts,
        )
