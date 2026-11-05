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
"""Long-sequence PCP/DCP matrix: CP-on versus CP-off greedy token parity.

Every case sends one deterministic long prompt twice: once with the
prefill/decode context-parallel switches disabled and once with the topology
under test. Greedy decoding must return exactly the same token ids in both
runs, for both eager mode and the full-decode-only graph.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_cp_matrix.py`.
"""

import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MAX_OUTPUT_TOKENS = 16
MAX_NUM_SEQS = 1
BLOCK_SIZE = 128
CP_KV_CACHE_INTERLEAVE_SIZE = 128
GRAPH_MODES = (None, "FULL_DECODE_ONLY")
CP_OFF_KWARGS = {"prefill_context_parallel_size": 1, "decode_context_parallel_size": 1}
ENVS = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "HCCL_BUFFSIZE": "1024",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
}
DSV3_2_MODEL = os.getenv(
    "DSV3_2_MODEL_PATH",
    "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
)
GQA_MODEL = os.getenv(
    "GQA_MODEL_PATH",
    "Qwen/Qwen3-8B",
)

MLA_RUNNER_KWARGS: dict[str, Any] = {
    "tensor_parallel_size": 2,
    "max_num_seqs": MAX_NUM_SEQS,
    "block_size": BLOCK_SIZE,
    "cp_kv_cache_interleave_size": CP_KV_CACHE_INTERLEAVE_SIZE,
    "enable_expert_parallel": True,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "gpu_memory_utilization": 0.85,
    "quantization": "ascend",
}

GQA_RUNNER_KWARGS: dict[str, Any] = {
    "tensor_parallel_size": 2,
    "max_num_seqs": MAX_NUM_SEQS,
    "block_size": BLOCK_SIZE,
    "cp_kv_cache_interleave_size": CP_KV_CACHE_INTERLEAVE_SIZE,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "gpu_memory_utilization": 0.85,
}


@dataclass(frozen=True)
class MatrixCase:
    name: str
    model: str
    input_tokens: int
    arch: str
    quantization: str
    runner_kwargs: dict[str, Any]
    cp_kwargs: dict[str, int]


MLA_PCP2_DCP1_CASE = MatrixCase(
    name="mla_pcp2_dcp1",
    model=DSV3_2_MODEL,
    input_tokens=32768,
    arch="moe",
    quantization="W8A8",
    runner_kwargs=MLA_RUNNER_KWARGS,
    cp_kwargs={"prefill_context_parallel_size": 2, "decode_context_parallel_size": 1},
)

MLA_PCP2_DCP2_CASE = MatrixCase(
    name="mla_pcp2_dcp2",
    model=DSV3_2_MODEL,
    input_tokens=32768,
    arch="moe",
    quantization="W8A8",
    runner_kwargs=MLA_RUNNER_KWARGS,
    cp_kwargs={"prefill_context_parallel_size": 2, "decode_context_parallel_size": 2},
)

MLA_PCP2_DCP4_CASE = MatrixCase(
    name="mla_pcp2_dcp4",
    model=DSV3_2_MODEL,
    input_tokens=32768,
    arch="moe",
    quantization="W8A8",
    runner_kwargs=MLA_RUNNER_KWARGS,
    # DCP reuses the TP x PCP workers, so DCP=PCP*TP is the widest MLA layout.
    cp_kwargs={"prefill_context_parallel_size": 2, "decode_context_parallel_size": 4},
)

# Standalone DCP keeps PCP=1; DCP groups live inside the TP workers, so this
# case runs the two-card TP2+DCP2 layout.
MLA_STANDALONE_DCP2_CASE = MatrixCase(
    name="mla_standalone_dcp2",
    model=DSV3_2_MODEL,
    input_tokens=8192,
    arch="moe",
    quantization="W8A8",
    runner_kwargs={**MLA_RUNNER_KWARGS, "tensor_parallel_size": 2},
    cp_kwargs={"prefill_context_parallel_size": 1, "decode_context_parallel_size": 2},
)

GQA_STANDALONE_DCP2_CASE = MatrixCase(
    name="gqa_standalone_dcp2",
    model=GQA_MODEL,
    input_tokens=32768,
    arch="dense",
    quantization="BF16",
    runner_kwargs=GQA_RUNNER_KWARGS,
    cp_kwargs={"prefill_context_parallel_size": 1, "decode_context_parallel_size": 2},
)

GQA_PCP2_DCP2_CASE = MatrixCase(
    name="gqa_pcp2_dcp2",
    model=GQA_MODEL,
    input_tokens=32768,
    arch="dense",
    quantization="BF16",
    runner_kwargs=GQA_RUNNER_KWARGS,
    # GQA stacking requires PCP == DCP.
    cp_kwargs={"prefill_context_parallel_size": 2, "decode_context_parallel_size": 2},
)


def _prompt_token_ids(token_count: int) -> list[int]:
    """Build a deterministic in-vocabulary prompt without downloading a tokenizer."""
    return [10 + index % 1000 for index in range(token_count)]


def _run_greedy(
    case: MatrixCase,
    *,
    cp_enabled: bool,
    graph_mode: str | None,
    prompt_token_ids: list[int],
) -> list[int]:
    runner_kwargs = dict(case.runner_kwargs)
    runner_kwargs["max_model_len"] = case.input_tokens + MAX_OUTPUT_TOKENS + 256
    runner_kwargs["max_num_batched_tokens"] = min(4096, case.input_tokens)
    runner_kwargs.update(case.cp_kwargs if cp_enabled else CP_OFF_KWARGS)
    compilation_config = {"cudagraph_mode": graph_mode, "cudagraph_capture_sizes": [MAX_NUM_SEQS]} if graph_mode else {}
    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=MAX_OUTPUT_TOKENS,
        detokenize=False,
    )
    with VllmRunner(
        case.model,
        enforce_eager=graph_mode is None,
        compilation_config=compilation_config,
        **runner_kwargs,
    ) as runner:
        outputs = runner.model.generate(
            [{"prompt_token_ids": prompt_token_ids}],
            sampling_params,
            use_tqdm=False,
        )
    assert len(outputs) == 1, f"{case.name}: expected one output, got {len(outputs)}"
    assert outputs[0].finished, f"{case.name}: generation did not finish"
    assert len(outputs[0].outputs) == 1, f"{case.name}: expected one completion, got {len(outputs[0].outputs)}"
    token_ids = list(outputs[0].outputs[0].token_ids)
    assert len(token_ids) == MAX_OUTPUT_TOKENS, f"{case.name}: generated {len(token_ids)} tokens"
    return token_ids


def _compare_cp_on_off(case: MatrixCase, graph_mode: str | None) -> None:
    prompt_token_ids = _prompt_token_ids(case.input_tokens)
    cp_off_tokens = _run_greedy(
        case,
        cp_enabled=False,
        graph_mode=graph_mode,
        prompt_token_ids=prompt_token_ids,
    )
    cp_on_tokens = _run_greedy(
        case,
        cp_enabled=True,
        graph_mode=graph_mode,
        prompt_token_ids=prompt_token_ids,
    )
    graph_mode_name = graph_mode or "eager"
    for position, (off_token, on_token) in enumerate(zip(cp_off_tokens, cp_on_tokens, strict=True)):
        assert off_token == on_token, (
            f"{case.name} ({graph_mode_name}): CP-on token {position} is {on_token}, "
            f"but CP-off generated {off_token}. "
            f"CP-off tokens: {cp_off_tokens}; CP-on tokens: {cp_on_tokens}"
        )


@pytest.mark.parametrize(
    "graph_mode",
    [
        pytest.param(
            graph_mode,
            id=f"mla-{graph_mode or 'eager'}",
            marks=pytest.mark.e2e_model(DSV3_2_MODEL),
        )
        for graph_mode in GRAPH_MODES
    ],
)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="long_sequence,chunked_prefill,prefix_caching",
    parallel="TP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager,full_decode_only",
)
@patch.dict(os.environ, ENVS)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mla_pcp2_dcp1_matches_cp_off(graph_mode: str | None) -> None:
    """MLA TP2 PCP2 DCP1 must match the CP-off greedy tokens on a 32k prompt."""
    _compare_cp_on_off(MLA_PCP2_DCP1_CASE, graph_mode)


@pytest.mark.parametrize(
    "graph_mode",
    [
        pytest.param(
            graph_mode,
            id=f"mla-{graph_mode or 'eager'}",
            marks=pytest.mark.e2e_model(DSV3_2_MODEL),
        )
        for graph_mode in GRAPH_MODES
    ],
)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="long_sequence,chunked_prefill,prefix_caching",
    parallel="TP,PCP,DCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager,full_decode_only",
)
@patch.dict(os.environ, ENVS)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mla_pcp2_dcp2_matches_cp_off(graph_mode: str | None) -> None:
    """MLA TP2 PCP2 DCP2 must match the CP-off greedy tokens on a 32k prompt."""
    _compare_cp_on_off(MLA_PCP2_DCP2_CASE, graph_mode)


@pytest.mark.parametrize(
    "graph_mode",
    [
        pytest.param(
            graph_mode,
            id=f"mla-{graph_mode or 'eager'}",
            marks=pytest.mark.e2e_model(DSV3_2_MODEL),
        )
        for graph_mode in GRAPH_MODES
    ],
)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="long_sequence,chunked_prefill,prefix_caching",
    parallel="TP,PCP,DCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager,full_decode_only",
)
@patch.dict(os.environ, ENVS)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mla_pcp2_dcp4_matches_cp_off(graph_mode: str | None) -> None:
    """MLA TP2 PCP2 DCP4 must match the CP-off greedy tokens on a 32k prompt."""
    _compare_cp_on_off(MLA_PCP2_DCP4_CASE, graph_mode)


@pytest.mark.parametrize(
    "graph_mode",
    [
        pytest.param(
            graph_mode,
            id=f"mla-{graph_mode or 'eager'}",
            marks=pytest.mark.e2e_model(DSV3_2_MODEL),
        )
        for graph_mode in GRAPH_MODES
    ],
)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="long_sequence,chunked_prefill,prefix_caching",
    parallel="TP,DCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager,full_decode_only",
)
@patch.dict(os.environ, ENVS)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mla_standalone_dcp2_matches_cp_off(graph_mode: str | None) -> None:
    """MLA TP2 DCP2 with PCP=1 must match the CP-off greedy tokens on an 8k prompt."""
    _compare_cp_on_off(MLA_STANDALONE_DCP2_CASE, graph_mode)


@pytest.mark.parametrize(
    "graph_mode",
    [
        pytest.param(
            graph_mode,
            id=f"gqa-{graph_mode or 'eager'}",
            marks=pytest.mark.e2e_model(GQA_MODEL),
        )
        for graph_mode in GRAPH_MODES
    ],
)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="long_sequence,chunked_prefill,prefix_caching",
    parallel="TP,DCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager,full_decode_only",
)
@patch.dict(os.environ, ENVS)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_gqa_standalone_dcp2_matches_cp_off(graph_mode: str | None) -> None:
    """GQA TP2 DCP2 with PCP=1 must match the CP-off greedy tokens on a 32k prompt."""
    _compare_cp_on_off(GQA_STANDALONE_DCP2_CASE, graph_mode)


@pytest.mark.parametrize(
    "graph_mode",
    [
        pytest.param(
            graph_mode,
            id=f"gqa-{graph_mode or 'eager'}",
            marks=pytest.mark.e2e_model(GQA_MODEL),
        )
        for graph_mode in GRAPH_MODES
    ],
)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="long_sequence,chunked_prefill,prefix_caching",
    parallel="TP,PCP,DCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager,full_decode_only",
)
@patch.dict(os.environ, ENVS)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_gqa_pcp2_dcp2_matches_cp_off(graph_mode: str | None) -> None:
    """GQA TP2 PCP2 DCP2 must match the CP-off greedy tokens on a 32k prompt."""
    _compare_cp_on_off(GQA_PCP2_DCP2_CASE, graph_mode)
