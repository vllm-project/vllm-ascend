# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import math
from collections.abc import Callable

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.qwen4_exp.gbsa import qsa_gbsa_attention
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: E402,F401

ROWS = 2
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 128
PAGE_SIZE = 128
ATOL = 2e-2
RTOL = 2e-2
GbsaCase = tuple[
    Callable[[], torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _compact_indices() -> torch.Tensor:
    groups = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="npu")
    residual_slots = torch.full((ROWS, 3), -1, dtype=torch.int32, device="npu")
    return torch.cat((groups, residual_slots), dim=1)


def _reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for row in range(query.shape[0]):
        request = int(token_to_req[row])
        length = int(seq_lens[request])
        tokens = torch.arange(length, device=query.device)
        pages = block_table[request, tokens // PAGE_SIZE].long()
        offsets = tokens % PAGE_SIZE
        keys = key_cache[pages, offsets].float().repeat_interleave(2, dim=1)
        values = value_cache[pages, offsets].float().repeat_interleave(2, dim=1)
        scores = torch.einsum("hd,shd->hs", query[row].float(), keys)
        probabilities = (scores / math.sqrt(HEAD_DIM)).softmax(-1)
        rows.append(torch.einsum("hs,shd->hd", probabilities, values))
    return torch.stack(rows)


def _make_case() -> GbsaCase:
    torch.manual_seed(20261009)
    query = torch.randn((ROWS, NUM_Q_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    key_cache = torch.randn((4, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="npu")
    value_cache = torch.randn_like(key_cache)
    compact_indices = _compact_indices()
    block_table = torch.tensor([[2], [3]], dtype=torch.int32, device="npu")
    token_to_req = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([9, 11], dtype=torch.int32, device="npu")
    logical_positions = seq_lens - 1
    output = torch.empty_like(query)

    def call() -> torch.Tensor:
        return qsa_gbsa_attention(
            query,
            key_cache,
            value_cache,
            compact_indices,
            block_table,
            token_to_req,
            logical_positions,
            seq_lens,
            output,
        )

    return call, query, key_cache, value_cache, block_table, token_to_req, seq_lens


def _expected(case: GbsaCase) -> torch.Tensor:
    _, query, key_cache, value_cache, block_table, token_to_req, seq_lens = case
    return _reference(query, key_cache, value_cache, block_table, token_to_req, seq_lens)


@pytest.mark.parametrize("execution", ("eager", "compile"))
@torch.inference_mode()
def test_qwen4_exp_gbsa_matches_dense_reference(execution: str) -> None:
    case = _make_case()
    call = case[0]
    expected = _expected(case)
    if execution == "compile":
        call = torch.compile(call, backend="npugraph_ex", fullgraph=True, dynamic=False)

    actual = call().clone()
    torch.npu.synchronize()

    torch.testing.assert_close(actual.float(), expected, rtol=RTOL, atol=ATOL)


@torch.inference_mode()
def test_qwen4_exp_gbsa_npugraph_replays_updated_input() -> None:
    case = _make_case()
    call, query, *_ = case
    call()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured_output = call()

    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured_output.float(), _expected(case), rtol=RTOL, atol=ATOL)

    query.mul_(0.5)
    expected_replay = _expected(case)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured_output.float(), expected_replay, rtol=RTOL, atol=ATOL)
