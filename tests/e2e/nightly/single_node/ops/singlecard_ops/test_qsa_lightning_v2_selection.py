# SPDX-License-Identifier: Apache-2.0
"""NPU selector A/B tests; require both the old LI and CANN LI V2.

Run explicitly after syncing the QSA adaptation and building the V2 bindings:
pytest -o addopts= -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_qsa_lightning_v2_selection.py
Missing CANN entry points are failures, not skips. CPU calculations below are
independent position-based references, not mocks of the NPU selector.
"""

import math
from dataclasses import dataclass

import pytest
import torch

pytest.importorskip("torch_npu")
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401
from vllm_ascend.models.qwen4_exp.lightning_indexer import (  # noqa: E402
    qsa_select_paged_tokens_lightning,
    qsa_select_paged_tokens_lightning_v2,
)

HEAD_DIM = 128
COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
GROUP_TOPK = TOKEN_TOPK // COMPRESS_RATIO
OUTPUT_WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1
PACKED_WIDTH = GROUP_TOPK + COMPRESS_RATIO - 1
# This tolerance applies only to deliberately random, possibly near-tied scores.
# Separated-score cases below require exact top-k sets and descending order.
RANDOM_SCORE_ATOL = 0.05


@dataclass
class SelectionCase:
    query: torch.Tensor
    cache: torch.Tensor
    table: torch.Tensor
    requests: torch.Tensor
    positions: torch.Tensor
    lengths: torch.Tensor
    boundaries: torch.Tensor
    query_cpu: torch.Tensor
    logical_keys_cpu: list[torch.Tensor]
    positions_cpu: torch.Tensor
    requests_cpu: torch.Tensor
    lengths_cpu: tuple[int, ...]

    def arguments(self):
        return (self.query, self.cache, self.table, self.requests, self.positions,
                self.lengths, self.boundaries, TOKEN_TOPK, COMPRESS_RATIO)


def make_selection_case(lengths, query_lengths, *, heads=4, page_size=192, random_scores=False):
    generator = torch.Generator().manual_seed(1031)
    assert len(lengths) == len(query_lengths)
    assert all(0 < q <= length for length, q in zip(lengths, query_lengths))
    boundaries = [0]
    for count in query_lengths:
        boundaries.append(boundaries[-1] + count)
    positions = torch.cat([torch.arange(length - count, length) for length, count in zip(lengths, query_lengths)])
    requests = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(query_lengths))
    pages_per_request = max(1, math.ceil(max(lengths) / COMPRESS_RATIO / page_size))
    pages = len(lengths) * pages_per_request
    # Logical requests use distinct, permuted pages. Padding values are poison
    # for accidental reads, and stepping dim 0 preserves descriptor-like gaps.
    table = torch.randperm(pages, generator=generator).reshape(len(lengths), pages_per_request).to(torch.int32)
    storage_cpu = torch.full((pages * 2, page_size, 1, HEAD_DIM), 97, dtype=torch.bfloat16)
    logical_keys = []
    for request, length in enumerate(lengths):
        count = length // COMPRESS_RATIO
        if random_scores:
            keys = torch.randn(count, HEAD_DIM, generator=generator).to(torch.bfloat16)
        else:
            keys = torch.zeros(count, HEAD_DIM, dtype=torch.bfloat16)
            score_ids = torch.arange(count) if request % 2 == 0 else torch.arange(count).flip(0)
            # A BF16-exact base-128 encoding gives strictly separated scores,
            # including beyond 512 groups, without relying on random tie breaks.
            keys[:, 0] = (score_ids % HEAD_DIM).to(torch.bfloat16)
            keys[:, 1] = (score_ids // HEAD_DIM).to(torch.bfloat16)
        logical_keys.append(keys)
        for group in range(count):
            page = int(table[request, group // page_size])
            storage_cpu[page * 2, group % page_size, 0] = keys[group]
    if random_scores:
        query_cpu = torch.randn(boundaries[-1], heads, HEAD_DIM, generator=generator).to(torch.bfloat16)
    else:
        query_cpu = torch.zeros(boundaries[-1], heads, HEAD_DIM, dtype=torch.bfloat16)
        query_cpu[:, 0, 0] = 1
        query_cpu[:, 0, 1] = HEAD_DIM
        # Negative head contributions must be removed by ReLU before reduction.
        query_cpu[:, 1, :2] = -query_cpu[:, 0, :2]
    device = "npu:0"
    return SelectionCase(
        query_cpu.to(device), storage_cpu.to(device)[::2], table.to(device),
        requests.to(device), positions.to(device), torch.tensor(lengths, dtype=torch.int32, device=device),
        torch.tensor(boundaries, dtype=torch.int32, device=device), query_cpu, logical_keys,
        positions, requests, tuple(lengths),
    )


@pytest.mark.parametrize("head_stride", [0, 1])
def test_v2_accepts_noncanonical_singleton_cache_stride(head_stride):
    case = make_selection_case((2053, 2056), (3, 2))
    strides = list(case.cache.stride())
    strides[2] = head_stride
    case.cache = case.cache.as_strided(case.cache.shape, strides)
    original_strides = case.cache.stride()
    original_pointer = case.cache.data_ptr()

    result = qsa_select_paged_tokens_lightning_v2(*case.arguments())

    assert_selection(case, result)
    assert case.cache.stride() == original_strides
    assert case.cache.data_ptr() == original_pointer


def row_reference_scores(case, row):
    request = int(case.requests_cpu[row])
    position = int(case.positions_cpu[row])
    visible = min((position + 1) // COMPRESS_RATIO, case.lengths_cpu[request] // COMPRESS_RATIO)
    keys = case.logical_keys_cpu[request][:visible].float()
    # Reference uses explicit row positions and logical keys, independently of
    # V2's suffix-length mask and physical page mapping.
    weight = float(torch.tensor(1 / math.sqrt(HEAD_DIM), dtype=torch.bfloat16))
    return ((case.query_cpu[row].float() @ keys.T).relu() * weight).sum(0)


def assert_selection(case, result, *, score_atol=0.0):
    actual = result.cpu()
    assert actual.shape == (len(case.query), OUTPUT_WIDTH)
    assert actual.dtype == torch.int32
    for row, position in enumerate(case.positions_cpu.tolist()):
        scores = row_reference_scores(case, row)
        count = min(GROUP_TOPK, len(scores))
        selected = actual[row, :count].long()
        assert ((selected >= 0) & (selected < len(scores))).all()
        assert selected.unique().numel() == count
        assert (actual[row, count:GROUP_TOPK] == -1).all()
        if count:
            selected_scores = scores[selected]
            # Order matters when GBSA drops the last full group for a tail.
            assert (selected_scores[1:] <= selected_scores[:-1] + score_atol).all()
            threshold = scores.topk(count).values[-1]
            assert (selected_scores >= threshold - score_atol).all()
            must_select = set(torch.where(scores > threshold + score_atol)[0].tolist())
            assert must_select <= set(selected.tolist())
        tail_start = (position + 1) // COMPRESS_RATIO * COMPRESS_RATIO
        tail_end = min(position + 1, case.lengths_cpu[int(case.requests_cpu[row])])
        tail = list(range(tail_start, tail_end))
        assert actual[row, GROUP_TOPK:GROUP_TOPK + len(tail)].tolist() == tail
        assert (actual[row, GROUP_TOPK + len(tail):] == -1).all()
    return actual


@pytest.mark.parametrize("lengths,query_lengths", [
    ((1, 2, 3, 4, 7, 11), (1, 2, 3, 4, 7, 11)),  # prefill, including zero complete groups
    ((3, 4, 7, 8, 9, 2044, 2048, 2052), (1, 1, 1, 1, 1, 1, 1, 1)),  # decode
    ((7, 11, 2053, 2056), (1, 4, 3, 2)),  # mixed / chunked suffixes / multi-token decode
    ((2047, 2048, 2049, 2051), (3, 3, 3, 3)),  # 511/512/513 visibility and tails
])
@pytest.mark.parametrize("heads,page_size", [(4, 192), (12, 384)])
def test_selectors_match_position_reference(lengths, query_lengths, heads, page_size):
    case = make_selection_case(lengths, query_lengths, heads=heads, page_size=page_size)
    assert not case.cache.is_contiguous()
    results = []
    for selector in (qsa_select_paged_tokens_lightning, qsa_select_paged_tokens_lightning_v2):
        buffer = torch.full((len(case.query), OUTPUT_WIDTH), 123, dtype=torch.int32, device=case.query.device)
        result = selector(*case.arguments(), out=buffer)
        assert result is buffer
        results.append(assert_selection(case, result))
    # Independently validated sets are exact for these separated-score inputs.
    old, new = results
    torch.testing.assert_close(old[:, :GROUP_TOPK].sort(1).values, new[:, :GROUP_TOPK].sort(1).values)
    torch.testing.assert_close(old[:, GROUP_TOPK:], new[:, GROUP_TOPK:])


def test_random_multihead_scores_against_independent_reference():
    case = make_selection_case((2053, 2081), (3, 2), heads=12, random_scores=True)
    for selector in (qsa_select_paged_tokens_lightning, qsa_select_paged_tokens_lightning_v2):
        assert_selection(case, selector(*case.arguments()), score_atol=RANDOM_SCORE_ATOL)


def test_v2_padding_and_empty_batch():
    case = make_selection_case((7, 11), (2, 3))
    expected = qsa_select_paged_tokens_lightning_v2(*case.arguments()).clone()
    extra_rows = 4
    padded_query = torch.cat((case.query, torch.zeros_like(case.query[:extra_rows])))
    padded_table = torch.cat((case.table, torch.zeros_like(case.table[:1])))
    padded_requests = torch.cat((case.requests, torch.full((extra_rows,), 2, device=case.query.device)))
    padded_positions = torch.cat((case.positions, torch.full((extra_rows,), -1, device=case.query.device)))
    padded_lengths = torch.cat((case.lengths, torch.ones(1, dtype=torch.int32, device=case.query.device)))
    padded_boundaries = torch.cat((case.boundaries, case.boundaries[-1:] + extra_rows))
    buffer = torch.full((len(padded_query), OUTPUT_WIDTH), 99, dtype=torch.int32, device=case.query.device)
    result = qsa_select_paged_tokens_lightning_v2(
        padded_query, case.cache, padded_table, padded_requests, padded_positions,
        padded_lengths, padded_boundaries, TOKEN_TOPK, COMPRESS_RATIO, buffer,
        num_actual_requests=2, num_actual_tokens=len(case.query),
    )
    assert result is buffer
    torch.testing.assert_close(result[:len(case.query)], expected)
    assert (result[len(case.query):] == -1).all()
    empty = qsa_select_paged_tokens_lightning_v2(
        case.query[:0], case.cache, case.table[:0], case.requests[:0], case.positions[:0],
        case.lengths[:0], case.boundaries[:1], TOKEN_TOPK, COMPRESS_RATIO,
    )
    assert empty.shape == (0, OUTPUT_WIDTH)
