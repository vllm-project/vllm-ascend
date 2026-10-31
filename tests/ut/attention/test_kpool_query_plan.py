# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.kpool_query_plan import build_kpool_query_plan


@pytest.mark.parametrize(
    "contexts,queries,topk",
    [
        ([0], [8192], 2048),
        ([0, 7, 8, 9, 100], [2, 5, 3, 1, 2], 8),
        ([0, 15, 16, 17], [17, 3, 2, 1], 16),
        ([3, 4, 5, 7, 8], [1, 4, 3, 2, 1], 8),
        ([0, 0], [0, 0], 8),
        ([0, 0, 0], [1, 2, 3], 8),
        ([2048, 4096], [128, 1024], 2048),
        ([1024, 1024], [1536, 1536], 2048),
    ],
)
def test_query_plan_matches_per_position_visibility(contexts, queries, topk):
    starts = torch.tensor([0, *torch.tensor(queries).cumsum(0).tolist()], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=len(queries),
        num_actual_tokens=sum(queries),
        query_start_loc_cpu=starts,
        seq_lens_cpu=torch.tensor([c + q for c, q in zip(contexts, queries)], dtype=torch.int32),
    )
    expected_full: list[int] = []
    expected_sparse: list[int] = []
    expected_ends: list[int] = []
    row = 0
    for context, query in zip(contexts, queries):
        for position in range(context, context + query):
            (expected_full if position < topk else expected_sparse).append(row)
            row += 1
        expected_ends.append(len(expected_sparse))
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")):
        plan = build_kpool_query_plan(common, topk, torch.device("cpu"), min_partial_rows=0)
    if not expected_full:
        assert plan is None
        return
    assert plan is not None
    assert plan.num_tokens == sum(queries)
    assert [row for begin, end in plan.full_visible_ranges for row in range(begin, end)] == expected_full
    assert plan.selective_rows.tolist() == expected_sparse
    assert plan.cum_query_lens.tolist() == expected_ends
    assert plan.selection_index[expected_full].tolist() == [-1] * len(expected_full)
    assert plan.selection_index[expected_sparse].tolist() == list(range(len(expected_sparse)))
    # Removing only request prefixes preserves every selective row's causal
    # position when the original final key lengths are passed to CANN.
    begin = 0
    for request, end in enumerate(expected_ends):
        count = end - begin
        expected_positions = list(
            range(contexts[request] + queries[request] - count, contexts[request] + queries[request])
        )
        original = [contexts[request] + row - int(starts[request]) for row in expected_sparse[begin:end]]
        assert original == expected_positions
        begin = end


@pytest.mark.parametrize("mirror", [None, torch.empty(2, device="meta")])
def test_missing_host_lengths_does_not_fetch_device_metadata(mirror):
    common = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 3]), seq_lens_cpu=mirror)
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")):
        assert build_kpool_query_plan(common, 8, torch.device("cpu")) is None


def test_query_plan_excludes_padded_requests_and_rejects_invalid_lengths():
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=3,
        query_start_loc_cpu=torch.tensor([0, 2, 3, 8]),
        seq_lens_cpu=torch.tensor([4, 9, 0]),
    )
    plan = build_kpool_query_plan(common, 8, torch.device("cpu"), min_partial_rows=0)
    assert plan is not None
    assert plan.full_visible_ranges == ((0, 2),)
    assert plan.selective_rows.tolist() == [2]
    assert plan.cum_query_lens.tolist() == [0, 1, 1]
    common.seq_lens_cpu[0] = 1
    assert build_kpool_query_plan(common, 8, torch.device("cpu")) is None


def test_host_upper_bound_conservatively_shortens_full_visible_prefix():
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=9,
        query_start_loc_cpu=torch.tensor([0, 6, 9]),
        seq_lens_cpu=torch.tensor([11, 20]),
    )
    plan = build_kpool_query_plan(common, 8, torch.device("cpu"), min_partial_rows=0)
    assert plan is not None
    assert plan.full_visible_ranges == ((0, 3),)
    assert plan.selective_rows.tolist() == [3, 4, 5, 6, 7, 8]
    # Actual histories can be shorter than the bound; bypassed rows must
    # remain full-visible for every possible length covered by that bound.
    for actual_length in range(6, 12):
        assert all(actual_length - 6 + row < 8 for row in range(3))


@pytest.mark.parametrize("context,query,admitted", [(0, 33, True), (1025, 8192, False), (1024, 8192, True)])
def test_partial_bypass_requires_enough_work_to_amortize_packing(context, query, admitted):
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=query,
        query_start_loc_cpu=torch.tensor([0, query]),
        seq_lens_cpu=torch.tensor([context + query]),
    )
    assert (build_kpool_query_plan(common, 2048, torch.device("cpu")) is not None) is admitted


def test_query_plan_staging_ignores_default_tensor_device():
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=3,
        query_start_loc_cpu=torch.tensor([0, 3]),
        seq_lens_cpu=torch.tensor([3]),
    )
    with torch.device("meta"):
        plan = build_kpool_query_plan(common, 8, torch.device("cpu"))
    assert plan is not None
    assert plan.selection_index.tolist() == [-1, -1, -1]
    assert plan.cum_query_lens.tolist() == [0]
