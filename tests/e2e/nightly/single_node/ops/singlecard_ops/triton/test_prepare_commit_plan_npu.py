# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integer reference checks for the RecoverSSM commit planner."""

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.mamba.prepare_commit_plan import prepare_commit_plan  # noqa: E402


@pytest.fixture
def npu():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    return torch.device("npu")


def assert_plan(plan, rows):
    expected = torch.tensor(rows, dtype=plan.dtype, device=plan.device).T
    batch = len(rows)
    torch.testing.assert_close(plan[:, :batch], expected, atol=0, rtol=0)
    torch.testing.assert_close(plan[1, :batch] + plan[2, :batch], plan[0, :batch], atol=0, rtol=0)
    assert not (plan[1, :batch].bool() & plan[2, :batch].bool()).any()
    torch.testing.assert_close(plan[:, batch:], torch.full_like(plan[:, batch:], -1), atol=0, rtol=0)


def test_prepare_commit_plan_clamps_accepted_prefix(npu):
    indices = torch.tensor([1, 2, 3, 4, 0, 6], dtype=torch.int32, device=npu)
    offsets = torch.tensor([0, 6, 8, 10, 12, 14, 14], dtype=torch.int32, device=npu)
    counts = torch.tensor([9, 9, -1, 0, 2, 2], dtype=torch.int32, device=npu)
    plan = torch.full((6, 8), -1, dtype=torch.int32, device=npu)
    prepare_commit_plan(counts, indices, offsets, *plan.unbind(), spec_query_len=5, null_block_id=0)
    # Accepted, state-only, boundary, final page, boundary page, boundary prefix.
    assert_plan(plan, [(5, 5, 0, 1, 0, 0), (2, 2, 0, 2, 0, 0), *([(0, 0, 0, 0, 0, 0)] * 4)])


def test_prepare_commit_plan_routes_aligned_pages(npu):
    indices = torch.tensor([1, 2, 3, 4, 0], dtype=torch.int32, device=npu)
    offsets = torch.tensor([0, -1, 5, -1, 10, -1, 15, -1, 20, -1, 25], dtype=torch.int64, device=npu)[::2]
    mapping = torch.tensor([1, -1, 3, -1, 0, -1, 4, -1, 2], dtype=torch.int32, device=npu)[::2]
    counts = torch.tensor([2, 4, 4, 2, 0], dtype=torch.int32, device=npu)
    computed = torch.tensor([1, 6, 6, 6, 6], dtype=torch.int32, device=npu)
    table = torch.tensor([[11, 12], [21, 22], [31, 32], [41, 42], [51, 52]], dtype=torch.int64, device=npu)
    table = table.repeat_interleave(2, dim=1)[:, ::2]
    plan = torch.full((6, 6), -1, dtype=torch.int32, device=npu)
    prepare_commit_plan(
        counts,
        indices,
        offsets,
        *plan.unbind(),
        spec_query_len=5,
        null_block_id=0,
        request_indices=mapping,
        block_table=table,
        num_computed_tokens=computed,
        mamba_block_size=8,
    )
    assert_plan(
        plan,
        [
            (4, 0, 4, 22, 21, 2),  # Crosses the boundary.
            (2, 0, 2, 41, 41, 2),  # Exactly ends at the boundary.
            (2, 2, 0, 11, 0, 7),  # No boundary page, with seven tokens to the next boundary.
            (0, 0, 0, 0, 0, 0),  # Zero acceptance.
            (0, 0, 0, 0, 0, 0),  # Null source page.
        ],
    )
