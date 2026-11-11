# SPDX-License-Identifier: Apache-2.0
"""Check union membership against the original per-query sparse selection."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.qwen4_exp.gbsa_union import build_qsa_union_inputs


@pytest.mark.parametrize("rows,topk", [(1, 8), (7, 8), (8, 8), (9, 8), (17, 512)])
def test_union_preserves_exact_membership(rows, topk):
    generator = torch.Generator().manual_seed(811)
    indices = torch.randint(0, 1024, (1, rows, topk), generator=generator, dtype=torch.int32)
    counts = torch.randint(0, topk + 1, (1, rows), generator=generator, dtype=torch.int32)
    visible = torch.arange(rows, dtype=torch.int32) + 4000
    # Include empty rows, duplicate selections, padding and partial final groups.
    counts[0, 0] = 0
    for row in range(rows):
        indices[0, row, counts[0, row] :] = -1
    actual = build_qsa_union_inputs(indices.npu(), counts.npu(), visible.npu())
    union_idx, union_count, exact = [tensor.cpu() for tensor in actual]
    for group in range((rows + 7) // 8):
        start, end = group * 8, min(rows, group * 8 + 8)
        selected = [set(indices[0, row, : counts[0, row]].tolist()) for row in range(start, end)]
        expected_union = sorted(set().union(*selected))
        count = int(union_count[0, group])
        assert count == len(expected_union)
        assert union_idx[0, group, :count].tolist() == expected_union
        assert (union_idx[0, group, count:] == -1).all()
        for rank, block in enumerate(expected_union):
            for lane in range(4):
                expected = sum(
                    1 << query_lane
                    for query_lane, blocks in enumerate(selected)
                    if block in blocks and block * 4 + lane < int(visible[start + query_lane])
                )
                assert int(exact[0, group, rank * 4 + lane]) == expected
        assert (exact[0, group, count * 4 :] == 0).all()


def test_large_union_matches_cpu_reference():
    # Exercise the expanded scatter index at the real chunk shape. Keeping that
    # index contiguous is required by the current A3 backend at this size.
    rows, topk = 4096, 512
    indices = torch.empty((1, rows, topk), dtype=torch.int32)
    indices[0, :, :384] = torch.arange(384, dtype=torch.int32)
    indices[0, :, 384:] = (
        512 + torch.arange(rows, dtype=torch.int32).remainder(8)[:, None] * 128 + torch.arange(128, dtype=torch.int32)
    )
    counts = torch.full((1, rows), topk, dtype=torch.int32)
    visible = torch.arange(rows, dtype=torch.int32) + 126977
    expected = build_qsa_union_inputs(indices, counts, visible)
    actual = build_qsa_union_inputs(indices.npu(), counts.npu(), visible.npu())
    for reference, candidate in zip(expected, actual):
        assert torch.equal(reference, candidate.cpu())
