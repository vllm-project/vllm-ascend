# SPDX-License-Identifier: Apache-2.0
"""Single-operator accuracy test for the Step5 append_current_region Triton
op (the GBSA id pack front-end).

The op is a pure integer pack (count + copy + current-region append), so
both outputs are compared exactly against a pure-torch reference.
"""

import gc

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.step5.summary_update import append_current_region


def make_ids(positions, groups, topk, region_size, device):
    """Ascending ids 0..live-1 with -1 tail, as indexer_score_topk yields."""
    live = (positions.to(torch.int32) // region_size).clamp(max=topk)
    cols = torch.arange(topk, device=device, dtype=torch.int32)
    mask = cols.unsqueeze(0) < live.unsqueeze(1)  # [tokens, topk]
    ids = torch.where(mask, cols, torch.full_like(cols, -1))
    return ids.repeat(groups, 1).contiguous()  # group-major [G*T, topk]


def refappend_current_region(ids, positions, groups, region_size):
    rows, topk = ids.shape
    tokens = rows // groups
    live = (ids != -1).sum(dim=1)
    wide = torch.full((rows, topk + 1), -1, dtype=torch.int32, device=ids.device)
    wide[:, :topk] = ids
    current = (positions.to(torch.int32) // region_size).repeat(groups)
    wide[torch.arange(rows, device=ids.device), live.clamp(max=topk)] = current
    counts = (live + 1).to(torch.int32)
    return wide.view(groups, tokens, topk + 1), counts.view(groups, tokens)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
@pytest.mark.parametrize(
    "tokens, groups, topk, max_pos, pos_dtype",
    [
        # Real Step5 DSA shape: topk = 512, region_size = 8, decode batch.
        (64, 4, 512, 8192, torch.int64),
        # Single-token decode (tokens == 1 specialization class).
        (1, 1, 512, 8192, torch.int64),
        # Long prefill chunk, single group.
        (2048, 1, 512, 16384, torch.int32),
        # Small / odd topk values (block_k tile variations).
        (17, 2, 64, 1024, torch.int64),
        (33, 3, 3, 256, torch.int64),
        (16, 1, 1, 64, torch.int32),
        # Positions below one region: live == 0 everywhere (current region
        # lands at slot 0, counts == 1).
        (9, 2, 64, 8, torch.int64),
        # Positions >= topk * region_size: full selections (live == topk),
        # the append overwrites the extra column.
        (16, 2, 64, 4096, torch.int64),
    ],
)
def testappend_current_region(tokens, groups, topk, max_pos, pos_dtype):
    torch.manual_seed(0)
    device = "npu"
    region_size = 8
    positions = torch.randint(0, max_pos, (tokens,), device=device).to(pos_dtype)
    ids = make_ids(positions, groups, topk, region_size, device)

    wide, counts = append_current_region(ids, positions, groups=groups, region_size=region_size)

    wide_ref, counts_ref = refappend_current_region(ids, positions, groups, region_size)
    torch.testing.assert_close(wide, wide_ref, rtol=0, atol=0, msg="packed ids mismatch")
    torch.testing.assert_close(counts, counts_ref, rtol=0, atol=0, msg="block counts mismatch")

    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
def test_append_current_region_empty():
    device = "npu"
    ids = torch.zeros(0, 8, dtype=torch.int32, device=device)
    positions = torch.zeros(0, dtype=torch.int64, device=device)
    wide, counts = append_current_region(ids, positions, groups=1, region_size=8)
    assert wide.shape == (1, 0, 9)
    assert counts.shape == (1, 0)
