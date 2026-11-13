# SPDX-License-Identifier: Apache-2.0
"""Exact FP32 ordering, payload and range coverage for fused candidate packing."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.dcp_test_utils import local_to_global
from vllm_ascend.ops.triton.sharded_indexer import pack_local_candidates


@pytest.mark.parametrize("rows", [48, 127, 144, 152, 161])
@pytest.mark.parametrize("column_stride", [1, 2])
@pytest.mark.parametrize("large_id", [1 << 24, (1 << 25) - 2, 1 << 26])
def test_fp32_sorted_pack_exact_order_and_payload(rows, column_stride, large_id):
    torch.npu.set_device(0)
    world, rank, interleave, k = 8, 7, 768, 2048
    generator = torch.Generator().manual_seed(20878)
    ids = torch.arange(k - 1, -1, -1, dtype=torch.int32).repeat(rows, 1).view(rows, 1, k)
    scores = torch.randint(-2, 3, ids.shape, generator=generator).float()
    # The high ID ties with low IDs, exposing truncated radix ID digits.
    ids[..., -1] = large_id
    scores[..., -1] = 1.0
    ids[0] = -1
    scores[..., :7] = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.0, -0.0, 1.0, -1.0])
    scores[..., 7] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    valid = (ids >= 0) & scores.isfinite()
    values = torch.where(valid, scores, -torch.inf)
    mapped = torch.where(valid, local_to_global(ids, world, rank, interleave), -1)
    by_id = torch.where(valid, ids, torch.iinfo(torch.int32).max).argsort(stable=True)
    order = by_id.gather(-1, values.gather(-1, by_id).argsort(descending=True, stable=True))
    expected = torch.stack((mapped.gather(-1, order), values.gather(-1, order).contiguous().view(torch.int32)), -1)

    device_ids = torch.empty((rows, 1, k * column_stride), device="npu", dtype=torch.int32)[..., ::column_stride]
    # Allocate scores separately so both input column strides are exercised.
    device_scores = torch.empty((rows, 1, k * column_stride), device="npu")[..., ::column_stride]
    device_ids.copy_(ids)
    device_scores.copy_(scores)
    actual = pack_local_candidates(
        device_ids, device_scores, world, rank, interleave, sort_candidates=True, fuse_sort=True
    )
    assert torch.equal(actual.cpu(), expected[:, 0])
