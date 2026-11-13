# SPDX-License-Identifier: Apache-2.0
"""Exact BF16/FP32 ordering, payload and range coverage for fused candidate packing."""

import pytest
import torch
import torch_npu  # noqa: F401
from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.dcp_test_utils import local_to_global
from vllm_ascend.ops.triton.sharded_indexer import pack_local_candidates


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
@pytest.mark.parametrize("rows", [0, 48])
def test_pack_rejects_unsupported_score_dtype(dtype, rows):
    indices = torch.empty((rows, 1, 2048), dtype=torch.int32)
    scores = torch.empty(indices.shape, dtype=dtype)
    with pytest.raises(TypeError, match="scores must have dtype torch.bfloat16 or torch.float32"):
        pack_local_candidates(indices, scores, 8, 7, 768, sort_candidates=True, fuse_sort=True)


@pytest.mark.parametrize("rows", [48, 127, 144, 152, 161])
@pytest.mark.parametrize("column_stride", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("id_case", ["short_max", "short_boundary", "2**24", "full_max", "fallback"])
def test_sorted_pack_exact_order_and_payload(rows, column_stride, dtype, id_case):
    torch.npu.set_device(0)
    world, rank, interleave, k = 8, 7, 768, 2048
    generator = torch.Generator().manual_seed(20878)
    ids = torch.arange(k - 1, -1, -1, dtype=torch.int32).repeat(rows, 1).view(rows, 1, k)
    scores = torch.randint(-2, 3, ids.shape, generator=generator).to(dtype)
    short_bits = 23 if dtype == torch.bfloat16 else 20
    large_id = {
        "short_max": (1 << short_bits) - 1,
        "short_boundary": 1 << short_bits,
        "2**24": 1 << 24,
        "full_max": (1 << 25) - 2,
        "fallback": 1 << 26,
    }[id_case]
    # The high ID ties with low IDs, exposing truncated radix ID digits.
    ids[..., -1] = large_id
    scores[..., -1] = 1.0
    ids[0] = -1
    scores[..., :7] = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.0, -0.0, 1.0, -1.0])
    scores[..., 7] = torch.nextafter(torch.tensor(1.0, dtype=dtype), torch.tensor(2.0, dtype=dtype))
    scores = scores.float()
    valid = (ids >= 0) & scores.isfinite()
    values = torch.where(valid, scores, -torch.inf)
    mapped = torch.where(valid, local_to_global(ids, world, rank, interleave), -1)
    by_id = torch.where(valid, ids, torch.iinfo(torch.int32).max).argsort(stable=True)
    order = by_id.gather(-1, values.gather(-1, by_id).argsort(descending=True, stable=True))
    expected = torch.stack((mapped.gather(-1, order), values.gather(-1, order).contiguous().view(torch.int32)), -1)

    device_ids = torch.empty((rows, 1, k * column_stride), device="npu", dtype=torch.int32)[..., ::column_stride]
    # Allocate scores separately so both input column strides are exercised.
    device_scores = torch.empty((rows, 1, k * column_stride), device="npu", dtype=dtype)[..., ::column_stride]
    device_ids.copy_(ids)
    device_scores.copy_(scores)
    actual = pack_local_candidates(
        device_ids, device_scores, world, rank, interleave, sort_candidates=True, fuse_sort=True
    )
    assert torch.equal(actual.cpu(), expected[:, 0])
