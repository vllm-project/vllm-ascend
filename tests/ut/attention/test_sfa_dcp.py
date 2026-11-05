# SPDX-License-Identifier: Apache-2.0
"""Public DCP eligibility and exact integer-layout contracts."""

import pytest
import torch

from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import can_exchange
from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton


@pytest.mark.parametrize(
    "tokens,heads,width,ranks,dtype,expected",
    [
        (1, 16, 512, 8, torch.bfloat16, True),
        (257, 128, 512, 2, torch.bfloat16, True),
        (0, 16, 512, 8, torch.bfloat16, False),
        (1, 16, 512, 1, torch.bfloat16, False),
        (1, 12, 512, 3, torch.bfloat16, False),
        (1, 96, 512, 8, torch.bfloat16, False),
        (1, 16, 513, 8, torch.bfloat16, False),
        (1, 16, 512, 8, torch.float16, False),
    ],
)
def test_exchange_layout_contract(tokens, heads, width, ranks, dtype, expected):
    output = torch.empty(tokens, heads, width, dtype=dtype)
    lse = torch.empty(tokens, heads, 1)
    assert can_exchange(output, lse, ranks) == expected


@pytest.mark.parametrize("shape", [(0, 2048), (2, 0), (2, 3, 0)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_empty_remap_preserves_input(shape, dtype):
    indices = torch.empty(shape, dtype=dtype)
    assert remap_sparse_indices_triton(indices, 8, 0, 128) is indices


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rank", [0, 2])
def test_non_power_of_two_layout_keeps_exact_integer_boundaries(dtype, rank):
    values = [-1, 0, 126, 127, 381, 2**24 - 1, 2**24, 2**24 + 1, 2**31 - 1]
    if dtype == torch.int64:
        values += [2**31, 2**40, 2**62 - 1]
    source = torch.tensor([values, values[::-1]], dtype=dtype)
    result = remap_sparse_indices_triton(source, 3, rank, 127)
    expected = torch.full_like(source, -1)
    for row, target in zip(source.tolist(), expected):
        owned = [x // 381 * 127 + x % 127 for x in row if x >= 0 and x // 127 % 3 == rank]
        target[: len(owned)] = torch.tensor(owned, dtype=dtype)
    assert torch.equal(result, expected)
    assert result.dtype == dtype


@pytest.mark.parametrize("world,rank,interleave", [(0, 0, 128), (2, 2, 128), (2, 0, 0)])
def test_invalid_remap_geometry_is_rejected(world, rank, interleave):
    with pytest.raises(ValueError, match="DCP size/interleave"):
        remap_sparse_indices_triton(torch.zeros(1, 7, dtype=torch.int32), world, rank, interleave)


def test_remap_rejects_silent_dtype_conversion():
    with pytest.raises(TypeError, match="INT32 or INT64"):
        remap_sparse_indices_triton(torch.zeros(1, 7), 8, 0, 128)
