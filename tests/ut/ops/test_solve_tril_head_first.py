# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.fla.solve_tril import solve_tril
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices


def _has_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="solve_tril requires an NPU device")


def _make_strict_lower(seqlens: list[int], heads: int, block_size: int) -> torch.Tensor:
    torch.manual_seed(0)
    total = sum(seqlens)
    matrix = torch.zeros((1, heads, total, block_size), dtype=torch.bfloat16, device="npu")
    start = 0
    for length in seqlens:
        for block_start in range(0, length, block_size):
            valid = min(block_size, length - block_start)
            values = torch.randn((heads, valid, block_size), dtype=torch.float32, device="npu") * 0.02
            matrix[0, :, start + block_start : start + block_start + valid] = torch.tril(values, diagonal=-1).to(
                matrix.dtype
            )
        start += length
    return matrix


def _golden(matrix: torch.Tensor, seqlens: list[int]) -> torch.Tensor:
    _, heads, _, block_size = matrix.shape
    expected = torch.empty_like(matrix, device="cpu")
    source = matrix.cpu().float()
    eye = torch.eye(block_size, dtype=torch.float32)
    start = 0
    for length in seqlens:
        for block_start in range(0, length, block_size):
            valid = min(block_size, length - block_start)
            for head in range(heads):
                block = torch.zeros((block_size, block_size), dtype=torch.float32)
                block[:valid] = source[0, head, start + block_start : start + block_start + valid]
                expected[0, head, start + block_start : start + block_start + valid] = torch.linalg.inv(eye + block)[:valid]
        start += length
    return expected


@pytest.mark.parametrize("seqlens", [[128], [77, 118]])
def test_solve_tril_head_first_matches_golden(seqlens: list[int]):
    block_size = 64
    matrix = _make_strict_lower(seqlens, heads=4, block_size=block_size)
    expected = _golden(matrix, seqlens)
    cu_seqlens = None
    chunk_indices = None
    if len(seqlens) > 1:
        cu_seqlens = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], dtype=torch.int64, device="npu")
        chunk_indices = {
            str(size): prepare_chunk_indices(cu_seqlens, size)
            for size in (32, 64, 1216)
        }

    actual = solve_tril(
        matrix,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        output_dtype=matrix.dtype,
    )
    torch.npu.synchronize()

    assert actual.shape == matrix.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2)
