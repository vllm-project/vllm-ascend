import gc
import random

import numpy as np
import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

SEED = 45
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@pytest.fixture(autouse=True, scope="module")
def _init_triton_device():
    """Initialize device properties before compiling the Triton kernels."""
    init_device_properties_triton()


def _make_strictly_lower(
    batch_size: int,
    total_length: int,
    num_heads: int,
    block_size: int,
    dtype: torch.dtype,
    sequence_lengths: list[int] | None = None,
) -> torch.Tensor:
    """Create chunk-local strictly lower-triangular matrices."""
    matrix = torch.zeros(
        batch_size,
        total_length,
        num_heads,
        block_size,
        dtype=torch.float32,
    )

    def fill_batch(batch_index: int, start: int, length: int) -> None:
        for chunk_start in range(0, length, block_size):
            chunk_length = min(block_size, length - chunk_start)
            block = torch.randn(num_heads, chunk_length, chunk_length) * 0.05
            block = torch.tril(block, diagonal=-1)
            row_start = start + chunk_start
            row_end = row_start + chunk_length
            matrix[batch_index, row_start:row_end, :, :chunk_length] = block.permute(1, 0, 2)

    if sequence_lengths is None:
        for batch_index in range(batch_size):
            fill_batch(batch_index, 0, total_length)
    else:
        offset = 0
        for sequence_length in sequence_lengths:
            fill_batch(0, offset, sequence_length)
            offset += sequence_length

    return matrix.to(dtype)


def _solve_tril_golden(
    matrix: torch.Tensor,
    sequence_lengths: list[int] | None = None,
) -> torch.Tensor:
    """Compute chunk-local ``(I + A)^-1`` using PyTorch."""
    batch_size, total_length, num_heads, block_size = matrix.shape
    golden = torch.zeros_like(matrix, dtype=torch.float32)
    matrix = matrix.to(torch.float32)

    def fill_batch(batch_index: int, start: int, length: int) -> None:
        for chunk_start in range(0, length, block_size):
            chunk_length = min(block_size, length - chunk_start)
            row_start = start + chunk_start
            row_end = row_start + chunk_length
            for head_index in range(num_heads):
                block = matrix[batch_index, row_start:row_end, head_index, :chunk_length]
                identity = torch.eye(chunk_length, dtype=torch.float32)
                golden[batch_index, row_start:row_end, head_index, :chunk_length] = torch.linalg.inv(identity + block)

    if sequence_lengths is None:
        for batch_index in range(batch_size):
            fill_batch(batch_index, 0, total_length)
    else:
        offset = 0
        for sequence_length in sequence_lengths:
            fill_batch(0, offset, sequence_length)
            offset += sequence_length

    return golden


def _assert_chunk_inverses_close(
    actual: torch.Tensor,
    golden: torch.Tensor,
    sequence_lengths: list[int] | None = None,
    atol: float = 2e-4,
    rtol: float = 2e-4,
) -> None:
    """Compare valid lower triangles; upper-triangular storage is undefined."""
    batch_size, total_length, num_heads, block_size = actual.shape

    def check_batch(batch_index: int, start: int, length: int) -> None:
        for chunk_start in range(0, length, block_size):
            chunk_length = min(block_size, length - chunk_start)
            row_start = start + chunk_start
            row_end = row_start + chunk_length
            for head_index in range(num_heads):
                actual_block = actual[batch_index, row_start:row_end, head_index, :chunk_length]
                golden_block = golden[batch_index, row_start:row_end, head_index, :chunk_length]
                torch.testing.assert_close(
                    torch.tril(actual_block),
                    golden_block,
                    atol=atol,
                    rtol=rtol,
                )

    if sequence_lengths is None:
        for batch_index in range(batch_size):
            check_batch(batch_index, 0, total_length)
    else:
        offset = 0
        for sequence_length in sequence_lengths:
            check_batch(0, offset, sequence_length)
            offset += sequence_length


@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_solve_tril_fixed_length(block_size, input_dtype):
    from vllm_ascend.ops.triton.fla.solve_tril import solve_tril

    batch_size, total_length, num_heads = 2, block_size + 7, 2
    matrix = _make_strictly_lower(
        batch_size,
        total_length,
        num_heads,
        block_size,
        input_dtype,
    )
    golden = _solve_tril_golden(matrix)

    actual = solve_tril(matrix.npu(), output_dtype=torch.float32)

    assert actual.dtype == torch.float32
    _assert_chunk_inverses_close(actual.cpu(), golden)

    del actual, matrix, golden
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_solve_tril_variable_length(block_size, output_dtype):
    from vllm_ascend.ops.triton.fla.solve_tril import solve_tril

    sequence_lengths = [block_size - 3, block_size + 5, 2 * block_size + 1]
    total_length = sum(sequence_lengths)
    matrix = _make_strictly_lower(
        1,
        total_length,
        2,
        block_size,
        torch.bfloat16,
        sequence_lengths,
    )
    golden = _solve_tril_golden(matrix, sequence_lengths).to(output_dtype)
    cu_seqlens = torch.tensor(
        [0] + list(np.cumsum(sequence_lengths)),
        dtype=torch.int32,
        device="npu",
    )

    actual = solve_tril(
        matrix.npu(),
        cu_seqlens=cu_seqlens,
        output_dtype=output_dtype,
    )

    assert actual.dtype == output_dtype
    _assert_chunk_inverses_close(
        actual.cpu(),
        golden,
        sequence_lengths,
        atol=5e-3,
        rtol=5e-3,
    )

    del actual, cu_seqlens, matrix, golden
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
