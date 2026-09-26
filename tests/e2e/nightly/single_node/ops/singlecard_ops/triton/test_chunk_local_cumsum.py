# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum_scalar as fla_chunk_local_cumsum_scalar
from vllm_ascend.ops.triton.kda.cumsum import chunk_local_cumsum_scalar as kda_chunk_local_cumsum_scalar
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DEVICE = "npu"
RTOL = 1e-5
ATOL = 1e-4


def reference_chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    scale: float | None,
    cu_seqlens: list[int] | None,
    head_first: bool,
) -> torch.Tensor:
    """Compute a float32, per-sequence and per-chunk cumsum on CPU."""
    g = g.detach().to(device="cpu", dtype=torch.float32)
    if head_first:
        g = g.transpose(1, 2)

    sequence_ranges = [(0, g.shape[1])]
    if cu_seqlens is not None:
        sequence_ranges = list(zip(cu_seqlens[:-1], cu_seqlens[1:]))

    output = torch.empty_like(g)
    for sequence_start, sequence_end in sequence_ranges:
        for chunk_start in range(sequence_start, sequence_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, sequence_end)
            chunk = g[:, chunk_start:chunk_end]
            if reverse:
                chunk = chunk.flip(1)
            chunk = chunk.cumsum(1)
            if reverse:
                chunk = chunk.flip(1)
            if scale is not None:
                chunk = chunk * scale
            output[:, chunk_start:chunk_end] = chunk

    if head_first:
        output = output.transpose(1, 2)
    return output.contiguous()


@pytest.mark.parametrize(
    "cumsum_scalar",
    [
        pytest.param(fla_chunk_local_cumsum_scalar, id="fla"),
        pytest.param(kda_chunk_local_cumsum_scalar, id="kda"),
    ],
)
@pytest.mark.parametrize(
    ("shape", "chunk_size", "reverse", "scale", "cu_seqlens", "head_first"),
    [
        pytest.param((4, 2048, 8), 128, False, None, None, False, id="multi-batch"),
        pytest.param((2, 513, 4), 64, True, 0.25, None, False, id="tail-reverse-scale"),
        pytest.param((2, 4, 513), 64, False, None, None, True, id="head-first-tail"),
        pytest.param((1, 300, 8), 64, True, None, [0, 15, 100, 300], False, id="varlen"),
        pytest.param((1, 4, 300), 64, False, None, [0, 15, 100, 300], True, id="head-first-varlen"),
    ],
)
@torch.inference_mode()
def test_chunk_local_cumsum_scalar_kernel(
    cumsum_scalar: Callable[..., torch.Tensor],
    shape: tuple[int, int, int],
    chunk_size: int,
    reverse: bool,
    scale: float | None,
    cu_seqlens: list[int] | None,
    head_first: bool,
) -> None:
    init_device_properties_triton()
    torch.manual_seed(42)
    g = torch.randn(shape, dtype=torch.float32, device=DEVICE) * 0.1
    cu_seqlens_tensor = None
    if cu_seqlens is not None:
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int64, device=DEVICE)

    actual = cumsum_scalar(
        g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens_tensor,
        head_first=head_first,
        output_dtype=torch.float32,
    )
    expected = reference_chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
    )

    assert actual.shape == g.shape
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, rtol=RTOL, atol=ATOL)
