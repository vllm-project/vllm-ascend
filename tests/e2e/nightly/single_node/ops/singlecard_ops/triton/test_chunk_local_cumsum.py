# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Correctness tests for the FLA ``chunk_local_cumsum`` Triton op on NPU.

The op sizes its tile with a ``BLOCK_T`` heuristic derived from the number of v-heads.
Before the clamp, any model with enough v-heads (``H >= 125`` for the GDN
``chunk_size`` of 64) got ``BLOCK_T < chunk_size``, hence ``N_CHUNKS == 0`` in the kernel
and an immediate reshape failure on the first prefill. These tests compare the op against
a plain ``torch`` chunk-local cumsum reference for small and large head counts.
"""

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

CHUNK_SIZE = 64
# Smallest v-head count whose unclamped BLOCK_T heuristic drops below CHUNK_SIZE.
FIRST_COLLAPSING_HEAD_COUNT = 125
ATOL = 1e-5

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)


def _chunk_local_cumsum_reference(gate, chunk_size, cu_seqlens=None, reverse=False, scale=None, head_first=False):
    """Chunk-local cumulative sum in fp32, one independent cumsum per chunk."""
    seq_first = gate.transpose(1, 2) if head_first else gate
    source = seq_first.float()
    reference = torch.empty_like(source)
    num_tokens = source.shape[1]
    if cu_seqlens is None:
        seq_ranges = [(batch, 0, num_tokens) for batch in range(source.shape[0])]
    else:
        seq_ranges = [(0, int(start), int(end)) for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
    for batch, seq_start, seq_end in seq_ranges:
        for chunk_start in range(seq_start, seq_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_end)
            chunk = source[batch, chunk_start:chunk_end]
            if reverse:
                chunk = chunk.flip(0)
            accumulated = chunk.cumsum(0)
            if reverse:
                accumulated = accumulated.flip(0)
            if scale is not None:
                accumulated = accumulated * scale
            reference[batch, chunk_start:chunk_end] = accumulated
    return reference.transpose(1, 2) if head_first else reference


def _make_gate(num_heads, num_tokens, head_first, device="npu"):
    torch.manual_seed(20261115 + num_heads + num_tokens)
    shape = (1, num_heads, num_tokens) if head_first else (1, num_tokens, num_heads)
    return (-torch.rand(shape, dtype=torch.float32, device=device) * 0.05).to(device)


@pytest.mark.parametrize("num_heads", [64, FIRST_COLLAPSING_HEAD_COUNT, 128, 256])
@pytest.mark.parametrize("num_tokens", [32, CHUNK_SIZE, 192])
@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_local_cumsum_matches_reference(num_heads, num_tokens, head_first):
    gate = _make_gate(num_heads, num_tokens, head_first)

    out = chunk_local_cumsum(gate, chunk_size=CHUNK_SIZE, head_first=head_first)
    reference = _chunk_local_cumsum_reference(gate.cpu(), CHUNK_SIZE, head_first=head_first)

    assert out.shape == gate.shape
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), reference, atol=ATOL, rtol=0)


@pytest.mark.parametrize("num_heads", [64, FIRST_COLLAPSING_HEAD_COUNT, 128, 256])
def test_chunk_local_cumsum_with_scale_and_reverse(num_heads):
    gate = _make_gate(num_heads, 192, head_first=False)
    scale = 1.4426950408889634

    out = chunk_local_cumsum(gate, chunk_size=CHUNK_SIZE, scale=scale, reverse=True)
    reference = _chunk_local_cumsum_reference(gate.cpu(), CHUNK_SIZE, reverse=True, scale=scale)

    torch.testing.assert_close(out.cpu(), reference, atol=ATOL, rtol=0)


@pytest.mark.parametrize("num_heads", [FIRST_COLLAPSING_HEAD_COUNT, 128, 256])
@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_local_cumsum_varlen_matches_reference(num_heads, head_first):
    # cu_seqlens requires batch size 1; sequences are deliberately not chunk-aligned.
    # Restricted to head counts whose clamped BLOCK_T equals CHUNK_SIZE, so the varlen tile
    # grid is chunk-aligned exactly like the reference below.
    cu_seqlens = torch.tensor([0, 37, 96, 192], dtype=torch.int32)
    gate = _make_gate(num_heads, 192, head_first)

    out = chunk_local_cumsum(gate, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, head_first=head_first)
    reference = _chunk_local_cumsum_reference(
        gate.cpu(), CHUNK_SIZE, cu_seqlens=cu_seqlens.tolist(), head_first=head_first
    )

    torch.testing.assert_close(out.cpu(), reference, atol=ATOL, rtol=0)


def test_chunk_local_cumsum_rejects_unsupported_rank():
    gate = torch.rand((1, 8, 8, 64), dtype=torch.float32, device="npu")
    with pytest.raises(ValueError, match="Unsupported input shape"):
        chunk_local_cumsum(gate, chunk_size=CHUNK_SIZE)
