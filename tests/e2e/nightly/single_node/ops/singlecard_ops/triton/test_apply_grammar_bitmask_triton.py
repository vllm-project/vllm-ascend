# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import (
    _apply_grammar_bitmask_kernel,
)

ROWS = 64
VOCAB_SIZE = 151936
BLOCK_SIZE = 8192
BITMASK_WORDS = (VOCAB_SIZE + 31) // 32


def _build_reference(
    logits: torch.Tensor,
    logits_indices: torch.Tensor,
    bitmask: torch.Tensor,
) -> torch.Tensor:
    bit_values = torch.ones(32, dtype=torch.int64) << torch.arange(32, dtype=torch.int64)
    blocked = (bitmask.to(torch.int64)[:, :, None] & bit_values[None, None, :]) == 0
    blocked = blocked.reshape(ROWS, -1)[:, :VOCAB_SIZE]

    expected = logits.clone()
    indices = logits_indices.to(torch.int64)
    mapped = expected.index_select(0, indices)
    mapped.masked_fill_(blocked, -float("inf"))
    expected.index_copy_(0, indices, mapped)
    return expected


def test_apply_grammar_bitmask_business_shape():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    logits = torch.randn(
        (ROWS, VOCAB_SIZE),
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)

    # Use a non-identity mapping to cover the compact-mask-row -> logits-row
    # contract used by structured output.
    logits_indices = torch.arange(
        ROWS - 1,
        -1,
        -1,
        dtype=torch.int32,
    )

    bitmask = torch.randint(
        -(2**31),
        2**31 - 1,
        (ROWS, BITMASK_WORDS),
        dtype=torch.int32,
        generator=generator,
    )

    # Explicitly cover fully allowed, fully blocked and alternating packed
    # words while keeping the real business shape.
    bitmask[:, 0] = -1
    bitmask[:, 1] = 0
    bitmask[:, 2] = 0x55555555
    bitmask[:, 3] = -1431655766  # int32 representation of 0xAAAAAAAA

    expected = _build_reference(
        logits,
        logits_indices,
        bitmask,
    )

    device = torch.device("npu:0")
    actual = logits.to(device)
    logits_indices_npu = logits_indices.to(device)
    bitmask_npu = bitmask.to(device)

    grid = (
        ROWS,
        triton.cdiv(VOCAB_SIZE, BLOCK_SIZE),
    )
    _apply_grammar_bitmask_kernel[grid](
        actual,
        actual.stride(0),
        logits_indices_npu,
        bitmask_npu,
        bitmask_npu.stride(0),
        VOCAB_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    assert torch.equal(actual.cpu(), expected)
