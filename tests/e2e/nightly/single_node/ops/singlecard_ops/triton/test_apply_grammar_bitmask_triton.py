# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import (
    _apply_grammar_bitmask_kernel,
)

ROWS = 64
NUM_REQS = 16
LOGITS_PER_REQ = 4
VOCAB_SIZE = 151936
BLOCK_SIZE = 8192
BITMASK_WORDS = (VOCAB_SIZE + 31) // 32
# Must exceed the largest position packed into the mapping.
MASK_STRIDE = 8


def _build_reference(
    logits: torch.Tensor,
    logit_rows: torch.Tensor,
    active: torch.Tensor,
    bitmask: torch.Tensor,
) -> torch.Tensor:
    bit_values = torch.ones(32, dtype=torch.int64) << torch.arange(32, dtype=torch.int64)
    blocked = (bitmask.to(torch.int64)[:, :, None] & bit_values[None, None, :]) == 0
    blocked = blocked.reshape(ROWS, -1)[:, :VOCAB_SIZE]
    # Inactive mapping entries must not touch the logits.
    blocked &= active[:, None]

    expected = logits.clone()
    indices = logit_rows.to(torch.int64)
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

    # cu_num_logits: cumulative logit rows per request (NUM_REQS + 1 entries).
    cu_num_logits = torch.arange(
        0,
        (NUM_REQS + 1) * LOGITS_PER_REQ,
        LOGITS_PER_REQ,
        dtype=torch.int32,
    )

    # Mapping packs (req_idx, position) as req_idx * MASK_STRIDE + position.
    # Mask rows 0..59 use positions 0..3 of requests 0..14 (all active); the
    # last 4 rows point at out-of-range positions of request 15, which must be
    # skipped (position_is_active=False).
    req_idx = torch.arange(ROWS) // LOGITS_PER_REQ
    position = torch.arange(ROWS) % LOGITS_PER_REQ
    position[60:] += LOGITS_PER_REQ  # positions 4..7 -> inactive
    mapping = req_idx * MASK_STRIDE + position
    logit_rows = req_idx * LOGITS_PER_REQ + torch.clamp(position, max=LOGITS_PER_REQ - 1)
    active = position < LOGITS_PER_REQ
    mapping = mapping.to(torch.int32)

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
        logit_rows,
        active,
        bitmask,
    )

    device = torch.device("npu:0")
    actual = logits.to(device)
    mapping_npu = mapping.to(device)
    cu_num_logits_npu = cu_num_logits.to(device)
    bitmask_npu = bitmask.to(device)

    grid = (
        ROWS,
        triton.cdiv(VOCAB_SIZE, BLOCK_SIZE),
    )
    _apply_grammar_bitmask_kernel[grid](
        actual,
        actual.stride(0),
        mapping_npu,
        cu_num_logits_npu,
        bitmask_npu,
        bitmask_npu.stride(0),
        VOCAB_SIZE,
        MASK_STRIDE=MASK_STRIDE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    torch.npu.synchronize()

    assert torch.equal(actual.cpu(), expected)
