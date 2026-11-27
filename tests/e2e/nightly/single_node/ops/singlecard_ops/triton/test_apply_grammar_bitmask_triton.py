# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.apply_grammar_bitmask import (
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


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
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


@pytest.mark.parametrize(
    "dtype,vocab_size",
    [
        (torch.float32, 1),
        (torch.float16, 31),
        (torch.bfloat16, 33),
        (torch.float32, 8193),
        (torch.bfloat16, 151936),
        (torch.float16, 256001),
    ],
)
def test_grammar_bitmask_boundaries(dtype, vocab_size):
    generator = torch.Generator().manual_seed(14160)
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    width = torch.iinfo(bits).bits
    raw = torch.randint(
        -(2 ** (width - 1)), 2 ** (width - 1) - 1, (6, vocab_size + 17), dtype=bits, generator=generator
    )
    # Mixed rows preserve signed zero, subnormals and quiet/signalling NaN payloads.
    payloads = {
        torch.float32: [0, -2147483648, 1, 0x7FC00001, 0x7F800001],
        torch.float16: [0, -32768, 1, 0x7E01, 0x7C01],
        torch.bfloat16: [0, -32768, 1, 0x7FC1, 0x7F81],
    }
    raw[:, :5] = torch.tensor(payloads[dtype], dtype=bits)
    storage = raw.view(dtype)
    cu = torch.tensor([0, 2, 2, 3, 6], dtype=torch.int32)
    # Includes inactive positions aliasing live rows and positions past the end.
    mapping = torch.tensor([25, 0, 16, 8, 3, 24, 31], dtype=torch.int32)
    words = triton.cdiv(vocab_size, 32)
    masks = torch.full((len(mapping), words + 3), 0x55555555, dtype=torch.int32)
    masks[0, :words] = -1
    masks[1, :words] = 0
    masks[2:, 0] |= 31
    expected = storage.clone()
    columns = torch.arange(vocab_size)
    for i, entry in enumerate(mapping.tolist()):
        req, pos = divmod(entry, 8)
        if pos < int(cu[req + 1] - cu[req]):
            blocked = ((masks[i, columns // 32].to(torch.int64) >> (columns % 32)) & 1) == 0
            expected[int(cu[req]) + pos, :vocab_size].masked_fill_(blocked, -float("inf"))
    actual = storage.npu()
    _apply_grammar_bitmask_kernel[(len(mapping), triton.cdiv(vocab_size, 8192))](
        actual,
        actual.stride(0),
        mapping.npu(),
        cu.npu(),
        masks.npu(),
        masks.stride(0),
        vocab_size,
        MASK_STRIDE=8,
        BLOCK_SIZE=8192,
    )
    torch.npu.synchronize()
    assert torch.equal(actual.cpu().view(bits), expected.view(bits))


@pytest.mark.parametrize("num_masks,vocab_size", [(0, 8192), (3, 0), (0, 0)])
def test_grammar_bitmask_empty(num_masks, vocab_size):
    logits = torch.empty((3, vocab_size), device="npu")
    mapping = torch.zeros(num_masks, dtype=torch.int32, device="npu")
    cu = torch.arange(4, dtype=torch.int32, device="npu")
    masks = torch.empty((num_masks, triton.cdiv(vocab_size, 32)), dtype=torch.int32, device="npu")
    _apply_grammar_bitmask_kernel[(num_masks, triton.cdiv(vocab_size, 8192))](
        logits,
        logits.stride(0),
        mapping,
        cu,
        masks,
        masks.stride(0),
        vocab_size,
        MASK_STRIDE=1,
        BLOCK_SIZE=8192,
    )
    torch.npu.synchronize()
