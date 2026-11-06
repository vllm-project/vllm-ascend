# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.apply_grammar_bitmask import _apply_grammar_bitmask_kernel


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
