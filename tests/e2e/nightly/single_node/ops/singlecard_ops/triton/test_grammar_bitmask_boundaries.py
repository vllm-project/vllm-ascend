# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import _apply_grammar_bitmask_kernel


@pytest.mark.parametrize("vocab_size", [1, 31, 32, 33, 1023, 8191, 8192, 8193, 151936, 248320, 256001])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["allowed", "blocked", "mixed"])
def test_grammar_bitmask_boundaries(vocab_size, dtype, pattern):
    """Cover tail bits, padded strides and adaptive verification mapping."""
    generator = torch.Generator().manual_seed(14160)
    # Requests have 2, 0, 1, 3 logit rows. Include inactive positions that
    # alias another request's live row, and inactive positions past the end.
    cu = torch.tensor([0, 2, 2, 3, 6], dtype=torch.int32)
    mapping = torch.tensor([25, 0, 16, 8, 3, 24, 31], dtype=torch.int32)
    stride = 8
    storage = torch.randn(6, vocab_size + 17, generator=generator).to(dtype)
    storage[:, : min(vocab_size, 5)] = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")], dtype=dtype
    )[: min(vocab_size, 5)]
    words = triton.cdiv(vocab_size, 32)
    masks = torch.randint(-(2**31), 2**31 - 1, (len(mapping), words + 3), dtype=torch.int32, generator=generator)
    if pattern == "allowed":
        masks.fill_(-1)
    elif pattern == "blocked":
        masks.zero_()
    expected = storage.clone()
    columns = torch.arange(vocab_size)
    for i, entry in enumerate(mapping.tolist()):
        req, pos = divmod(entry, stride)
        if pos >= int(cu[req + 1] - cu[req]):
            continue
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
        MASK_STRIDE=stride,
        BLOCK_SIZE=8192,
    )
    torch.npu.synchronize()
    # Integer views check NaN payloads, signed zero, and row-padding canaries.
    bits = torch.int32 if dtype == torch.float32 else torch.int16
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_grammar_bitmask_preserves_raw_float_bits(dtype):
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    width = 32 if dtype == torch.float32 else 16
    generator = torch.Generator().manual_seed(31)
    raw = torch.randint(-(2 ** (width - 1)), 2 ** (width - 1) - 1, (1, 513), dtype=bits, generator=generator)
    # Signed zeros, subnormals, quiet and signalling NaN payloads.
    patterns = [0, 1, -(2 ** (width - 1)), -1]
    patterns += [0x7FC00001, 0x7F800001] if width == 32 else [0x7FC1, 0x7C01]
    raw[0, : len(patterns)] = torch.tensor(patterns, dtype=bits)
    logits = raw.view(dtype)
    expected = logits.clone()
    expected[0, 20] = -float("inf")
    masks = torch.full((1, 17), -1, dtype=torch.int32, device="npu")
    masks[0, 0] = ~(1 << 20)
    actual = logits.npu()
    _apply_grammar_bitmask_kernel[(1, 1)](
        actual,
        513,
        torch.tensor([0], dtype=torch.int32, device="npu"),
        torch.tensor([0, 1], dtype=torch.int32, device="npu"),
        masks,
        17,
        513,
        MASK_STRIDE=1,
        BLOCK_SIZE=8192,
    )
    torch.npu.synchronize()
    assert torch.equal(actual.cpu().view(bits), expected.view(bits))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grammar_bitmask_large_batch_tail(dtype):
    rows, vocab_size = 256, 256001
    words = triton.cdiv(vocab_size, 32)
    generator = torch.Generator().manual_seed(256)
    logits = torch.randn(rows, vocab_size, generator=generator).to(dtype)
    masks = torch.randint(-(2**31), 2**31 - 1, (rows, words), dtype=torch.int32, generator=generator)
    blocked = ((masks.to(torch.int64).unsqueeze(-1) >> torch.arange(32)) & 1).reshape(rows, -1)
    expected = logits.masked_fill(blocked[:, :vocab_size] == 0, -float("inf"))
    actual = logits.npu()
    _apply_grammar_bitmask_kernel[(rows, triton.cdiv(vocab_size, 8192))](
        actual,
        vocab_size,
        torch.arange(rows, dtype=torch.int32, device="npu"),
        torch.arange(rows + 1, dtype=torch.int32, device="npu"),
        masks.npu(),
        words,
        vocab_size,
        MASK_STRIDE=1,
        BLOCK_SIZE=8192,
    )
    torch.npu.synchronize()
    assert torch.equal(actual.cpu(), expected)
