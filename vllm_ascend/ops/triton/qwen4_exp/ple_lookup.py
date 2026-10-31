# SPDX-License-Identifier: Apache-2.0
"""Gather BF16 PLE rows from registered host memory."""

from vllm.triton_utils import tl, triton


@triton.jit
def _lookup(
    ptrs,
    ids,
    out,
    START: tl.constexpr,
    END: tl.constexpr,
    WIDTH: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(ids + row).to(tl.int64)
    owned = (index >= START) & (index < END)
    local = tl.where(owned, index - START, 0)
    base = tl.load(ptrs + local // CHUNK).to(tl.pointer_type(tl.bfloat16))
    col = tl.arange(0, BLOCK)
    value = tl.load(base + (local % CHUNK) * WIDTH + col, mask=owned & (col < WIDTH), other=0)
    tl.store(out + row * WIDTH + col, value, mask=col < WIDTH)


def lookup(ptrs, ids, output, start, end, chunk):
    if ids.numel() == 0:
        return
    ids = ids.contiguous()
    width = output.shape[-1]
    _lookup[(ids.numel(),)](ptrs, ids, output, start, end, width, chunk, triton.next_power_of_2(width))
