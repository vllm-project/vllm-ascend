# SPDX-License-Identifier: Apache-2.0
"""Experimental Ascend add+argmax; validate on the target NPU before use."""

import torch
from vllm.triton_utils import tl, triton

BLOCK_SIZE = 4096
# Host-side capacity validation only. JIT functions use literal constants.
SENTINEL = 2147483647


@triton.jit
def _reduce_first_argmax(values, indices):
    # Standard reductions avoid the custom multi-result tl.reduce lowering.
    sentinel = 2147483647
    is_nan = values != values
    first_nan = tl.min(tl.where(is_nan, indices, sentinel), axis=0)
    has_nan = first_nan != sentinel
    maximum = tl.max(tl.where(is_nan, -float("inf"), values), axis=0)
    first_max = tl.min(
        tl.where((~is_nan) & (values == maximum), indices, sentinel),
        axis=0,
    )
    value = tl.where(has_nan, float("nan"), maximum)
    index = tl.where(has_nan, first_nan, first_max)
    return value, index


@triton.jit
def _load_sum(A, B, row, cols, SA, SB, V: tl.constexpr):
    valid = cols < V
    a = tl.load(A + row.to(tl.int64) * SA + cols, valid, other=0)
    b = tl.load(B + row.to(tl.int64) * SB + cols, valid, other=0)
    # Reproduce materialized torch.add with matching source dtypes.
    z = (a.to(tl.float32) + b.to(tl.float32)).to(A.dtype.element_ty).to(tl.float32)
    return tl.where(valid, z, -float("inf")), tl.where(valid, cols, 2147483647)


@triton.jit
def _partial(A, B, PV, PI, SA, SB, V: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    values, indices = _load_sum(A, B, row, cols, SA, SB, V)
    value, index = _reduce_first_argmax(values, indices)
    offset = row.to(tl.int64) * PARTS + part
    tl.store(PV + offset, value)
    tl.store(PI + offset, index)


@triton.jit
def _finish(PV, PI, OUT, SO, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    valid = part < PARTS
    value = tl.load(PV + row.to(tl.int64) * PARTS + part, valid, other=-float("inf"))
    index = tl.load(PI + row.to(tl.int64) * PARTS + part, valid, other=2147483647)
    _, winner = _reduce_first_argmax(value, index)
    tl.store(OUT + row.to(tl.int64) * SO, winner.to(OUT.dtype.element_ty))


class TwoStageAddArgmax:
    """Allocate before graph capture; share only on one ordered stream.

    output must not alias either input or scratch. Concurrent graph executions
    require separate workspaces. No allocation or host read occurs in run().
    """

    def __init__(self, max_rows, max_vocab, device, block=BLOCK_SIZE):
        if max_rows < 1 or not 0 < max_vocab < SENTINEL:
            raise ValueError("Invalid workspace capacity")
        if block < 1 or block & (block - 1):
            raise ValueError("block must be a positive power of two")
        self.max_rows = max_rows
        self.max_vocab = max_vocab
        self.block = block
        capacity = max_rows * triton.cdiv(max_vocab, block)
        self.values = torch.empty(capacity, dtype=torch.float32, device=device)
        self.indices = torch.empty(capacity, dtype=torch.int32, device=device)

    def supports(self, a, b, out):
        return (
            a.ndim == b.ndim == 2
            and a.shape == b.shape
            and a.device.type in ("npu", "privateuseone")
            and a.dtype == b.dtype
            and a.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and a.stride(1) == b.stride(1) == 1
            and a.shape[0] <= self.max_rows
            and 0 < a.shape[1] <= self.max_vocab
            and out.ndim == 1
            and out.shape[0] == a.shape[0]
            and out.dtype in (torch.int32, torch.int64)
            and out.stride(0) > 0
            and a.device == b.device == out.device == self.values.device
        )

    def run(self, a, b, out):
        if not self.supports(a, b, out):
            raise ValueError("Unsupported inputs; caller must use original path")
        rows, vocab = a.shape
        if rows == 0:
            return out
        parts = triton.cdiv(vocab, self.block)
        _partial[(rows, parts)](
            a,
            b,
            self.values,
            self.indices,
            a.stride(0),
            b.stride(0),
            vocab,
            parts,
            self.block,
        )
        _finish[(rows,)](
            self.values,
            self.indices,
            out,
            out.stride(0),
            parts,
            max(16, triton.next_power_of_2(parts)),
        )
        return out
