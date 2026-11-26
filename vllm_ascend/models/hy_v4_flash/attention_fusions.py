# SPDX-License-Identifier: Apache-2.0
"""FIA output correction with a fused FP32 sink and masked BF16 store."""

import torch
from vllm.triton_utils import tl, triton

HEADS_PER_PROGRAM = 8


@triton.jit
def _sink_context_kernel(
    context,
    lse,
    sink,
    selected,
    output,
    heads: tl.constexpr,
    width: tl.constexpr,
    candidates: tl.constexpr,
    candidate_block: tl.constexpr,
    head_block: tl.constexpr,
    value_block: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * head_block + tl.arange(0, head_block)
    col = tl.arange(0, value_block)
    valid_head = head < heads
    index = tl.arange(0, candidate_block)
    position = tl.load(selected + row * candidates + index, index < candidates, -1)
    valid_row = tl.sum((position >= 0).to(tl.int32), 0) > 0
    normalization = tl.load(lse + row * heads + head, valid_head, 0).to(tl.float32)
    bias = tl.load(sink + head, valid_head, 0).to(tl.float32)
    ratio = tl.sigmoid(normalization - bias)
    offset = (row * heads + head[:, None]) * width + col[None, :]
    live = valid_head[:, None] & (col[None, :] < width)
    value = tl.load(context + offset, live, 0).to(tl.float32)
    corrected = tl.where(valid_row, value * ratio[:, None], 0.0)
    tl.store(output + offset, corrected, live)


def fused_sink_context(context, lse, sink, selected):
    rows, heads, width = context.shape
    output = torch.empty(context.shape, device=context.device, dtype=context.dtype)
    _sink_context_kernel[(rows, triton.cdiv(heads, HEADS_PER_PROGRAM))](
        context.contiguous(),
        lse.reshape(rows, heads).contiguous(),
        sink.contiguous(),
        selected.contiguous(),
        output,
        heads,
        width,
        selected.shape[-1],
        triton.next_power_of_2(selected.shape[-1]),
        HEADS_PER_PROGRAM,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return output


GATE_BLOCK_SIZE = 1024


@triton.jit
def _output_gate_kernel(value, logits, output, count: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    live = index < count
    x = tl.load(value + index, live, 0).to(tl.float32)
    gate = tl.load(logits + index, live, 0).to(tl.float32)
    # Preserve the BF16 sigmoid intermediate used by unquantized projection.
    probability = tl.sigmoid(gate).to(output.dtype.element_ty).to(tl.float32)
    tl.store(output + index, x * probability, live)


def fused_output_gate(value, logits):
    output = torch.empty_like(value)
    _output_gate_kernel[(triton.cdiv(value.numel(), GATE_BLOCK_SIZE),)](
        value,
        logits,
        output,
        value.numel(),
        GATE_BLOCK_SIZE,
        enable_fp_fusion=False,
    )
    return output
