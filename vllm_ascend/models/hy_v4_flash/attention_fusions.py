# SPDX-License-Identifier: Apache-2.0
"""Output gate fusion preserving the BF16 sigmoid intermediate."""

import torch
from vllm.triton_utils import tl, triton

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
