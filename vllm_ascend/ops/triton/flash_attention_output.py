# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

SMALL_GATE_ELEMENTS = 8192
MEDIUM_GATE_ELEMENTS = 65536
LARGE_GATE_ELEMENTS = 1048576


@triton.jit(do_not_specialize=["numel"])
def _flash_attention_gate_kernel(
    projected,
    gate,
    numel,
    HIDDEN: tl.constexpr,
    PROJECTED_STRIDE: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    CONTIGUOUS: tl.constexpr = PROJECTED_STRIDE == HIDDEN and GATE_STRIDE == HIDDEN
    if CONTIGUOUS:
        tasks = tl.cdiv(numel, BLOCK)
    else:
        tasks = numel // HIDDEN * tl.cdiv(HIDDEN, BLOCK)
    for block in range(tl.program_id(0), tasks, tl.num_programs(0)):
        if CONTIGUOUS:
            # Cross row boundaries without vector division or partial row tiles.
            offsets = block * BLOCK + tl.arange(0, BLOCK)
            projected_offsets = offsets
            gate_offsets = offsets
            mask = offsets < numel
        else:
            row = block // tl.cdiv(HIDDEN, BLOCK)
            columns = block % tl.cdiv(HIDDEN, BLOCK) * BLOCK + tl.arange(0, BLOCK)
            projected_offsets = row * PROJECTED_STRIDE + columns
            gate_offsets = row * GATE_STRIDE + columns
            mask = columns < HIDDEN
        values = tl.load(projected + projected_offsets, mask, other=0)
        gates = tl.load(gate + gate_offsets, mask, other=0).to(tl.float32)
        # Preserve the intermediate rounding of torch.sigmoid(gate).
        scale = tl.sigmoid(gates).to(gate.dtype.element_ty)
        tl.store(projected + projected_offsets, values * scale, mask)


def flash_attention_gate(projected: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply output * sigmoid(gate) in place, preserving sigmoid rounding."""
    if projected.ndim != 2 or gate.ndim != 2:
        raise ValueError("Projected and gate tensors must be 2-D")
    if projected.shape != gate.shape:
        raise ValueError(f"Gate shape {gate.shape} must match projected shape {projected.shape}")
    if projected.dtype != gate.dtype:
        raise ValueError("Projected and gate dtypes must match")
    if projected.numel() == 0:
        return projected
    if projected.stride(1) != 1 or gate.stride(1) != 1:
        raise ValueError("Projected and gate columns must be contiguous")
    init_device_properties_triton()
    numel = projected.numel()
    # Small decode tiles limit vector work; larger tiles amortize prefill loops.
    if numel <= SMALL_GATE_ELEMENTS:
        block = 512
    elif numel < MEDIUM_GATE_ELEMENTS:
        block = 2048
    elif numel < LARGE_GATE_ELEMENTS:
        block = 4096
    else:
        block = 8192
    if projected.stride(0) == gate.stride(0) == projected.shape[1]:
        tasks = triton.cdiv(numel, block)
    else:
        tasks = projected.shape[0] * triton.cdiv(projected.shape[1], block)
    grid = (min(tasks, get_vectorcore_num()),)
    _flash_attention_gate_kernel[grid](
        projected,
        gate,
        numel,
        HIDDEN=projected.shape[1],
        PROJECTED_STRIDE=projected.stride(0),
        GATE_STRIDE=gate.stride(0),
        BLOCK=block,
        multibuffer=False,
    )
    return projected
