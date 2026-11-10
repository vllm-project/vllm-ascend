# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

SMALL_GATE_ELEMENTS = 8192
MEDIUM_GATE_ELEMENTS = 65536
LARGE_GATE_ELEMENTS = 1048576
TRITON_POINTER_ALIGNMENT = 16


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


def _flash_attention_gate_block_size(numel: int) -> int:
    # Small decode tiles limit vector work; larger tiles amortize prefill loops.
    if numel <= SMALL_GATE_ELEMENTS:
        return 512
    if numel < MEDIUM_GATE_ELEMENTS:
        return 2048
    if numel < LARGE_GATE_ELEMENTS:
        return 4096
    return 8192


class FlashAttentionGateKernel(AscendTritonJitKernel["FlashAttentionGateKernel.CompileKey"]):
    kernel = _flash_attention_gate_kernel
    # Bind explicitly to avoid signature inspection on the eager hot path.
    bind_launch_inputs = False

    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        hidden: int
        projected_stride: int
        gate_stride: int
        block: int
        projected_aligned: bool
        gate_aligned: bool

    def dispatch(
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        projected_stride: int,
        gate_stride: int,
        block: int,
        projected_aligned: bool,
        gate_aligned: bool,
    ) -> CompileKey:
        return self.CompileKey(
            dtype=dtype,
            hidden=hidden,
            projected_stride=projected_stride,
            gate_stride=gate_stride,
            block=block,
            projected_aligned=projected_aligned,
            gate_aligned=gate_aligned,
        )

    def get_warmup_keys(
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        projected_stride: int,
        gate_stride: int,
        projected_aligned: bool = True,
        gate_aligned: bool = True,
    ) -> list[CompileKey]:
        if hidden <= 0:
            return []
        # Token count is dynamic, including prefill outside graph capture.
        # The first complete row and the first row count crossing each threshold
        # cover every reachable BLOCK without enumerating capture batch sizes.
        boundaries = (hidden, SMALL_GATE_ELEMENTS + 1, MEDIUM_GATE_ELEMENTS, LARGE_GATE_ELEMENTS)
        blocks = sorted({_flash_attention_gate_block_size(triton.cdiv(n, hidden) * hidden) for n in boundaries})
        return self._trace_dispatch(self.dispatch)(
            dtype=dtype,
            hidden=hidden,
            projected_stride=projected_stride,
            gate_stride=gate_stride,
            block=tuple(blocks),
            projected_aligned=projected_aligned,
            gate_aligned=gate_aligned,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            projected=TritonWarmupTensor(
                compile_key.dtype,
                aligned=compile_key.projected_aligned,
                shape=(1, compile_key.hidden),
                strides=(compile_key.projected_stride, 1),
            ),
            gate=TritonWarmupTensor(
                compile_key.dtype,
                aligned=compile_key.gate_aligned,
                shape=(1, compile_key.hidden),
                strides=(compile_key.gate_stride, 1),
            ),
            numel=compile_key.hidden,
            HIDDEN=compile_key.hidden,
            BLOCK=compile_key.block,
            grid=(1,),
        )

    @kernel_launcher
    def __call__(
        self,
        projected: torch.Tensor,
        gate: torch.Tensor,
        numel: int,
        HIDDEN: int,
        BLOCK: int,
        grid: tuple[int, ...],
    ) -> LaunchSpec:
        # Keep the caller's runtime grid, including its vector-core cap.
        return grid, dict(
            projected=projected,
            gate=gate,
            numel=numel,
            HIDDEN=HIDDEN,
            BLOCK=BLOCK,
            PROJECTED_STRIDE=projected.stride(0),
            GATE_STRIDE=gate.stride(0),
            multibuffer=False,
        )


_FLASH_ATTENTION_GATE_KERNEL = FlashAttentionGateKernel()


def register_flash_attention_gate_warmup(dtype: torch.dtype, hidden: int, gate_prefix: int = 0) -> None:
    """Register the owner's standalone or trailing fused-projection gate layout.

    Attention outputs and projection allocations are aligned. A trailing gate
    view retains the full projection row stride and offsets its base pointer.
    Registration is a no-op outside the existing model-construction window.
    """
    _FLASH_ATTENTION_GATE_KERNEL.register_warmup(
        dtype=dtype,
        hidden=hidden,
        projected_stride=hidden,
        gate_stride=hidden + gate_prefix,
        gate_aligned=(gate_prefix * dtype.itemsize) % TRITON_POINTER_ALIGNMENT == 0,
    )


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
    block = _flash_attention_gate_block_size(numel)
    if projected.stride(0) == gate.stride(0) == projected.shape[1]:
        tasks = triton.cdiv(numel, block)
    else:
        tasks = projected.shape[0] * triton.cdiv(projected.shape[1], block)
    grid = (min(tasks, get_vectorcore_num()),)
    _FLASH_ATTENTION_GATE_KERNEL(
        projected=projected,
        gate=gate,
        numel=numel,
        HIDDEN=projected.shape[1],
        BLOCK=block,
        grid=grid,
    )
    return projected
