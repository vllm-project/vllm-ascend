# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Fused relative-LSE preparation for bounded attention updates."""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

# Native npu_attention_update accepts at most 16 output/LSE pairs.
ATTENTION_UPDATE_MAX_INPUTS = 16


# Batch size varies across prefill steps; reuse the compiled kernel.
@triton.jit(do_not_specialize=["ROWS"])
def _prepare_attention_update_kernel(
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
    L10,
    L11,
    L12,
    L13,
    L14,
    L15,
    Strides: tl.constexpr,
    PreviousAnchor,
    Relative,
    Anchor,
    ROWS,
    COUNT: tl.constexpr,
    HAS_ANCHOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lses = (L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15)
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row < ROWS
    # The first group contains current causal attention, including each
    # query's own key. Its anchor is finite for ordinary model inputs.
    if HAS_ANCHOR:
        previous = tl.load(PreviousAnchor + row, mask, other=0)
        anchor = previous
    else:
        anchor = tl.load(L0 + row * Strides[0], mask, other=0).to(tl.float32)
    for i in tl.static_range(1, COUNT):
        score = tl.load(lses[i] + row * Strides[i], mask, other=0).to(tl.float32)
        # Native update ignores infinite history LSEs. They cannot anchor
        # normalization; leave their scores unchanged apart from translation.
        anchor = tl.maximum(anchor, tl.where(tl.abs(score) < float("inf"), score, -float("inf")))
    tl.store(Anchor + row, anchor, mask)
    for i in tl.static_range(COUNT):
        score = tl.load(lses[i] + row * Strides[i], mask, other=0).to(tl.float32)
        if HAS_ANCHOR and i == 0:
            # Keep the relative mass separate from the common offset.
            score = score + (previous - anchor)
        else:
            score = score - anchor
        tl.store(Relative + i * ROWS + row, score, mask)


class PrepareAttentionUpdateKernel(AscendTritonJitKernel["PrepareAttentionUpdateKernel.CompileKey"]):
    kernel = _prepare_attention_update_kernel
    BLOCK = 128
    bind_launch_inputs = False

    @dataclass(frozen=True)
    class CompileKey:
        count: int
        has_anchor: bool

    def dispatch(self, *, count: int, has_anchor: bool) -> "PrepareAttentionUpdateKernel.CompileKey":
        return self.CompileKey(count=count, has_anchor=has_anchor)

    def get_warmup_keys(self) -> list["PrepareAttentionUpdateKernel.CompileKey"]:
        # The first flush has 16 pairs. Subsequent flushes and the final
        # residual merge have an anchor and anywhere from 2 to 16 pairs.
        # Production LSEs are contiguous FP32; ROWS is do_not_specialize.
        return self._trace_dispatch(self.dispatch)(
            count=ATTENTION_UPDATE_MAX_INPUTS, has_anchor=False
        ) + self._trace_dispatch(self.dispatch)(count=tuple(range(2, ATTENTION_UPDATE_MAX_INPUTS + 1)), has_anchor=True)

    def warmup_inputs(self, compile_key: "PrepareAttentionUpdateKernel.CompileKey") -> dict[str, Any]:
        lse = TritonWarmupTensor(torch.float32, shape=(self.BLOCK,))
        return dict(
            lses=[lse] * compile_key.count,
            previous_anchor=lse if compile_key.has_anchor else None,
            relative=TritonWarmupTensor(torch.float32, shape=(compile_key.count, self.BLOCK)),
            anchor=lse,
            rows=self.BLOCK,
        )

    @kernel_launcher
    def __call__(
        self,
        lses: list[torch.Tensor],
        previous_anchor: torch.Tensor | None,
        relative: torch.Tensor,
        anchor: torch.Tensor,
        rows: int,
    ) -> LaunchSpec:
        # Explicit pointers also support Ascend launchers that cannot unpack
        # tuples of device pointers. Unused pointers are compiled away.
        pointers = (*lses, *((lses[0],) * (ATTENTION_UPDATE_MAX_INPUTS - len(lses))))
        return (triton.cdiv(rows, self.BLOCK),), dict(
            {f"L{i}": pointer for i, pointer in enumerate(pointers)},
            Strides=tuple(x.stride(0) for x in lses),
            PreviousAnchor=previous_anchor,
            Relative=relative,
            Anchor=anchor,
            ROWS=rows,
            COUNT=len(lses),
            HAS_ANCHOR=previous_anchor is not None,
            BLOCK=self.BLOCK,
        )


_PREPARE_ATTENTION_UPDATE_KERNEL = PrepareAttentionUpdateKernel()


def prepare_attention_update(
    lses: list[torch.Tensor], previous_anchor: torch.Tensor | None
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Normalize FP32 score vectors with one device launch.

    The first group must contain current causal attention. Its LSE is
    relative when previous_anchor is supplied; the remaining LSEs are
    absolute. Every real query has a finite current-chunk prefix LSE.
    """
    rows = lses[0].numel()
    relative = torch.empty((len(lses), rows), dtype=torch.float32, device=lses[0].device)
    anchor = lses[0].new_empty(rows)
    if rows:
        pointers = (*lses, *((lses[0],) * (ATTENTION_UPDATE_MAX_INPUTS - len(lses))))
        block = PrepareAttentionUpdateKernel.BLOCK
        _prepare_attention_update_kernel[(triton.cdiv(rows, block),)](
            *pointers,
            tuple(x.stride(0) for x in lses),
            previous_anchor,
            relative,
            anchor,
            rows,
            len(lses),
            previous_anchor is not None,
            block,
        )
    return relative.unbind(0), anchor
