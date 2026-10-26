# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Fused relative-LSE preparation for bounded attention updates."""

import torch
from vllm.triton_utils import tl, triton


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
        # Explicit arguments work with Ascend launchers that do not unpack
        # tuples of device pointers. Unused arguments are eliminated at compile time.
        pointers = (*lses, *((lses[0],) * (16 - len(lses))))
        _prepare_attention_update_kernel[(triton.cdiv(rows, 128),)](
            *pointers,
            tuple(x.stride(0) for x in lses),
            previous_anchor,
            relative,
            anchor,
            rows,
            len(lses),
            previous_anchor is not None,
            128,
        )
    return relative.unbind(0), anchor
