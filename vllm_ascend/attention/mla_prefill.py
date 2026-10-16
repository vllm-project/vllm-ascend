# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Bounded MLA prefill attention updates with relative intermediate LSEs."""

import torch
import torch_npu

ATTENTION_UPDATE_MAX_INPUTS = 16


def merge_prefill_chunks(
    out_list: list[torch.Tensor],
    lse_list: list[torch.Tensor],
    lse_anchor: torch.Tensor | None,
    *,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Merge FP32 pairs, retaining a separate per-row anchor for staged updates.

    If an anchor is supplied, the first LSE is relative to that anchor; all
    remaining LSEs are absolute. Never add the anchor to the returned LSE:
    rounding that sum before the next update loses small relative weights when
    the inputs have a large common offset.
    """
    assert 1 < len(out_list) == len(lse_list) <= ATTENTION_UPDATE_MAX_INPUTS
    if lse_anchor is None and not return_lse:
        output, lse = torch_npu.npu_attention_update(tuple(lse_list), tuple(out_list), 0)
        return output, lse, None

    if lse_anchor is None:
        candidates = torch.stack(lse_list)
    else:
        # An all-invalid partial result must not anchor a later valid chunk.
        previous_anchor = torch.where(torch.isfinite(lse_list[0]), lse_anchor, lse_list[0])
        candidates = torch.stack([previous_anchor, *lse_list[1:]])
    anchor = candidates.masked_fill(~torch.isfinite(candidates), -torch.inf).amax(dim=0)
    anchor = anchor.masked_fill(~torch.isfinite(anchor), 0)
    # Subtract only finite anchors. Leave non-finite LSEs/outputs to the native
    # operator, preserving its masked-row semantics.
    relative_lse = [lse - anchor for lse in lse_list]
    if lse_anchor is not None:
        relative_lse[0] = lse_list[0] + (lse_anchor - anchor)
    if return_lse:
        # Native all-infinite rows yield NaN LSE, which would poison a later
        # valid chunk. Retain finite provisional outputs (NaN/Inf inputs still
        # propagate), and mark these rows with an empty-state LSE of -inf.
        # The loop restores native NaN output at an exact final flush.
        all_infinite = torch.isinf(candidates).all(dim=0)
        relative_lse = [lse.masked_fill(all_infinite, 0) for lse in relative_lse]
    output, lse = torch_npu.npu_attention_update(tuple(relative_lse), tuple(out_list), int(return_lse))
    if return_lse:
        lse = lse.masked_fill(all_infinite, -torch.inf)
    return output, lse, anchor
