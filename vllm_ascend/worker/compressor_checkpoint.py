# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-page tail copies using the caller's current device stream."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch

    from vllm_ascend.core.compressor_checkpoint import TailRestorePlan, TailSavePlan


def copy_compressor_tail_pages(
    plans: Sequence[TailSavePlan | TailRestorePlan],
    pages_by_group: Mapping[int, Sequence[torch.Tensor]],
) -> None:
    """Copy each ring slot and its padding without allocating a gather buffer.

    Each tensor is a raw per-layer cache viewed as [num_blocks, page_bytes].
    This copies both KV and score state, including indexer compressor layers.
    Completion/publication is the caller's responsibility.
    """
    for plan in plans:
        for operation in plan.copies:
            for pages in pages_by_group[operation.group_id]:
                pages[operation.destination].copy_(pages[operation.source], non_blocking=True)
