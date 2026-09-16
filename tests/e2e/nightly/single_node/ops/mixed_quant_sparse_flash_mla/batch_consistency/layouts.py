#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Layout-aware batch selection / reordering primitives.

Three KV layouts are supported:

* **BSND** – batch is explicit dim-0: ``[B, S, N, D]``.
* **TND**  – batch is implicit via ``cu_seqlens``: ``[T, N, D]``.
* **PA_BBND** – KV is a physical cache ``[block_num, block_size, N, D]``;
  the logical batch is expressed only through the block table rows.

Every function is pure (returns cloned or sliced views) and never mutates
its arguments.
"""

from typing import List, Optional, Tuple

import torch


# ------------------------------------------------------------------ #
#  BSND
# ------------------------------------------------------------------ #


def select_bsnd(
    tensor: Optional[torch.Tensor], batch_ids: List[int]
) -> Optional[torch.Tensor]:
    """Select (and optionally reorder) batch rows from a BSND tensor.

    Only valid for tensors whose dim-0 is the explicit batch dimension.
    Must NOT be used for ``sinks``, physical cache, or scalar attributes.
    """
    if tensor is None:
        return None
    index = torch.as_tensor(batch_ids, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, index).contiguous()


# ------------------------------------------------------------------ #
#  TND
# ------------------------------------------------------------------ #


def split_tnd(
    tensor: Optional[torch.Tensor], cu_seqlens: torch.Tensor
) -> Optional[List[torch.Tensor]]:
    """Split a TND tensor ``[T, ...]`` into per-batch segments using ``cu_seqlens``.

    Returns a list of ``B`` slices (each ``segment_i = tensor[offset_i:offset_{i+1}]``).
    """
    if tensor is None:
        return None
    offsets = cu_seqlens.to("cpu", torch.int64).tolist()
    segments = []
    for i in range(len(offsets) - 1):
        segments.append(tensor[offsets[i] : offsets[i + 1]].contiguous())
    return segments


def concat_tnd_segments(segments: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
    """Concatenate TND segments and return the new prefix-sum offsets.

    Returns ``(concatenated_tensor, cu_seqlens)`` where ``cu_seqlens`` has
    ``len(segments) + 1`` elements starting at 0.
    """
    if not segments:
        raise ValueError("concat_tnd_segments: segments list is empty")
    concatenated = torch.cat(segments, dim=0).contiguous()
    prefix = [0]
    for seg in segments:
        prefix.append(prefix[-1] + seg.shape[0])
    return concatenated, prefix


def select_tnd(
    tensor: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    batch_ids: List[int],
) -> Tuple[Optional[torch.Tensor], List[int]]:
    """Select and reorder TND segments.

    Returns ``(reordered_tensor, new_cu_seqlens)``.
    """
    if tensor is None:
        return None, [0]
    segments = split_tnd(tensor, cu_seqlens)
    selected = [segments[i] for i in batch_ids]
    return concat_tnd_segments(selected)


# ------------------------------------------------------------------ #
#  PA_BBND
# ------------------------------------------------------------------ #


def select_pa_block_table(
    block_table: Optional[torch.Tensor], batch_ids: List[int]
) -> Optional[torch.Tensor]:
    """Select / reorder block table rows.

    The physical KV cache is NOT touched – only the logical-to-physical
    mapping rows are selected.
    """
    if block_table is None:
        return None
    return select_bsnd(block_table, batch_ids)


def validate_block_ids(
    block_table: Optional[torch.Tensor], cache_block_num: int
) -> None:
    """Assert every non-negative entry in ``block_table`` is within cache range."""
    if block_table is None:
        return
    valid_mask = block_table >= 0
    if not valid_mask.any():
        return
    max_id = int(block_table[valid_mask].max().item())
    if max_id >= cache_block_num:
        raise ValueError(
            f"Block id {max_id} exceeds physical cache block_num {cache_block_num}"
        )


# ------------------------------------------------------------------ #
#  Per-batch vector helpers (1-D ``[B]`` tensors)
# ------------------------------------------------------------------ #


def select_per_batch_vec(
    vec: Optional[torch.Tensor], batch_ids: List[int]
) -> Optional[torch.Tensor]:
    """Select / reorder a 1-D per-batch vector of length ``B``."""
    if vec is None:
        return None
    return select_bsnd(vec, batch_ids)


def recompute_cu_seqlens(lengths: List[int]) -> List[int]:
    """Build a prefix-sum list from per-batch lengths."""
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return cu


def lengths_from_cu_seqlens(cu_seqlens: torch.Tensor) -> List[int]:
    """Convert ``cu_seqlens [B+1]`` to per-batch lengths ``[B]``."""
    cu_list = cu_seqlens.to("cpu", torch.int64).tolist()
    return [cu_list[i + 1] - cu_list[i] for i in range(len(cu_list) - 1)]


def seqused_to_lengths(seqused: Optional[torch.Tensor]) -> Optional[List[int]]:
    """Convert a ``[B]`` seqused tensor to a Python list."""
    if seqused is None:
        return None
    return seqused.to("cpu", torch.int64).tolist()
