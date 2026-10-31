# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-page tail copies using the caller's current device stream."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs

from vllm_ascend.core.compressor_tail import AscendCompressorTailSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm_ascend.core.compressor_checkpoint import TailRestorePlan, TailSavePlan


def compressor_checkpoint_page_views(
    config: KVCacheConfig,
    kv_caches: Mapping[str, Sequence[torch.Tensor]],
) -> dict[int, list[torch.Tensor]]:
    """Recover whole physical pages from MRV2's strided DSV4 cache views.

    The first component starts at the layer's raw allocation. Include all
    KV/score components and page padding, also for mixed-width uniform groups.
    """
    result = {}
    for group_id, group in enumerate(config.kv_cache_groups):
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if not isinstance(spec, AscendCompressorTailSpec):
                continue
            first = kv_caches[name][0]
            page_bytes = spec.page_size_bytes
            if first.shape[0] != config.num_blocks or first.stride(0) * first.element_size() != page_bytes:
                raise ValueError("Compressor checkpoint requires the physical per-page DSV4 stride")
            pages = first.view(torch.uint8).as_strided((config.num_blocks, page_bytes), (page_bytes, 1))
            result.setdefault(group_id, []).append(pages)
    return result


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
