# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""One C8 mapping producer per captured forward and upstream index group."""

import weakref

import torch

_MAX_CAPTURE_SCOPES = 32


class C8MappingScope:
    """Own output only; never retain an input view or a forward context."""

    def __init__(self, context, stream, source, block, physical_bound):
        self.context = weakref.ref(context)
        self.stream = stream
        self.source_start = source.data_ptr()
        self.source_end = self.source_start + source.numel() * source.element_size()
        self.device = source.device
        self.block = block
        self.physical_bound = physical_bound
        self.key = None
        self.result = None
        self.hits = 0

    def binding(self, indices, block, page_stride, capacity):
        """Accept only the current group's shared, already-physical selection."""
        if (
            indices.device != self.device
            or indices.dtype != torch.int32
            or indices.ndim != 2
            or not indices.is_contiguous()
            or not 0 < indices.shape[0] <= 128
            or indices.shape[1] != 2048
            or block != self.block
            or page_stride != block
            or capacity < self.physical_bound
        ):
            return None
        start = indices.data_ptr()
        if start < self.source_start or start + indices.numel() * 4 > self.source_end:
            return None
        key = (start, tuple(indices.shape), tuple(indices.stride()), block, self.physical_bound)
        return key if self.key is None or self.key == key else None


def begin_c8_mapping_scope(group, context, *, enabled, cache_handles):
    """Reset at the existing group leader; retain independent graph outputs."""
    group._c8_mapping_scope = None
    if not enabled or not torch.npu.is_current_stream_capturing():
        return
    owners = getattr(group, "_c8_mapping_capture_owners", None)
    if owners is None:
        group._c8_mapping_capture_owners = owners = []
    if len(owners) >= _MAX_CAPTURE_SCOPES:
        return
    source = group.shared_topk.physical_topk_indices
    hot = group.leader.hot
    cache = hot.cache
    if (
        source.dtype != torch.int32
        or not source.is_contiguous()
        or cache.dtype != torch.float8_e4m3fn
        or cache.ndim != 3
        or cache.shape[-1] != 656
        or cache.shape[0] <= 0
        or hot.block_size <= 0
        or hot.attention_block_stride < hot.block_size
    ):
        return
    # Upstream resolves once per group. All its Hot/Resident physical page IDs
    # address this full backing view. Layer aliases have different tail sizes,
    # but every compatible alias must cover this shared physical-row domain.
    if len(cache_handles) < 2:
        return
    bounds = []
    capacities = []
    for handle in cache_handles:
        if handle.runtime.index_group is not group:
            return
        member = handle.runtime.hot
        if member.block_size != hot.block_size or member.attention_block_stride != hot.attention_block_stride:
            return
        views = [member]
        if handle.view is not None:
            if handle.view.attention_cache.data_ptr() != member.attention_cache.data_ptr():
                return  # Resident and Hot indices must address the same layer origin.
            views.append(handle.view)
        for view in views:
            if (
                view.block_size != hot.block_size
                or view.attention_block_stride != hot.attention_block_stride
                or view.cache.dtype != torch.float8_e4m3fn
                or view.cache.shape[-1] != 656
                or view.cache.shape[0] <= 0
                or not view.attention_cache.is_contiguous()
            ):
                return
            bounds.append((view.cache.shape[0] - 1) * view.attention_block_stride + view.block_size)
            capacities.append(view.attention_cache.numel() // 656)
    bound = max(bounds)
    if any(capacity < bound for capacity in capacities):
        return
    if not 0 < bound <= 2**31 - 1:
        return
    try:
        scope = C8MappingScope(context, torch.npu.current_stream().npu_stream, source, hot.block_size, bound)
    except TypeError:
        return  # A context without weak-reference support keeps the old path.
    scope.domain_proof = dict(
        cache_count=len(cache_handles),
        physical_bound=bound,
        minimum_alias_capacity=min(capacities),
        block_stride=hot.attention_block_stride,
    )
    owners.append(scope)
    group._c8_mapping_scope = scope


def current_c8_mapping_scope(group, context):
    scope = getattr(group, "_c8_mapping_scope", None)
    if (
        scope is None
        or scope.context() is not context
        or not torch.npu.is_current_stream_capturing()
        or scope.stream != torch.npu.current_stream().npu_stream
    ):
        return None
    return scope
