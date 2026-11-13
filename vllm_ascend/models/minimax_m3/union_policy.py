# SPDX-License-Identifier: Apache-2.0
"""Request-local, device-side routing of aligned eight-query sparse groups."""

import torch
from vllm.triton_utils import tl, triton

_UNION_SIZE = 8
_MERGE_TILE = 1024


@triton.jit
def _route_union_groups(
    UnionCount,
    Indices,
    Counts,
    CuQuery,
    RoutedUnionCount,
    NativeIndices,
    NativeCounts,
    TokenRoute,
    BATCH: tl.constexpr,
    TOKENS: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    THRESHOLD: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    # UnionSparseGen packs ceil(query_len / 8) groups per request. Locate
    # this group's request on device; groups must never cross requests.
    base = 0
    start = 0
    stop = 0
    for request in range(BATCH):
        q_start = tl.load(CuQuery + request).to(tl.int32)
        q_end = tl.load(CuQuery + request + 1).to(tl.int32)
        groups = (q_end - q_start + GROUP_SIZE - 1) // GROUP_SIZE
        owns_group = (group >= base) & (group < base + groups)
        start = tl.where(owns_group, q_start + (group - base) * GROUP_SIZE, start)
        stop = tl.where(owns_group, q_end, stop)
        base += groups
    rows = start + tl.arange(0, GROUP_SIZE)
    valid = (rows < stop) & (rows < TOKENS)
    query_count = tl.minimum(GROUP_SIZE, stop - start)
    union_count = tl.load(UnionCount + group)
    # D8 = |union(selected blocks)| / (8 * topk_blocks). A final short
    # group uses its actual query count, so decode is not biased by padding.
    use_union = union_count * 100 <= query_count * TOPK * THRESHOLD
    tl.store(RoutedUnionCount + group, tl.where(use_union, union_count, 0))
    counts = tl.load(Counts + rows, mask=valid, other=0)
    tl.store(NativeCounts + rows, tl.where(use_union, 0, counts), mask=valid)
    tl.store(TokenRoute + rows, use_union, mask=valid)
    slots = tl.arange(0, WIDTH)
    offsets = rows[:, None] * TOPK + slots[None, :]
    indices = tl.load(Indices + offsets, mask=valid[:, None] & (slots[None, :] < TOPK), other=-1)
    native = tl.where(use_union | (slots[None, :] >= counts[:, None]), -1, indices)
    tl.store(NativeIndices + offsets, native, mask=valid[:, None] & (slots[None, :] < TOPK))


@triton.jit
def _merge_union_groups(
    Union,
    Native,
    TokenRoute,
    Counts,
    Output,
    ELEMENTS: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = offsets < ELEMENTS
    row = offsets // ROW_SIZE
    use_union = tl.load(TokenRoute + row, mask=valid, other=0) != 0
    nonempty = tl.load(Counts + row, mask=valid, other=0) > 0
    union = tl.load(Union + offsets, mask=valid & use_union & nonempty, other=0)
    native = tl.load(Native + offsets, mask=valid & (~use_union) & nonempty, other=0)
    tl.store(Output + offsets, tl.where(use_union, union, native), mask=valid)


def route_union_groups(union_count, indices, counts, cu_seqlens_q, threshold_pct):
    """Partition real selections; unassigned branches receive empty groups.

    Both attention entry points launch, but each receives only its assigned
    edges. All selection-dependent values stay on device, including replay.
    """
    routed_union_count = torch.empty_like(union_count)
    native_indices = torch.empty_like(indices)
    native_counts = torch.empty_like(counts)
    token_route = torch.empty(indices.shape[1], dtype=torch.bool, device=indices.device)
    _route_union_groups[(union_count.numel(),)](
        union_count,
        indices,
        counts,
        cu_seqlens_q,
        routed_union_count,
        native_indices,
        native_counts,
        token_route,
        BATCH=cu_seqlens_q.numel() - 1,
        TOKENS=indices.shape[1],
        TOPK=indices.shape[2],
        WIDTH=triton.next_power_of_2(indices.shape[2]),
        THRESHOLD=threshold_pct,
        GROUP_SIZE=_UNION_SIZE,
    )
    return routed_union_count, native_indices, native_counts, token_route


def merge_union_groups(union, native, token_route, counts, output):
    _merge_union_groups[(triton.cdiv(output.numel(), _MERGE_TILE),)](
        union,
        native,
        token_route,
        counts,
        output,
        ELEMENTS=output.numel(),
        ROW_SIZE=output.shape[1] * output.shape[2],
        TILE=_MERGE_TILE,
    )
