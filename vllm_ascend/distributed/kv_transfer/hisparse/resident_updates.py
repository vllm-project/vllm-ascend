# SPDX-License-Identifier: Apache-2.0
"""Detached Ascend candidate: fuse Resident state-row lookup and table stores.

No installer. Writes the exact host matrix directly into a fresh host tensor.
Uses the unchanged async H2D helper and the same fused update kernel as v1.
Requires actual NPU and model qualification before deployment. No scheduler,
allocation, page durability, event or MTP/count-copy ordering change.
"""

import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d


def pack_current_updates(num_groups, updates, request_ids, out=None):
    """Build the upstream matrix from current metadata, never a cached value."""
    batch_rows = {request_id: row for row, request_id in enumerate(request_ids)}
    rows = []
    pages = []
    block_ids = [[] for _ in range(num_groups)]
    for request_id, update in updates.items():
        rows.append(batch_rows[request_id])
        pages.extend(update.pages)
        for group_block_ids, update_block_ids in zip(block_ids, update.block_ids, strict=True):
            group_block_ids.extend(update_block_ids)
    shape = (len(pages), 2 + num_groups)
    if out is None:
        host_values = np.empty(shape, dtype=np.int32)
    else:
        if out.shape != shape or out.dtype != np.int32 or not out.flags.c_contiguous:
            raise ValueError("Resident update output buffer geometry changed")
        host_values = out
    host_values[:, 0] = np.repeat(rows, [len(update.pages) for update in updates.values()])
    host_values[:, 1] = pages
    host_values[:, 2:] = np.asarray(block_ids, dtype=np.int32).T
    return host_values


def fresh_host_updates(num_groups, updates, request_ids, *, allocator=None):
    """No bank reuse: every call owns a new host allocation and current values.

    The same async_tensor_h2d helper tracks the fresh pinned allocation through
    nonblocking DMA. Nothing overwrites a prior call's allocation. Injecting an
    allocator is only for CPU-contract tests, never the serving call path.
    """
    count = sum(len(update.pages) for update in updates.values())
    allocate = torch.empty if allocator is None else allocator
    host_tensor = allocate((count, 2 + num_groups), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY)
    matrix = pack_current_updates(num_groups, updates, request_ids, out=host_tensor.numpy())
    return host_tensor, matrix


def unique_destinations(host_values):
    keys = host_values[:, :2].tolist()
    return len({(row, page) for row, page in keys}) == len(keys)


def eligible_update_matrix(host_values, request_ids, mapping_rows, page_count):
    if len(request_ids) != len(set(request_ids)) or len(request_ids) > mapping_rows:
        return False
    seen = set()
    for row, page in host_values[:, :2].tolist():
        if row < 0 or row >= len(request_ids) or page < 0 or page >= page_count:
            return False
        key = (row, page)
        if key in seen:
            return False
        seen.add(key)
    return True


@triton.jit(do_not_specialize=["N"])
def _apply_updates(
    VALUES,
    STATE_ROWS,
    RESIDENT,
    N,
    G: tl.constexpr,
    VSTRIDE: tl.constexpr,
    MAPPING_STRIDE: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    STATES: tl.constexpr,
    PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = linear // G
    group = linear % G
    valid = row < N
    batch_row = tl.load(VALUES + row * VSTRIDE, mask=valid, other=0)
    page = tl.load(VALUES + row * VSTRIDE + 1, mask=valid, other=0)
    state = tl.load(STATE_ROWS + batch_row * MAPPING_STRIDE, mask=valid, other=0)
    # Match valid negative Python tensor indices as well as ordinary indices.
    state = tl.where(state < 0, state + STATES, state)
    page = tl.where(page < 0, page + PAGES, page)
    value = tl.load(VALUES + row * VSTRIDE + 2 + group, mask=valid, other=0)
    tl.store(RESIDENT + state.to(tl.int64) * R0 + group * R1 + page * R2, value, mask=valid)


def candidate_update(residency, updates, request_ids, request_state_indices, original):
    """Explicit candidate only; unsupported/ambiguous inputs use original."""
    if not updates:
        return
    if (
        residency.device.type != "npu"
        or residency.dtype != torch.int32
        or residency.ndim != 3
        or request_state_indices.ndim != 1
        or request_state_indices.dtype not in (torch.int32, torch.int64)
        or request_state_indices.device != residency.device
    ):
        return original(residency, updates, request_ids, request_state_indices)
    host_tensor, host = fresh_host_updates(residency.shape[1], updates, request_ids)
    if not eligible_update_matrix(host, request_ids, request_state_indices.numel(), residency.shape[2]):
        return original(residency, updates, request_ids, request_state_indices)
    if host.shape[0] == 0:
        return
    values = async_tensor_h2d(host_tensor, device=request_state_indices.device, dtype=torch.int32)
    n, width = host.shape
    groups = residency.shape[1]
    _apply_updates[(triton.cdiv(n * groups, 128),)](
        values,
        request_state_indices,
        residency,
        n,
        groups,
        width,
        request_state_indices.stride(0),
        *residency.stride(),
        residency.shape[0],
        residency.shape[2],
        128,
    )
