# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Invalidate rewritten slots across upstream HiSparse index groups in one op."""

import torch
from vllm.triton_utils import tl, triton

_INVALIDATION_TILE = 256


@triton.jit
def invalidate_groups_kernel(
    group_tables,
    request_states,
    request_ids,
    written_slots,
    REQUESTS: tl.constexpr,
    STATES: tl.constexpr,
    HOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    offset = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(request_ids + token)
    state = tl.load(request_states + request, mask=(request >= 0) & (request < REQUESTS), other=-1)
    written = tl.load(written_slots + token)
    valid = (state >= 0) & (state < STATES) & (written >= 0) & (written <= 2147483647) & (offset < HOT)
    # Each address belongs to this invocation's actual index-group table.
    # The common invalidator builds and owns the address table outside replay.
    table = tl.load(group_tables + group).to(tl.pointer_type(tl.int32))
    row = table + state.to(tl.int64) * HOT + offset
    cached = tl.load(row, mask=valid, other=-1)
    tl.store(row, -1, mask=valid & (cached == written))


def invalidate_written_groups(
    group_tables,
    request_states,
    request_ids,
    written_slots,
    num_state_rows: int,
    region_stride: int,
) -> None:
    """Implement PR60103's six-argument cache-op contract with integer IDs."""
    device = group_tables.device
    if device.type != "npu":
        raise ValueError("HiSparse grouped invalidation requires NPU tables")
    tensors = (group_tables, request_states, request_ids, written_slots)
    if any(t.device != device or t.ndim != 1 or not t.is_contiguous() for t in tensors):
        raise ValueError("HiSparse invalidation needs contiguous vectors on the same NPU")
    if (
        group_tables.dtype != torch.int64
        or request_states.dtype != torch.int32
        or request_ids.dtype != torch.int32
        or written_slots.dtype not in (torch.int32, torch.int64)
        or request_ids.shape != written_slots.shape
        or num_state_rows < 0
        or region_stride < 0
    ):
        raise ValueError("Invalid HiSparse group addresses, request IDs or table geometry")
    tokens = written_slots.numel()
    groups = group_tables.numel()
    if not tokens or not groups or not num_state_rows or not region_stride:
        return
    invalidate_groups_kernel[(tokens, groups, triton.cdiv(region_stride, _INVALIDATION_TILE))](
        group_tables,
        request_states,
        request_ids,
        written_slots,
        REQUESTS=request_states.numel(),
        STATES=num_state_rows,
        HOT=region_stride,
        BLOCK=_INVALIDATION_TILE,
        multibuffer=False,
    )
