# SPDX-License-Identifier: Apache-2.0

"""Pack DCP FlashMLA inputs directly into their preallocated workspaces."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _prepare_flashmla_kernel(
    q_nope_ptr,
    q_pe_ptr,
    query_ptr,
    k_nope_ptr,
    k_pe_ptr,
    cache_ptr,
    slots_ptr,
    q_nope_stride_t,
    q_nope_stride_h,
    q_nope_stride_d,
    q_pe_stride_t,
    q_pe_stride_h,
    q_pe_stride_d,
    query_stride_t,
    query_stride_h,
    query_stride_d,
    k_nope_stride_t,
    k_nope_stride_d,
    k_pe_stride_t,
    k_pe_stride_d,
    cache_stride_page,
    cache_stride_block,
    cache_stride_d,
    slots_stride,
    num_heads,
    total_rows,
    cache_block_size,
    cache_num_slots,
    NOPE_DIM: tl.constexpr,
    PE_DIM: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_PE: tl.constexpr,
    PREPARE_KV: tl.constexpr,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    nope_offsets = tl.arange(0, BLOCK_NOPE)
    pe_offsets = tl.arange(0, BLOCK_PE)
    nope_mask = nope_offsets < NOPE_DIM
    pe_mask = pe_offsets < PE_DIM

    # Bound the launch to the vector cores, as in DCP output packing. Each
    # program copies successive rows without creating one task per Q head.
    for row_idx in range(program_idx, total_rows, num_programs):
        token_idx = (row_idx // num_heads).to(tl.int64)
        head_idx = (row_idx % num_heads).to(tl.int64)
        q_nope = tl.load(
            q_nope_ptr + token_idx * q_nope_stride_t + head_idx * q_nope_stride_h + nope_offsets * q_nope_stride_d,
            mask=nope_mask,
            other=0.0,
        )
        q_pe = tl.load(
            q_pe_ptr + token_idx * q_pe_stride_t + head_idx * q_pe_stride_h + pe_offsets * q_pe_stride_d,
            mask=pe_mask,
            other=0.0,
        )
        query_base = token_idx * query_stride_t + head_idx * query_stride_h
        tl.store(query_ptr + query_base + nope_offsets * query_stride_d, q_nope, mask=nope_mask)
        tl.store(query_ptr + query_base + (NOPE_DIM + pe_offsets) * query_stride_d, q_pe, mask=pe_mask)

        if PREPARE_KV:
            # Only one Q head owns this token's KV write. Other heads still pack Q.
            if head_idx == 0:
                slot = tl.load(slots_ptr + token_idx * slots_stride).to(tl.int64)
                valid_slot = (slot >= 0) & (slot < cache_num_slots)
                safe_slot = tl.where(valid_slot, slot, 0)
                cache_base = (safe_slot // cache_block_size) * cache_stride_page + (
                    safe_slot % cache_block_size
                ) * cache_stride_block
                k_nope = tl.load(
                    k_nope_ptr + token_idx * k_nope_stride_t + nope_offsets * k_nope_stride_d,
                    mask=nope_mask & valid_slot,
                    other=0.0,
                )
                k_pe = tl.load(
                    k_pe_ptr + token_idx * k_pe_stride_t + pe_offsets * k_pe_stride_d,
                    mask=pe_mask & valid_slot,
                    other=0.0,
                )
                tl.store(cache_ptr + cache_base + nope_offsets * cache_stride_d, k_nope, mask=nope_mask & valid_slot)
                tl.store(
                    cache_ptr + cache_base + (NOPE_DIM + pe_offsets) * cache_stride_d,
                    k_pe,
                    mask=pe_mask & valid_slot,
                )


def _check_query(q_nope: torch.Tensor, q_pe: torch.Tensor, query_out: torch.Tensor) -> None:
    assert q_nope.ndim == q_pe.ndim == query_out.ndim == 3
    assert q_nope.shape[:2] == q_pe.shape[:2] == query_out.shape[:2]
    assert query_out.shape[2] == q_nope.shape[2] + q_pe.shape[2]
    assert q_nope.dtype == q_pe.dtype == query_out.dtype
    assert q_nope.device == q_pe.device == query_out.device
    assert q_nope.dtype in (torch.float16, torch.bfloat16)
    assert q_nope.shape[1] > 0 and q_nope.shape[2] > 0 and q_pe.shape[2] > 0


def prepare_flashmla_query(q_nope: torch.Tensor, q_pe: torch.Tensor, query_out: torch.Tensor) -> None:
    """Copy strided Q components into ``query_out[T, H, Dn + Dr]``.

    All storage is caller-owned. Sources must not overlap the destination.
    This operation enqueues one kernel and allocates no tensor storage.
    """
    _check_query(q_nope, q_pe, query_out)
    num_tokens, num_heads, nope_dim = q_nope.shape
    if num_tokens == 0:
        return
    total_rows = num_tokens * num_heads
    init_device_properties_triton()
    grid_size = min(total_rows, get_vectorcore_num())
    _prepare_flashmla_kernel[(grid_size,)](
        q_nope,
        q_pe,
        query_out,
        q_nope,
        q_pe,
        query_out,
        query_out,
        *q_nope.stride(),
        *q_pe.stride(),
        *query_out.stride(),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        num_heads,
        total_rows,
        1,
        0,
        NOPE_DIM=nope_dim,
        PE_DIM=q_pe.shape[2],
        BLOCK_NOPE=triton.next_power_of_2(nope_dim),
        BLOCK_PE=triton.next_power_of_2(q_pe.shape[2]),
        PREPARE_KV=False,
    )


def prepare_flashmla_current(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    query_out: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Pack local Q and scatter current KV in a single kernel.

    K components have shape ``[T, 1, Dn/Dr]`` and ``cache`` has shape
    ``[pages, block_size, 1, Dn + Dr]``. Only in-range ``slots[T]`` are written;
    a padding slot of -1 leaves cache unchanged, while its Q is still copied.
    Valid slots must be unique, and inputs must not overlap the outputs.
    Output elements must also have distinct storage locations.
    Sources and destinations may be strided. No host read of slots is needed.
    """
    _check_query(q_nope, q_pe, query_out)
    num_tokens, num_heads, nope_dim = q_nope.shape
    assert k_nope.shape == (num_tokens, 1, nope_dim)
    assert k_pe.shape == (num_tokens, 1, q_pe.shape[2])
    assert cache.ndim == 4 and cache.shape[2:] == (1, query_out.shape[2])
    assert cache.shape[1] > 0
    assert slots.shape == (num_tokens,) and slots.dtype in (torch.int32, torch.int64)
    assert k_nope.dtype == k_pe.dtype == cache.dtype == query_out.dtype
    assert k_nope.device == k_pe.device == cache.device == slots.device == query_out.device
    if num_tokens == 0:
        return
    total_rows = num_tokens * num_heads
    init_device_properties_triton()
    grid_size = min(total_rows, get_vectorcore_num())
    _prepare_flashmla_kernel[(grid_size,)](
        q_nope,
        q_pe,
        query_out,
        k_nope,
        k_pe,
        cache,
        slots,
        *q_nope.stride(),
        *q_pe.stride(),
        *query_out.stride(),
        k_nope.stride(0),
        k_nope.stride(2),
        k_pe.stride(0),
        k_pe.stride(2),
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        slots.stride(0),
        num_heads,
        total_rows,
        cache.shape[1],
        cache.shape[0] * cache.shape[1],
        NOPE_DIM=nope_dim,
        PE_DIM=q_pe.shape[2],
        BLOCK_NOPE=triton.next_power_of_2(nope_dim),
        BLOCK_PE=triton.next_power_of_2(q_pe.shape[2]),
        PREPARE_KV=True,
    )
