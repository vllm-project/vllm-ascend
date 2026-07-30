# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def dspark_masked_cache_store_kernel(
    cache_ptr,
    shared_kv_ptr,
    positions_ptr,
    slot_mapping_ptr,
    num_cache_blocks,
    cache_block_size,
    cache_stride_block,
    cache_stride_token,
    cache_stride_dim,
    shared_kv_stride_token,
    shared_kv_stride_dim,
    positions_stride,
    slot_mapping_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    dims = tl.arange(0, BLOCK_DIM)
    dim_mask = dims < HEAD_DIM

    position = tl.load(positions_ptr + token_idx * positions_stride)
    # Match the previous PyTorch path, which promoted slots to int64 before
    # deriving paged-cache addresses.
    slot = tl.load(slot_mapping_ptr + token_idx * slot_mapping_stride).to(tl.int64)
    num_cache_slots = num_cache_blocks * cache_block_size
    valid = (position >= 0) & (slot >= 0) & (slot < num_cache_slots)

    # Keep address calculation in bounds even when the masked store is disabled.
    safe_slot = tl.where(slot < 0, 0, slot)
    safe_slot = tl.where(safe_slot >= num_cache_slots, num_cache_slots - 1, safe_slot)
    block_id = safe_slot // cache_block_size
    block_offset = safe_slot % cache_block_size

    values = tl.load(
        shared_kv_ptr + token_idx * shared_kv_stride_token + dims * shared_kv_stride_dim,
        mask=dim_mask,
        other=0.0,
    )
    cache_offsets = block_id * cache_stride_block + block_offset * cache_stride_token + dims * cache_stride_dim
    # The scheduler owns writable paged blocks, so valid slots are unique
    # within one launch and do not require atomic stores.
    tl.store(cache_ptr + cache_offsets, values, mask=valid & dim_mask)


def dspark_masked_cache_store_impl(
    kv_cache: torch.Tensor,
    shared_kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if positions.numel() == 0:
        return
    if not HAS_TRITON:
        raise RuntimeError("DSpark masked cache store requires Triton.")
    if kv_cache.device != shared_kv.device or kv_cache.device != positions.device:
        raise ValueError("DSpark cache, shared KV, and positions must be on the same device.")
    if kv_cache.device != slot_mapping.device:
        raise ValueError("DSpark cache and slot mapping must be on the same device.")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"DSpark positions must be int32 or int64, got dtype={positions.dtype}.")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"DSpark slot mapping must be int32 or int64, got dtype={slot_mapping.dtype}.")
    if kv_cache.dtype != shared_kv.dtype:
        raise ValueError(
            "DSpark cache and shared KV must have the same dtype: "
            f"cache={kv_cache.dtype}, shared_kv={shared_kv.dtype}."
        )
    if kv_cache.ndim < 3:
        raise ValueError(f"DSpark paged cache must have at least 3 dimensions, got shape={tuple(kv_cache.shape)}.")
    if kv_cache.shape[0] == 0 or kv_cache.shape[1] == 0:
        raise ValueError(f"DSpark paged cache must not be empty, got shape={tuple(kv_cache.shape)}.")
    if any(size != 1 for size in kv_cache.shape[2:-1]):
        raise ValueError(f"DSpark paged cache requires one KV head, got shape={tuple(kv_cache.shape)}.")
    if not shared_kv.is_contiguous() or not positions.is_contiguous() or not slot_mapping.is_contiguous():
        raise ValueError("DSpark masked cache store requires contiguous shared KV, positions, and slot mapping.")

    head_dim = shared_kv.shape[-1]
    if head_dim == 0:
        raise ValueError("DSpark shared KV head dimension must not be empty.")
    shared_kv_rows = shared_kv.numel() // head_dim
    if positions.numel() != shared_kv_rows or slot_mapping.numel() != shared_kv_rows:
        raise ValueError(
            "DSpark masked cache store input size mismatch: "
            f"shared_kv_rows={shared_kv_rows}, positions={positions.numel()}, "
            f"slot_mapping={slot_mapping.numel()}."
        )
    if kv_cache.shape[-1] < head_dim:
        raise ValueError(
            f"DSpark paged cache head dimension is too small: cache={kv_cache.shape[-1]}, shared_kv={head_dim}."
        )

    shared_kv_2d = shared_kv.view(shared_kv_rows, head_dim)
    positions_1d = positions.view(-1)
    slot_mapping_1d = slot_mapping.view(-1)
    block_dim = triton.next_power_of_2(head_dim)
    dspark_masked_cache_store_kernel[(shared_kv_rows,)](
        kv_cache,
        shared_kv_2d,
        positions_1d,
        slot_mapping_1d,
        kv_cache.shape[0],
        kv_cache.shape[1],
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(-1),
        shared_kv_2d.stride(0),
        shared_kv_2d.stride(1),
        positions_1d.stride(0),
        slot_mapping_1d.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_DIM=block_dim,
    )


def dspark_masked_cache_store_fake(
    kv_cache: torch.Tensor,
    shared_kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="dspark_masked_cache_store",
    op_func=dspark_masked_cache_store_impl,
    mutates_args=["kv_cache"],
    fake_impl=dspark_masked_cache_store_fake,
    dispatch_key="PrivateUse1",
)


def dspark_masked_cache_store(
    kv_cache: torch.Tensor,
    shared_kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Store unique valid slots without dynamic-shape indexing."""
    torch.ops.vllm.dspark_masked_cache_store(kv_cache, shared_kv, positions, slot_mapping)
