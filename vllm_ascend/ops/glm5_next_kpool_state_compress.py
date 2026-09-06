# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused GLM5 Next compressor-state write + KPool compression op.

Replaces the torch glue sequence (cat/scatter/bucketize/gather/where/compress)
that used to run before ``glm5_next_lightning_indexer`` on A3: those ops lower
to an aclnnIndex/SearchSorted small-op flood per sparse layer per step.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import direct_register_custom_op

TRITON_MAX_POOL_SIZE = 64
TRITON_MAX_HEAD_DIM = 1024

if HAS_TRITON:
    from vllm_ascend.ops.triton.glm5_next_kpool_state_compress import (
        glm5_next_kpool_state_compress_and_write_cache_triton,
    )
else:
    glm5_next_kpool_state_compress_and_write_cache_triton = None


def _validate_inputs(
    state_cache: torch.Tensor,
    indexer_cache: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    state_block_table: torch.Tensor,
    indexer_slot_mapping: torch.Tensor,
    index_kpool: int,
) -> None:
    if k.ndim != 2:
        raise ValueError(f"k must be [N,D], got {k.shape}.")
    if k.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"k must be bfloat16 or float32, got {k.dtype}.")
    if gate_score.shape != k.shape or gate_score.dtype != k.dtype:
        raise ValueError(
            "gate_score must match k, got "
            f"{gate_score.shape} {gate_score.dtype}."
        )
    num_tokens, head_dim = k.shape
    if index_kpool <= 0:
        raise ValueError(f"index_kpool must be positive, got {index_kpool}.")
    if ape.shape != (index_kpool, head_dim):
        raise ValueError(
            f"ape must be [P,D]={(index_kpool, head_dim)}, got {ape.shape}."
        )
    if ape.dtype != torch.float32:
        raise TypeError(f"ape must be float32, got {ape.dtype}.")
    if state_cache.ndim != 3 or state_cache.shape[2] != 2 * head_dim:
        raise ValueError(
            f"state_cache must be [blocks,block,{2 * head_dim}], "
            f"got {state_cache.shape}."
        )
    if indexer_cache.ndim != 4 or indexer_cache.shape[2:] != (
        1,
        head_dim,
    ):
        raise ValueError(
            f"indexer_cache must be [blocks,block,1,{head_dim}], "
            f"got {indexer_cache.shape}."
        )
    if indexer_cache.dtype != torch.bfloat16:
        raise TypeError(
            f"indexer_cache must be bfloat16, got {indexer_cache.dtype}."
        )
    for name, tensor in (
        ("positions", positions),
        ("state_slot_mapping", state_slot_mapping),
        ("indexer_slot_mapping", indexer_slot_mapping),
    ):
        if tensor.shape != (num_tokens,):
            raise ValueError(
                f"{name} must have shape {(num_tokens,)}, got {tensor.shape}."
            )
        if tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{name} must be int32/int64, got {tensor.dtype}.")
    if cum_query_lens.ndim != 1:
        raise ValueError(
            f"cum_query_lens must be 1-D, got {cum_query_lens.shape}."
        )
    num_reqs = cum_query_lens.shape[0]
    if seq_lens.shape != (num_reqs,):
        raise ValueError(
            f"seq_lens must have shape {(num_reqs,)}, got {seq_lens.shape}."
        )
    if (
        state_block_table.ndim != 2
        or state_block_table.shape[0] != num_reqs
    ):
        raise ValueError(
            f"state_block_table must be [R={num_reqs},pages], "
            f"got {state_block_table.shape}."
        )
    for name, tensor in (
        ("state_cache", state_cache),
        ("indexer_cache", indexer_cache),
        ("gate_score", gate_score),
        ("ape", ape),
        ("positions", positions),
        ("cum_query_lens", cum_query_lens),
        ("seq_lens", seq_lens),
        ("state_slot_mapping", state_slot_mapping),
        ("state_block_table", state_block_table),
        ("indexer_slot_mapping", indexer_slot_mapping),
    ):
        if tensor.device != k.device:
            raise ValueError(
                f"{name} must be on {k.device}, got {tensor.device}."
            )


def _can_use_triton(k: torch.Tensor, index_kpool: int) -> bool:
    if (
        not HAS_TRITON
        or glm5_next_kpool_state_compress_and_write_cache_triton is None
    ):
        return False
    if k.device.type != "npu":
        return False
    if k.shape[0] == 0:
        return False
    if index_kpool > TRITON_MAX_POOL_SIZE:
        return False
    return k.shape[1] <= TRITON_MAX_HEAD_DIM


def _scatter_paged_rows(
    cache: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Scatter rows into a paged cache, treating invalid slots as no-ops.

    Fixed-shape graph-safe version: invalid rows dump into row zero, which is
    restored immediately afterwards.
    """
    block_size = cache.shape[1]
    num_slots = cache.shape[0] * block_size
    valid = (slots >= 0) & (slots < num_slots)
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    block_ids = torch.div(safe_slots, block_size, rounding_mode="floor")
    block_offsets = torch.remainder(safe_slots, block_size)
    row_mask = valid.view(-1, *([1] * (values.ndim - 1)))
    row_zero = cache[0, 0].clone()
    safe_values = torch.where(row_mask, values, row_zero.unsqueeze(0))
    row_zero_mask = valid & (slots == 0)
    update_zero = torch.where(
        row_zero_mask.view(-1, *([1] * (values.ndim - 1))),
        values,
        torch.zeros_like(values),
    ).sum(dim=0)
    expected_zero = torch.where(row_zero_mask.any(), update_zero, row_zero)
    cache[block_ids, block_offsets] = safe_values
    cache[0, 0].copy_(expected_zero)


def _fallback_kpool_state_compress_and_write_cache(
    state_cache: torch.Tensor,
    indexer_cache: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    state_block_table: torch.Tensor,
    indexer_slot_mapping: torch.Tensor,
    index_kpool: int,
) -> None:
    num_tokens, head_dim = k.shape
    if num_tokens == 0:
        return
    device = k.device

    current_state = torch.cat([k, gate_score], dim=-1).to(
        state_cache.dtype
    )
    _scatter_paged_rows(state_cache, state_slot_mapping, current_state)

    token_ids = torch.arange(num_tokens, device=device)
    request_ids = torch.bucketize(
        token_ids, cum_query_lens, right=True
    ).clamp_max(cum_query_lens.shape[0] - 1)
    state_block_size = state_cache.shape[1]
    offsets = torch.arange(index_kpool - 1, -1, -1, device=device)
    pool_positions = positions[:, None] - offsets[None, :]
    safe_positions = pool_positions.clamp_min(0)
    pages = torch.div(
        safe_positions,
        state_block_size,
        rounding_mode="floor",
    ).clamp_max(state_block_table.shape[1] - 1)
    page_offsets = torch.remainder(safe_positions, state_block_size)
    physical_blocks = state_block_table[
        request_ids[:, None],
        pages,
    ].clamp(min=0, max=state_cache.shape[0] - 1)
    pool_state = state_cache[physical_blocks.long(), page_offsets]

    query_ends = cum_query_lens
    query_offsets = torch.cat(
        [torch.zeros_like(query_ends[:1]), query_ends[:-1]]
    )
    query_lens = query_ends - query_offsets
    request_query_starts = seq_lens[request_ids] - query_lens[request_ids]
    local_positions = pool_positions - request_query_starts[:, None]
    current_mask = (local_positions >= 0) & (
        local_positions < query_lens[request_ids, None]
    )
    current_indices = (
        query_offsets[request_ids, None] + local_positions.clamp_min(0)
    ).long().clamp(0, num_tokens - 1)
    current_pool_state = current_state[current_indices]
    pool_state = torch.where(
        current_mask.unsqueeze(-1),
        current_pool_state,
        pool_state,
    )

    pool_k, pool_gate = pool_state.split(head_dim, dim=-1)
    scores = pool_gate.float() + ape.float().unsqueeze(0)
    compressed = (
        (torch.softmax(scores, dim=1) * pool_k.float())
        .sum(dim=1)
        .to(indexer_cache.dtype)
    )
    _scatter_paged_rows(
        indexer_cache,
        indexer_slot_mapping,
        compressed.reshape(num_tokens, *indexer_cache.shape[2:]),
    )


def glm5_next_kpool_state_compress_and_write_cache(
    state_cache: torch.Tensor,
    indexer_cache: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    state_block_table: torch.Tensor,
    indexer_slot_mapping: torch.Tensor,
    *,
    index_kpool: int,
) -> None:
    """Write compressor states, compress completed pools, write indexer cache."""
    _validate_inputs(
        state_cache,
        indexer_cache,
        k,
        gate_score,
        ape,
        positions,
        cum_query_lens,
        seq_lens,
        state_slot_mapping,
        state_block_table,
        indexer_slot_mapping,
        index_kpool,
    )
    if _can_use_triton(k, index_kpool):
        assert glm5_next_kpool_state_compress_and_write_cache_triton is not None
        glm5_next_kpool_state_compress_and_write_cache_triton(
            state_cache,
            indexer_cache,
            k,
            gate_score,
            ape,
            positions,
            cum_query_lens,
            seq_lens,
            state_slot_mapping,
            state_block_table,
            indexer_slot_mapping,
            index_kpool,
        )
        return

    _fallback_kpool_state_compress_and_write_cache(
        state_cache,
        indexer_cache,
        k,
        gate_score,
        ape,
        positions,
        cum_query_lens,
        seq_lens,
        state_slot_mapping,
        state_block_table,
        indexer_slot_mapping,
        index_kpool,
    )


def glm5_next_kpool_state_compress_and_write_cache_fake(
    state_cache: torch.Tensor,
    indexer_cache: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    state_block_table: torch.Tensor,
    indexer_slot_mapping: torch.Tensor,
    *,
    index_kpool: int,
) -> None:
    return


direct_register_custom_op(
    op_name="glm5_next_kpool_state_compress_and_write_cache",
    op_func=glm5_next_kpool_state_compress_and_write_cache,
    mutates_args=["state_cache", "indexer_cache"],
    fake_impl=glm5_next_kpool_state_compress_and_write_cache_fake,
    dispatch_key="PrivateUse1",
)
