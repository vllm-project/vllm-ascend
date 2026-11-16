# SPDX-License-Identifier: Apache-2.0
"""Adapt the generic CANN GBSA operator to Qwen4Exp QSA semantics."""

import torch
from vllm import envs

from vllm_ascend.ops.gbsa import (
    generic_block_sparse_attention,
    generic_block_sparse_attention_metadata,
)
from vllm_ascend.ops.triton.qwen4_exp.gbsa import prepare_qsa_gbsa_inputs
from vllm_ascend.ops.triton.qwen4_exp.gbsa_packing import pack_selected_kv

GBSA_COMPRESS_RATIO = 4
GBSA_RESIDUAL_TOKENS = GBSA_COMPRESS_RATIO - 1
GBSA_PAGE_SIZE = 128
_checked_metadata_devices: set[torch.device] = set()


def _use_batch_invariant_gbsa_schedule(metadata: torch.Tensor) -> None:
    """Select the CANN v8 non-split schedule: a single core owns each (query, KV head)."""
    # generic_block_sparse_attention_metadata_kernel.h defines the 1024-int v8 wire format.
    # Reject an unfamiliar layout before writing to the scheduling fields.
    if metadata.dtype != torch.int32 or metadata.shape != (1024,):
        raise RuntimeError("Unexpected CANN GBSA metadata layout in batch-invariant mode")
    if metadata.device not in _checked_metadata_devices:
        # All calls on this worker use the same loaded CANN library. Check once
        # during warm-up; a D2H read on every decode step serializes the stream.
        if metadata[:3].tolist() != [0x5341534D, 8, 272]:
            raise RuntimeError("Batch-invariant GBSA requires the CANN v8 scheduling metadata")
        _checked_metadata_devices.add(metadata.device)
    # fdActiveCoreNum, decodePerCoreTaskNum, combineTaskNum, fdScheduleFlags,
    # fdTotalFlatTaskNum, fdPartialTaskNum. Retain FD_ACTUAL_BLOCK_PREFIX.
    metadata[5:11].zero_()
    metadata[8].fill_(2)


def qsa_gbsa_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    compact_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("QSA GBSA requires 4-D K/V caches")
    if key_cache.shape != value_cache.shape:
        raise ValueError("QSA GBSA requires matching K/V cache shapes")
    # The table width counts logical kernel pages; pool rows also include
    # packed gaps and therefore do not describe the sequence-length bound.
    logical_max_seqlen_kv = block_table.shape[1] * GBSA_PAGE_SIZE
    num_kv_heads = key_cache.shape[2]
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("QSA GBSA requires FP16/BF16 query")
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise ValueError("QSA GBSA requires matching Q/K/V dtypes")
    if compact_indices.shape[1] <= GBSA_RESIDUAL_TOKENS:
        raise ValueError("QSA GBSA requires compact LI-v2 indices")
    if min(key_cache.shape[:3]) == 0 or min(block_table.shape) == 0:
        raise ValueError("QSA GBSA requires nonempty KV cache and block table")
    if key_cache.shape != value_cache.shape or key_cache.shape[-1] != query.shape[-1]:
        raise ValueError("QSA GBSA requires matching KV shapes and Q/K head dimensions")
    if key_cache.shape[1] != GBSA_PAGE_SIZE:
        raise ValueError("QSA GBSA requires 128-token paged KV cache")
    if query.shape[1] % num_kv_heads:
        raise ValueError("QSA GBSA requires query heads divisible by KV heads")
    if seq_lens.shape != (block_table.shape[0],):
        raise ValueError("QSA GBSA sequence lengths must match block table rows")
    if token_to_req.shape != (query.shape[0],):
        raise ValueError("QSA GBSA request mapping must match query rows")
    if logical_positions.shape != (query.shape[0],):
        raise ValueError("QSA GBSA logical positions must match query rows")

    sparse_idx, sparse_count, row_visible_lens, cu_q, row_block_table = prepare_qsa_gbsa_inputs(
        compact_indices, logical_positions, seq_lens, token_to_req, block_table, num_kv_heads
    )
    if envs.VLLM_BATCH_INVARIANT:
        # Limit temporary KV storage; each query sees all of its selected blocks.
        for first in range(0, query.shape[0], 32):
            rows = slice(first, first + 32)
            packed_k, packed_v, idx, lens, table = pack_selected_kv(
                key_cache, value_cache, row_block_table[rows], sparse_idx[:, rows],
                sparse_count[:, rows], row_visible_lens[rows],
            )
            size = lens.shape[0]
            output[rows].copy_(_run_gbsa(
                query[rows], packed_k, packed_v, idx, sparse_count[:, rows].contiguous(),
                lens, cu_q[:size + 1], table, table.shape[1] * GBSA_PAGE_SIZE,
            ))
    else:
        output.copy_(_run_gbsa(
            query, key_cache, value_cache, sparse_idx, sparse_count,
            row_visible_lens, cu_q, row_block_table, logical_max_seqlen_kv,
        ))
    return output


def _run_gbsa(query, key_cache, value_cache, sparse_idx, sparse_count, row_visible_lens,
              cu_q, row_block_table, logical_max_seqlen_kv):
    num_kv_heads = key_cache.shape[2]
    common = dict(
        cu_seqlens_q=cu_q,
        seqused_kv=row_visible_lens,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=4,
        mask_mode=1,
        quant_mode=0,
        softmax_precision=0,
        residual_block_mode=0,
        is_consistent_topk=False,
    )
    metadata = generic_block_sparse_attention_metadata(
        sparse_idx,
        sparse_count,
        query.shape[1],
        num_kv_heads,
        query.shape[-1],
        [1, GBSA_COMPRESS_RATIO],
        max_seqlen_q=1,
        max_seqlen_kv=int(logical_max_seqlen_kv),
        **common,
    )
    if envs.VLLM_BATCH_INVARIANT:
        _use_batch_invariant_gbsa_schedule(metadata)
    result, _ = generic_block_sparse_attention(
        query.contiguous(),
        key_cache,
        value_cache,
        sparse_idx,
        sparse_count,
        [1, GBSA_COMPRESS_RATIO],
        metadata=metadata,
        block_table=row_block_table,
        softmax_scale=query.shape[-1] ** -0.5,
        **common,
    )
    return result


__all__ = ["qsa_gbsa_attention"]
