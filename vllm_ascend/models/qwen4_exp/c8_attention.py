# SPDX-License-Identifier: Apache-2.0
"""Native INT8 GBSA with dynamic Q/K and calibrated per-channel V.

Pack only selected INT8 codes and K scales. CANN's fixed dense K-scale
addressing cannot consume MRV2's strided scale pages directly.
"""

import torch

from vllm_ascend.ops.gbsa import GBSAQuantMode, generic_block_sparse_attention, generic_block_sparse_attention_metadata
from vllm_ascend.ops.triton.qwen4_exp.c8 import pack_qsa_c8_rows
from vllm_ascend.ops.triton.qwen4_exp.gbsa import prepare_qsa_gbsa_inputs

from .gbsa import GBSA_COMPRESS_RATIO, GBSA_PAGE_SIZE
from .quantization import qsa_c8_cache_views, quantize_qsa_rows

# Each tile packs at most 2048 selected INT8 tokens per query. Batching 128
# queries amortizes Python dispatch and GBSA metadata preparation in chunked
# prefill while keeping scratch memory independent of the full KV context.
# At TP8 for Qwen3.8 (one local KV head, D=256), K/V scratch is 128 MiB.
C8_QUERY_TILE = 128
C8_PROBABILITY_QUANT_SCALE = 127.0


def qsa_c8_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    compact_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: torch.Tensor,
) -> torch.Tensor:
    """Run quantized QK/PV in GBSA with context-independent packing memory."""
    if query.ndim != 3 or query.dtype != torch.bfloat16 or output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError("QSA C8 attention requires matching BF16 [tokens,heads,dim] query/output")
    key_codes, _ = qsa_c8_cache_views(key_cache)
    qsa_c8_cache_views(value_cache)
    heads = key_cache.shape[2]
    if key_cache.shape != value_cache.shape or key_cache.shape[1] != GBSA_PAGE_SIZE:
        raise ValueError("QSA C8 attention requires matching 128-token K/V pages")
    if query.shape[-1] != key_codes.shape[-1] or query.shape[1] % heads:
        raise ValueError("QSA C8 attention has incompatible query/KV heads")
    if compact_indices.ndim != 2 or not 3 < compact_indices.shape[1] <= 515:
        raise ValueError("QSA C8 requires at most 512 compact groups plus three residual columns")
    if compact_indices.shape[0] != query.shape[0] or token_to_req.shape != (query.shape[0],):
        raise ValueError("QSA C8 selection/request mapping must match query rows")
    if logical_positions.shape != token_to_req.shape or block_table.ndim != 2 or not block_table.shape[1]:
        raise ValueError("QSA C8 requires row positions and a nonempty page table")
    if seq_lens.shape != (block_table.shape[0],):
        raise ValueError("QSA C8 sequence lengths must match page table requests")
    if v_scale.shape != (heads, query.shape[-1]) or v_scale.dtype != torch.float32 or not v_scale.is_contiguous():
        raise ValueError("QSA native INT8 GBSA requires dense FP32 [KV heads,dim] V scale")
    if p_scale.shape != (1,) or p_scale.dtype != torch.float32 or not p_scale.is_contiguous():
        raise ValueError("QSA native INT8 GBSA requires one FP32 P quantization factor")
    output.zero_()
    if not query.shape[0]:
        return output
    groups, counts, visible, _, row_table = prepare_qsa_gbsa_inputs(
        compact_indices, logical_positions, seq_lens, token_to_req, block_table, heads
    )
    group_capacity = groups.shape[-1]
    tokens = ((group_capacity * GBSA_COMPRESS_RATIO + GBSA_PAGE_SIZE - 1) // GBSA_PAGE_SIZE) * GBSA_PAGE_SIZE
    pages_per_row = tokens // GBSA_PAGE_SIZE
    # Allocate once per forward and reuse across fixed-size query tiles.
    tile_rows = min(query.shape[0], C8_QUERY_TILE)
    workspace_k = torch.empty((tile_rows, tokens, heads, query.shape[-1]), dtype=torch.int8, device=query.device)
    workspace_v = torch.empty_like(workspace_k)
    workspace_scale = torch.empty(
        (tile_rows * pages_per_row, heads, GBSA_PAGE_SIZE), dtype=torch.float32, device=query.device
    )
    local_groups = torch.arange(group_capacity, dtype=torch.int32, device=query.device)
    for start in range(0, query.shape[0], C8_QUERY_TILE):
        end = min(start + C8_QUERY_TILE, query.shape[0])
        rows = end - start
        k, v = workspace_k[:rows], workspace_v[:rows]
        tile_counts = counts[:, start:end]
        tile_groups = groups[:, start:end]
        tile_visible = visible[start:end]
        k_scale = workspace_scale[: rows * pages_per_row]
        pack_qsa_c8_rows(key_cache, tile_groups, tile_counts, tile_visible, row_table[start:end], k, k_scale)
        pack_qsa_c8_rows(value_cache, tile_groups, tile_counts, tile_visible, row_table[start:end], v)
        # Preparation puts the residual group last; all preceding packed
        # groups contain four valid tokens. Rebuild causal lengths after gather.
        tail = tile_visible.remainder(GBSA_COMPRESS_RATIO)
        packed_lengths = (
            tile_counts[0] * GBSA_COMPRESS_RATIO - torch.where(tail > 0, GBSA_COMPRESS_RATIO - tail, 0)
        ).clamp_min(0)
        packed_indices = local_groups.view(1, 1, -1).expand(heads, rows, -1).contiguous()
        local_table = torch.arange(rows * pages_per_row, dtype=torch.int32, device=query.device).view(
            rows, pages_per_row
        )
        codes_q, scales_q = quantize_qsa_rows(query[start:end])
        common = dict(
            cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int64, device=query.device),
            seqused_kv=packed_lengths,
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_sparse_pattern=4,
            mask_mode=1,
            quant_mode=GBSAQuantMode.INT8_QK_DYNAMIC_V_STATIC,
            softmax_precision=0,
            residual_block_mode=0,
            is_consistent_topk=False,
        )
        metadata = generic_block_sparse_attention_metadata(
            packed_indices,
            tile_counts.contiguous(),
            query.shape[1],
            heads,
            query.shape[-1],
            [1, GBSA_COMPRESS_RATIO],
            max_seqlen_q=1,
            max_seqlen_kv=tokens,
            **common,
        )
        result, _ = generic_block_sparse_attention(
            codes_q,
            k.view(-1, GBSA_PAGE_SIZE, heads, query.shape[-1]),
            v.view(-1, GBSA_PAGE_SIZE, heads, query.shape[-1]),
            packed_indices,
            tile_counts.contiguous(),
            [1, GBSA_COMPRESS_RATIO],
            metadata=metadata,
            q_dequant_scale=scales_q,
            k_dequant_scale=k_scale,
            v_dequant_scale=v_scale,
            p_quant_scale=p_scale,
            attention_out_dtype=query.dtype,
            block_table=local_table,
            softmax_scale=query.shape[-1] ** -0.5,
            **common,
        )
        output[start:end].copy_(torch.where((tile_counts[0] > 0).view(-1, 1, 1), result, 0))
    return output
