# SPDX-License-Identifier: Apache-2.0
"""Experimental prefill-only union GBSA integration for Qwen4Exp.

The private custom-op package is required; decode keeps the stock GBSA path.
Union construction stays on the device, including the exact per-query mask.
"""

from types import ModuleType

import torch

from vllm_ascend.ops.triton.qwen4_exp.gbsa import prepare_qsa_gbsa_inputs

UNION_SIZE = 8
BLOCK_Y = 4
INVALID_KEY = 1 << 30
GBSA_PAGE_SIZE = 128


def build_qsa_union_inputs(
    sparse_idx: torch.Tensor,
    sparse_count: torch.Tensor,
    visible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group one request without altering its selected blocks or tail policy."""
    if sparse_idx.ndim != 3 or sparse_idx.shape[0] != 1:
        raise ValueError("Union preparation requires one KV head")
    rows, topk = sparse_idx.shape[1:]
    groups = (rows + UNION_SIZE - 1) // UNION_SIZE
    width = UNION_SIZE * topk
    padded = torch.full((groups * UNION_SIZE, topk), INVALID_KEY, dtype=torch.int32, device=sparse_idx.device)
    columns = torch.arange(topk, device=sparse_idx.device)
    lanes = torch.arange(rows, dtype=torch.int32, device=sparse_idx.device).remainder(UNION_SIZE)
    valid = (columns.unsqueeze(0) < sparse_count[0].unsqueeze(1)) & (sparse_idx[0] >= 0)
    padded[:rows] = torch.where(valid, sparse_idx[0] * UNION_SIZE + lanes.unsqueeze(1), INVALID_KEY)
    sorted_keys = padded.reshape(groups, width).float().sort(dim=-1).values
    keys = sorted_keys.to(torch.int32)
    # Scaling the already-FP32 sort keys by a power of two is exact. All keys
    # are nonnegative; conversion to int32 therefore gives floor(key / 8).
    block_id = (sorted_keys * (1.0 / UNION_SIZE)).to(torch.int32)
    valid = keys < INVALID_KEY
    previous = torch.nn.functional.pad(keys[:, :-1], (1, 0), value=-1)
    previous_block = torch.nn.functional.pad(block_id[:, :-1], (1, 0), value=-1)
    first = valid & (block_id != previous_block)
    rank = first.cumsum(-1, dtype=torch.int32) - 1
    union_count = (rank[:, -1] + 1).unsqueeze(0)
    destinations = torch.where(valid, rank, width).long()
    table = torch.full((groups, width + 1), -1, dtype=torch.int32, device=sparse_idx.device)
    # All duplicate destinations have the same block ID. Invalid entries write
    # only to a discarded sentinel column, never to the last valid union block.
    table.scatter_(1, destinations, torch.where(valid, block_id, -1))
    union_idx = table[:, :width].contiguous().unsqueeze(0)
    if rows % UNION_SIZE == 0:
        padded_visible = visible
    else:
        padded_visible = torch.zeros(groups * UNION_SIZE, dtype=visible.dtype, device=visible.device)
        padded_visible[:rows] = visible
    query_lane = keys.remainder(UNION_SIZE)
    visible_rows = padded_visible.reshape(groups, UNION_SIZE)
    query_lane_long = query_lane.long()
    row_visible = visible_rows.gather(1, query_lane_long)
    unique = valid & (keys != previous)
    bit_table = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.int32, device=sparse_idx.device)
    bit_rows = bit_table.unsqueeze(0).expand(groups, -1).contiguous()
    bits = bit_rows.gather(1, query_lane_long)
    # Full 4-token blocks share one bitset. Only the at-most-one partial
    # block per query requires per-token handling.
    full = unique & ((block_id + 1) * BLOCK_Y <= row_visible)
    full_mask = torch.zeros_like(table)
    full_mask.scatter_add_(1, destinations, torch.where(full, bits, 0))
    mask = full_mask.unsqueeze(-1).expand(-1, -1, BLOCK_Y).contiguous()
    mask = mask.reshape(groups, (width + 1) * BLOCK_Y)

    tail_block = visible_rows.div(BLOCK_Y, rounding_mode="floor")
    tail_key = tail_block * UNION_SIZE + torch.arange(UNION_SIZE, dtype=torch.int32, device=sparse_idx.device)
    location = torch.searchsorted(keys, tail_key).clamp_max(width - 1)
    found = keys.gather(1, location) == tail_key
    tail_rank = rank.gather(1, location)
    tail_length = visible_rows.remainder(BLOCK_Y)
    partial = found & (tail_length > 0)
    token = torch.arange(BLOCK_Y, dtype=torch.int32, device=sparse_idx.device)
    allowed_tail = partial.unsqueeze(-1) & (token < tail_length.unsqueeze(-1))
    tail_dest = torch.where(partial, tail_rank, width).long().unsqueeze(-1) * BLOCK_Y + token
    tail_bits = torch.where(allowed_tail, bit_table.reshape(1, UNION_SIZE, 1), 0)
    mask.scatter_add_(
        1, tail_dest.reshape(groups, UNION_SIZE * BLOCK_Y), tail_bits.reshape(groups, UNION_SIZE * BLOCK_Y)
    )
    exact = mask[:, : width * BLOCK_Y].contiguous().unsqueeze(0)

    return union_idx, union_count, exact


def _run_union_gbsa(
    union_ops: ModuleType,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    union_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cu_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    output: torch.Tensor,
    *,
    max_query_length: int,
) -> torch.Tensor:
    """Use the same native attention contract for single and packed requests."""
    union_idx, union_count, exact = union_inputs
    common = dict(
        cu_seqlens_q=cu_q,
        seqused_kv=seq_lens,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=4,
        mask_mode=3,
        quant_mode=0,
        softmax_precision=0,
        win_left=-1,
        win_right=-1,
        residual_block_mode=0,
        is_consistent_topk=False,
    )
    shape = [UNION_SIZE, BLOCK_Y]
    metadata = union_ops.generic_block_sparse_attention_metadata(
        union_idx,
        union_count,
        query.shape[1],
        1,
        query.shape[-1],
        shape,
        max_seqlen_q=max_query_length,
        max_seqlen_kv=block_table.shape[1] * GBSA_PAGE_SIZE,
        **common,
    )
    result, _ = union_ops.generic_block_sparse_attention(
        query.contiguous(),
        key_cache,
        value_cache,
        union_idx,
        union_count,
        shape,
        metadata=metadata,
        block_table=block_table.contiguous(),
        attn_mask=exact,
        softmax_scale=query.shape[-1] ** -0.5,
        return_softmax_lse=False,
        **common,
    )
    output.copy_(result)
    return output


def qsa_union_gbsa_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    compact_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    request_index: int,
) -> torch.Tensor:
    # Lazy import isolates the optional vendor package from model discovery and
    # from stock decode/graph capture. Missing support must fail visibly.
    import cann_ops_transformer_gbsa_union as union_ops

    if key_cache.shape[2] != 1 or query.shape[1] > 16:
        raise ValueError("Union GBSA requires one KV head and at most 16 Q heads per launch")
    if block_table.shape[1] * GBSA_PAGE_SIZE > (1 << 23):
        raise ValueError("Union index encoding exceeds exact FP32 sort range")
    sparse_idx, sparse_count, visible, _, _ = prepare_qsa_gbsa_inputs(
        compact_indices, logical_positions, seq_lens, token_to_req, block_table, 1, write_row_metadata=False
    )
    union_inputs = build_qsa_union_inputs(sparse_idx, sparse_count, visible)
    rows = query.shape[0]
    cu_q = torch.tensor([0, rows], dtype=torch.int64, device=query.device)
    return _run_union_gbsa(
        union_ops,
        query,
        key_cache,
        value_cache,
        union_inputs,
        cu_q,
        seq_lens[request_index : request_index + 1].contiguous(),
        block_table[request_index : request_index + 1],
        output,
        max_query_length=rows,
    )


def build_qsa_union_batch_inputs(
    sparse_idx: torch.Tensor,
    sparse_count: torch.Tensor,
    visible: torch.Tensor,
    query_starts: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch union construction while keeping every request's groups separate."""
    lengths = tuple(b - a for a, b in zip(query_starts, query_starts[1:]))
    if (
        len(query_starts) < 2
        or query_starts[0] != 0
        or query_starts[-1] != sparse_idx.shape[1]
        or any(n <= 0 for n in lengths)
    ):
        raise ValueError("Packed query starts must cover all nonempty requests")
    if all(n % UNION_SIZE == 0 for n in lengths):
        return build_qsa_union_inputs(sparse_idx, sparse_count, visible)
    # Padding is per request, never across a request boundary. Keep actual Q
    # unpadded; metadata uses its original cu_seqlens_q to locate union groups.
    padded_rows = 0
    destinations = []
    for length in lengths:
        start = padded_rows
        destinations.extend(range(start, start + length))
        padded_rows += (length + UNION_SIZE - 1) // UNION_SIZE * UNION_SIZE
    destination = torch.tensor(destinations, dtype=torch.int64, device=sparse_idx.device)
    padded_idx = torch.full(
        (1, padded_rows, sparse_idx.shape[-1]),
        -1,
        dtype=sparse_idx.dtype,
        device=sparse_idx.device,
    )
    padded_count = torch.zeros((1, padded_rows), dtype=sparse_count.dtype, device=sparse_count.device)
    padded_visible = torch.zeros(padded_rows, dtype=visible.dtype, device=visible.device)
    padded_idx[0].index_copy_(0, destination, sparse_idx[0])
    padded_count[0].index_copy_(0, destination, sparse_count[0])
    padded_visible.index_copy_(0, destination, visible)
    return build_qsa_union_inputs(padded_idx, padded_count, padded_visible)


def qsa_union_gbsa_attention_batch(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    compact_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    query_starts: tuple[int, ...],
    request_indices: tuple[int, ...],
) -> torch.Tensor:
    """Run adjacent packed Prefill requests in one union GBSA launch."""
    import cann_ops_transformer_gbsa_union as union_ops

    if not request_indices or len(request_indices) + 1 != len(query_starts):
        raise ValueError("Each packed query segment must have one request")
    if key_cache.shape[2] != 1 or query.shape[1] > 16:
        raise ValueError("Union GBSA requires one KV head and at most 16 Q heads")
    if block_table.shape[1] * GBSA_PAGE_SIZE > (1 << 23):
        raise ValueError("Union index encoding exceeds exact FP32 sort range")
    sparse_idx, sparse_count, visible, _, _ = prepare_qsa_gbsa_inputs(
        compact_indices,
        logical_positions,
        seq_lens,
        token_to_req,
        block_table,
        1,
        write_row_metadata=False,
    )
    union_inputs = build_qsa_union_batch_inputs(sparse_idx, sparse_count, visible, query_starts)
    cu_q = torch.tensor(query_starts, dtype=torch.int64, device=query.device)
    selected = torch.tensor(request_indices, dtype=torch.int64, device=query.device)
    table = block_table.index_select(0, selected)
    lengths = seq_lens.index_select(0, selected)
    return _run_union_gbsa(
        union_ops,
        query,
        key_cache,
        value_cache,
        union_inputs,
        cu_q,
        lengths,
        table,
        output,
        max_query_length=max(b - a for a, b in zip(query_starts, query_starts[1:])),
    )
