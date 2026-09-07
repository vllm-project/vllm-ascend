# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pool scoring and selection for the GLM-5.3-Flash kpool indexer on Ascend.

``npu_lightning_indexer`` scores a paged key cache and returns the ids of the
highest-scoring entries. Because the GLM index cache holds one entry per pool of
``index_kpool`` tokens, those ids are pool ids, and the sparse attention needs
token ids -- so selection is scoring plus an expansion.

Two properties of the operator shape everything here, both established by
probing an Ascend 950:

* It takes one key length per *request*, and its ``sparse_mode=3`` mask moves
  the boundary by one key per query row. With pooled keys a key is
  ``index_kpool`` tokens, so that boundary advances four times too fast. Asking
  for a key length per query row is rejected outright (error 561002).
* ``sparse_mode=0`` applies no mask at all and matches an unmasked reference.

So selection runs unmasked over the pool prefix that is causally valid for
*every* query row of the request, and the expansion names the remaining recent
tokens explicitly. That is exact -- nothing is dropped and no future token is
named -- and it covers a draft-verify batch, whose rows sit at different
positions, for the same reason it covers a prefill chunk.
"""

import torch

from vllm_ascend.models.glm5next.ops.kpool_compress import (
    expand_pools_and_append_tail,
    shared_pool_prefix,
)

# npu_lightning_indexer reports the key page's pool axis as ``block_size`` and
# requires it to be a multiple of 16 in (0, 1024]. A 160-pool page (block_size
# 640, index_kpool 4) is valid; a 165-pool page is not.
LIGHTNING_INDEXER_POOLS_ALIGNMENT = 16
LIGHTNING_INDEXER_MAX_POOLS = 1024

# npu_sparse_flash_attention reads exactly the first ``actual_seq_lengths_kv``
# entries of the index list and never looks past them, so the columns beyond a
# row's selection can hold anything. Probing an Ascend 950 confirmed the value
# makes no difference; keying it above every real token id is what lets the sort
# in `compact_selection` pack the valid ids to the front.
SPARSE_INDEX_PAD = 2**31 - 1


def pa_bsnd_keys(index_cache: torch.Tensor, head_dim: int, pools_per_block: int) -> tuple[torch.Tensor, int]:
    """View the pooled key cache as ``[blocks, pools_per_block, 1, head_dim]``.

    ``DeepseekV32IndexerCache.bind_kv_cache`` squeezes a size-1 head axis at
    dim 1, so the tensor may arrive as ``[blocks, pools, C]`` or as
    ``[blocks, pools, 1, C]``. Inferring the pool axis with ``view(..., -1,
    1, head_dim)`` is unsafe: a 132-wide FP8-plus-scale page (160 cells)
    viewed as 128-wide bf16 becomes 165 pools, which the operator then
    rejects as ``block_size must be a multiple of 16``.

    Returns the view and how many of its pages one scheduler block spans, which
    the caller multiplies into the block table.
    """
    if index_cache.ndim == 3:
        index_cache = index_cache.unsqueeze(2)
    if index_cache.ndim != 4:
        raise RuntimeError(f"kpool indexer key cache must be 3-D or 4-D PA_BSND, got {tuple(index_cache.shape)}")

    _, dim1, dim2, content = index_cache.shape
    if dim2 == 1:
        page_pools = dim1
    elif dim1 == 1:
        page_pools = dim2
        index_cache = index_cache.permute(0, 2, 1, 3).contiguous()
    else:
        raise RuntimeError(f"kpool indexer key cache needs a size-1 head axis, got {tuple(index_cache.shape)}")

    if content != head_dim:
        inferred_pools = index_cache.numel() // (index_cache.shape[0] * head_dim)
        raise RuntimeError(
            f"kpool indexer key cache content width is {content}, expected "
            f"{head_dim}. Viewing that page as {head_dim}-wide would invent "
            f"{inferred_pools} pools (165 for a 160-pool FP8-plus-scale page) "
            "and npu_lightning_indexer would reject the shape."
        )

    # Unifying page sizes across the cache groups pads this page out to the far
    # wider MLA one, leaving a block's pools in its first ``pools_per_block``
    # slots and padding behind them. The operator caps a page at 1024 pools and
    # would reject the padded width, but splitting the page into that many
    # sub-pages is a plain reshape over the same bytes, and a block table scaled
    # by the split still lands each block on its own pools.
    if page_pools % pools_per_block:
        raise RuntimeError(
            f"kpool indexer page holds {page_pools} pools, which is not a whole "
            f"number of the {pools_per_block} one scheduler block addresses."
        )
    block_stride = page_pools // pools_per_block
    if block_stride > 1:
        index_cache = index_cache.reshape(-1, pools_per_block, 1, head_dim)

    if pools_per_block % LIGHTNING_INDEXER_POOLS_ALIGNMENT != 0 or not (
        0 < pools_per_block <= LIGHTNING_INDEXER_MAX_POOLS
    ):
        raise RuntimeError(
            "npu_lightning_indexer requires the key block_size (pools per page) "
            f"to be a multiple of {LIGHTNING_INDEXER_POOLS_ALIGNMENT} in "
            f"(0, {LIGHTNING_INDEXER_MAX_POOLS}], got {pools_per_block}."
        )
    return index_cache, block_stride


def score_and_select_pools(
    query: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cumulative_query_lens: torch.Tensor,
    selectable_pools: torch.Tensor,
    select_pools: int,
) -> torch.Tensor:
    """Pick the highest-scoring pools for each query row.

    Args:
        query: ``[num_tokens, index_n_heads, head_dim]`` bf16, TND layout.
        weights: ``[num_tokens, index_n_heads]`` -- the per-head gate the
            scores are summed under.
        index_cache: ``[num_blocks, pools_per_block, 1, head_dim]`` bf16,
            PA_BSND layout.
        block_table: ``[num_requests, max_blocks]``.
        cumulative_query_lens: ``[num_requests]`` -- TND wants the query
            lengths cumulated.
        selectable_pools: ``[num_requests]`` -- pools offered to each request,
            from :func:`shared_pool_prefix`.
        select_pools: budget in pools, i.e. ``index_topk // index_kpool``.

    Returns:
        ``[num_tokens, 1, select_pools]`` int32 pool ids, padded with -1.
    """
    import torch_npu

    pool_ids, _ = torch_npu.npu_lightning_indexer(
        query=query,
        key=index_cache,
        weights=weights,
        actual_seq_lengths_query=cumulative_query_lens,
        actual_seq_lengths_key=selectable_pools,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=select_pools,
        # Not sparse_mode=3: see the module docstring. The prefix handed over
        # in `selectable_pools` is already causally valid for every query row,
        # so a mask would only ever remove pools the request may legitimately
        # see.
        sparse_mode=0,
    )
    return pool_ids


def select_token_ids(
    query: torch.Tensor,
    weights: torch.Tensor,
    index_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
    pools_per_block: int,
    topk_tokens: int,
    max_query_len: int,
) -> torch.Tensor:
    """Score pools and expand the winners into token ids.

    Args:
        query: ``[num_tokens, index_n_heads, head_dim]`` bf16.
        weights: ``[num_tokens, index_n_heads]``.
        index_cache: the pooled bf16 cache in PA_BSND layout.
        block_table: ``[num_requests, max_blocks]``.
        query_lens: ``[num_requests]`` -- query rows per request this step, on
            ``query``'s device. Both length arguments have to be there already:
            an H2D copy would record the host address it read during ACL-graph
            capture and every replay would then read whatever occupies it.
        seq_lens: ``[num_requests]`` -- tokens known after this step, likewise
            on ``query``'s device.
        pool_size: the checkpoint's ``index_kpool``.
        pools_per_block: pools one scheduler block addresses, from the cache
            layer. Not inferred from the allocation: page-size unification pads
            this cache out to the MLA page, and the padded width would name a
            page far wider than the operator accepts.
        topk_tokens: the checkpoint's ``index_topk``.
        max_query_len: the batch's widest request, taken from the attention
            metadata rather than from ``query_lens`` -- reading a device tensor
            to size the output would abort ACL-graph capture.

    Returns:
        ``[num_tokens, topk_tokens + tail_width]`` int32 token ids relative to
        each request's start, valid ids packed to the front, plus the
        ``[num_tokens]`` count of valid ids per row.
    """
    assert topk_tokens % pool_size == 0, (topk_tokens, pool_size)

    index_cache, block_stride = pa_bsnd_keys(index_cache, query.shape[-1], pools_per_block)
    if block_stride > 1:
        # npu_lightning_indexer treats -1 as an unallocated slot. Multiplying
        # the table by the split would turn that sentinel into -stride and the
        # operator would then index a real page.
        block_table = torch.where(block_table >= 0, block_table * block_stride, block_table)

    # Copying a length in would work while eager and go silently wrong once
    # captured, so refuse it at the boundary instead of reading a stale replay.
    if query_lens.device != query.device or seq_lens.device != query.device:
        raise RuntimeError(
            "kpool selection needs its lengths on the query's device, got "
            f"query_lens={query_lens.device}, seq_lens={seq_lens.device}, query={query.device}."
        )

    selectable = shared_pool_prefix(seq_lens, query_lens, pool_size)
    pool_ids = score_and_select_pools(
        query,
        weights,
        index_cache,
        block_table=block_table,
        cumulative_query_lens=query_lens.cumsum(0).to(torch.int32),
        selectable_pools=selectable,
        select_pools=topk_tokens // pool_size,
    )

    # The operator answers per query row, so the request-granular boundary and
    # sequence length both have to be spread out to one entry per row. The width
    # is passed in because deriving it from a device tensor is a host read.
    rows = request_index_per_row(query_lens, query.shape[0])
    return compact_selection(
        expand_pools_and_append_tail(
            pool_ids.reshape(query.shape[0], -1),
            row_seq_lens(seq_lens, query_lens, rows=rows),
            pool_size,
            selectable_pools=selectable[rows],
            tail_width=tail_width_for(pool_size, max_query_len),
        )
    )


def compact_selection(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack each row's valid token ids to the front and count them.

    The expansion leaves the valid ids scattered: the selected history occupies
    the front of a budget-wide block and the tail sits after the whole block. A
    row that filled 5 of 512 budget slots therefore has its 20 oldest tokens in
    columns 0-19 and its newest two in columns 2048-2049.

    That layout is unusable, because the operator reads a prefix of the list
    rather than the whole of it: with the count it would stop at column 20 and
    never reach the tail, which is exactly the newest tokens the model cannot do
    without. Sorting with the invalid entries keyed above every real id closes
    the gap and leaves the ids in ascending token order, without assuming
    anything about where top-k put its own padding.
    """
    counts = (token_ids >= 0).sum(dim=-1, dtype=torch.int32)
    keyed = torch.where(token_ids >= 0, token_ids, SPARSE_INDEX_PAD)
    ordered, _ = keyed.sort(dim=-1)
    return ordered.to(torch.int32), counts


def rows_as_batch_entries(
    block_table: torch.Tensor, num_tokens: int, num_requests: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give every query row its own entry of the attention operator's batch.

    Returns the block table restated per row and the cumulative query lengths
    that size the batch, to be passed alongside `compact_selection`'s per-row
    counts.

    The operator takes one key length per batch entry, and a draft-verify step
    gives a request several rows whose selections differ in length, an earlier
    row having less history to choose from. One entry per row is what lets each
    row state its own length. Padding a request's rows to a common one is no
    alternative, because the operator reads a prefix of the index list and no id
    costs nothing: repeating a selected token just weighs it twice in the
    softmax. A plain decode step, one row per request, is the case where the two
    readings coincide -- which is why passing the per-row counts worked until
    drafts made a request span several rows.

    Nothing is lost by splitting a request up. A row may only see its own
    history, and the selection already enforced that when it built the row's
    list, so the causal crop across a request's rows has nothing left to do;
    with a single row per entry it spans nothing.

    Both results are built on ``block_table``'s device. The decode metadata's
    sequence lengths are a host tensor, and deriving these from those would
    strand them on the CPU, where an ACL-graph capture records the host address
    and every later replay reads whatever occupies it by then.
    """
    # A decode batch gives every request the same number of rows, which is what
    # lets a row name its request by division rather than a length scan.
    rows_per_request = max(num_tokens // max(num_requests, 1), 1)
    rows = torch.arange(num_tokens, device=block_table.device)
    # TND cumulates the query lengths, and one row each cumulates to 1, 2, 3...
    return block_table[rows // rows_per_request], (rows + 1).to(torch.int32)


def request_index_per_row(query_lens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Which request each query row belongs to, ``[num_tokens]`` int64.

    Spelling the source out is not redundant: the one-argument
    ``repeat_interleave(repeats)`` selects the ``aten::repeat_interleave.Tensor``
    overload, which has no NPU kernel and silently falls back to the CPU. That
    drags the lengths off the device and stalls the step, once per layer. The
    two-argument overload is the one the rest of this repo relies on.
    """
    return torch.repeat_interleave(
        torch.arange(query_lens.shape[0], dtype=torch.int64, device=query_lens.device),
        query_lens,
        output_size=num_tokens,
    )


def row_seq_lens(
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sequence length as of each query row, not as of each request.

    ``seq_lens`` counts the tokens a request knows once the whole step has been
    applied, so a row ``i`` places before the end of its request knows ``i``
    fewer of them.

    Args:
        seq_lens: ``[num_requests]``.
        query_lens: ``[num_requests]``.
        rows: ``[num_tokens]`` request index per query row, if already built.
    """
    if rows is None:
        # Only for callers that are not on the decode path: sizing the output
        # from a device tensor is a host read.
        rows = request_index_per_row(query_lens, int(query_lens.sum()))
    query_lens = query_lens.to(torch.int64)
    starts = torch.cumsum(query_lens, 0) - query_lens
    within = torch.arange(rows.shape[0], device=rows.device) - starts[rows]
    return (seq_lens[rows] - query_lens[rows] + 1 + within).to(seq_lens.dtype)


def tail_width_for(pool_size: int, max_query_len: int) -> int:
    """Output width the tail expansion needs for a shared selection boundary.

    The boundary is at most ``pool_size - 1`` tokens behind the earliest row of
    the request, and the last row is ``max_query_len - 1`` further along, so
    that many token ids can fall outside the selectable prefix.
    """
    return pool_size - 1 + max_query_len - 1
