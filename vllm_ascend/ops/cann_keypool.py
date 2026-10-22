# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State gathering and nonempty input adaptation for CANN KeyPool."""

import torch

from vllm_ascend.ops.triton.prepare_nonempty_keypool_inputs import prepare_nonempty_keypool_inputs


def _cache_rows(cache):
    width = cache.shape[-1]
    if cache.stride(0) % width:
        raise ValueError("KeyPool cache pages must align to complete rows.")
    rows = (cache.shape[0] - 1) * (cache.stride(0) // width) + cache.shape[1]
    return cache.as_strided((rows, width), (width, 1))


def prepare_keypool_state(tail, starts, tail_metadata, pool):
    n, width = starts.numel(), tail.shape[-1]
    inputs = tail_metadata.cann_inputs
    if inputs.page_rows * width != tail.stride(0):
        raise ValueError("KeyPool tail metadata must preserve the physical cache page stride.")
    state = tail.new_empty((1 + 2 * n, pool, width))
    torch.index_select(
        _cache_rows(tail), 0, inputs.history_indices.flatten(), out=state[1 : n + 1].view(n * pool, width)
    )
    return state


def gather_retained_hidden(hidden, indices):
    return torch.index_select(hidden, 0, indices.flatten())


def pool_key_indexer_nonempty(
    op,
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q,
    actual_seq_k,
    block_table,
    topk,
    pool_size,
    layout_q="TND",
    layout_k="PA_BBND",
    mask_mode=3,
):
    """Keep CANN's TND indexer out of its unsafe zero-work paths.

    Short DSA-CP shards can have queries but no completed pool. On A3 the
    zero-work kernel initializes B*T output rows although TND only allocates T.
    A trailing dummy request guarantees work, including fully padded replay.
    Empty real histories use one physical pool; index packing discards these
    candidates using the original positions, so they cannot enter attention.
    All metadata stays on device and the shapes remain graph-capturable.
    """
    if layout_q != "TND" or layout_k != "PA_BBND" or mask_mode != 3:
        raise ValueError("The KPool empty-work adapter requires causal TND/PA_BBND inputs.")
    tokens = query.shape[0]
    query_padded, weights_padded, query_ends, key_lengths, tails, tables = prepare_nonempty_keypool_inputs(
        query, weights, actual_seq_q, actual_seq_k, pool_tail_k, block_table
    )
    indices, values = op(
        query_padded,
        pool_key,
        weights_padded,
        tails,
        actual_seq_q=query_ends,
        actual_seq_k=key_lengths,
        block_table=tables,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
    )
    return indices[:tokens], values
