# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare nonempty inputs for the CANN pooled key indexer."""

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["tokens", "requests", "table_rows"])
def _prepare_nonempty_keypool_inputs_kernel(
    query,
    weights,
    ends,
    lengths,
    tails,
    table,
    padded_query,
    padded_weights,
    padded_ends,
    padded_lengths,
    padded_tails,
    padded_table,
    tokens,
    requests,
    table_rows,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    if block * BLOCK < (tokens + 1) * HEADS * DIM:
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(query + offsets, offsets < tokens * HEADS * DIM, other=0)
        tl.store(padded_query + offsets, value, offsets < (tokens + 1) * HEADS * DIM)
    if block * BLOCK < (tokens + 1) * HEADS:
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(weights + offsets, offsets < tokens * HEADS, other=0)
        tl.store(padded_weights + offsets, value, offsets < (tokens + 1) * HEADS)
    # Int64 vector arithmetic is scalarized on A3. Keep these small metadata
    # tiles separate from the wide BF16/int32 copies. Tile boundaries also
    # keep different cores from writing the same 32-byte output cache line.
    META_BLOCK: tl.constexpr = 16
    if block * META_BLOCK < requests + 1:
        offsets = block * META_BLOCK + tl.arange(0, META_BLOCK)
        end = tl.load(ends + offsets, offsets < requests, other=0)
        length = tl.load(lengths + offsets, offsets < requests, other=0)
        tail = tl.load(tails + offsets, offsets < requests, other=0)
        tl.store(padded_ends + offsets, tl.where(offsets < requests, end, tokens + 1), offsets < requests + 1)
        tl.store(padded_lengths + offsets, tl.maximum(length, 1), offsets < requests + 1)
        tl.store(padded_tails + offsets, tail, offsets < requests + 1)
    if block * BLOCK < (table_rows + 1) * TABLE_COLS:
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(table + offsets, offsets < table_rows * TABLE_COLS, other=0)
        tl.store(padded_table + offsets, value, offsets < (table_rows + 1) * TABLE_COLS)


def prepare_nonempty_keypool_inputs(query, weights, ends, lengths, tails, table):
    """Append safe CANN sentinel inputs in one launch, without temporary fills.

    Allocate outputs per invocation so independently captured graphs and layers
    cannot alias a mutable workspace. Graph replay reuses captured allocations.
    Contiguous inputs use a single vector-copy launch. Normalize strided views
    first to avoid scalar gather lowering on A3; contiguous() is a no-op on the
    ordinary backend inputs. All operator inputs returned here are contiguous.
    """
    tokens, heads, dim = query.shape
    requests = ends.numel()
    table_rows, table_cols = table.shape
    padded_query = query.new_empty((tokens + 1, heads, dim))
    padded_weights = weights.new_empty((tokens + 1, heads))
    padded_ends = ends.new_empty((requests + 1,))
    padded_lengths = lengths.new_empty((requests + 1,))
    padded_tails = tails.new_empty((requests + 1,))
    padded_table = table.new_empty((table_rows + 1, table_cols))
    block = 8192
    size = max(padded_query.numel(), padded_weights.numel(), requests + 1, padded_table.numel())
    grid = max(triton.cdiv(size, block), triton.cdiv(requests + 1, 16))
    _prepare_nonempty_keypool_inputs_kernel[(grid,)](
        query.contiguous(),
        weights.contiguous(),
        ends.contiguous(),
        lengths.contiguous(),
        tails.contiguous(),
        table.contiguous(),
        padded_query,
        padded_weights,
        padded_ends,
        padded_lengths,
        padded_tails,
        padded_table,
        tokens,
        requests,
        table_rows,
        heads,
        dim,
        table_cols,
        block,
    )
    return padded_query, padded_weights, padded_ends, padded_lengths, padded_tails, padded_table
