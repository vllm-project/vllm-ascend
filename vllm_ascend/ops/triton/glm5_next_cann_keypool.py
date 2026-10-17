# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Three fused adapters for CANN pooled indexing: inputs, cache writes and indices."""

from types import SimpleNamespace

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["tail_rows", "tail_blocks", "cache_blocks", "INPUT_STRIDE", "INPUT_WIDTH"])
def _write_keypool_caches_kernel(
    tail,
    cache,
    key,
    gate,
    pooled,
    writes,
    state_rows,
    state,
    slots,
    tail_rows,
    HEAD_DIM: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    tail_blocks,
    CAPACITY: tl.constexpr,
    cache_blocks,
    BLOCK_SIZE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    POOLED_STRIDE: tl.constexpr,
    INPUT_STRIDE,
    INPUT_WIDTH,
    SPECULATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    if row < tail_rows:
        input_row = row // INPUT_WIDTH * INPUT_STRIDE + row % INPUT_WIDTH
        dest = tl.load(writes + input_row).to(tl.int64)
        valid = (dest >= 0) & (dest // PAGE_ROWS < tail_blocks) & (dest % PAGE_ROWS < CAPACITY)
        if SPECULATIVE:
            k = tl.load(key + row * HEAD_DIM + cols, valid & (cols < HEAD_DIM), other=0)
            g = tl.load(
                gate + row * HEAD_DIM + cols - HEAD_DIM, valid & (cols >= HEAD_DIM) & (cols < 2 * HEAD_DIM), other=0
            )
            tail_value = tl.where(cols < HEAD_DIM, k, g)
        else:
            source = tl.load(state_rows + input_row, valid, other=0).to(tl.int64)
            tail_value = tl.load(state + source * 2 * HEAD_DIM + cols, valid & (cols < 2 * HEAD_DIM), other=0)
        tl.store(tail + dest * 2 * HEAD_DIM + cols, tail_value, valid & (cols < 2 * HEAD_DIM))
    else:
        pool_row = row - tail_rows
        slot = tl.load(slots + pool_row).to(tl.int64)
        valid = (slot >= 0) & (slot < cache_blocks * BLOCK_SIZE)
        pool_value = tl.load(pooled + pool_row * POOLED_STRIDE + cols, valid & (cols < HEAD_DIM), other=0)
        address = cache + slot // BLOCK_SIZE * CACHE_STRIDE + slot % BLOCK_SIZE * HEAD_DIM + cols
        tl.store(address, pool_value, valid & (cols < HEAD_DIM))


def write_keypool_caches(tail, cache, key, gate, pooled, metadata, tail_metadata, state):
    inputs = tail_metadata.cann_inputs
    writes = inputs.write_indices
    tail_rows = writes.numel()
    pooled_rows = pooled.shape[0]
    # Each program writes one row: tail rows first, then pooled rows.
    _write_keypool_caches_kernel[(tail_rows + pooled_rows,)](
        tail,
        cache,
        key,
        gate,
        pooled,
        writes,
        inputs.state_indices,
        state,
        metadata.pooled_slot_mapping,
        tail_rows,
        cache.shape[-1],
        inputs.page_rows,
        tail.shape[0],
        tail.shape[1],
        cache.shape[0],
        cache.shape[1],
        cache.stride(0),
        pooled.stride(0),
        writes.stride(0),
        writes.shape[1],
        key is not None,
        triton.next_power_of_2(2 * cache.shape[-1]),
    )


@triton.jit(do_not_specialize=["num_reqs"], do_not_specialize_on_alignment=["positions"])
def _pack_indices_kernel(
    indices,
    positions,
    cumulative,
    output,
    INPUT_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    num_reqs,
    OUTPUT_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid_row = row < tl.load(cumulative + num_reqs - 1)
    position = tl.load(positions + row, valid_row, other=0).to(tl.int64)
    tail_start = (position + 1) // POOL_SIZE * POOL_SIZE
    history = tl.minimum(tail_start, TOPK).to(tl.int32)
    value = tl.load(indices + row * INPUT_STRIDE + cols, valid_row & (cols < history), other=-1)
    tail_columns = cols - history
    tail_count = (position + 1 - tail_start).to(tl.int32)
    # Keep wide index vectors in int32: Ascend scalarizes int64 vector
    # conversion to the int32 SFA output. Only position arithmetic is int64.
    tail_value = tail_start.to(tl.int32) + tail_columns
    value = tl.where(
        valid_row & (tail_columns >= 0) & (tail_columns < tail_count) & (cols < TOPK + POOL_SIZE - 1),
        tail_value,
        value,
    )
    tl.store(output + row * OUTPUT_STRIDE + cols, value, cols < OUTPUT_COLS)


def pack_keypool_indices(indices, metadata, topk, pool_size, output_buffer=None):
    positions, cumulative = metadata.positions, metadata.cum_query_lens
    output = indices if output_buffer is None else output_buffer[: indices.shape[0]]
    _pack_indices_kernel[(indices.shape[0],)](
        indices,
        positions,
        cumulative,
        output,
        indices.stride(0),
        output.stride(0),
        topk,
        pool_size,
        cumulative.numel(),
        output.shape[1],
        triton.next_power_of_2(output.shape[1]),
    )
    return output[:, : topk + pool_size - 1].unsqueeze(1)


METADATA_BLOCK = 64
METADATA_MIN_BLOCK = 8
METADATA_PROGRAMS = 32
# Bound the request inversion tile to fit Ascend unified buffer storage.
METADATA_REQUEST_BLOCK = 128


@triton.jit(do_not_specialize=["n", "tokens", "source", "actual", "pooled_rows", "cp_start", "cp_end", "WIDTH"])
def _build_keypool_metadata_kernel(
    raw_query,
    raw_lengths,
    raw_positions,
    raw_slots,
    pages,
    positions,
    slots,
    seq,
    raw,
    query,
    ends,
    starts,
    tails,
    offsets,
    compact,
    history,
    writes,
    state,
    retained,
    n,
    tokens,
    source,
    actual,
    pooled_rows,
    QUERY_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    LOGICAL: tl.constexpr,
    POOL: tl.constexpr,
    CAPACITY: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    WIDTH,
    TAIL_MODE: tl.constexpr,
    SPECULATIVE: tl.constexpr,
    BLOCK: tl.constexpr,
    REQUESTS: tl.constexpr,
    COLUMNS: tl.constexpr,
    cp_ends,
    cp_seq,
    cp_tails,
    cp_start,
    cp_end,
    CP_MODE: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q0 = tl.load(raw_query + row * QUERY_STRIDE, row <= n, other=0)
    q0 = tl.minimum(tl.where(q0 < 0, actual, q0), actual)
    q1 = tl.load(raw_query + (row + 1) * QUERY_STRIDE, row < n, other=0)
    q1 = tl.minimum(tl.where(q1 < 0, actual, q1), actual)
    length = q1 - q0
    raw_len = tl.load(raw_lengths + row * SEQ_STRIDE, row < n, other=0)
    start = raw_len - length
    if TAIL_MODE:
        cols = tl.arange(0, COLUMNS)
        page = tl.load(pages + row * PAGE_STRIDE, row < n, other=0).to(tl.int64)
        hist = tl.maximum(page, 0)[:, None] * PAGE_ROWS + ((start - start % POOL)[:, None] + cols[None, :]) % CAPACITY
        tl.store(history + row[:, None] * POOL + cols[None, :], hist, (row[:, None] < n) & (cols[None, :] < POOL))
        if SPECULATIVE:
            source_row = q1[:, None] - WIDTH + cols[None, :]
            safe = tl.maximum(source_row, 0)
            tl.store(
                retained + row[:, None] * WIDTH + cols[None, :],
                safe.to(tl.int64),
                (row[:, None] < n) & (cols[None, :] < WIDTH),
            )
            slot = tl.load(
                raw_slots + safe * SLOT_STRIDE, (row[:, None] < n) & (cols[None, :] < WIDTH) & (safe < actual), other=-1
            ).to(tl.int64)
            pos = tl.load(
                raw_positions + safe * POS_STRIDE,
                (row[:, None] < n) & (cols[None, :] < WIDTH) & (safe < actual),
                other=0,
            ).to(tl.int64)
            address = page[:, None] * PAGE_ROWS + pos % CAPACITY
            address = tl.where((source_row >= q0[:, None]) & (slot >= 0) & (page[:, None] >= 0), address, -1)
        else:
            end = start % POOL + length
            state_row = (row + 1 + tl.where(end >= POOL, n, 0))[:, None] * POOL + cols[None, :]
            tl.store(
                state + row[:, None] * WIDTH + cols[None, :],
                state_row.to(tl.int64),
                (row[:, None] < n) & (cols[None, :] < WIDTH),
            )
            pos = (start + length - end % POOL)[:, None] + cols[None, :]
            address = page[:, None] * PAGE_ROWS + pos % CAPACITY
            address = tl.where(
                (length[:, None] > 0) & (cols[None, :] < (end % POOL)[:, None]) & (page[:, None] >= 0), address, -1
            )
        tl.store(writes + row[:, None] * WIDTH + cols[None, :], address, (row[:, None] < n) & (cols[None, :] < WIDTH))
    else:
        pos = tl.load(raw_positions + row * POS_STRIDE, row < source, other=0).to(tl.int64)
        slot = tl.load(raw_slots + row * SLOT_STRIDE, row < source, other=-1).to(tl.int64)
        compressed = slot // LOGICAL * (LOGICAL // POOL) + slot % LOGICAL // POOL
        tl.store(positions + row, pos, row < tokens)
        tl.store(slots + row, tl.where((slot >= 0) & ((pos + 1) % POOL == 0), compressed, -1), row < tokens)
        tl.store(query + row, q0, row <= n)
        tl.store(seq + row, raw_len // POOL, row < n)
        tl.store(raw + row, raw_len, row < n)
        tl.store(ends + row, q1, row < n)
        tl.store(starts + row, start, row < n)
        tl.store(tails + row, raw_len % POOL, row < n)
        tl.store(offsets + row, start % POOL, row < n)
        if CP_MODE:
            local_end = tl.minimum(q1, cp_end)
            has_query = local_end > tl.maximum(q0, cp_start)
            local_length = tl.where(has_query, tl.maximum(raw_len - tl.maximum(q1 - cp_end, 0), 0), 0)
            tl.store(cp_ends + row, tl.maximum(local_end - cp_start, 0), row < n)
            tl.store(cp_seq + row, local_length // POOL, row < n)
            tl.store(cp_tails + row, local_length % POOL, row < n)
        # Invert request pool counts to locate each compact output's boundary
        # token directly; evicted slots still occupy their CANN output row.
        prefix = tl.full((), 0, tl.int32)
        compact_address = tl.full((BLOCK,), -1, tl.int64)
        for request_start in range(0, n, REQUESTS):
            req = request_start + tl.arange(0, REQUESTS)
            begin = tl.load(raw_query + req * QUERY_STRIDE, req < n, other=0)
            finish = tl.load(raw_query + (req + 1) * QUERY_STRIDE, req < n, other=0)
            begin = tl.minimum(tl.where(begin < 0, actual, begin), actual)
            finish = tl.minimum(tl.where(finish < 0, actual, finish), actual)
            total = tl.load(raw_lengths + req * SEQ_STRIDE, req < n, other=0)
            first = total - (finish - begin)
            count = total // POOL - first // POOL
            inclusive = prefix + tl.cumsum(count, 0)
            prior = inclusive - count
            match = (req[None, :] < n) & (row[:, None] >= prior[None, :]) & (row[:, None] < inclusive[None, :])
            boundary = begin[None, :] + POOL - first[None, :] % POOL - 1 + (row[:, None] - prior[None, :]) * POOL
            token = tl.sum(tl.where(match, boundary, 0), 1)
            valid = tl.sum(match.to(tl.int32), 1) > 0
            pool_slot = tl.load(
                raw_slots + token * SLOT_STRIDE, valid & (token < source) & (row < pooled_rows), other=-1
            ).to(tl.int64)
            address = pool_slot // LOGICAL * (LOGICAL // POOL) + pool_slot % LOGICAL // POOL
            compact_address = tl.where(valid & (pool_slot >= 0), address, compact_address)
            prefix += tl.sum(count, 0)
        tl.store(compact + row + 1, compact_address, row < pooled_rows)
        if tl.program_id(0) == 0:
            tl.store(compact, -1)


def build_keypool_metadata(common, buffers, tokens, source, logical, ratio, *, query_metadata=None, local_start=0):
    n = common.num_reqs
    pooled = min(tokens, tokens // ratio + n)
    # Fully padded buckets can have empty source tensors. Keep masked input
    # pointers backed by allocated storage without adding a device operation.
    source_positions = common.positions if source else buffers.positions
    source_slots = common.slot_mapping if source else buffers.slot_mapping
    block = min(
        METADATA_BLOCK,
        max(METADATA_MIN_BLOCK, triton.next_power_of_2(triton.cdiv(max(tokens, n + 1), METADATA_PROGRAMS))),
    )
    _build_keypool_metadata_kernel[(triton.cdiv(max(tokens, n + 1), block),)](
        common.query_start_loc,
        common.seq_lens,
        source_positions,
        source_slots,
        None,
        buffers.positions,
        buffers.slot_mapping,
        buffers.seq_lens,
        buffers.raw_seq_lens,
        buffers.query_start_loc,
        buffers.cum_query_lens,
        buffers.start_pos,
        buffers.pool_tail,
        buffers.start_offsets,
        buffers.pooled_slot_mapping,
        None,
        None,
        None,
        None,
        n,
        tokens,
        source,
        common.num_actual_tokens,
        pooled,
        common.query_start_loc.stride(0),
        common.seq_lens.stride(0),
        source_positions.stride(0),
        source_slots.stride(0),
        0,
        logical,
        ratio,
        0,
        0,
        0,
        False,
        False,
        block,
        METADATA_REQUEST_BLOCK,
        1,
        query_metadata.cum_query_lens if query_metadata is not None else None,
        query_metadata.seq_lens if query_metadata is not None else None,
        query_metadata.pool_tail if query_metadata is not None else None,
        local_start,
        local_start + query_metadata.positions.numel() if query_metadata is not None else 0,
        query_metadata is not None,
    )
    return (
        buffers.positions[:tokens],
        buffers.slot_mapping[:tokens],
        buffers.seq_lens[:n],
        buffers.raw_seq_lens[:n],
        buffers.query_start_loc[: n + 1],
        buffers.cum_query_lens[:n],
        buffers.start_pos[:n],
        buffers.pool_tail[:n],
    )


def build_tail_inputs(common, pool, capacity, page_rows, retained_width, buffers):
    n = common.num_reqs
    width = retained_width or pool
    history = buffers["history"][:n]
    writes = buffers["writes"].view(-1)[: n * width].view(n, width)
    state = buffers["state"].view(-1)[: n * width].view(n, width)
    retained = buffers["retained"].view(-1)[: n * width].view(n, width) if retained_width else None
    _build_keypool_metadata_kernel[(triton.cdiv(n + 1, METADATA_MIN_BLOCK),)](
        common.query_start_loc,
        common.seq_lens,
        common.positions,
        common.slot_mapping,
        common.block_table_tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        history,
        writes,
        state,
        retained,
        n,
        0,
        0,
        common.num_actual_tokens,
        0,
        common.query_start_loc.stride(0),
        common.seq_lens.stride(0),
        common.positions.stride(0),
        common.slot_mapping.stride(0),
        common.block_table_tensor.stride(0),
        0,
        pool,
        capacity,
        page_rows,
        width,
        True,
        retained_width > 0,
        METADATA_MIN_BLOCK,
        1,
        triton.next_power_of_2(max(pool, capacity)),
        None,
        None,
        None,
        0,
        0,
        False,
    )
    return SimpleNamespace(
        history_indices=history,
        write_indices=writes,
        state_indices=state,
        retained_indices=retained,
        page_rows=page_rows,
    )
