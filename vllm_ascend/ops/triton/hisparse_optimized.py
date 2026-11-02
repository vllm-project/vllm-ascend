# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-only HiSparse planning and mapped-host row copies for Ascend.

Device sorts and binary searches form the union of a request's contiguous
verification rows. All selected slots are protected before any eviction.
Scratch belongs to the caller and is allocated before graph capture.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _translate(
    token,
    request,
    state,
    source,
    resident,
    HOST_ROWS: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    SOURCE_COLS: tl.constexpr,
    SOURCE_BLOCK: tl.constexpr,
    RESIDENT_STRIDE: tl.constexpr,
    RESIDENT_ROWS: tl.constexpr,
    RESIDENT_COLS: tl.constexpr,
    RESIDENT_BLOCK: tl.constexpr,
    RESIDENT_NULL: tl.constexpr,
    HAS_SOURCE: tl.constexpr,
    HAS_RESIDENT: tl.constexpr,
):
    host = token
    device = tl.full(token.shape, -1, tl.int32)
    if HAS_SOURCE:
        col = token // SOURCE_BLOCK
        valid = (token >= 0) & (request >= 0) & (request < SOURCE_ROWS) & (col < SOURCE_COLS)
        block = tl.load(source + request.to(tl.int64) * SOURCE_STRIDE + col, mask=valid, other=0)
        host = tl.where(valid & (block > 0), block * SOURCE_BLOCK + token % SOURCE_BLOCK, -1)
        if HAS_RESIDENT:
            col = token // RESIDENT_BLOCK
            valid = (token >= 0) & (state >= 0) & (state < RESIDENT_ROWS) & (col < RESIDENT_COLS)
            block = tl.load(resident + state.to(tl.int64) * RESIDENT_STRIDE + col, mask=valid, other=RESIDENT_NULL)
            device = tl.where(
                valid & (block >= 0) & (block != RESIDENT_NULL),
                block * RESIDENT_BLOCK + token % RESIDENT_BLOCK,
                -1,
            )
    host = tl.where((host >= 0) & (host < HOST_ROWS), host, -1)
    return host, device


@triton.jit
def _lookup_with_position(sorted_keys, value, active, N: tl.constexpr):
    # Keys pack (nonnegative host row, original position). Lower-bound also
    # identifies the first reference when a query contains duplicate indices.
    low = tl.full(value.shape, 0, tl.int32)
    high = tl.full(value.shape, N, tl.int32)
    target = value.to(tl.int64) << 32
    for _ in range(0, tl.constexpr(N.bit_length())):
        middle = (low + high) // 2
        key = tl.load(sorted_keys + middle, mask=active & (middle < N), other=9223372036854775807)
        left = key < target
        low = tl.where(left, middle + 1, low)
        high = tl.where(left, high, middle)
    key = tl.load(sorted_keys + low, mask=active & (low < N), other=9223372036854775807)
    found = active & (low < N) & ((key >> 32) == value)
    return (key & 4294967295).to(tl.int32), found, low


@triton.jit
def _lookup(sorted_keys, value, active, N: tl.constexpr):
    slot, found, _ = _lookup_with_position(sorted_keys, value, active, N)
    return slot, found


@triton.jit
def prepare_residency_kernel(
    indices,
    requests,
    states,
    source,
    resident,
    dgi,
    hot_indices,
    resolved,
    counts,
    row_keys,
    cache_keys,
    metadata,
    num_valid_rows,
    HAS_VALID_ROWS: tl.constexpr,
    ROWS: tl.constexpr,
    TOP_K: tl.constexpr,
    HOT_SIZE: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    COUNT_STRIDE: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    STATE_COUNT: tl.constexpr,
    HOST_ROWS: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    SOURCE_ROWS: tl.constexpr,
    SOURCE_COLS: tl.constexpr,
    SOURCE_BLOCK: tl.constexpr,
    RESIDENT_STRIDE: tl.constexpr,
    RESIDENT_ROWS: tl.constexpr,
    RESIDENT_COLS: tl.constexpr,
    RESIDENT_BLOCK: tl.constexpr,
    RESIDENT_NULL: tl.constexpr,
    HAS_REQUESTS: tl.constexpr,
    HAS_STATE: tl.constexpr,
    HAS_SOURCE: tl.constexpr,
    HAS_RESIDENT: tl.constexpr,
    HAS_RESOLVED: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    valid_rows = tl.maximum(0, tl.minimum(ROWS, tl.load(num_valid_rows))) if HAS_VALID_ROWS else ROWS
    request = tl.load(requests + row) if HAS_REQUESTS else row
    request = tl.where(row < valid_rows, request, -1)
    first = row
    end = row + 1
    if HAS_REQUESTS:
        searching = (first > 0) & (row < valid_rows)
        while searching:
            same = tl.load(requests + first - 1, mask=first > 0, other=-2147483648) == request
            same = same & (first > 0)
            first -= same.to(tl.int32)
            searching = same
        searching = end < valid_rows
        while searching:
            same = tl.load(requests + end, mask=end < valid_rows, other=-2147483648) == request
            same = same & (end < valid_rows)
            end += same.to(tl.int32)
            searching = same
    state = tl.load(states + request, mask=(request >= 0) & (request < STATE_COUNT), other=-1) if HAS_STATE else request
    valid_state = (state >= 0) & (state < STATE_ROWS)
    m = tl.arange(0, 8)
    meta = tl.where(m == 0, request, tl.where(m == 1, state, tl.where(m == 2, first, end)))
    tl.store(metadata + row * 8 + m, meta)
    i = tl.arange(0, BK)
    token = tl.load(indices + row * INPUT_STRIDE + i, mask=(i < TOP_K) & valid_state, other=-1)
    host, device = _translate(
        token,
        request,
        state,
        source,
        resident,
        HOST_ROWS,
        SOURCE_STRIDE,
        SOURCE_ROWS,
        SOURCE_COLS,
        SOURCE_BLOCK,
        RESIDENT_STRIDE,
        RESIDENT_ROWS,
        RESIDENT_COLS,
        RESIDENT_BLOCK,
        RESIDENT_NULL,
        HAS_SOURCE,
        HAS_RESIDENT,
    )
    tl.store(hot_indices + row * TOP_K + i, device, mask=i < TOP_K)
    if HAS_RESOLVED:
        tl.store(resolved + row * TOP_K + i, host, mask=i < TOP_K)
    if HAS_COUNTS:
        tl.store(counts + row * COUNT_STRIDE, tl.sum(((i < TOP_K) & ((device >= 0) | (host >= 0))).to(tl.int32), 0))
    cold = (i < TOP_K) & (device < 0) & (host >= 0)
    key = tl.where(cold, (host.to(tl.int64) << 32) | i.to(tl.int64), 9223372036854775807)
    tl.store(row_keys + row * TOP_K + i, key, mask=i < TOP_K)
    for start in range(0, HOT_SIZE, 256):
        p = start + tl.arange(0, 256)
        cached = tl.load(
            dgi + state.to(tl.int64) * HOT_SIZE + p, mask=valid_state & (row == first) & (p < HOT_SIZE), other=-1
        )
        cache_key = tl.where(cached >= 0, (cached.to(tl.int64) << 32) | p.to(tl.int64), 9223372036854775807)
        tl.store(cache_keys + row * HOT_SIZE + p, cache_key, mask=p < HOT_SIZE)


@triton.jit
def classify_residency_kernel(
    row_keys,
    sorted_rows,
    sorted_cache,
    metadata,
    hot_table,
    hot_indices,
    first_refs,
    first_misses,
    row_stats,
    TOP_K: tl.constexpr,
    HOT_SIZE: tl.constexpr,
    HOT_BLOCK: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    BK: tl.constexpr,
    protected=None,
    USE_PROTECTED: tl.constexpr = False,
):
    row = tl.program_id(0)
    request = tl.load(metadata + row * 8)
    first = tl.load(metadata + row * 8 + 2)
    i = tl.arange(0, BK)
    key = tl.load(row_keys + row * TOP_K + i, mask=i < TOP_K, other=9223372036854775807)
    host = (key >> 32).to(tl.int32)
    active = (i < TOP_K) & (host != 2147483647)
    if USE_PROTECTED:
        slot, hit, position = _lookup_with_position(sorted_cache + first * HOT_SIZE, host, active, HOT_SIZE)
        # Multiple proposal rows may select the same slot. Atomic max gives
        # the exact set union without depending on their launch order.
        tl.atomic_max(protected + first.to(tl.int64) * HOT_SIZE + slot, 1, mask=hit, sem="relaxed")
        # Scalar loop-carried progress avoids the Ascend compiler failure
        # on vector while iter_args. Sorted equal IDs form a contiguous range.
        offset = 1
        remaining = tl.sum(hit.to(tl.int32), 0)
        while remaining > 0:
            cursor = position + offset
            duplicate_key = tl.load(
                sorted_cache + first * HOT_SIZE + cursor, mask=hit & (cursor < HOT_SIZE), other=9223372036854775807
            )
            duplicate = hit & (cursor < HOT_SIZE) & ((duplicate_key >> 32).to(tl.int32) == host)
            other_slot = (duplicate_key & 4294967295).to(tl.int32)
            tl.atomic_max(protected + first.to(tl.int64) * HOT_SIZE + other_slot, 1, mask=duplicate, sem="relaxed")
            remaining = tl.sum(duplicate.to(tl.int32), 0)
            offset += 1
    else:
        slot, hit = _lookup(sorted_cache + first * HOT_SIZE, host, active, HOT_SIZE)
    reference = tl.full((BK,), 2147483647, tl.int32)
    for query in range(first, row + 1):
        pos, found = _lookup(sorted_rows + query * TOP_K, host, active, TOP_K)
        reference = tl.minimum(reference, tl.where(found, query * TOP_K + pos, 2147483647))
    miss = active & ~hit & (reference == row * TOP_K + i)
    tl.store(first_refs + row * TOP_K + i, tl.where(active, reference, -1), mask=i < TOP_K)
    tl.store(first_misses + row * TOP_K + i, miss, mask=i < TOP_K)
    block = tl.load(hot_table + request.to(tl.int64) * TABLE_STRIDE + slot // HOT_BLOCK, mask=hit, other=0)
    tl.store(hot_indices + row * TOP_K + i, block * HOT_BLOCK + slot % HOT_BLOCK, mask=hit)
    # Independent padded records avoid adjacent programs sharing a transfer line.
    s = tl.arange(0, 4)
    misses = tl.sum(miss.to(tl.int32), 0)
    references = tl.sum(active.to(tl.int32), 0)
    tl.store(row_stats + row * 4 + s, tl.where(s == 0, references - misses, tl.where(s == 1, misses, 0)))


@triton.jit
def protect_residency_kernel(
    sorted_rows,
    metadata,
    dgi,
    lru,
    protected,
    TOP_K: tl.constexpr,
    HOT_SIZE: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    first = tl.program_id(0)
    tile = tl.program_id(1)
    leader = tl.load(metadata + first * 8 + 2)
    p = tile * BLOCK + tl.arange(0, BLOCK)
    hit = tl.full((BLOCK,), False, tl.int1)
    if first == leader:
        state = tl.load(metadata + first * 8 + 1)
        end = tl.load(metadata + first * 8 + 3)
        valid = (state >= 0) & (state < STATE_ROWS) & (p < HOT_SIZE)
        slot = tl.load(lru + state.to(tl.int64) * HOT_SIZE + p, mask=valid, other=-1).to(tl.int32)
        valid = valid & (slot >= 0) & (slot < HOT_SIZE)
        cached = tl.load(dgi + state.to(tl.int64) * HOT_SIZE + slot, mask=valid, other=-1)
        for row in range(first, end):
            _, found = _lookup(sorted_rows + row * TOP_K, cached, valid & (cached >= 0), TOP_K)
            hit = hit | found
    tl.store(protected + first * HOT_SIZE + p, hit, mask=p < HOT_SIZE)


@triton.jit
def partition_residency_kernel(
    metadata,
    protected,
    hit_prefix,
    row_stats,
    lru,
    temp_lru,
    errors,
    HOT_SIZE: tl.constexpr,
    MAX_UNION: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    first = tl.program_id(0)
    tile = tl.program_id(1)
    leader = tl.load(metadata + first * 8 + 2)
    if first == leader:
        state = tl.load(metadata + first * 8 + 1)
        end = tl.load(metadata + first * 8 + 3)
        p = tile * BLOCK + tl.arange(0, BLOCK)
        valid = (state >= 0) & (state < STATE_ROWS) & (p < HOT_SIZE)
        hit = tl.load(protected + first * HOT_SIZE + p, mask=valid, other=0) != 0
        hits = tl.load(hit_prefix + first * HOT_SIZE + HOT_SIZE - 1)
        misses = 0
        for row in range(first, end):
            misses += tl.load(row_stats + row * 4 + 1).to(tl.int32)
        union = hits + misses
        failed = union > MAX_UNION
        if tile == 0:
            err = tl.arange(0, 8)
            tl.store(errors + first * 8 + err, tl.where(failed, union, 0))
            tl.store(metadata + first * 8 + 4, hits)
            tl.store(metadata + first * 8 + 5, misses)
        tl.device_assert(~failed, "HiSparse request union exceeds hot capacity")
        if (union > 0) & ~failed:
            slot = tl.load(lru + state.to(tl.int64) * HOT_SIZE + p, mask=valid, other=-1).to(tl.int32)
            prefix = tl.load(hit_prefix + first * HOT_SIZE + p, mask=valid, other=0)
            hit_rank = prefix - 1
            evictable = valid & ~hit
            evict_rank = p - prefix
            temp = temp_lru + first * HOT_SIZE
            tl.store(temp + hit_rank, slot, mask=hit)
            tl.store(temp + HOT_SIZE - 1 - evict_rank, slot, mask=evictable)


@triton.jit
def update_lru_kernel(metadata, temp_lru, lru, errors, HOT_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    first = tl.program_id(0)
    tile = tl.program_id(1)
    leader = tl.load(metadata + first * 8 + 2)
    if first == leader:
        hits = tl.load(metadata + first * 8 + 4)
        misses = tl.load(metadata + first * 8 + 5)
        failed = tl.load(errors + first * 8) != 0
        if ((hits + misses) > 0) & ~failed:
            state = tl.load(metadata + first * 8 + 1)
            p = tile * BLOCK + tl.arange(0, BLOCK)
            remaining = HOT_SIZE - hits - misses
            source = tl.where(
                p < remaining,
                HOT_SIZE - 1 - misses - p,
                tl.where(p < remaining + misses, HOT_SIZE - 1 - (p - remaining), p - remaining - misses),
            )
            updated = tl.load(temp_lru + first * HOT_SIZE + source, mask=p < HOT_SIZE, other=0)
            tl.store(lru + state.to(tl.int64) * HOT_SIZE + p, updated, mask=p < HOT_SIZE)


@triton.jit
def protected_lru_tile_prefix_kernel(
    protected,
    metadata,
    row_stats,
    lru,
    hit_prefix,
    tile_counts,
    HOT_SIZE: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    COUNT_STRIDE: tl.constexpr = 4,
):
    first = tl.program_id(0)
    tile = tl.program_id(1)
    leader = tl.load(metadata + first * 8 + 2)
    p = tile * BLOCK + tl.arange(0, BLOCK)
    hit = tl.full((BLOCK,), False, tl.int1)
    if first == leader:
        state = tl.load(metadata + first * 8 + 1)
        end = tl.load(metadata + first * 8 + 3)
        references = 0
        for row in range(first, end):
            references |= (tl.load(row_stats + row * 4) + tl.load(row_stats + row * 4 + 1) > 0).to(tl.int32)
        if references > 0:
            valid = (state >= 0) & (state < STATE_ROWS) & (p < HOT_SIZE)
            slot = tl.load(lru + state.to(tl.int64) * HOT_SIZE + p, mask=valid, other=-1).to(tl.int32)
            hit = (
                tl.load(
                    protected + first.to(tl.int64) * HOT_SIZE + slot,
                    mask=valid & (slot >= 0) & (slot < HOT_SIZE),
                    other=0,
                )
                != 0
            )
    prefix = tl.cumsum(hit.to(tl.int32), 0)
    tl.store(hit_prefix + first.to(tl.int64) * HOT_SIZE + p, prefix, mask=p < HOT_SIZE)
    # cache_keys is no longer read after classification. Each tile owns one
    # scalar here; its old packed keys are rebuilt by the next resolver call.
    tl.store(tile_counts + first.to(tl.int64) * HOT_SIZE + tile * COUNT_STRIDE, tl.sum(hit.to(tl.int32), 0))


@triton.jit
def protected_lru_partition_kernel(
    protected,
    metadata,
    row_stats,
    lru,
    hit_prefix,
    tile_counts,
    temp_lru,
    errors,
    HOT_SIZE: tl.constexpr,
    MAX_UNION: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    COUNT_BLOCK: tl.constexpr,
    COUNT_STRIDE: tl.constexpr = 4,
):
    first = tl.program_id(0)
    tile = tl.program_id(1)
    leader = tl.load(metadata + first * 8 + 2)
    if first == leader:
        state = tl.load(metadata + first * 8 + 1)
        end = tl.load(metadata + first * 8 + 3)
        t = tl.arange(0, COUNT_BLOCK)
        counts = tl.load(tile_counts + first.to(tl.int64) * HOT_SIZE + t * COUNT_STRIDE, mask=t < TILES, other=0).to(
            tl.int32
        )
        hits = tl.sum(counts, 0)
        preceding = tl.sum(tl.where(t < tile, counts, 0), 0)
        misses = 0
        for row in range(first, end):
            misses += tl.load(row_stats + row * 4 + 1).to(tl.int32)
        union = hits + misses
        failed = union > MAX_UNION
        if tile == 0:
            tl.store(metadata + first * 8 + 4, hits)
            tl.store(metadata + first * 8 + 5, misses)
            err = tl.arange(0, 8)
            tl.store(errors + first * 8 + err, tl.where(failed, union, 0))
        tl.device_assert(~failed, "HiSparse request union exceeds hot capacity")
        if (union > 0) & ~failed:
            p = tile * BLOCK + tl.arange(0, BLOCK)
            valid = (state >= 0) & (state < STATE_ROWS) & (p < HOT_SIZE)
            slot = tl.load(lru + state.to(tl.int64) * HOT_SIZE + p, mask=valid, other=-1).to(tl.int32)
            hit = (
                tl.load(
                    protected + first.to(tl.int64) * HOT_SIZE + slot,
                    mask=valid & (slot >= 0) & (slot < HOT_SIZE),
                    other=0,
                )
                != 0
            )
            prefix = tl.load(hit_prefix + first.to(tl.int64) * HOT_SIZE + p, mask=valid, other=0) + preceding
            temp = temp_lru + first.to(tl.int64) * HOT_SIZE
            tl.store(temp + prefix - 1, slot, mask=valid & hit)
            tl.store(temp + HOT_SIZE - 1 - (p - prefix), slot, mask=valid & ~hit)


@triton.jit
def protect_and_update_lru_kernel(
    sorted_rows,
    metadata,
    dgi,
    lru,
    row_stats,
    temp_lru,
    errors,
    TOP_K: tl.constexpr,
    HOT_SIZE: tl.constexpr,
    MAX_UNION: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    protected=None,
    USE_PROTECTED: tl.constexpr = False,
):
    first = tl.program_id(0)
    leader = tl.load(metadata + first * 8 + 2)
    if first == leader:
        state = tl.load(metadata + first * 8 + 1)
        end = tl.load(metadata + first * 8 + 3)
        misses = 0
        references = 0
        for row in range(first, end):
            misses += tl.load(row_stats + row * 4 + 1).to(tl.int32)
            references |= (tl.load(row_stats + row * 4) + tl.load(row_stats + row * 4 + 1) > 0).to(tl.int32)
        hits = 0
        p = tl.arange(0, BLOCK)
        temp = temp_lru + first * HOT_SIZE
        live_lru = lru + state.to(tl.int64) * HOT_SIZE
        # Bound live lookup/scan vectors by BLOCK, independently of HOT_SIZE.
        # Each request leader owns both passes; no inter-program barrier or
        # in-place LRU permutation is needed.
        if references > 0:
            for start in range(0, HOT_SIZE, BLOCK):
                position = start + p
                valid = (state >= 0) & (state < STATE_ROWS) & (position < HOT_SIZE)
                slot = tl.load(live_lru + position, mask=valid, other=-1).to(tl.int32)
                if USE_PROTECTED:
                    hit = (
                        tl.load(
                            protected + first.to(tl.int64) * HOT_SIZE + slot,
                            mask=valid & (slot >= 0) & (slot < HOT_SIZE),
                            other=0,
                        )
                        != 0
                    )
                else:
                    cached = tl.load(
                        dgi + state.to(tl.int64) * HOT_SIZE + slot,
                        mask=valid & (slot >= 0) & (slot < HOT_SIZE),
                        other=-1,
                    )
                    hit = tl.full((BLOCK,), False, tl.int1)
                    for row in range(first, end):
                        _, found = _lookup(sorted_rows + row * TOP_K, cached, valid & (cached >= 0), TOP_K)
                        hit = hit | found
                prefix = tl.cumsum(hit.to(tl.int32), 0) + hits
                # Hit prefix and reverse victim suffix are disjoint even
                # before the final request-wide hit count is known.
                tl.store(temp + prefix - 1, slot, mask=valid & hit)
                tl.store(temp + HOT_SIZE - 1 - (position - prefix), slot, mask=valid & ~hit)
                hits += tl.sum(hit.to(tl.int32), 0)
        union = hits + misses
        failed = union > MAX_UNION
        tl.device_assert(~failed, "HiSparse request union exceeds hot capacity")
        # An all-resident request leaves the hot map and LRU untouched.
        if (union > 0) & ~failed:
            remaining = HOT_SIZE - hits - misses
            # Complete protection for the whole request before publishing
            # any new LRU order. Assignment consumes the same victim plan.
            for start in range(0, HOT_SIZE, BLOCK):
                position = start + p
                source = tl.where(
                    position < remaining,
                    HOT_SIZE - 1 - misses - position,
                    tl.where(
                        position < remaining + misses,
                        HOT_SIZE - 1 - (position - remaining),
                        position - remaining - misses,
                    ),
                )
                updated = tl.load(temp + source, mask=position < HOT_SIZE, other=0)
                tl.store(live_lru + position, updated, mask=position < HOT_SIZE)
        tl.store(metadata + first * 8 + 4, hits)
        tl.store(metadata + first * 8 + 5, misses)
        err = tl.arange(0, 8)
        tl.store(errors + first * 8 + err, tl.where(union > MAX_UNION, union, 0))


@triton.jit
def assign_residency_kernel(
    row_keys,
    metadata,
    first_misses,
    hot_table,
    hot_indices,
    dgi,
    temp_lru,
    errors,
    swap_host,
    swap_device,
    swap_counts,
    TOP_K: tl.constexpr,
    HOT_SIZE: tl.constexpr,
    HOT_BLOCK: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    HAS_SWAPS: tl.constexpr,
    BK: tl.constexpr,
):
    first = tl.program_id(0)
    leader = tl.load(metadata + first * 8 + 2)
    if first == leader:
        request = tl.load(metadata + first * 8)
        state = tl.load(metadata + first * 8 + 1)
        end = tl.load(metadata + first * 8 + 3)
        misses = tl.load(metadata + first * 8 + 5)
        failed = tl.load(errors + first * 8) != 0
        if HAS_SWAPS:
            for row in range(first, end):
                count = tl.minimum(TOP_K, tl.maximum(0, misses - (row - first) * TOP_K))
                tl.store(swap_counts + row, tl.where(failed, 0, count))
        if (misses > 0) & ~failed:
            i = tl.arange(0, BK)
            consumed = 0
            temp = temp_lru + first * HOT_SIZE
            for row in range(first, end):
                miss = tl.load(first_misses + row * TOP_K + i, mask=i < TOP_K, other=0) != 0
                rank = tl.cumsum(miss.to(tl.int32), 0) - 1 + consumed
                evict = tl.load(temp + HOT_SIZE - 1 - rank, mask=miss, other=-1)
                good = miss & (evict >= 0) & (evict < HOT_SIZE)
                key = tl.load(row_keys + row * TOP_K + i, mask=miss, other=0)
                host = (key >> 32).to(tl.int32)
                tl.store(dgi + state.to(tl.int64) * HOT_SIZE + evict, host, mask=good)
                block = tl.load(
                    hot_table + request.to(tl.int64) * TABLE_STRIDE + evict // HOT_BLOCK, mask=good, other=0
                )
                physical = tl.where(good, block * HOT_BLOCK + evict % HOT_BLOCK, -1)
                tl.store(hot_indices + row * TOP_K + i, physical, mask=miss)
                if HAS_SWAPS:
                    tl.store(swap_host + first * TOP_K + rank, host, mask=miss)
                    tl.store(swap_device + first * TOP_K + rank, physical, mask=miss)
                consumed += tl.sum(miss.to(tl.int32), 0)


@triton.jit
def finish_residency_kernel(
    metadata,
    first_refs,
    first_misses,
    hot_indices,
    attention,
    miss_mask,
    counts,
    errors,
    TOP_K: tl.constexpr,
    HOT_BLOCK: tl.constexpr,
    ATTENTION_BLOCK: tl.constexpr,
    ATTENTION_STRIDE: tl.constexpr,
    COUNT_STRIDE: tl.constexpr,
    HAS_ATTENTION: tl.constexpr,
    HAS_MISS_MASK: tl.constexpr,
    HAS_COUNTS: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BK)
    first = tl.load(metadata + row * 8 + 2)
    failed = tl.load(errors + first * 8) != 0
    reference = tl.load(first_refs + row * TOP_K + i, mask=i < TOP_K, other=-1)
    own = tl.load(hot_indices + row * TOP_K + i, mask=i < TOP_K, other=-1)
    physical = tl.load(hot_indices + reference, mask=(i < TOP_K) & (reference >= 0), other=-1)
    physical = tl.where(failed, -1, tl.where(reference >= 0, physical, own))
    tl.store(hot_indices + row * TOP_K + i, physical, mask=i < TOP_K)
    if HAS_ATTENTION:
        encoded = tl.where(physical >= 0, physical // HOT_BLOCK * ATTENTION_BLOCK + physical % HOT_BLOCK, -1)
        tl.store(attention + row * ATTENTION_STRIDE + i, encoded, mask=i < TOP_K)
    if HAS_MISS_MASK:
        miss = tl.load(first_misses + row * TOP_K + i, mask=i < TOP_K, other=0)
        tl.store(miss_mask + row * TOP_K + i, tl.where(failed, 0, miss), mask=i < TOP_K)
    if HAS_COUNTS:
        if failed:
            tl.store(counts + row * COUNT_STRIDE, 0)


@triton.jit
def residency_stats_kernel(row_stats, stats, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    hit = tl.load(row_stats + i * 4, mask=i < ROWS, other=0)
    miss = tl.load(row_stats + i * 4 + 1, mask=i < ROWS, other=0)
    tl.store(stats, tl.load(stats) + tl.sum(hit, 0))
    tl.store(stats + 1, tl.load(stats + 1) + tl.sum(miss, 0))


@triton.jit
def compact_physical_indices_kernel(
    indices,
    selected,
    counts,
    INPUT_STRIDE: tl.constexpr,
    TOP_K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(indices + row.to(tl.int64) * INPUT_STRIDE + col, mask=col < TOP_K, other=-1)
    valid = (col < TOP_K) & (value >= 0) & (value.to(tl.int64) < CAPACITY)
    prefix = tl.cumsum(valid.to(tl.int32), 0)
    count = tl.sum(valid.to(tl.int32), 0)
    # Valid entries own [0, count); tail padding owns [count, TOP_K).
    # Disjoint stores preserve both selection order and duplicate entries.
    tl.store(selected + row.to(tl.int64) * TOP_K + prefix - 1, value, mask=valid)
    tl.store(selected + row.to(tl.int64) * TOP_K + col, -1, mask=(col >= count) & (col < TOP_K))
    tl.store(counts + row, count)


@triton.jit
def gather_rows_kernel(
    host_address,
    hot,
    source,
    destination,
    counts,
    mask_ptr,
    attention,
    state,
    HOST_ROWS: tl.constexpr,
    HOT_ROWS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    HOT_BLOCK: tl.constexpr,
    HOT_STRIDE_BYTES: tl.constexpr,
    TOP_K: tl.constexpr,
    ATTENTION_BLOCK: tl.constexpr,
    COMPACT: tl.constexpr,
    HAS_ATTENTION: tl.constexpr,
    HAS_STATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offset = tl.arange(0, BLOCK)
    index = row * TOP_K + col
    dst = tl.load(destination + index)
    padding = tl.load(state + row) < 0 if HAS_STATE else False
    if HAS_ATTENTION:
        value = tl.where((dst >= 0) & ~padding, dst // HOT_BLOCK * ATTENTION_BLOCK + dst % HOT_BLOCK, -1)
        tl.store(attention + index, value)
    if COMPACT:
        copy = col < tl.minimum(TOP_K, tl.maximum(tl.load(counts + row), 0))
    else:
        copy = tl.load(mask_ptr + index) != 0
    if copy & ~padding:
        src = tl.load(source + index)
        valid = (src >= 0) & (dst >= 0) & (dst < HOT_ROWS)
        if valid:
            host = host_address.to(tl.pointer_type(tl.uint8))
            destination_offset = dst.to(tl.int64) // HOT_BLOCK * HOT_STRIDE_BYTES + dst % HOT_BLOCK * ROW_BYTES
            for start in range(0, ROW_BYTES, BLOCK):
                payload = tl.load(
                    host + src.to(tl.int64) * ROW_BYTES + start + offset,
                    mask=(src < HOST_ROWS) & (start + offset < ROW_BYTES),
                    other=0,
                )
                tl.store(hot + destination_offset + start + offset, payload, mask=start + offset < ROW_BYTES)


@triton.jit
def invalidate_kernel(
    dgi,
    states,
    requests,
    written,
    REQUESTS: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    REGION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(requests + token)
    state = tl.load(states + request, mask=(request >= 0) & (request < REQUESTS), other=-1)
    value = tl.load(written + token)
    valid = (state >= 0) & (state < STATE_ROWS) & (value >= 0) & (offset < REGION)
    ptr = dgi + state.to(tl.int64) * REGION + offset
    cached = tl.load(ptr, mask=valid, other=-1)
    tl.store(ptr, -1, mask=valid & (cached == value))


@triton.jit
def pack_attention_rows_kernel(
    cache,
    indices,
    latent,
    rope,
    K: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    LATENT: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    i = tl.arange(0, BLOCK)
    physical = tl.load(indices + row * INDEX_STRIDE + col)
    valid = (physical >= 0) & (physical < CACHE_ROWS)
    base = physical.to(tl.int64) // CACHE_BLOCK * CACHE_STRIDE + physical % CACHE_BLOCK * WIDTH
    value = tl.load(cache + base + i, mask=valid & (i < WIDTH), other=0)
    tl.store(latent + (row * K + col).to(tl.int64) * LATENT + i, value, mask=i < LATENT)
    tl.store(
        rope + (row * K + col).to(tl.int64) * (WIDTH - LATENT) + i - LATENT, value, mask=(i >= LATENT) & (i < WIDTH)
    )


@triton.jit
def scatter_packed_kv_kernel(
    latent,
    rope,
    cache,
    slots,
    valid_rows,
    CACHE_ROWS: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    LATENT: tl.constexpr,
    LATENT_STRIDE: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    HAS_VALID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    slot = tl.load(slots + row)
    valid = (slot >= 0) & (slot < CACHE_ROWS)
    if HAS_VALID:
        valid = valid & (row < tl.load(valid_rows))
    a = tl.load(latent + row.to(tl.int64) * LATENT_STRIDE + i, mask=valid & (i < LATENT), other=0)
    b = tl.load(rope + row.to(tl.int64) * ROPE_STRIDE + i - LATENT, mask=valid & (i >= LATENT) & (i < WIDTH), other=0)
    base = slot.to(tl.int64) // CACHE_BLOCK * CACHE_STRIDE + slot % CACHE_BLOCK * WIDTH
    tl.store(cache + base + i, tl.where(i < LATENT, a, b), mask=valid & (i < WIDTH))
