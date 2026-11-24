# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-atomic suffix corpus ingestion and exact segmented index reduction."""

from vllm.triton_utils import tl, triton


@triton.jit
def _bucket(hv, NUM_BUCKETS: tl.constexpr):
    return ((hv >> 29) ^ hv) & (NUM_BUCKETS - 1)


@triton.jit
def _segmented_max_combine(left_value, left_head, right_value, right_head):
    # Associative summary of a sequence: its last segment max, and whether
    # any segment starts inside it. Identity is (-1, 0).
    value = tl.where(right_head != 0, right_value, tl.maximum(left_value, right_value))
    head = left_head | right_head
    return value, head


@triton.jit(do_not_specialize=["count"])
def segment_local_scan(
    sorted_keys_ptr,
    permutation_ptr,
    original_values_ptr,
    prefixes_ptr,
    summary_values_ptr,
    summary_heads_ptr,
    count,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    n = tl.cast(count, tl.int64)
    offsets = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = offsets < n
    # Keep even masked address calculations inside their allocation.
    safe_offsets = tl.minimum(offsets, n - 1)
    keys = tl.load(sorted_keys_ptr + safe_offsets, mask=valid, other=SENTINEL)
    previous_offsets = tl.maximum(safe_offsets - 1, 0)
    previous_keys = tl.load(
        sorted_keys_ptr + previous_offsets,
        mask=valid & (offsets > 0),
        other=-1.0,
    )
    heads = tl.where(valid, (keys != previous_keys).to(tl.int32), 0)
    permutation = tl.load(permutation_ptr + safe_offsets, mask=valid, other=0)
    values = tl.load(original_values_ptr + permutation, mask=valid, other=-1)
    prefixes, _ = tl.associative_scan((values, heads), axis=0, combine_fn=_segmented_max_combine)
    tl.store(prefixes_ptr + safe_offsets, prefixes, mask=valid)

    last_offset = tl.minimum((tile + 1) * BLOCK, n) - 1
    # Select the LAST valid prefix, not the maximum of every segment in a
    # tile. A new segment must discard a previous segment's larger value.
    summary_value = tl.max(tl.where(offsets == last_offset, prefixes, -1), axis=0)
    summary_head = tl.max(heads, axis=0)
    tl.store(summary_values_ptr + tile, summary_value)
    tl.store(summary_heads_ptr + tile, summary_head)


@triton.jit
def segment_scan_carries(
    summary_values_ptr,
    summary_heads_ptr,
    exclusive_carries_ptr,
    NUM_TILES: tl.constexpr,
    CARRY_BLOCK: tl.constexpr,
):
    indices = tl.arange(0, CARRY_BLOCK).to(tl.int64)
    valid = indices < NUM_TILES
    previous = tl.minimum(tl.maximum(indices - 1, 0), NUM_TILES - 1)
    has_previous = valid & (indices > 0)
    # Scan shifted summaries: lane t receives the global last-segment max
    # through tile t-1. This handles segments spanning ANY number of tiles.
    values = tl.load(summary_values_ptr + previous, mask=has_previous, other=-1)
    heads = tl.load(summary_heads_ptr + previous, mask=has_previous, other=0)
    carries, _ = tl.associative_scan((values, heads), axis=0, combine_fn=_segmented_max_combine)
    safe_indices = tl.minimum(indices, NUM_TILES - 1)
    tl.store(exclusive_carries_ptr + safe_indices, carries, mask=valid)


@triton.jit(do_not_specialize=["count"])
def segment_tail_merge(
    sorted_keys_ptr,
    prefixes_ptr,
    exclusive_carries_ptr,
    table_ptr,
    count,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    n = tl.cast(count, tl.int64)
    first_offset = tile * BLOCK
    offsets = first_offset + tl.arange(0, BLOCK).to(tl.int64)
    valid = offsets < n
    safe_offsets = tl.minimum(offsets, n - 1)
    keys = tl.load(sorted_keys_ptr + safe_offsets, mask=valid, other=SENTINEL)
    first_key = tl.load(sorted_keys_ptr + first_offset)
    previous_key = tl.load(sorted_keys_ptr + tl.maximum(first_offset - 1, 0))
    carry = tl.load(exclusive_carries_ptr + tile)
    prefixes = tl.load(prefixes_ptr + safe_offsets, mask=valid, other=-1)
    # Sorted order ensures only the leading equal-key run can need a carry.
    continues = (tile > 0) & (first_key == previous_key) & (keys == first_key)
    values = tl.where(continues, tl.maximum(prefixes, carry), prefixes)

    next_offsets = tl.minimum(offsets + 1, n - 1)
    next_keys = tl.load(
        sorted_keys_ptr + next_offsets,
        mask=valid & (offsets + 1 < n),
        other=-1.0,
    )
    write = valid & (keys < SENTINEL) & (keys != next_keys)
    # sentinel is one past the table; clamp it BEFORE forming a pointer,
    # even though its lane will not write.
    addresses = tl.where(keys < SENTINEL, keys, 0.0).to(tl.int64)
    previous_values = tl.load(table_ptr + addresses, mask=write, other=-1)
    tl.store(table_ptr + addresses, tl.maximum(previous_values, values), mask=write)


@triton.jit(do_not_specialize=["num_docs"])
def _suffix_ingest_emit_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    docs_ptr,  # *int32  [F, 2]  (slot, start) of finished requests, sorted
    num_docs,
    total_len_ptr,  # *int32  [max_num_reqs]
    corpus_ptr,  # *int32  [corpus_len]
    corpus_len,
    head_ptr,  # *i64 [1]  next write offset (input snapshot)
    doc_seq_ptr,  # *i64 [1]  documents ingested so far (input snapshot)
    head_out_ptr,  # *i64 [1]  staged new head
    doc_seq_out_ptr,  # *i64 [1]  staged new doc_seq
    emit_keys_ptr,  # *fp32 [C]  record table addresses (SENTINEL padding)
    emit_values_ptr,  # *i64  [C]  record payloads (-1 padding)
    HASH_MUL: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
    P: tl.constexpr,  # per-doc record plane = cdiv(max_model_len, BLOCK)*BLOCK
    SENTINEL: tl.constexpr,  # == 2 * NUM_BUCKETS * SLOTS, float32-exact
):
    """Append finished responses to the corpus and EMIT index records.

    Same corpus placement as the atomic original (placement replay, single
    write per position, staged head/doc_seq out), but table updates are
    emitted as dense SoA records for the sort + segmented-max path instead
    of device atomics. Each (f, blk) program writes exactly one real-or-
    padding record per family slot in its tile (single store per slot, no
    fill-then-overwrite); record slots never depend on the previous batch
    or on any allocator or counter.
    """
    f = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    cap = tl.cast(corpus_len, tl.int64)
    req = tl.load(docs_ptr + 2 * f).to(tl.int64)
    end = tl.load(total_len_ptr + req).to(tl.int64)
    start = tl.maximum(tl.load(docs_ptr + 2 * f + 1).to(tl.int64), end - (cap - 1))
    n = tl.maximum(end - start, 0)
    head = tl.load(head_ptr)
    for i in range(f):
        req_i = tl.load(docs_ptr + 2 * i).to(tl.int64)
        end_i = tl.load(total_len_ptr + req_i).to(tl.int64)
        start_i = tl.maximum(tl.load(docs_ptr + 2 * i + 1).to(tl.int64), end_i - (cap - 1))
        n_i = tl.maximum(end_i - start_i, 0)
        head = tl.where(head + n_i + 1 > cap, 0, head) + n_i + 1
    head = tl.where(head + n + 1 > cap, 0, head)
    doc_seq = tl.load(doc_seq_ptr) + f

    j = blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    m = j < n
    pos = head + j
    sep = head + n
    later = head + n + 1
    excl_hi = head * 0  # i64 zero: post-wrap exclusion prefix upper bound
    wrapped = 0
    for g in range(f + 1, num_docs):
        req_g = tl.load(docs_ptr + 2 * g).to(tl.int64)
        end_g = tl.load(total_len_ptr + req_g).to(tl.int64)
        start_g = tl.maximum(tl.load(docs_ptr + 2 * g + 1).to(tl.int64), end_g - (cap - 1))
        n_g = tl.maximum(end_g - start_g, 0)
        later = tl.where(later + n_g + 1 > cap, 0, later)
        wrapped = tl.where(later == 0, 1, wrapped)
        excl_hi = tl.where(wrapped == 1, tl.maximum(excl_hi, later + n_g + 1), excl_hi)
        later += n_g + 1
    keep = m & (pos >= excl_hi)
    keep_sep = sep >= excl_hi
    if blk == 0:
        tl.store(corpus_ptr + sep, -1, mask=keep_sep)
        if f == num_docs - 1:
            tl.store(head_out_ptr, sep + 1)
            tl.store(doc_seq_out_ptr, doc_seq + 1)

    # ---- record emit: dense two-family planes ----
    # r4(f, j) = f * P + j                  (4-token family)
    # r2(f, j) = num_docs * P + f * P + j   (2-token family; second plane
    #                                    starts at this batch's F, not max F)
    # Masks do not short-circuit address formation on Ascend. Keep the row
    # base in its allocation even for empty documents and inactive tail tiles.
    row = token_ids_ptr + req * token_ids_stride
    token_offset = tl.minimum(tl.maximum(start + j, 0), token_ids_stride - 1)
    corpus_offset = tl.minimum(tl.maximum(pos, 0), cap - 1)
    tokens = tl.load(row + token_offset, mask=m, other=0)
    tl.store(corpus_ptr + corpus_offset, tokens, mask=keep)
    slot = doc_seq % SLOTS
    val = (doc_seq << 32) | pos
    r4 = f * P + j
    r2 = num_docs * P + r4
    valid4 = m & (j >= 4)
    valid2 = m & (j >= 2)
    pad_k = tl.full([BLOCK], SENTINEL, tl.float32)
    pad_v = tl.full([BLOCK], -1, tl.int64)
    hv = tl.zeros([BLOCK], tl.int64)
    for d in tl.static_range(4):
        # Preserve the raw document-relative mask, but clamp both ends of
        # the address before forming a pointer into the request-state row.
        _hp = j - 1 - d
        history_offset = tl.minimum(tl.maximum(start + _hp, 0), token_ids_stride - 1)
        t = tl.load(row + history_offset, mask=m & (_hp >= 0), other=0)
        hv = (hv ^ t.to(tl.int64)) * HASH_MUL
        if d == 1:
            # Emit the 2-token family record: hv holds tokens [j-2, j).
            a2 = (NUM_BUCKETS + _bucket(hv, NUM_BUCKETS)) * SLOTS + slot
            tl.store(emit_keys_ptr + r2, tl.where(valid2, a2.to(tl.float32), pad_k))
            tl.store(emit_values_ptr + r2, tl.where(valid2, val, pad_v))
        if d == 3:
            # Emit the 4-token family record: hv holds tokens [j-4, j).
            a4 = _bucket(hv, NUM_BUCKETS) * SLOTS + slot
            tl.store(emit_keys_ptr + r4, tl.where(valid4, a4.to(tl.float32), pad_k))
            tl.store(emit_values_ptr + r4, tl.where(valid4, val, pad_v))
