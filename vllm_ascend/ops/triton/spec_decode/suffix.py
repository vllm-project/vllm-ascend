# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Ascend MRV2 suffix speculative decoding.

Adapted from vLLM #60339, merge 190df0b90007eead441d97d3abf6270f84f2c63e.
Includes local suffix scanning, corpus lookup and voting, draft finalization,
acceptance observation and corpus ingestion.
"""

from vllm.triton_utils import tl, triton

# Odd 64-bit multiplier (golden ratio), as a signed int64.
_HASH_MUL = 0x9E3779B97F4A7C15 - (1 << 64)
# Corpus slots kept per hash bucket: the newest occurrence in each of the
# last _BUCKET_SLOTS documents that contain the bucket's key tokens. Entries
# pack (document sequence number << 32 | corpus position).
_BUCKET_SLOTS = 8
# Prompt tokens stored ahead of each response (longest hash key length - 1).
_PROMPT_CONTEXT = 3
_INGEST_BLOCK = 1024
# Adaptive verification: acceptance is tracked per (draft source, match-length
# bucket, position). Match lengths 1-6 get their own bucket, then 7-10, 11-20
# and 21+. Counts are halved every _STATS_HALF_LIFE steps so the estimate
# follows the workload.
_NUM_LEN_BUCKETS = tl.constexpr(9)
_STATS_HALF_LIFE = 1024


@triton.jit
def _len_bucket(L):
    return tl.where(L <= 6, L - 1, tl.where(L <= 10, 6, tl.where(L <= 20, 7, 8)))


@triton.jit
def _bucket(hv, NUM_BUCKETS: tl.constexpr):
    return ((hv >> 29) ^ hv) & (NUM_BUCKETS - 1)


@triton.jit
def _suffix_local_scan_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int32  [B]  batch_idx -> req_state_idx
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    scratch_ptr,  # *int64  [B, scratch_stride]  (output)
    scratch_stride,
    Lp1,  # int64 packing radix (> any position)
    MAX_DEPTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per (request, block): best (match_len, end_pos) packed into one int64.

    end_pos is the exclusive end of an earlier occurrence of the request's
    suffix in its own history; the draft continues from end_pos. Longer
    matches win, then later positions.
    """
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + req).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)
    row = token_ids_ptr + req * token_ids_stride

    scratch_off = b * scratch_stride + blk
    # Blocks past the last candidate (an occurrence must leave a token).
    if blk * BLOCK >= seq_len:
        tl.store(scratch_ptr + scratch_off, tl.zeros((), tl.int64))
        return

    pos = blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    active = (pos >= 1) & (pos < seq_len) & (num_sampled > 0)
    # Keep indirect-load masks independent of loop-carried match state so
    # Triton-Ascend can lower the loop. Matching still stops at the first miss.
    candidate_mask = active
    match_len = tl.zeros([BLOCK], dtype=tl.int64)
    for t in range(MAX_DEPTH):
        # Request token t back from its end (newest first).
        tail_pos = seq_len - 1 - t
        tail_tok = tl.load(row + tl.maximum(tail_pos, 0), mask=tail_pos >= 0, other=-2)
        src_pos = pos - 1 - t
        load_mask = candidate_mask & (src_pos >= 0) & (tail_pos >= 0)
        tok = tl.load(row + tl.minimum(tl.maximum(src_pos, 0), token_ids_stride - 1), mask=load_mask, other=-1)
        active = active & load_mask & (tok == tail_tok)
        match_len += active.to(tl.int64)

    score = tl.where(match_len > 0, match_len * Lp1 + pos, 0)
    tl.store(scratch_ptr + scratch_off, tl.max(score, axis=0))


@triton.jit
def _corpus_draft(
    row,
    seq_len,
    table_ptr,
    corpus_ptr,
    corpus_len,
    HASH_MUL: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    SLOTS: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    DEPTH_PO2: tl.constexpr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
):
    """(match_len, draft, num_tokens) from the corpus for one request.

    Candidates are the slots of the request's 4-token and 2-token buckets.
    Each is verified and extended against the corpus, so hash collisions and
    entries for overwritten documents only cost a slot. The draft is then
    chosen by frequency, as in SuffixDecoding: for each candidate match
    length L, the candidates matching at least L tokens vote token by token
    (prefix-consistent; ties go to the newest candidate), and the chain with
    the highest sum of per-token probabilities wins (ties: longer L).
    """
    C: tl.constexpr = 2 * SLOTS
    c_iota = tl.arange(0, C).to(tl.int64)
    d_iota = tl.arange(0, DEPTH_PO2).to(tl.int64)
    k_iota = tl.arange(0, K_PO2).to(tl.int64)

    tail_pos = seq_len - 1 - d_iota
    tail = tl.load(row + tl.maximum(tail_pos, 0), mask=(d_iota < MAX_DEPTH) & (tail_pos >= 0), other=-2)
    # Hash the trailing tokens newest-first, as ingestion does.
    hv = tl.zeros((), tl.int64)
    for d in tl.static_range(4):
        tok = tl.load(row + tl.maximum(seq_len - 1 - d, 0), mask=seq_len - 1 - d >= 0, other=0)
        hv = (hv ^ tok.to(tl.int64)) * HASH_MUL
        if d == 1:
            bucket2 = _bucket(hv, NUM_BUCKETS)
    bucket4 = _bucket(hv, NUM_BUCKETS)
    is4 = c_iota < SLOTS
    key_len = tl.where(is4, 4, 2)
    slot = tl.where(is4, bucket4 * SLOTS + c_iota, (NUM_BUCKETS + bucket2) * SLOTS + c_iota - SLOTS)
    vals = tl.load(table_ptr + slot)
    cpos = vals & 0xFFFFFFFF

    src_pos = cpos[:, None] - 1 - d_iota[None, :]
    src = tl.load(
        corpus_ptr + tl.minimum(tl.maximum(src_pos, 0), corpus_len - 1),
        mask=(vals[:, None] >= 0) & (src_pos >= 0) & (src_pos < corpus_len) & (d_iota[None, :] < MAX_DEPTH),
        other=-1,
    )
    miss = (src != tail[None, :]).to(tl.int32)
    c_len = tl.sum((tl.cumsum(miss, axis=1) == 0).to(tl.int64), axis=1)
    nxt = tl.load(corpus_ptr + tl.minimum(cpos, corpus_len - 1), mask=(vals >= 0) & (cpos < corpus_len), other=-1)
    ok = (vals >= 0) & (nxt >= 0) & (c_len >= key_len) & (seq_len >= key_len)
    # Both buckets often hold the same position; count it once.
    dup = (cpos[:, None] == cpos[None, :]) & ok[:, None] & (c_iota[:, None] < c_iota[None, :])
    ok = ok & (tl.max(dup.to(tl.int32), axis=0) == 0)

    # Continuations, cut at the first document separator.
    cont_pos = cpos[:, None] + k_iota[None, :]
    cont = tl.load(
        corpus_ptr + tl.minimum(cont_pos, corpus_len - 1),
        mask=ok[:, None] & (k_iota[None, :] < K) & (cont_pos < corpus_len),
        other=-1,
    ).to(tl.int64)
    cont = tl.where(tl.cumsum((cont < 0).to(tl.int32), axis=1) == 0, cont, -1)

    # Candidates sharing a match length draft identical chains; score each
    # distinct length once.
    same_len = (c_len[:, None] == c_len[None, :]) & ok[:, None] & (c_iota[:, None] < c_iota[None, :])
    first = ok & (tl.max(same_len.to(tl.int32), axis=0) == 0)

    # A token's probability is its vote count over the candidates still
    # alive, which is the previous token's count, so a chain's score
    # telescopes to (sum of its vote counts) / (candidates at the start).
    # Scores are compared as these integer fractions, so ties are exact.
    best_num = tl.full((), -1, tl.int64)
    best_den = tl.full((), 1, tl.int64)
    best_len = tl.zeros((), tl.int64)
    best_n = tl.zeros((), tl.int64)
    best_chain = tl.zeros([K_PO2], tl.int64)
    for i in range(C):
        sel = c_iota == i
        if tl.sum(tl.where(sel, first.to(tl.int32), 0), axis=0) > 0:
            L = tl.sum(tl.where(sel, c_len, 0), axis=0)
            alive = ok & (c_len >= L)
            den = tl.maximum(tl.sum(alive.to(tl.int64), axis=0), 1)
            go = L > 0
            num = tl.zeros((), tl.int64)
            n_emit = tl.zeros((), tl.int64)
            chain = tl.zeros([K_PO2], tl.int64)
            for d in tl.static_range(K):
                col = tl.sum(tl.where(k_iota[None, :] == d, cont, 0), axis=1)
                has = alive & (col >= 0)
                cnt = tl.sum(((col[:, None] == col[None, :]) & has[None, :]).to(tl.int32), axis=1)
                cnt = tl.where(has, cnt, 0)
                cmax = tl.max(cnt, axis=0)
                # Most frequent next token; ties go to the newest candidate.
                newest = tl.max(tl.where(has & (cnt == cmax), vals, -1), axis=0)
                next_tok = tl.sum(tl.where(has & (vals == newest), col, 0), axis=0)
                go = go & (cmax > 0)
                num = tl.where(go, num + cmax.to(tl.int64), num)
                chain = tl.where((k_iota == d) & go, next_tok, chain)
                n_emit += go.to(tl.int64)
                alive = alive & has & (col == next_tok)
            lhs = num * best_den
            rhs = best_num * den
            take = (lhs > rhs) | ((lhs == rhs) & (best_len < L))
            best_num = tl.where(take, num, best_num)
            best_den = tl.where(take, den, best_den)
            best_len = tl.where(take, L, best_len)
            best_n = tl.where(take, n_emit, best_n)
            best_chain = tl.where(take, chain, best_chain)
    return best_len, best_chain, best_n


@triton.jit
def _suffix_finalize_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int32  [B]
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    last_sampled_ptr,  # *int64  [max_num_reqs]
    scratch_ptr,  # *int64  [B, scratch_stride]
    scratch_stride,
    num_blocks,
    Lp1,
    corpus_ptr,  # *int32  [corpus_len]
    corpus_len,
    table_ptr,  # *int64  [2, NUM_BUCKETS, SLOTS]
    drafts_ptr,  # *int64  [B, K]  (output)
    num_valid_ptr,  # *int32  [B]  (output)
    stats_ptr,  # *int64  [2 * _NUM_LEN_BUCKETS, K, 2]  (accepted, observed)
    last_bucket_ptr,  # *int32  [max_num_reqs]  (output)
    last_valid_ptr,  # *int32  [max_num_reqs]  (output)
    conf_ptr,  # *fp32  [B, K]  (output)
    HAS_CORPUS: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    HASH_MUL: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    SLOTS: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    DEPTH_PO2: tl.constexpr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    NB_PO2: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + req).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)
    last_tok = tl.load(last_sampled_ptr + req)
    row = token_ids_ptr + req * token_ids_stride

    # Best match in the request's own history.
    nb = tl.arange(0, NB_PO2).to(tl.int64)
    l_score = tl.max(
        tl.load(scratch_ptr + b * scratch_stride + nb, mask=nb < num_blocks, other=0),
        axis=0,
    )
    l_len = l_score // Lp1
    l_pos = l_score - l_len * Lp1

    k_iota = tl.arange(0, K_PO2).to(tl.int64)
    k_mask = k_iota < K
    idx = l_pos + k_iota
    draft_ok = k_mask & (idx < seq_len) & (l_len > 0)
    draft = tl.load(row + tl.minimum(idx, token_ids_stride - 1), mask=draft_ok, other=0).to(tl.int64)

    if HAS_CORPUS:
        g_len, g_draft, g_n = _corpus_draft(
            row,
            seq_len,
            table_ptr,
            corpus_ptr,
            corpus_len,
            HASH_MUL,
            NUM_BUCKETS,
            SLOTS,
            MAX_DEPTH,
            DEPTH_PO2,
            K,
            K_PO2,
        )
        # The longer match wins; ties go to the request's own history.
        use_corpus = g_len > l_len
        draft = tl.where(use_corpus, g_draft, draft)
        draft_ok = tl.where(use_corpus, k_iota < g_n, draft_ok)
        match_len = tl.where(use_corpus, g_len, l_len)
        source = use_corpus.to(tl.int64)
    else:
        match_len = l_len
        source = tl.zeros((), tl.int64)

    draft_ok = draft_ok & (num_sampled > 0)
    num_valid = tl.sum(draft_ok.to(tl.int32), axis=0)
    # Invalid slots fall back to the last sampled token; they are verified as
    # ordinary (rejectable) drafts, so the fill value only affects efficiency.
    out = tl.where(draft_ok, draft, last_tok)
    tl.store(drafts_ptr + b * K + k_iota, out, mask=k_mask)
    tl.store(num_valid_ptr + b, num_valid)

    if ADAPTIVE:
        # Per-position confidence for adaptive verification: the observed
        # acceptance rate for drafts like this one (Laplace-smoothed); filler
        # slots get 0 so they are trimmed first.
        bucket = source * _NUM_LEN_BUCKETS + _len_bucket(tl.maximum(match_len, 1))
        if K != K_PO2:
            # As in the observer, avoid padded strided statistics accesses
            # for non-power-of-two draft lengths on the current backend.
            for d in tl.static_range(K):
                stat = stats_ptr + (bucket * K + d) * 2
                accepted = tl.load(stat).to(tl.float32)
                observed = tl.load(stat + 1).to(tl.float32)
                is_valid = tl.sum(tl.where(k_iota == d, draft_ok.to(tl.int32), 0), axis=0) > 0
                conf = tl.where(is_valid, (accepted + 1.0) / (observed + 2.0), 0.0)
                tl.store(conf_ptr + b * K + d, conf)
        else:
            stat = stats_ptr + (bucket * K + k_iota) * 2
            accepted = tl.load(stat, mask=k_mask, other=0).to(tl.float32)
            observed = tl.load(stat + 1, mask=k_mask, other=0).to(tl.float32)
            conf = tl.where(draft_ok, (accepted + 1.0) / (observed + 2.0), 0.0)
            tl.store(conf_ptr + b * K + k_iota, conf, mask=k_mask)
        tl.store(last_bucket_ptr + req, tl.where(num_valid > 0, bucket, -1).to(tl.int32))
        tl.store(last_valid_ptr + req, num_valid)


@triton.jit
def _suffix_observe_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    stats_ptr,
    last_bucket_ptr,
    last_valid_ptr,
    num_reqs,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    REQS_PO2: tl.constexpr,
):
    """One program owns each statistics bucket; no contended int64 atomics.

    Reduce batch-local counts in int32, then accumulate in int64. This avoids
    the divergent early return / vector-atomic path in the GPU observer and
    gives every TP rank the same integer result.
    """
    bucket = tl.program_id(0)
    b = tl.arange(0, REQS_PO2)
    req = tl.load(idx_mapping_ptr + b, mask=b < num_reqs, other=0).to(tl.int64)
    previous_bucket = tl.load(last_bucket_ptr + req)
    valid = tl.load(last_valid_ptr + req)
    sampled = tl.load(num_sampled_ptr + b, mask=b < num_reqs, other=0)
    rejected = tl.load(num_rejected_ptr + b, mask=b < num_reqs, other=0)
    accepted = sampled - 1
    verified = accepted + rejected
    if K != K_PO2:
        # The current backend fails on the padded 2D reduction / strided
        # int64 update path for K=3. Use scalar updates for unaligned K,
        # keeping the vectorized path for power-of-two draft lengths.
        for k in tl.static_range(K):
            observed = (
                (b < num_reqs)
                & (previous_bucket == bucket)
                & (sampled > 0)
                & (k <= accepted)
                & (k < verified)
                & (k < valid)
            )
            num_observed = tl.sum(observed.to(tl.int32), axis=0).to(tl.int64)
            num_accepted = tl.sum((observed & (k < accepted)).to(tl.int32), axis=0).to(tl.int64)
            stat = stats_ptr + (bucket * K + k) * 2
            # Load both fields before either store; interleaving them also
            # triggers a BiShengHIR pipeline failure on the current backend.
            old_accepted = tl.load(stat)
            old_observed = tl.load(stat + 1)
            tl.store(stat, old_accepted + num_accepted)
            tl.store(stat + 1, old_observed + num_observed)
    else:
        k = tl.arange(0, K_PO2)
        observed = (
            (b[:, None] < num_reqs)
            & (previous_bucket[:, None] == bucket)
            & (sampled[:, None] > 0)
            & (k[None, :] < K)
            & (k[None, :] <= accepted[:, None])
            & (k[None, :] < verified[:, None])
            & (k[None, :] < valid[:, None])
        )
        num_observed = tl.sum(observed.to(tl.int32), axis=0).to(tl.int64)
        num_accepted = tl.sum((observed & (k[None, :] < accepted[:, None])).to(tl.int32), axis=0).to(tl.int64)
        stat = stats_ptr + (bucket * K + k) * 2
        old_accepted = tl.load(stat, mask=k < K, other=0)
        old_observed = tl.load(stat + 1, mask=k < K, other=0)
        tl.store(stat, old_accepted + num_accepted, mask=k < K)
        tl.store(stat + 1, old_observed + num_observed, mask=k < K)


@triton.jit(do_not_specialize=["num_docs"])
def _suffix_ingest_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    docs_ptr,  # *int32  [F, 2]  (slot, start) of finished requests, sorted
    num_docs,
    total_len_ptr,  # *int32  [max_num_reqs]
    corpus_ptr,  # *int32  [corpus_len]
    corpus_len,
    head_ptr,  # *int64  [1]  next write offset
    doc_seq_ptr,  # *int64  [1]  documents ingested so far
    head_out_ptr,  # *int64  [1]
    doc_seq_out_ptr,  # *int64  [1]
    table_ptr,  # *int64  [2, NUM_BUCKETS, SLOTS]
    HASH_MUL: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Append finished responses to the corpus ring and index them, with one
    program per (document, block).

    Documents are placed one after another. A document that does not fit
    before the end of the ring wraps to offset 0, so no document straddles
    the boundary. Older documents are overwritten in FIFO order (including
    earlier documents of the same batch, if the batch exceeds the ring);
    their index entries become stale and are rejected by verification at
    lookup. Each program replays the batch's placement to find its offset
    and skips positions a later document of the batch overwrites, so every
    position is written by exactly one program and the result does not
    depend on execution order.
    """
    f = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    cap = tl.cast(corpus_len, tl.int64)
    req = tl.load(docs_ptr + 2 * f).to(tl.int64)
    end = tl.load(total_len_ptr + req).to(tl.int64)
    # Keep the newest cap - 1 tokens of an over-long response.
    start = tl.maximum(tl.load(docs_ptr + 2 * f + 1).to(tl.int64), end - (cap - 1))
    n = tl.maximum(end - start, 0)
    # Block 0 also writes the separator.
    if (blk > 0) & (blk * BLOCK >= n):
        return
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
    # Carry integer overlap counts instead of write masks through the loop:
    # Triton-Ascend cannot lower loop-carried masks on indirect stores.
    overwritten = tl.zeros([BLOCK], tl.int32)
    sep_overwritten = tl.zeros((), tl.int32)
    later = head + n + 1
    for g in range(f + 1, num_docs):
        req_g = tl.load(docs_ptr + 2 * g).to(tl.int64)
        end_g = tl.load(total_len_ptr + req_g).to(tl.int64)
        start_g = tl.maximum(tl.load(docs_ptr + 2 * g + 1).to(tl.int64), end_g - (cap - 1))
        n_g = tl.maximum(end_g - start_g, 0)
        later = tl.where(later + n_g + 1 > cap, 0, later)
        # Document g and its separator occupy [later, later + n_g].
        overwritten += ((pos >= later) & (pos <= later + n_g)).to(tl.int32)
        sep_overwritten += ((sep >= later) & (sep <= later + n_g)).to(tl.int32)
        later += n_g + 1
    keep = m & (overwritten == 0)
    keep_sep = sep_overwritten == 0
    if blk == 0:
        tl.store(corpus_ptr + sep, -1, mask=keep_sep)
        if f == num_docs - 1:
            tl.store(head_out_ptr, sep + 1)
            tl.store(doc_seq_out_ptr, doc_seq + 1)

    row = token_ids_ptr + req * token_ids_stride + start
    tokens = tl.load(row + tl.minimum(j, token_ids_stride - start - 1), mask=m, other=0)
    tl.store(corpus_ptr + tl.minimum(pos, cap - 1), tokens, mask=keep)
    # Index each position j under its preceding 2 and 4 tokens, when they lie
    # in the document. Entries order by (document, position), so atomic_max
    # keeps the newest occurrence regardless of execution order.
    slot = doc_seq % SLOTS
    val = (doc_seq << 32) | pos
    hv = tl.zeros([BLOCK], tl.int64)
    for d in tl.static_range(4):
        t = tl.load(
            row + tl.minimum(tl.maximum(j - 1 - d, 0), token_ids_stride - start - 1),
            mask=m & (j - 1 - d >= 0),
            other=0,
        )
        hv = (hv ^ t.to(tl.int64)) * HASH_MUL
        if d == 1:
            tl.atomic_max(
                table_ptr + (NUM_BUCKETS + _bucket(hv, NUM_BUCKETS)) * SLOTS + slot,
                val,
                mask=m & (j >= 2),
            )
        if d == 3:
            tl.atomic_max(
                table_ptr + _bucket(hv, NUM_BUCKETS) * SLOTS + slot,
                val,
                mask=m & (j >= 4),
            )
