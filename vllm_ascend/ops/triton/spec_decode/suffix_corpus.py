# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Corpus lookup, voting and ingestion for MRV2 suffix decoding.

Adapted from vLLM #60339, merge 190df0b90007eead441d97d3abf6270f84f2c63e.
Host launch wrappers retain the upstream positional ABI and derive NPU launch
metadata from tensors and scalar arguments. Request lifecycle and execution
ordering remain with the upstream speculator.
"""

import torch
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


@triton.jit
def _bucket(hv, NUM_BUCKETS: tl.constexpr):
    return ((hv >> 29) ^ hv) & (NUM_BUCKETS - 1)


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
    # Count each position once, keeping its newest valid index entry after wraparound.
    # At the same position, packed values differ only in their document sequence.
    newer = vals[:, None] > vals[None, :]
    same_entry_earlier = (vals[:, None] == vals[None, :]) & (c_iota[:, None] < c_iota[None, :])
    dup = (cpos[:, None] == cpos[None, :]) & ok[:, None] & (newer | same_entry_earlier)
    ok = ok & (tl.max(dup.to(tl.int32), axis=0) == 0)

    # Continuations, cut at the first document separator.
    cont_pos = cpos[:, None] + k_iota[None, :]
    cont = tl.load(
        corpus_ptr + tl.minimum(cont_pos, corpus_len - 1),
        mask=ok[:, None] & (k_iota[None, :] < K) & (cont_pos < corpus_len),
        other=-1,
    ).to(tl.int64)
    # A5 can miscompile the compare/cumsum/select path for narrow rows
    # (K_PO2=2 or 4). Find the first separator instead of scanning its mask.
    stop = tl.min(tl.where(cont < 0, k_iota[None, :], K), axis=1)
    cont = tl.where(k_iota[None, :] < stop[:, None], cont, -1)

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


def launch_suffix_ingest(
    token_ids: torch.Tensor,
    _token_ids_stride: int,
    docs: torch.Tensor,
    num_docs: int,
    total_len: torch.Tensor,
    corpus: torch.Tensor,
    corpus_len: int,
    head: torch.Tensor,
    doc_seq: torch.Tensor,
    head_out: torch.Tensor,
    doc_seq_out: torch.Tensor,
    table: torch.Tensor,
    **launch_options,
) -> None:
    """Append completed responses and build indexes without owning their lifecycle."""
    grid = (num_docs, triton.cdiv(token_ids.shape[1], launch_options["BLOCK"]))
    _suffix_ingest_kernel[grid](
        token_ids,
        token_ids.stride(0),
        docs,
        num_docs,
        total_len,
        corpus,
        corpus_len,
        head,
        doc_seq,
        head_out,
        doc_seq_out,
        table,
        **launch_options,
    )
