# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU suffix decoding speculator for the V2 model runner.

Extends the n-gram speculator with the two parts of SuffixDecoding
(https://arxiv.org/abs/2411.04975) that carry most of its acceptance gain:

* Variable-length matching: drafts continue the occurrence whose preceding
  tokens match the longest suffix of the request's history (up to
  ``suffix_decoding_max_tree_depth``), instead of a fixed n-gram window.
* Cross-request memory: responses of finished requests are appended to a
  device-resident token corpus. A hash index keyed on the last 2 and last 4
  tokens gives candidate positions, and each candidate is extended to the
  longest suffix match, so the index is only an entry point for suffix
  matching.

Ingestion reads token history and lengths directly from ``RequestState`` and
processes finished requests in sorted id order. Exact index reduction and
staged corpus state updates run in sequence on the current stream, so each
TP rank builds the same corpus without a collective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

from vllm_ascend.ops.triton.spec_decode.suffix import (
    _bucket,
    _suffix_ingest_emit_kernel,
    segment_local_scan,
    segment_scan_carries,
    segment_tail_merge,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.sample.states import SamplingStates
    from vllm.v1.worker.gpu.states import RequestState


# Odd 64-bit multiplier (golden ratio), as a signed int64.
_HASH_MUL = 0x9E3779B97F4A7C15 - (1 << 64)
# Corpus slots kept per hash bucket: the newest occurrence in each of the
# last _BUCKET_SLOTS documents that contain the bucket's key tokens. Entries
# pack (document sequence number << 32 | corpus position).
_BUCKET_SLOTS = 8
# Prompt tokens stored ahead of each response (longest hash key length - 1).
_PROMPT_CONTEXT = 3
_INGEST_BLOCK = 1024
# Address keys (including the one-past-table sentinel) must be float32-exact.
_MAX_EXACT_TABLE_CELLS = 1 << 24
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
    b = tl.program_id(0)
    blk = tl.program_id(1)
    req = tl.load(idx_mapping_ptr + b)
    seq_len = tl.load(total_len_ptr + req)
    num_sampled = tl.load(num_sampled_ptr + b)

    scratch_off = b * scratch_stride + blk
    base_pos = blk * BLOCK
    if base_pos >= seq_len or num_sampled <= 0 or seq_len <= 0:
        tl.store(scratch_ptr + scratch_off, 0)
        return

    row = token_ids_ptr + req.to(tl.int64) * token_ids_stride
    offs = tl.arange(0, BLOCK)
    pos = base_pos + offs

    # Fast path for strictly interior blocks where no candidate or lookback
    # reaches outside [0, seq_len).
    if (blk > 0) & (base_pos >= MAX_DEPTH) & (base_pos + BLOCK <= seq_len):
        tail0 = tl.load(row + seq_len - 1)
        tok0 = tl.load(row + (base_pos - 1) + offs)
        act0 = tok0 == tail0
        if tl.max(act0.to(tl.int32), axis=0) == 0:
            tl.store(scratch_ptr + scratch_off, 0)
            return
        best1 = tl.max(tl.where(act0, Lp1 + pos.to(tl.int64), 0), axis=0)
        tail1 = tl.load(row + seq_len - 2)
        tok1 = tl.load(row + (base_pos - 2) + offs)
        active = act0 & (tok1 == tail1)
        if tl.max(active.to(tl.int32), axis=0) == 0:
            tl.store(scratch_ptr + scratch_off, best1)
            return
        match_len = tl.where(active, 2, 0)
        t = 2
        while (t < MAX_DEPTH) & (tl.max(active.to(tl.int32), axis=0) > 0):
            tail_tok = tl.load(row + seq_len - 1 - t)
            tok = tl.load(row + (base_pos - 1 - t) + offs)
            active = active & (tok == tail_tok)
            match_len += active.to(tl.int32)
            t += 1
        best2 = tl.max(
            tl.where(
                match_len >= 2,
                match_len.to(tl.int64) * Lp1 + pos.to(tl.int64),
                0,
            ),
            axis=0,
        )
        tl.store(scratch_ptr + scratch_off, tl.maximum(best1, best2))
        return

    # Boundary blocks clamp all load addresses into [0, token_ids_stride) because
    # NPU masked loads still resolve out-of-range vector addresses.
    in_seq = pos < seq_len
    tail0 = tl.load(row + seq_len - 1)
    src0 = pos - 1
    _sp0 = tl.maximum(tl.minimum(src0, token_ids_stride - 1), 0)
    tok0 = tl.load(row + _sp0, mask=in_seq & (src0 >= 0), other=-1)
    act0 = in_seq & (src0 >= 0) & (tok0 == tail0)
    if tl.max(act0.to(tl.int32), axis=0) == 0:
        tl.store(scratch_ptr + scratch_off, 0)
        return

    best1 = tl.max(tl.where(act0, Lp1 + pos.to(tl.int64), 0), axis=0)
    if seq_len < 2:
        tl.store(scratch_ptr + scratch_off, best1)
        return

    tail1 = tl.load(row + seq_len - 2)
    src1 = pos - 2
    _sp1 = tl.maximum(tl.minimum(src1, token_ids_stride - 1), 0)
    tok1 = tl.load(row + _sp1, mask=in_seq & (src1 >= 0), other=-1)
    active = act0 & (src1 >= 0) & (tok1 == tail1)
    if tl.max(active.to(tl.int32), axis=0) == 0:
        tl.store(scratch_ptr + scratch_off, best1)
        return

    match_len = tl.where(active, 2, 0)
    max_t = tl.minimum(MAX_DEPTH, seq_len)
    t = 2
    while (t < max_t) & (tl.max(active.to(tl.int32), axis=0) > 0):
        tail_tok = tl.load(row + seq_len - 1 - t)
        src_pos = pos - 1 - t
        _sp = tl.maximum(tl.minimum(src_pos, token_ids_stride - 1), 0)
        tok = tl.load(row + _sp, mask=in_seq & (src_pos >= 0), other=-1)
        active = active & (src_pos >= 0) & (tok == tail_tok)
        match_len += active.to(tl.int32)
        t += 1

    best2 = tl.max(
        tl.where(
            match_len >= 2,
            match_len.to(tl.int64) * Lp1 + pos.to(tl.int64),
            0,
        ),
        axis=0,
    )
    tl.store(scratch_ptr + scratch_off, tl.maximum(best1, best2))


@triton.jit
def _finalize_and_corpus_a_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, Lp1]
    token_ids_stride,  # int64
    idx_mapping_ptr,  # *int32  [B]
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    last_sampled_ptr,  # *int64  [max_num_reqs]
    scratch_ptr,  # *int64  [B, NB_MAX]
    scratch_stride,  # int64
    num_blocks,  # int32
    Lp1,  # int64 (max_model_len + 1)
    corpus_ptr,  # *int32  [corpus_len]
    corpus_len,  # int32
    table_ptr,  # *int64  [2, NUM_BUCKETS, SLOTS]
    drafts_ptr,  # *int64  [B, K]  (output)
    req_drafts_ptr,  # *int64  [max_num_reqs, K]  (output)
    num_valid_ptr,  # *int32  [B]  (output)
    stats_ptr,  # *int64  [2 * _NUM_LEN_BUCKETS, K, 2]  (accepted, observed)
    last_bucket_ptr,  # *int32  [max_num_reqs]  (output)
    last_valid_ptr,  # *int32  [max_num_reqs]  (output)
    conf_ptr,  # *fp32  [B, K]  (output)
    seeded_ptr,  # *bool  [B]  seeded request sampled at temperature > 0
    out_clen_ptr,  # *int32  [B, 2 * SLOTS]
    out_vals_ptr,  # *int64  [B, 2 * SLOTS]
    out_cont_t_ptr,  # *int32  [B, K_PO2, 2 * SLOTS]
    out_llen_ptr,  # *int32  [B] (-1 means skip corpus stage B)
    req_drafts_stride: tl.constexpr,
    HAS_CORPUS: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    WRITE_REQ_DRAFTS: tl.constexpr,
    HASH_MUL: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    SLOTS: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    DEPTH_PO2: tl.constexpr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    NB_PO2: tl.constexpr,
):
    """Finalize local suffix draft and prepare transposed corpus candidates."""
    b = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + b)
    seq_len = tl.load(total_len_ptr + req)
    num_sampled = tl.load(num_sampled_ptr + b)
    last_tok = tl.load(last_sampled_ptr + req)
    row = token_ids_ptr + req.to(tl.int64) * token_ids_stride

    # Best match in the request's own history.
    nb = tl.arange(0, NB_PO2)
    l_score = tl.max(
        tl.load(scratch_ptr + b * scratch_stride + nb, mask=nb < num_blocks, other=0),
        axis=0,
    )
    l_len = (l_score // Lp1).to(tl.int32)
    l_pos = (l_score - l_len.to(tl.int64) * Lp1).to(tl.int32)

    k_iota = tl.arange(0, K_PO2)
    k_mask = k_iota < K
    idx = l_pos + k_iota
    # [NPU] clamp load addresses into the row.
    _ix = tl.where(idx < 0, 0, idx)
    _ix = tl.where(_ix >= token_ids_stride, token_ids_stride - 1, _ix)
    draft_ok = k_mask & (idx < seq_len) & (l_len > 0) & (num_sampled > 0)
    draft = tl.load(row + _ix, mask=draft_ok, other=0)

    num_valid = tl.sum(draft_ok.to(tl.int32), axis=0)
    # Invalid slots fall back to the last sampled token; they are verified as
    # ordinary (rejectable) drafts, so the fill value only affects efficiency.
    out = tl.where(draft_ok, draft.to(tl.int64), last_tok)
    tl.store(drafts_ptr + b * K + k_iota, out, mask=k_mask)
    if WRITE_REQ_DRAFTS:
        tl.store(req_drafts_ptr + req.to(tl.int64) * req_drafts_stride + k_iota, out, mask=k_mask)
    tl.store(num_valid_ptr + b, num_valid)

    if ADAPTIVE:
        # Per-position confidence for adaptive verification: the observed
        # acceptance rate for drafts like this one (Laplace-smoothed); filler
        # slots get 0 so they are trimmed first.
        bucket = _len_bucket(tl.maximum(l_len.to(tl.int64), 1))
        stat = stats_ptr + (bucket * K + k_iota) * 2
        accepted = tl.load(stat, mask=k_mask, other=0).to(tl.float32)
        observed = tl.load(stat + 1, mask=k_mask, other=0).to(tl.float32)
        conf = tl.where(draft_ok, (accepted + 1.0) / (observed + 2.0), 0.0)
        tl.store(conf_ptr + b * K + k_iota, conf, mask=k_mask)
        tl.store(last_bucket_ptr + req, tl.where(num_valid > 0, bucket, -1).to(tl.int32))
        tl.store(last_valid_ptr + req, num_valid)

    if not HAS_CORPUS:
        return

    # Corpus drafts change as other requests finish, and the rejection
    # sampler's draws depend on the drafts, so seeded sampled requests
    # draft only from their own history to stay reproducible.
    seeded = tl.load(seeded_ptr + b).to(tl.int32)
    if (num_sampled <= 0) | (seeded != 0) | (l_len >= MAX_DEPTH):
        tl.store(out_llen_ptr + b, -1)
        return

    C: tl.constexpr = 2 * SLOTS
    c_iota = tl.arange(0, C)
    d_iota = tl.arange(0, DEPTH_PO2)

    # Hash the trailing tokens newest-first, as ingestion does.
    hv = tl.zeros((), tl.int64)
    bucket2 = tl.zeros((), tl.int64)
    for d in tl.static_range(4):
        _tp_d = tl.where(seq_len - 1 - d < 0, 0, seq_len - 1 - d)
        _tp_d = tl.where(_tp_d >= token_ids_stride, token_ids_stride - 1, _tp_d)
        tok = tl.load(row + _tp_d, mask=seq_len - 1 - d >= 0, other=0)
        hv = (hv ^ tok.to(tl.int64)) * HASH_MUL
        if d == 1:
            bucket2 = _bucket(hv, NUM_BUCKETS)
    bucket4 = _bucket(hv, NUM_BUCKETS)
    is4 = c_iota < SLOTS
    key_len = tl.where(is4, 4, 2)
    slot = tl.where(
        is4,
        bucket4.to(tl.int32) * SLOTS + c_iota,
        (NUM_BUCKETS + bucket2.to(tl.int32)) * SLOTS + (c_iota - SLOTS),
    )
    vals = tl.load(table_ptr + slot)
    if tl.max((vals >= 0).to(tl.int32), axis=0) == 0:
        tl.store(out_llen_ptr + b, -1)
        return

    tail_pos = seq_len - 1 - d_iota
    _tp = tl.where(tail_pos < 0, 0, tail_pos)
    _tp = tl.where(_tp >= token_ids_stride, token_ids_stride - 1, _tp)
    tail = tl.load(row + _tp, mask=(d_iota < MAX_DEPTH) & (tail_pos >= 0), other=-2)

    cpos = (vals & 0xFFFFFFFF).to(tl.int32)
    # [NPU] masked load does not short-circuit address computation: clamp all
    # addresses into [0, corpus_len).
    _cpos = tl.where((cpos < 0) | (cpos >= corpus_len), 0, cpos)
    raw_src_pos = _cpos[:, None] - 1 - d_iota[None, :]
    src_pos = tl.where(raw_src_pos < 0, 0, raw_src_pos)
    src_pos = tl.where(src_pos >= corpus_len, corpus_len - 1, src_pos)
    src = tl.load(
        corpus_ptr + src_pos,
        mask=(vals[:, None] >= 0) & (cpos[:, None] >= 1) & (raw_src_pos >= 0) & (d_iota[None, :] < MAX_DEPTH),
        other=-1,
    )
    miss = (src != tail[None, :]).to(tl.int32)
    c_len = tl.sum((tl.cumsum(miss, axis=1) == 0).to(tl.int32), axis=1)
    nxt = tl.load(corpus_ptr + _cpos, mask=(vals >= 0) & (cpos >= 0) & (cpos < corpus_len), other=-1)
    ok = (vals >= 0) & (nxt >= 0) & (c_len >= key_len) & (seq_len >= key_len)
    if tl.max(ok.to(tl.int32), axis=0) == 0:
        tl.store(out_llen_ptr + b, -1)
        return

    # Both buckets often hold the same position; count it once.
    dup = (cpos[:, None] == cpos[None, :]) & ok[:, None] & (c_iota[:, None] < c_iota[None, :])
    ok = ok & (tl.max(dup.to(tl.int32), axis=0) == 0)
    c_len_ok = tl.where(ok, c_len, 0)
    if tl.max(c_len_ok, axis=0) <= l_len:
        tl.store(out_llen_ptr + b, -1)
        return

    # Continuations, cut at the first document separator.
    cont_pos = _cpos[:, None] + k_iota[None, :]
    cont_pos = tl.where(cont_pos >= corpus_len, corpus_len - 1, cont_pos)
    cont = tl.load(
        corpus_ptr + cont_pos,
        mask=ok[:, None] & (k_iota[None, :] < K),
        other=-1,
    )
    cont = tl.where(tl.cumsum((cont < 0).to(tl.int32), axis=1) == 0, cont, -1)

    tl.store(out_llen_ptr + b, l_len)
    tl.store(out_clen_ptr + b * C + c_iota, c_len_ok)
    tl.store(out_vals_ptr + b * C + c_iota, vals)
    d_k = tl.arange(0, K_PO2)
    tl.store(out_cont_t_ptr + b * (K_PO2 * C) + d_k[:, None] * C + c_iota[None, :], tl.trans(cont))


@triton.jit
def _corpus_b_1d_kernel(
    idx_mapping_ptr,  # *int32  [B]
    last_sampled_ptr,  # *int64  [max_num_reqs]
    in_clen_ptr,  # *int32  [B, 2 * SLOTS]
    in_vals_ptr,  # *int64  [B, 2 * SLOTS]
    in_cont_t_ptr,  # *int32  [B, K_PO2, 2 * SLOTS]
    in_llen_ptr,  # *int32  [B]
    drafts_ptr,  # *int64  [B, K]  (output)
    req_drafts_ptr,  # *int64  [max_num_reqs, K]  (output)
    num_valid_ptr,  # *int32  [B]  (output)
    stats_ptr,  # *int64  [2 * _NUM_LEN_BUCKETS, K, 2]
    last_bucket_ptr,  # *int32  [max_num_reqs]  (output)
    last_valid_ptr,  # *int32  [max_num_reqs]  (output)
    conf_ptr,  # *fp32  [B, K]  (output)
    req_drafts_stride: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    WRITE_REQ_DRAFTS: tl.constexpr,
    SLOTS: tl.constexpr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
):
    """Score corpus continuation candidates with 1D row loads to avoid UB spill."""
    b = tl.program_id(0)
    l_len = tl.load(in_llen_ptr + b)
    if l_len < 0:
        return

    C: tl.constexpr = 2 * SLOTS
    c_iota = tl.arange(0, C)
    c_len = tl.load(in_clen_ptr + b * C + c_iota)
    vals = tl.load(in_vals_ptr + b * C + c_iota)
    ok = c_len > 0

    same_len = (c_len[:, None] == c_len[None, :]) & ok[:, None] & (c_iota[:, None] < c_iota[None, :])
    first = ok & (tl.max(same_len.to(tl.int32), axis=0) == 0)

    best_num = tl.full((), -1, tl.int32)
    best_den = tl.full((), 1, tl.int32)
    best_len = tl.zeros((), tl.int32)
    best_n = tl.zeros((), tl.int32)
    best_chain = tl.zeros([K_PO2], tl.int32)

    k_iota = tl.arange(0, K_PO2)
    rem = first.to(tl.int32)
    cont_t_base = in_cont_t_ptr + b * (K_PO2 * C)

    while tl.max(rem, axis=0) > 0:
        i = tl.argmax(rem, axis=0)
        sel = c_iota == i
        rem = tl.where(sel, 0, rem)
        L = tl.sum(tl.where(sel, c_len, 0), axis=0)
        alive = ok & (c_len >= L)
        den = tl.maximum(tl.sum(alive.to(tl.int32), axis=0), 1)
        go = L > 0
        num = tl.zeros((), tl.int32)
        n_emit = tl.zeros((), tl.int32)
        chain = tl.zeros([K_PO2], tl.int32)
        d = tl.zeros((), tl.int32)
        while (d < K) & go:
            col = tl.load(cont_t_base + d * C + c_iota)
            has = alive & (col >= 0)
            cnt = tl.sum(((col[:, None] == col[None, :]) & has[None, :]).to(tl.int32), axis=1)
            cnt = tl.where(has, cnt, 0)
            cmax = tl.max(cnt, axis=0)
            newest = tl.max(tl.where(has & (cnt == cmax), vals, -1), axis=0)
            next_tok = tl.sum(tl.where(has & (vals == newest), col, 0), axis=0)
            go = go & (cmax > 0)
            num = tl.where(go, num + cmax, num)
            chain = tl.where((k_iota == d) & go, next_tok, chain)
            n_emit += go.to(tl.int32)
            alive = alive & has & (col == next_tok)
            d += 1
        lhs = num * best_den
        rhs = best_num * den
        take = (lhs > rhs) | ((lhs == rhs) & (best_len < L))
        best_len = tl.where(take, L, best_len)
        best_n = tl.where(take, n_emit, best_n)
        best_num = tl.where(take, num, best_num)
        best_den = tl.where(take, den, best_den)
        best_chain = tl.where(take, chain, best_chain)

    if best_len <= l_len:
        return

    req = tl.load(idx_mapping_ptr + b)
    last_tok = tl.load(last_sampled_ptr + req)
    k_mask = k_iota < K
    draft_ok = k_iota < best_n
    out = tl.where(draft_ok, best_chain.to(tl.int64), last_tok)
    tl.store(drafts_ptr + b * K + k_iota, out, mask=k_mask)
    if WRITE_REQ_DRAFTS:
        tl.store(req_drafts_ptr + req.to(tl.int64) * req_drafts_stride + k_iota, out, mask=k_mask)
    tl.store(num_valid_ptr + b, best_n)
    if ADAPTIVE:
        bucket = _NUM_LEN_BUCKETS + _len_bucket(tl.maximum(best_len.to(tl.int64), 1))
        stat = stats_ptr + (bucket * K + k_iota) * 2
        accepted = tl.load(stat, mask=k_mask, other=0).to(tl.float32)
        observed = tl.load(stat + 1, mask=k_mask, other=0).to(tl.float32)
        conf = tl.where(draft_ok, (accepted + 1.0) / (observed + 2.0), 0.0)
        tl.store(conf_ptr + b * K + k_iota, conf, mask=k_mask)
        tl.store(last_bucket_ptr + req, tl.where(best_n > 0, bucket, -1).to(tl.int32))
        tl.store(last_valid_ptr + req, best_n)


@triton.jit
def _suffix_observe_kernel(
    idx_mapping_ptr,  # *int32  [B]
    num_sampled_ptr,  # *int32  [B]
    num_rejected_ptr,  # *int32  [B]
    stats_ptr,  # *int64  [2 * _NUM_LEN_BUCKETS, K, 2]
    last_bucket_ptr,  # *int32  [max_num_reqs]
    last_valid_ptr,  # *int32  [max_num_reqs]
    K: tl.constexpr,
    K_PO2: tl.constexpr,
):
    """Fold the target's verdict on each request's last draft into the stats.

    Integer atomics are order-independent, so every TP rank keeps identical
    stats (and therefore identical confidences).
    """
    b = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    bucket = tl.load(last_bucket_ptr + req).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b).to(tl.int64)
    if (bucket < 0) | (num_sampled == 0):
        return
    num_accepted = num_sampled - 1
    num_verified = num_accepted + tl.load(num_rejected_ptr + b).to(tl.int64)
    num_valid = tl.load(last_valid_ptr + req).to(tl.int64)
    k_iota = tl.arange(0, K_PO2).to(tl.int64)
    # Accepted positions plus the first rejected one were observed.
    observed = (k_iota < K) & (k_iota <= num_accepted) & (k_iota < num_verified) & (k_iota < num_valid)
    stat = stats_ptr + (bucket * K + k_iota) * 2
    tl.atomic_add(stat, (k_iota < num_accepted).to(tl.int64), mask=observed)
    tl.atomic_add(stat + 1, tl.full([K_PO2], 1, tl.int64), mask=observed)
    tl.store(last_bucket_ptr + req, -1)


class SuffixSpeculator(BaseSpeculator):
    """V2-compatible GPU suffix decoding speculator."""

    supports_mm_inputs = False
    draft_logits = None
    _writes_req_draft_tokens = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        req_states: RequestState,
    ):
        if not HAS_TRITON:
            raise RuntimeError("suffix speculative decoding requires Triton.")
        spec = vllm_config.speculative_config
        assert spec is not None
        self.vllm_config = vllm_config
        self.device = device
        self.req_states = req_states
        self.num_speculative_steps: int = spec.num_speculative_tokens
        self.max_depth: int = spec.suffix_decoding_max_tree_depth
        self.max_num_reqs: int = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len: int = vllm_config.model_config.max_model_len

        L = self.max_model_len
        if L >= 256 and self.max_depth <= 128:
            self.block_l = 128
        elif L >= 1024:
            self.block_l = 256
        elif L >= 256:
            self.block_l = 128
        else:
            self.block_l = max(16, triton.next_power_of_2(max(L, 1)))
        self.n_blocks = triton.cdiv(L, self.block_l)

        # Corpus of finished responses separated by -1, plus two hash tables
        # keyed on the last 4 and last 2 tokens before each position. A
        # capacity of 0 disables it.
        self.corpus_len: int = (
            spec.suffix_decoding_corpus_tokens if spec.suffix_decoding_max_cached_requests != 0 else 0
        )
        if self.corpus_len > 0 and self.max_depth < 2:
            # Corpus matches need the 2-token hash key.
            logger.warning(
                "suffix_decoding_max_tree_depth=%d is too short for "
                "cross-request matching (needs >= 2); drafting from each "
                "request's own history only.",
                self.max_depth,
            )
            self.corpus_len = 0
        self.has_corpus = self.corpus_len > 0
        self.num_buckets = triton.next_power_of_2(max(16, self.corpus_len // 4))
        table_cells = 2 * (self.num_buckets if self.has_corpus else 1) * _BUCKET_SLOTS
        if table_cells > _MAX_EXACT_TABLE_CELLS:
            raise ValueError(
                "suffix corpus hash table exceeds 2**24 cells; "
                "float32 ingestion address keys would not be exact. "
                "Reduce suffix_decoding_corpus_tokens."
            )
        # Validate before allocating the corpus, table, or configuration-sized
        # scratch, including when Python assertions are disabled.
        self.scratch = torch.zeros((self.max_num_reqs, self.n_blocks), dtype=torch.int64, device=device)
        self.corpus = torch.full((max(self.corpus_len, 1),), -1, dtype=torch.int32, device=device)
        self.table = torch.full(
            (2, self.num_buckets if self.has_corpus else 1, _BUCKET_SLOTS),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self._corpus_state = torch.zeros(2, dtype=torch.int64, device=device)
        self.corpus_head = self._corpus_state[0:1]
        self.doc_seq = self._corpus_state[1:2]
        # Host mirror of doc_seq: index entries keep it in 31 bits.
        self.num_docs = 0
        self.ingest_out = torch.zeros(2, dtype=torch.int64, device=device)

        # Non-atomic ingest scratch (sort + segmented-max path). Sized by
        # configuration, not by any single batch: C_max = 2 * F_max * P.
        # Allocated only when the corpus is enabled; a disabled corpus never
        # runs the ingest chain, so no full-size buffers are held.
        c_slots = 2 * _BUCKET_SLOTS
        if self.has_corpus:
            self.emit_p = triton.cdiv(self.max_model_len, _INGEST_BLOCK) * _INGEST_BLOCK
            self.emit_c_max = 2 * self.max_num_reqs * self.emit_p
            cmax = max(self.emit_c_max, 1)
            self.emit_keys = torch.zeros(cmax, dtype=torch.float32, device=device)
            self.emit_values = torch.zeros(cmax, dtype=torch.int64, device=device)
            self.emit_prefixes = torch.zeros(cmax, dtype=torch.int64, device=device)
            t_max = triton.cdiv(cmax, _INGEST_BLOCK)
            self.emit_summary_values = torch.zeros(t_max, dtype=torch.int64, device=device)
            self.emit_summary_heads = torch.zeros(t_max, dtype=torch.int32, device=device)
            self.emit_carries = torch.zeros(t_max, dtype=torch.int64, device=device)
            k_po2 = max(1, triton.next_power_of_2(self.num_speculative_steps))
            self._corpus_clen = torch.zeros((self.max_num_reqs, c_slots), dtype=torch.int32, device=device)
            self._corpus_vals = torch.zeros((self.max_num_reqs, c_slots), dtype=torch.int64, device=device)
            self._corpus_cont_t = torch.zeros((self.max_num_reqs, k_po2, c_slots), dtype=torch.int32, device=device)
            self._corpus_llen = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        else:
            self._corpus_clen = torch.empty(1, dtype=torch.int32, device=device)
            self._corpus_vals = torch.empty(1, dtype=torch.int64, device=device)
            self._corpus_cont_t = torch.empty(1, dtype=torch.int32, device=device)
            self._corpus_llen = torch.empty(1, dtype=torch.int32, device=device)

        # Batch-ordered draft output, scattered into RequestState.draft_tokens
        # by the model runner (same contract as the model-based speculators).
        self.drafts = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.int64,
            device=device,
        )
        self._triton_kcache: dict[tuple[Any, ...], Any] = {}
        self._get_raw_stream: Any = None
        self._is_capturing: Any = None
        if device.type == "npu":
            import torch_npu

            self._get_raw_stream = torch_npu._C._npu_getCurrentRawStream
            self._is_capturing = torch_npu.npu.is_current_stream_capturing
        # Number of real (non-filler) draft tokens per batch row.
        self.num_valid = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)

        # Adaptive verification (optional): per-slot confidences from
        # acceptance statistics learned online.
        self.enable_adaptive_verification = spec.enable_adaptive_verification
        self.draft_token_confidence_probs = torch.zeros(
            (self.max_num_reqs, self.num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self.accept_stats = torch.zeros(
            (2 * _NUM_LEN_BUCKETS, self.num_speculative_steps, 2),
            dtype=torch.int64,
            device=device,
        )
        self.last_bucket = torch.full((self.max_num_reqs,), -1, dtype=torch.int32, device=device)
        self.last_valid = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self.num_steps = 0
        # Slots reassigned since the last propose; their stats bookkeeping
        # still belongs to the previous request.
        self.new_slots: list[int] = []
        self.sampling_states: SamplingStates | None = None
        self.not_seeded = torch.zeros(self.max_num_reqs, dtype=torch.bool, device=device)

    def set_sampling_states(self, sampling_states: SamplingStates) -> None:
        self.sampling_states = sampling_states

    def seeded_requests(self, req_indices: np.ndarray) -> np.ndarray | None:
        """Mask of requests with an explicit seed that sample (temperature >
        0), or None if there are none. Their drafts depend only on their own
        history, so verifying all of their slots keeps them reproducible.
        Greedy outputs don't depend on the drafts, so they keep the corpus."""
        states = self.sampling_states
        if states is None or not states.any_explicit_seed(req_indices):
            return None
        seeded = states.seeds_set[req_indices] & (states.temperature.np[req_indices] != 0.0)
        return seeded if seeded.any() else None

    def add_request(self, req_index: int) -> None:
        if self.enable_adaptive_verification:
            self.new_slots.append(req_index)

    def reset_state(self) -> None:
        """Forget everything learned so far (used after the warmup requests)."""
        self.corpus.fill_(-1)
        self.table.fill_(-1)
        self._corpus_state.zero_()
        self.num_docs = 0
        self.accept_stats.zero_()
        self.last_bucket.fill_(-1)
        self.last_valid.zero_()
        self.num_steps = 0
        self.new_slots.clear()

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Add finished responses to the corpus before their slots are freed."""
        if not self.has_corpus:
            return
        req_states = self.req_states
        prefill_len = getattr(req_states, "prefill_len", None)
        num_computed_np = getattr(req_states, "num_computed_tokens_np", None)
        # Sorted so every TP rank ingests documents in the same order. Starts
        # come from the host copy of prompt_len: the device-visible copy is
        # republished every step and may be reused for a new request before
        # this kernel runs.
        docs = []
        max_n = 1
        for req_id in sorted(finished_req_ids):
            slot = req_states.req_id_to_index.get(req_id)
            if slot is None:
                continue
            # Keep the prompt's last few tokens (typically the chat template's
            # assistant header) so the first response tokens are indexed by
            # the tokens that precede them.
            start = max(int(req_states.prompt_len.np[slot]) - _PROMPT_CONTEXT, 0)
            if prefill_len is not None and num_computed_np is not None:
                upper_end = min(
                    self.max_model_len,
                    max(int(prefill_len.np[slot]), int(num_computed_np[slot])) + self.num_speculative_steps + 2,
                )
                max_n = max(max_n, upper_end - start)
            else:
                max_n = self.max_model_len
            docs += [slot, start]
        if not docs:
            return
        max_n = min(max_n, self.corpus_len - 1 if self.corpus_len > 1 else max_n)
        max_emit_blocks = triton.cdiv(self.max_model_len, _INGEST_BLOCK)
        needed_emit_blocks = max(1, triton.cdiv(max_n, _INGEST_BLOCK))
        ingest_blocks = max_emit_blocks
        for _b in (1, 2, 3, 4, 6, 8, 10, 12):
            if needed_emit_blocks <= _b:
                ingest_blocks = _b
                break
        ingest_blocks = min(ingest_blocks, max_emit_blocks)
        active_emit_p = min(self.emit_p, ingest_blocks * _INGEST_BLOCK)

        num_docs = len(docs) // 2
        if self.num_docs + num_docs >= 1 << 31:
            # Index entries pack doc_seq into 31 bits; start a new index (the
            # corpus tokens stay, only their entries are dropped).
            self.table.fill_(-1)
            self.doc_seq.zero_()
            self.num_docs = 0
        self.num_docs += num_docs
        docs_gpu = async_tensor_h2d(docs, self.device, torch.int32)
        token_ids = self.req_states.all_token_ids.gpu
        # Dense record capacity for THIS batch; the second family plane starts
        # at this batch's F (num_docs), not at max_num_reqs.
        C = 2 * num_docs * active_emit_p
        T = triton.cdiv(C, _INGEST_BLOCK)
        tid_stride = token_ids.stride(0)
        tot_len_gpu = req_states.total_len.gpu
        head_out = self.ingest_out[0:1]
        doc_seq_out = self.ingest_out[1:2]
        ek = self.emit_keys[:C]
        ev = self.emit_values[:C]
        ep = self.emit_prefixes[:C]
        esv = self.emit_summary_values[:T]
        esh = self.emit_summary_heads[:T]
        eca = self.emit_carries[:T]
        sentinel = self.table.numel()
        carry_block = triton.next_power_of_2(max(T, 1))
        get_raw_stream = self._get_raw_stream
        is_capturing = self._is_capturing
        if get_raw_stream is None or is_capturing is None or is_capturing():
            _suffix_ingest_emit_kernel[(num_docs, ingest_blocks)](
                token_ids,
                tid_stride,
                docs_gpu,
                num_docs,
                tot_len_gpu,
                self.corpus,
                self.corpus_len,
                self.corpus_head,
                self.doc_seq,
                head_out,
                doc_seq_out,
                ek,
                ev,
                HASH_MUL=_HASH_MUL,
                NUM_BUCKETS=self.num_buckets,
                SLOTS=_BUCKET_SLOTS,
                BLOCK=_INGEST_BLOCK,
                P=active_emit_p,
                SENTINEL=sentinel,
                num_warps=4,
            )
            # Sort address keys only; values keep original record order and are
            # gathered on device inside segment_local_scan (no host-side gather).
            sorted_keys, permutation = torch.sort(ek)
            segment_local_scan[(T,)](
                sorted_keys,
                permutation,
                ev,
                ep,
                esv,
                esh,
                C,
                SENTINEL=sentinel,
                BLOCK=_INGEST_BLOCK,
                num_warps=4,
            )
            segment_scan_carries[(1,)](
                esv,
                esh,
                eca,
                NUM_TILES=T,
                CARRY_BLOCK=carry_block,
                num_warps=4,
            )
            segment_tail_merge[(T,)](
                sorted_keys,
                ep,
                eca,
                self.table,
                C,
                SENTINEL=sentinel,
                BLOCK=_INGEST_BLOCK,
                num_warps=4,
            )
        else:
            dev_idx = self.device.index if self.device.index is not None else 0
            emit_key = ("emit", dev_idx, active_emit_p, num_docs == 1, (docs_gpu.data_ptr() & 15) == 0)
            k_emit = self._triton_kcache.get(emit_key)
            if k_emit is None:
                k_emit = _suffix_ingest_emit_kernel[(num_docs, ingest_blocks)](
                    token_ids,
                    tid_stride,
                    docs_gpu,
                    num_docs,
                    tot_len_gpu,
                    self.corpus,
                    self.corpus_len,
                    self.corpus_head,
                    self.doc_seq,
                    head_out,
                    doc_seq_out,
                    ek,
                    ev,
                    HASH_MUL=_HASH_MUL,
                    NUM_BUCKETS=self.num_buckets,
                    SLOTS=_BUCKET_SLOTS,
                    BLOCK=_INGEST_BLOCK,
                    P=active_emit_p,
                    SENTINEL=sentinel,
                    num_warps=4,
                )
                self._triton_kcache[emit_key] = k_emit
            else:
                st = get_raw_stream(dev_idx)
                k_emit.run(
                    num_docs,
                    ingest_blocks,
                    1,
                    st,
                    k_emit.function,
                    k_emit.packed_metadata,
                    None,
                    None,
                    None,
                    token_ids,
                    tid_stride,
                    docs_gpu,
                    num_docs,
                    tot_len_gpu,
                    self.corpus,
                    self.corpus_len,
                    self.corpus_head,
                    self.doc_seq,
                    head_out,
                    doc_seq_out,
                    ek,
                    ev,
                    _HASH_MUL,
                    self.num_buckets,
                    _BUCKET_SLOTS,
                    _INGEST_BLOCK,
                    active_emit_p,
                    sentinel,
                )
            sorted_keys, permutation = torch.sort(ek)
            sort_align = (((sorted_keys.data_ptr() & 15) == 0) << 0) | (((permutation.data_ptr() & 15) == 0) << 1)
            sls_key = ("sls", dev_idx, sort_align)
            k_sls = self._triton_kcache.get(sls_key)
            if k_sls is None:
                k_sls = segment_local_scan[(T,)](
                    sorted_keys,
                    permutation,
                    ev,
                    ep,
                    esv,
                    esh,
                    C,
                    SENTINEL=sentinel,
                    BLOCK=_INGEST_BLOCK,
                    num_warps=4,
                )
                self._triton_kcache[sls_key] = k_sls
            else:
                st = get_raw_stream(dev_idx)
                k_sls.run(
                    T,
                    1,
                    1,
                    st,
                    k_sls.function,
                    k_sls.packed_metadata,
                    None,
                    None,
                    None,
                    sorted_keys,
                    permutation,
                    ev,
                    ep,
                    esv,
                    esh,
                    C,
                    sentinel,
                    _INGEST_BLOCK,
                )
            ssc_key = ("ssc", dev_idx, T, carry_block)
            k_ssc = self._triton_kcache.get(ssc_key)
            if k_ssc is None:
                k_ssc = segment_scan_carries[(1,)](
                    esv,
                    esh,
                    eca,
                    NUM_TILES=T,
                    CARRY_BLOCK=carry_block,
                    num_warps=4,
                )
                self._triton_kcache[ssc_key] = k_ssc
            else:
                st = get_raw_stream(dev_idx)
                k_ssc.run(
                    1,
                    1,
                    1,
                    st,
                    k_ssc.function,
                    k_ssc.packed_metadata,
                    None,
                    None,
                    None,
                    esv,
                    esh,
                    eca,
                    T,
                    carry_block,
                )
            stm_key = ("stm", dev_idx, (sorted_keys.data_ptr() & 15) == 0)
            k_stm = self._triton_kcache.get(stm_key)
            if k_stm is None:
                k_stm = segment_tail_merge[(T,)](
                    sorted_keys,
                    ep,
                    eca,
                    self.table,
                    C,
                    SENTINEL=sentinel,
                    BLOCK=_INGEST_BLOCK,
                    num_warps=4,
                )
                self._triton_kcache[stm_key] = k_stm
            else:
                st = get_raw_stream(dev_idx)
                k_stm.run(
                    T,
                    1,
                    1,
                    st,
                    k_stm.function,
                    k_stm.packed_metadata,
                    None,
                    None,
                    None,
                    sorted_keys,
                    ep,
                    eca,
                    self.table,
                    C,
                    sentinel,
                    _INGEST_BLOCK,
                )
        # Publish staged state for the next batch only after the whole chain.
        self._corpus_state.copy_(self.ingest_out)

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: Any,
        slot_mappings: Any,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if dummy_run:
            return self.drafts[:num_reqs]

        token_ids = self.req_states.all_token_ids.gpu
        total_len = self.req_states.total_len.gpu
        idx_mapping = input_batch.idx_mapping
        k_po2 = max(1, triton.next_power_of_2(self.num_speculative_steps))
        seeded = self.not_seeded
        if self.has_corpus:
            mask = self.seeded_requests(input_batch.idx_mapping_np)
            if mask is not None:
                seeded = async_tensor_h2d(mask, self.device, torch.bool)

        if self.enable_adaptive_verification:
            if self.new_slots:
                slots = async_tensor_h2d(self.new_slots, self.device, torch.int64)
                self.last_bucket.index_fill_(0, slots, -1)
                self.new_slots.clear()
            _suffix_observe_kernel[(num_reqs,)](
                idx_mapping,
                num_sampled,
                num_rejected,
                self.accept_stats,
                self.last_bucket,
                self.last_valid,
                K=self.num_speculative_steps,
                K_PO2=k_po2,
                num_warps=1,
            )
            self.num_steps += 1
            if self.num_steps % _STATS_HALF_LIFE == 0:
                self.accept_stats.bitwise_right_shift_(1)

        seq_lens_upper = getattr(input_batch, "seq_lens_cpu_upper_bound", None)
        if num_reqs > 0 and seq_lens_upper is not None:
            max_seq_len = int(seq_lens_upper[:num_reqs].max()) + self.num_speculative_steps + 2
            active_n_blocks = min(self.n_blocks, max(1, triton.cdiv(max_seq_len, self.block_l)))
        else:
            active_n_blocks = self.n_blocks
        active_nb_po2 = max(8, triton.next_power_of_2(active_n_blocks))

        tid_stride = token_ids.stride(0)
        scr_stride = self.scratch.stride(0)
        lp1 = self.max_model_len + 1
        last_sampled_flat = last_sampled.view(-1)
        req_drafts = getattr(self.req_states, "draft_tokens", None)
        write_req_drafts = req_drafts is not None
        req_drafts_buf = req_drafts if req_drafts is not None else self.drafts
        req_drafts_stride = req_drafts_buf.stride(0)
        depth_po2 = triton.next_power_of_2(self.max_depth)
        get_raw_stream = self._get_raw_stream
        is_capturing = self._is_capturing
        if get_raw_stream is None or is_capturing is None or is_capturing() or num_reqs <= 0:
            _suffix_local_scan_kernel[(num_reqs, active_n_blocks)](
                token_ids,
                tid_stride,
                idx_mapping,
                total_len,
                num_sampled,
                self.scratch,
                scr_stride,
                lp1,
                MAX_DEPTH=self.max_depth,
                BLOCK=self.block_l,
                num_warps=4,
            )
            _finalize_and_corpus_a_kernel[(num_reqs,)](
                token_ids,
                tid_stride,
                idx_mapping,
                total_len,
                num_sampled,
                last_sampled_flat,
                self.scratch,
                scr_stride,
                active_n_blocks,
                lp1,
                self.corpus,
                self.corpus.shape[0],
                self.table,
                self.drafts,
                req_drafts_buf,
                self.num_valid,
                self.accept_stats,
                self.last_bucket,
                self.last_valid,
                self.draft_token_confidence_probs,
                seeded,
                self._corpus_clen,
                self._corpus_vals,
                self._corpus_cont_t,
                self._corpus_llen,
                req_drafts_stride=req_drafts_stride,
                HAS_CORPUS=self.has_corpus,
                ADAPTIVE=self.enable_adaptive_verification,
                WRITE_REQ_DRAFTS=write_req_drafts,
                HASH_MUL=_HASH_MUL,
                NUM_BUCKETS=self.num_buckets,
                SLOTS=_BUCKET_SLOTS,
                MAX_DEPTH=self.max_depth,
                DEPTH_PO2=depth_po2,
                K=self.num_speculative_steps,
                K_PO2=k_po2,
                NB_PO2=active_nb_po2,
                num_warps=4,
            )
            if self.has_corpus:
                _corpus_b_1d_kernel[(num_reqs,)](
                    idx_mapping,
                    last_sampled_flat,
                    self._corpus_clen,
                    self._corpus_vals,
                    self._corpus_cont_t,
                    self._corpus_llen,
                    self.drafts,
                    req_drafts_buf,
                    self.num_valid,
                    self.accept_stats,
                    self.last_bucket,
                    self.last_valid,
                    self.draft_token_confidence_probs,
                    req_drafts_stride=req_drafts_stride,
                    ADAPTIVE=self.enable_adaptive_verification,
                    WRITE_REQ_DRAFTS=write_req_drafts,
                    SLOTS=_BUCKET_SLOTS,
                    K=self.num_speculative_steps,
                    K_PO2=k_po2,
                    num_warps=4,
                )
        else:
            dev_idx = self.device.index if self.device.index is not None else 0
            p_align = (
                (((idx_mapping.data_ptr() & 15) == 0) << 0)
                | (((num_sampled.data_ptr() & 15) == 0) << 1)
                | (((last_sampled_flat.data_ptr() & 15) == 0) << 2)
                | (((seeded.data_ptr() & 15) == 0) << 3)
            )
            ls_key = ("ls", dev_idx, p_align & 3)
            k_ls = self._triton_kcache.get(ls_key)
            if k_ls is None:
                k_ls = _suffix_local_scan_kernel[(num_reqs, active_n_blocks)](
                    token_ids,
                    tid_stride,
                    idx_mapping,
                    total_len,
                    num_sampled,
                    self.scratch,
                    scr_stride,
                    lp1,
                    MAX_DEPTH=self.max_depth,
                    BLOCK=self.block_l,
                    num_warps=4,
                )
                self._triton_kcache[ls_key] = k_ls
            else:
                st = get_raw_stream(dev_idx)
                k_ls.run(
                    num_reqs,
                    active_n_blocks,
                    1,
                    st,
                    k_ls.function,
                    k_ls.packed_metadata,
                    None,
                    None,
                    None,
                    token_ids,
                    tid_stride,
                    idx_mapping,
                    total_len,
                    num_sampled,
                    self.scratch,
                    scr_stride,
                    lp1,
                    self.max_depth,
                    self.block_l,
                )

            fa_key = ("fa", dev_idx, active_n_blocks == 1, active_nb_po2, write_req_drafts, p_align)
            k_fa = self._triton_kcache.get(fa_key)
            corpus_len = self.corpus.shape[0]
            if k_fa is None:
                k_fa = _finalize_and_corpus_a_kernel[(num_reqs,)](
                    token_ids,
                    tid_stride,
                    idx_mapping,
                    total_len,
                    num_sampled,
                    last_sampled_flat,
                    self.scratch,
                    scr_stride,
                    active_n_blocks,
                    lp1,
                    self.corpus,
                    corpus_len,
                    self.table,
                    self.drafts,
                    req_drafts_buf,
                    self.num_valid,
                    self.accept_stats,
                    self.last_bucket,
                    self.last_valid,
                    self.draft_token_confidence_probs,
                    seeded,
                    self._corpus_clen,
                    self._corpus_vals,
                    self._corpus_cont_t,
                    self._corpus_llen,
                    req_drafts_stride=req_drafts_stride,
                    HAS_CORPUS=self.has_corpus,
                    ADAPTIVE=self.enable_adaptive_verification,
                    WRITE_REQ_DRAFTS=write_req_drafts,
                    HASH_MUL=_HASH_MUL,
                    NUM_BUCKETS=self.num_buckets,
                    SLOTS=_BUCKET_SLOTS,
                    MAX_DEPTH=self.max_depth,
                    DEPTH_PO2=depth_po2,
                    K=self.num_speculative_steps,
                    K_PO2=k_po2,
                    NB_PO2=active_nb_po2,
                    num_warps=4,
                )
                self._triton_kcache[fa_key] = k_fa
            else:
                st = get_raw_stream(dev_idx)
                k_fa.run(
                    num_reqs,
                    1,
                    1,
                    st,
                    k_fa.function,
                    k_fa.packed_metadata,
                    None,
                    None,
                    None,
                    token_ids,
                    tid_stride,
                    idx_mapping,
                    total_len,
                    num_sampled,
                    last_sampled_flat,
                    self.scratch,
                    scr_stride,
                    active_n_blocks,
                    lp1,
                    self.corpus,
                    corpus_len,
                    self.table,
                    self.drafts,
                    req_drafts_buf,
                    self.num_valid,
                    self.accept_stats,
                    self.last_bucket,
                    self.last_valid,
                    self.draft_token_confidence_probs,
                    seeded,
                    self._corpus_clen,
                    self._corpus_vals,
                    self._corpus_cont_t,
                    self._corpus_llen,
                    req_drafts_stride,
                    self.has_corpus,
                    self.enable_adaptive_verification,
                    write_req_drafts,
                    _HASH_MUL,
                    self.num_buckets,
                    _BUCKET_SLOTS,
                    self.max_depth,
                    depth_po2,
                    self.num_speculative_steps,
                    k_po2,
                    active_nb_po2,
                )

            if self.has_corpus:
                cb_key = ("cb", dev_idx, write_req_drafts, p_align & 5)
                k_cb = self._triton_kcache.get(cb_key)
                if k_cb is None:
                    k_cb = _corpus_b_1d_kernel[(num_reqs,)](
                        idx_mapping,
                        last_sampled_flat,
                        self._corpus_clen,
                        self._corpus_vals,
                        self._corpus_cont_t,
                        self._corpus_llen,
                        self.drafts,
                        req_drafts_buf,
                        self.num_valid,
                        self.accept_stats,
                        self.last_bucket,
                        self.last_valid,
                        self.draft_token_confidence_probs,
                        req_drafts_stride=req_drafts_stride,
                        ADAPTIVE=self.enable_adaptive_verification,
                        WRITE_REQ_DRAFTS=write_req_drafts,
                        SLOTS=_BUCKET_SLOTS,
                        K=self.num_speculative_steps,
                        K_PO2=k_po2,
                        num_warps=4,
                    )
                    self._triton_kcache[cb_key] = k_cb
                else:
                    st = get_raw_stream(dev_idx)
                    k_cb.run(
                        num_reqs,
                        1,
                        1,
                        st,
                        k_cb.function,
                        k_cb.packed_metadata,
                        None,
                        None,
                        None,
                        idx_mapping,
                        last_sampled_flat,
                        self._corpus_clen,
                        self._corpus_vals,
                        self._corpus_cont_t,
                        self._corpus_llen,
                        self.drafts,
                        req_drafts_buf,
                        self.num_valid,
                        self.accept_stats,
                        self.last_bucket,
                        self.last_valid,
                        self.draft_token_confidence_probs,
                        req_drafts_stride,
                        self.enable_adaptive_verification,
                        write_req_drafts,
                        _BUCKET_SLOTS,
                        self.num_speculative_steps,
                        k_po2,
                    )
        return self.drafts[:num_reqs]
