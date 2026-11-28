# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 suffix corpus NPU tests adapted from vLLM #60339."""

import numpy as np
import pytest
import torch

from .suffix_test_utils import (
    _finish,
    _follows_in_corpus,
    _local_draft,
    _make,
    _propose,
    _set,
)

_HASH_MUL = 0x9E3779B97F4A7C15 - (1 << 64)


def test_corpus_wraps_without_straddling():
    k, depth, max_reqs, max_len = 4, 8, 2, 64
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=50)
    for rnd in range(5):
        _set(rs, 0, [100], list(range(rnd * 10, rnd * 10 + 20)))
        _finish(spec, rs, {f"r{rnd}": 0})
    corpus = spec.corpus.tolist()
    # Each document (1 prompt token + 20 response tokens + separator) takes
    # 22 slots: two fit, and every third wraps to offset 0, so documents 4
    # and 3 survive at offsets 0 and 22.
    assert int(spec.corpus_head) == 22
    assert int(spec.doc_seq) == 5
    assert corpus[:22] == [100] + list(range(40, 60)) + [-1]
    assert corpus[22:44] == [100] + list(range(30, 50)) + [-1]
    # Stale index entries for overwritten documents are never drafted.
    _set(rs, 1, [], [12, 13])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 0


def test_ingestion_is_deterministic():
    """The same finished set, reported in different orders, builds identical
    corpus and index state (what keeps TP ranks consistent)."""
    k, depth, max_reqs, max_len = 4, 8, 8, 256
    rng = np.random.default_rng(0)
    toks = [rng.integers(0, 50, 200) for _ in range(max_reqs)]
    states = []
    for order in (list(range(max_reqs)), list(reversed(range(max_reqs)))):
        spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1 << 14)
        for s in range(max_reqs):
            _set(rs, s, toks[s][:20], toks[s][20:])
        ids = [f"r{s}" for s in order]
        rs.req_id_to_index = {rid: int(rid[1:]) for rid in ids}
        spec.on_requests_finished(ids)
        states.append((spec.corpus.clone(), spec.table.clone()))
    assert torch.equal(states[0][0], states[1][0])
    assert torch.equal(states[0][1], states[1][1])


def _s64(x):
    x &= (1 << 64) - 1
    return x - (1 << 64) if x >> 63 else x


def _ref_ingest(state, docs, cap, num_buckets, slots=8):
    """Python model of ingestion: documents placed one after another, wrapping
    to 0 when they don't fit, each position indexed by its 2 and 4 tokens."""
    corpus, table = state["corpus"], state["table"]
    for doc in docs:
        doc = doc[-(cap - 1) :]
        n = len(doc)
        if state["head"] + n + 1 > cap:
            state["head"] = 0
        head, seq = state["head"], state["doc_seq"]
        for j in range(n):
            corpus[head + j] = doc[j]
            hv = 0
            for d in range(4):
                t = doc[j - 1 - d] if j - 1 - d >= 0 else 0
                hv = _s64((hv ^ t) * _HASH_MUL)
                b = ((hv >> 29) ^ hv) & (num_buckets - 1)
                if (d == 1 and j >= 2) or (d == 3 and j >= 4):
                    t_idx = (1 if d == 1 else 0, b, seq % slots)
                    table[t_idx] = max(table[t_idx], (seq << 32) | (head + j))
        corpus[head + n] = -1
        state["head"] = head + n + 1
        state["doc_seq"] = seq + 1


@pytest.mark.parametrize("max_len,corpus_tokens", [(64, 2), (64, 97), (2048, 300), (2048, 4096)])
def test_ingestion_matches_reference(max_len, corpus_tokens):
    """Bursts of finished requests, including bursts larger than the ring
    that overwrite their own earlier documents, give exactly the state of
    placing documents one after another."""
    k, depth, max_reqs = 4, 8, 8
    rng = np.random.default_rng(corpus_tokens)
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=corpus_tokens)
    nb = spec.num_buckets
    ref = dict(
        corpus=np.full(corpus_tokens, -1, dtype=np.int64),
        table=np.full((2, nb, 8), -1, dtype=np.int64),
        head=0,
        doc_seq=0,
    )
    for burst in range(40):
        ids, docs = {}, []
        for i in range(int(rng.integers(1, max_reqs + 1))):
            n = int(rng.integers(1, max_len))
            toks = rng.integers(0, 30, n)
            plen = int(rng.integers(0, n))
            _set(rs, i, toks[:plen], toks[plen:])
            ids[f"b{burst:02d}-{i}"] = i
            docs.append([int(t) for t in toks[max(plen - 3, 0) :]])
        _finish(spec, rs, ids)
        _ref_ingest(ref, docs, corpus_tokens, nb)
        assert spec.corpus.tolist() == ref["corpus"].tolist(), burst
        assert int(spec.corpus_head) == ref["head"]
        assert int(spec.doc_seq) == ref["doc_seq"]
    assert np.array_equal(spec.table.cpu().numpy(), ref["table"])


def test_corpus_drafts_are_real_continuations():
    """A draft is the exact local draft, or follows the request's trailing
    n-gram somewhere in a finished response, and is at least as long a
    match as the local one."""
    k, depth, max_reqs, max_len = 6, 8, 8, 300
    vocab = 16
    rng = np.random.default_rng(vocab)
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1 << 15)
    docs: list[list[int]] = []
    num_from_corpus = 0
    for rnd in range(6):
        hists = []
        for s in range(max_reqs):
            plen, olen = int(rng.integers(1, 60)), int(rng.integers(1, 200))
            toks = rng.integers(0, vocab, plen + olen)
            _set(rs, s, toks[:plen], toks[plen:])
            hists.append(toks)
        drafts, num_valid = _propose(spec, rs, list(range(max_reqs)))
        for s, hist in enumerate(hists):
            got = drafts[s, : num_valid[s]].tolist()
            if got == _local_draft(hist, k, depth):
                continue
            num_from_corpus += 1
            assert _follows_in_corpus(docs, hist, got), (rnd, s)
        ids = {f"r{rnd}-{s}": s for s in range(max_reqs)}
        _finish(spec, rs, ids)
        for rid in sorted(ids):
            plen = int(rs.prompt_len.np[ids[rid]])
            docs.append(hists[ids[rid]][max(plen - 3, 0) :].tolist())
    assert num_from_corpus > 0


def test_corpus_indexes_response_start():
    """The first response token is drafted from the shared prompt tail."""
    k, depth, max_reqs, max_len = 3, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    header = [90, 91, 92]
    _set(rs, 0, [1, 2] + header, [7, 8, 9])
    _finish(spec, rs, {"done": 0})
    # A different prompt with the same template tail, nothing generated yet
    # beyond the prefill token.
    _set(rs, 1, [60, 61] + header, [7])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 2
    assert drafts[0].tolist() == [8, 9, 7]


def test_corpus_draft_from_finished_response():
    k, depth, max_reqs, max_len = 5, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    _set(rs, 0, [1, 2, 3], [7, 8, 9, 10, 11, 12, 13, 14])
    _finish(spec, rs, {"done": 0})
    # A new request whose output starts like the finished response.
    _set(rs, 1, [50, 51], [7, 8, 9])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 5
    assert drafts[0].tolist() == [10, 11, 12, 13, 14]


def test_draft_stops_at_document_end():
    k, depth, max_reqs, max_len = 6, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    _set(rs, 0, [], [5, 6, 7, 8])
    _finish(spec, rs, {"a": 0})
    _set(rs, 0, [], [40, 41, 42])
    _finish(spec, rs, {"b": 0})
    _set(rs, 1, [], [5, 6])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 2
    assert drafts[0].tolist() == [7, 8, 6, 6, 6, 6]


def test_newest_document_wins_ties():
    k, depth, max_reqs, max_len = 2, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    for i, cont in enumerate(([30, 31], [40, 41], [50, 51])):
        _set(rs, 0, [], [5, 6] + cont)
        _finish(spec, rs, {f"d{i}": 0})
    _set(rs, 1, [], [5, 6])
    drafts, _ = _propose(spec, rs, [1])
    assert drafts[0].tolist() == [50, 51]


def test_most_frequent_continuation_wins():
    k, depth, max_reqs, max_len = 2, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    # Two older responses continue "5 6" with "30 31", the newest with "40 41".
    for i, cont in enumerate(([30, 31], [30, 31], [40, 41])):
        _set(rs, 0, [], [5, 6] + cont)
        _finish(spec, rs, {f"d{i}": 0})
    _set(rs, 1, [], [5, 6])
    drafts, _ = _propose(spec, rs, [1])
    assert drafts[0].tolist() == [30, 31]


def test_vote_ties_go_to_the_longer_match():
    """Equal vote scores must tie exactly (summed fp32 vote shares need not)."""
    k, depth, max_reqs, max_len = 2, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=4096)
    # Query "1 2 3 4 5 6". Document a matches all 6 tokens and continues with
    # one token: score 1/1. With "5 6", six more documents match; the seven
    # vote 6 of 7 for 30, then 1 of 6 for one continuation: score (6 + 1) / 7.
    _set(rs, 0, [], [1, 2, 3, 4, 5, 6, 70])
    _finish(spec, rs, {"a": 0})
    for i in range(6):
        _set(rs, 0, [], [90 + i, 5, 6, 30, 40 + i])
        _finish(spec, rs, {f"b{i}": 0})
    _set(rs, 1, [], [1, 2, 3, 4, 5, 6])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 1
    assert drafts[0].tolist() == [70, 6]


def test_vote_prefers_longer_supported_match():
    """A longer match is used when its continuation is as well supported."""
    k, depth, max_reqs, max_len = 2, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    _set(rs, 0, [], [9, 9, 5, 6, 30, 31])
    _finish(spec, rs, {"a": 0})
    _set(rs, 0, [], [1, 2, 3, 4, 5, 6, 70, 71])
    _finish(spec, rs, {"b": 0})
    _set(rs, 1, [], [1, 2, 3, 4, 5, 6])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 2
    assert drafts[0].tolist() == [70, 71]


@pytest.mark.parametrize("competitors,expected", [(1, [30, 31]), (2, [20, 21])])
def test_corpus_wraparound_keeps_one_vote_with_the_newest_sequence(competitors, expected):
    # Each document uses eight slots including the separator. The last document
    # wraps to offset zero, leaving both its entry and the original entry for pos=4.
    spec, rs = _make(2, 8, 2, 64, corpus_tokens=8 * (1 + competitors))
    prefix = [1, 2, 3, 4]
    for i, first in enumerate([10, *([20] * competitors), 30]):
        _set(rs, 0, [], prefix + [first, first + 1, first + 2])
        _finish(spec, rs, {f"r{i}": 0})

    _set(rs, 1, [], prefix)
    drafts, num_valid = _propose(spec, rs, [1])

    # With one competitor, the latest document wins the tie. With two, the
    # competitors win: the wrapped position must not get a duplicate vote.
    assert num_valid.tolist() == [2]
    assert drafts[0].tolist() == expected
