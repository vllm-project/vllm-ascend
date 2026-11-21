# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 suffix finalize NPU tests adapted from vLLM #60339."""

import numpy as np
import pytest

from .suffix_test_utils import (
    _finish,
    _local_draft,
    _make,
    _propose,
    _set,
)


@pytest.mark.parametrize("vocab", [4, 16, 1000])
def test_local_matches_reference(vocab):
    k, depth, max_reqs, max_len = 6, 8, 8, 300
    rng = np.random.default_rng(vocab)
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=0)
    for _ in range(4):
        hists = []
        for s in range(max_reqs):
            plen, olen = int(rng.integers(1, 60)), int(rng.integers(1, 200))
            toks = rng.integers(0, vocab, plen + olen)
            _set(rs, s, toks[:plen], toks[plen:])
            hists.append(toks)
        drafts, num_valid = _propose(spec, rs, list(range(max_reqs)))
        for s, hist in enumerate(hists):
            ref = _local_draft(hist, k, depth)
            assert num_valid[s] == len(ref)
            assert drafts[s, : len(ref)].tolist() == ref
            # Filler slots carry the last sampled token.
            assert (drafts[s, len(ref) :] == hist[-1]).all()


def test_longer_corpus_match_wins_over_local():
    k, depth, max_reqs, max_len = 3, 8, 2, 128
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    _set(rs, 0, [], [1, 2, 3, 4, 70, 71, 72])
    _finish(spec, rs, {"done": 0})
    # Locally only "4" recurs (continuing with 99); the corpus matches
    # "1 2 3 4".
    _set(rs, 1, [4, 99, 98], [1, 2, 3, 4])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 3
    assert drafts[0].tolist() == [70, 71, 72]


def test_no_draft_without_sampled_token():
    k, depth, max_reqs, max_len = 3, 8, 2, 64
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=0)
    _set(rs, 0, [], [1, 2, 3, 1, 2])
    # A partial prefill samples no token and must not draft.
    drafts, num_valid = _propose(spec, rs, [0], num_sampled=[0])
    assert num_valid[0] == 0
    drafts, num_valid = _propose(spec, rs, [0])
    assert drafts[0].tolist() == [3, 1, 2]


@pytest.mark.parametrize("length", [1, 2, 3])
def test_short_history_does_not_match_clamped_addresses(length):
    spec, rs = _make(3, 24, 4, 128, corpus_tokens=1024)
    _set(rs, 0, [], [5, 6, 7, 8])
    _finish(spec, rs, {"done": 0})
    history = [5, 6, 7][:length]
    _set(rs, 3, [], history)
    drafts, valid = _propose(spec, rs, [3])
    expected = [] if length == 1 else [7, 8] if length == 2 else [8]
    assert valid.tolist() == [len(expected)]
    assert drafts[0].tolist() == expected + [history[-1]] * (3 - len(expected))


@pytest.mark.parametrize("max_len,k", [(32, 1), (257, 3), (8192, 8)])
def test_sparse_slots_at_context_limit(max_len, k):
    spec, rs = _make(k, 24, 8, max_len, corpus_tokens=0)
    history = list(range(100, 100 + max_len))
    history[4:8] = [1, 2, 3, 4]
    history[-4:] = [1, 2, 3, 4]
    _set(rs, 6, [], history)
    _set(rs, 1, [], [7])
    drafts, valid = _propose(spec, rs, [6, 1])
    assert drafts[0].tolist() == _local_draft(history, k, 24)
    assert valid.tolist() == [k, 0]
    assert drafts[1].tolist() == [7] * k
