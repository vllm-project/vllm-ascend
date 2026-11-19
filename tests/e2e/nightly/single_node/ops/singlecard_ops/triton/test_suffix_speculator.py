# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 suffix speculator NPU tests adapted from vLLM #60339."""

import pytest

from .suffix_test_utils import (
    _finish,
    _local_draft,
    _make,
    _propose,
    _set,
)


@pytest.mark.parametrize(
    "k,depth,max_len,corpus_enabled",
    [
        (1, 1, 300, False),
        (1, 2, 300, True),
        (3, 8, 300, False),
        (3, 8, 300, True),
        (8, 24, 2048, False),
        (8, 24, 2048, True),
    ],
)
@pytest.mark.parametrize("adaptive", [False, True])
def test_kernel_feature_combinations(k, depth, max_len, corpus_enabled, adaptive):
    """Compile each feature path and check local, corpus and fallback outputs."""
    spec, rs = _make(k, depth, 8, max_len, 4096 if corpus_enabled else 0, adaptive=adaptive)
    key = list(range(100, 100 + depth))
    continuation = list(range(300, 300 + k))
    if corpus_enabled:
        _set(rs, 7, [], key + continuation)
        _finish(spec, rs, {"done": 7})

    # Three sparse slots exercise padded request reductions; histories use
    # both partial and full scan blocks, and K=3 requires padded draft lanes.
    local = list(range(1000, 1000 + max_len - 2 * depth - 1)) + key + [700] + key
    corpus_query = list(range(5000, 5000 + max_len - depth)) + key
    histories = [local, corpus_query, [991]]
    slots = [6, 1, 4]
    for slot, history in zip(slots, histories):
        _set(rs, slot, [], history)
    expected = [_local_draft(local, k, depth), continuation if corpus_enabled and depth >= 2 else [], []]
    drafts, valid = _propose(spec, rs, slots)
    for batch_idx, (history, reference) in enumerate(zip(histories, expected)):
        assert valid[batch_idx] == len(reference)
        assert drafts[batch_idx].tolist() == reference + [history[-1]] * (k - len(reference))
        if adaptive:
            confidence = spec.draft_token_confidence_probs[batch_idx].tolist()
            assert confidence == pytest.approx([0.5] * len(reference) + [0.0] * (k - len(reference)))


def test_reset_state_forgets_the_corpus():
    k, depth, max_reqs, max_len = 3, 8, 2, 64
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    _set(rs, 0, [], [1, 2, 3, 4, 70, 71, 72])
    _finish(spec, rs, {"warmup": 0})
    _set(rs, 1, [], [1, 2, 3, 4])
    _, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 3
    spec.reset_state()
    _, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 0
    assert int(spec.corpus_head) == 0 and int(spec.doc_seq) == 0


def test_document_counter_rollover_restarts_the_index():
    """Index entries hold the document number in 31 bits; when it would
    overflow, the index restarts and new documents are still found."""
    k, depth, max_reqs, max_len = 3, 8, 2, 64
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=1024)
    spec.num_docs = (1 << 31) - 1
    spec.doc_seq.fill_((1 << 31) - 1)
    _set(rs, 0, [], [1, 2, 3, 4, 70, 71, 72])
    _finish(spec, rs, {"a": 0})
    assert spec.num_docs == 1 and int(spec.doc_seq) == 1
    _set(rs, 1, [], [1, 2, 3, 4])
    drafts, num_valid = _propose(spec, rs, [1])
    assert num_valid[0] == 3
    assert drafts[0].tolist() == [70, 71, 72]


def test_short_tree_depth_disables_the_corpus():
    spec, _ = _make(3, 1, 2, 64, corpus_tokens=1024)
    assert not spec.has_corpus
