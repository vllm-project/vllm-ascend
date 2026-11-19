# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 suffix observe NPU tests adapted from vLLM #60339."""

import pytest

from .suffix_test_utils import (
    _make,
    _propose,
    _set,
)


def test_adaptive_verification_confidences():
    k, depth, max_reqs, max_len = 4, 8, 2, 64
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=0, adaptive=True)
    _set(rs, 0, [], [1, 2, 3, 1, 2])
    drafts, num_valid = _propose(spec, rs, [0])
    assert num_valid[0] == 3
    conf = spec.draft_token_confidence_probs[0].tolist()
    # No observations yet: smoothed 1/2 for real drafts, 0 for the filler.
    assert conf == pytest.approx([0.5, 0.5, 0.5, 0.0])

    # The target accepted 2 of the 3 real tokens and rejected the third.
    _set(rs, 0, [], [1, 2, 3, 1, 2, 3, 1, 9])
    _propose(spec, rs, [0], num_sampled=[3], num_rejected=[1])
    stats = spec.accept_stats.sum(dim=0).tolist()  # [K, (accepted, observed)]
    assert stats[:3] == [[1, 1], [1, 1], [0, 1]]
    assert stats[3] == [0, 0]


def test_reused_slot_does_not_inherit_verification_stats():
    k, depth, max_reqs, max_len = 4, 8, 2, 64
    spec, rs = _make(k, depth, max_reqs, max_len, corpus_tokens=0, adaptive=True)
    _set(rs, 0, [], [1, 2, 3, 1, 2])
    _propose(spec, rs, [0])  # the request finishes before this draft is verified
    # A new request takes slot 0; its first step verifies only placeholders.
    spec.add_request(0)
    _set(rs, 0, [], [5, 6, 7])
    _propose(spec, rs, [0], num_sampled=[1], num_rejected=[k])
    assert spec.accept_stats.sum().item() == 0


@pytest.mark.parametrize("num_reqs,k", [(64, 8), (65, 3)])
def test_observe_large_batches_match_reference(num_reqs, k):
    """Check production batch size and padded reductions with sparse slots."""
    spec, rs = _make(k, 24, 2 * num_reqs + 1, 64, corpus_tokens=0, adaptive=True)
    slots = list(range(2 * num_reqs - 1, 0, -2))
    for slot in slots:
        _set(rs, slot, [], [1, 2, 3, 1, 2])
    _propose(spec, rs, slots)
    sampled = [[3, 1, 0][i % 3] for i in range(num_reqs)]
    rejected = [[1, 3, 0][i % 3] for i in range(num_reqs)]
    _propose(spec, rs, slots, num_sampled=sampled, num_rejected=rejected)
    accepted_count = sampled.count(3)
    rejected_count = sampled.count(1)
    expected = [
        [accepted_count, accepted_count + rejected_count],
        [accepted_count, accepted_count],
        [0, accepted_count],
    ]
    expected += [[0, 0]] * (k - 3)
    assert spec.accept_stats.sum(dim=0).tolist() == expected


@pytest.mark.parametrize("k", [3, 5, 7, 9])
def test_adaptive_padded_drafts_match_learned_statistics(k):
    """Non-power-of-two drafts compile and use learned, nonuniform rates."""
    spec, rs = _make(k, 24, 8, 300, corpus_tokens=0, adaptive=True)
    slots = [6, 1, 4]
    for slot in slots:
        _set(rs, slot, [], [1, 2, 3, 1, 2])
    _propose(spec, rs, slots)
    _, valid = _propose(spec, rs, slots, num_sampled=[3, 1, 0], num_rejected=[1, 3, 0])
    assert valid.tolist() == [3, 3, 0]
    expected_stats = [[1, 2], [1, 1], [0, 1]] + [[0, 0]] * (k - 3)
    assert spec.accept_stats.sum(dim=0).tolist() == expected_stats
    expected_confidence = [0.5, 2 / 3, 1 / 3] + [0.0] * (k - 3)
    for batch_idx in (0, 1):
        assert spec.draft_token_confidence_probs[batch_idx].tolist() == pytest.approx(expected_confidence)
    assert spec.draft_token_confidence_probs[2].tolist() == [0.0] * k
