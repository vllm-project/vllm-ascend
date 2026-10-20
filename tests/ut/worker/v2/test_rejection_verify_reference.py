# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tests.ut.worker.v2.rejection_verify_reference import accepted_prefix, entropy_thresholds


@pytest.mark.parametrize("full_draft", [False, True])
def test_entropy_threshold_changes_acceptance(full_draft):
    p = np.array([[0.3, 0.7]] * 3, dtype=np.float32)
    q = np.array([[0.6, 0.4]] * 3, dtype=np.float32) if full_draft else None
    assert accepted_prefix(p, q, [0, 0, 0], [0.6] * 3) == 0
    assert accepted_prefix(p, q, [0, 0, 0], [0.6] * 3, thresholds=[0.4] * 3) == 3


def test_entropy_uses_each_position_and_stops_at_first_rejection():
    p = np.array([[0.3, 0.7]] * 3, dtype=np.float32)
    assert accepted_prefix(p, None, [0, 0, 0], [0.5] * 3, thresholds=[0.5, 0.8, 0.5]) == 1


def test_invalid_proposals_are_prefix_boundaries():
    p = np.array([[0.5, 0.5]] * 3, dtype=np.float32)
    assert accepted_prefix(p, p, [0, -1, 0], [0.5] * 3) == 1
    q = p.copy()
    q[1] = [0, 1]
    assert accepted_prefix(p, q, [0, 0, 0], [0.5] * 3) == 1
    assert accepted_prefix(p[:0], None, [], []) == 0


def test_entropy_alpha_zero_and_probability_zeros():
    p = np.array([[1, 0], [0.5, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(entropy_thresholds(p, 0.7, 0), [0.7, 0.7])
    np.testing.assert_allclose(entropy_thresholds(p, 1, 0), [1, 1])
    thresholds = entropy_thresholds(p)
    assert np.isfinite(thresholds).all()
    np.testing.assert_allclose(thresholds, [0.95, 2**-0.4], rtol=1e-6)
