# SPDX-License-Identifier: Apache-2.0
"""Exercise the real comparison, including regressions hidden by a tolerance."""

import numpy as np
import pytest

from tools.ci.glm53flash_precision import check_logits


def test_equal_logits_pass():
    logits = np.zeros((8, 154880), dtype=np.float32)
    assert check_logits(logits, logits, 0)["status"] == "PASS"


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_candidate_fails(value):
    reference = np.zeros((8, 154880), dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0] = value
    assert check_logits(reference, candidate, 1)["status"] == "FAIL"


def test_top1_change_fails_even_inside_tolerance():
    reference = np.zeros((8, 154880), dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 1] = 0.001
    assert check_logits(reference, candidate, 1)["status"] == "FAIL"


def test_shape_and_reference_finiteness_are_required():
    reference = np.zeros((8, 154880), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        check_logits(reference, reference[:1], 0)
    reference[0, 0] = np.nan
    with pytest.raises(ValueError, match="reference"):
        check_logits(reference, reference, 0)
