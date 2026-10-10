import pytest

from tools.spec_decode_metrics import validate_acceptance_rates


@pytest.mark.parametrize(
    "actual_rates, baseline_rates, tolerance",
    [
        pytest.param([0.8, 0.0], 0.8, 0.05, id="scalar-checks-only-position-zero"),
        pytest.param([1.0, 0.0], 1, 0.05, id="integer-scalar"),
        pytest.param([0.8, 0.5, 0.0], [0.8, 0.5], 0.05, id="list-checks-configured-prefix"),
        pytest.param([0.8, 0.5], [0.8, 0.5], 0.05, id="full-list"),
        pytest.param([0.4375], 0.5, 0.125, id="inclusive-lower-bound"),
        pytest.param([0.625], 0.5, 0.125, id="inclusive-double-tolerance-upper-bound"),
        pytest.param([0.8, 0.5], [0.8, 0.5], 0.0, id="zero-tolerance"),
    ],
)
def test_validate_acceptance_rates_accepts_configured_positions(actual_rates, baseline_rates, tolerance):
    validate_acceptance_rates(actual_rates, baseline_rates, tolerance)


@pytest.mark.parametrize(
    "actual_rates, baseline_rates, expected_error",
    [
        pytest.param([0.759], 0.8, "pos0", id="scalar-below-lower-bound"),
        pytest.param([0.881], 0.8, "pos0", id="scalar-above-upper-bound"),
        pytest.param([0.759, 0.5], [0.8, 0.5], "pos0", id="list-first-position-regression"),
        pytest.param([0.8, 0.474], [0.8, 0.5], "pos1", id="list-later-position-regression"),
        pytest.param([0.8, 0.551], [0.8, 0.5], "pos1", id="list-later-position-too-high"),
        pytest.param([0.8], [0.8, 0.5], "position count is insufficient", id="missing-position"),
        pytest.param([], 0.8, "position count is insufficient", id="missing-scalar-position"),
        pytest.param([0.8], [], "baseline must contain at least one position", id="empty-baseline"),
        pytest.param([float("nan")], 0.8, "pos0", id="invalid-measurement"),
    ],
)
def test_validate_acceptance_rates_rejects_invalid_measurements(actual_rates, baseline_rates, expected_error):
    with pytest.raises(AssertionError, match=expected_error):
        validate_acceptance_rates(actual_rates, baseline_rates)
