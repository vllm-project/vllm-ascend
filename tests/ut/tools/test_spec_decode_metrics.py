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
        pytest.param([0.625], 0.5, 0.125, id="scalar-above-baseline"),
        pytest.param([0.881], 0.8, 0.05, id="scalar-above-former-upper-bound"),
        pytest.param([0.8, 0.551], [0.8, 0.5], 0.05, id="list-above-former-upper-bound"),
        pytest.param([0.9, 0.6], [0.8, 0.5], 0.0, id="zero-tolerance-allows-improvement"),
        pytest.param(
            [
                0.9104925106738944,
                0.7979973247969786,
                0.6951013968204011,
                0.605072118669687,
                0.5265801711964286,
                0.4556003263250744,
                0.3917515953900371,
            ],
            [0.89, 0.79, 0.65, 0.55, 0.47, 0.4, 0.34],
            0.05,
            id="glm-dspark-v2-nightly-improvement",
        ),
    ],
)
def test_validate_acceptance_rates_accepts_configured_positions(actual_rates, baseline_rates, tolerance):
    validate_acceptance_rates(actual_rates, baseline_rates, tolerance)


@pytest.mark.parametrize(
    "actual_rates, baseline_rates, expected_error",
    [
        pytest.param([0.759], 0.8, "pos0", id="scalar-below-lower-bound"),
        pytest.param([0.759, 0.5], [0.8, 0.5], "pos0", id="list-first-position-regression"),
        pytest.param([0.8, 0.474], [0.8, 0.5], "pos1", id="list-later-position-regression"),
        pytest.param([0.8], [0.8, 0.5], "position count is insufficient", id="missing-position"),
        pytest.param([], 0.8, "position count is insufficient", id="missing-scalar-position"),
        pytest.param([0.8], [], "baseline must contain at least one position", id="empty-baseline"),
        pytest.param([float("nan")], 0.8, "pos0", id="invalid-measurement"),
    ],
)
def test_validate_acceptance_rates_rejects_invalid_measurements(actual_rates, baseline_rates, expected_error):
    with pytest.raises(AssertionError, match=expected_error):
        validate_acceptance_rates(actual_rates, baseline_rates)
