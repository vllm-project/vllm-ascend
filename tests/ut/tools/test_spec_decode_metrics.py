import pytest
from prometheus_client import CollectorRegistry, Counter, generate_latest

from tools import spec_decode_metrics as metrics
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


def _counter_metrics(drafts, accepted, created=None):
    """Build actual Prometheus counter exposition, including optional timestamps."""
    registry = CollectorRegistry()
    draft_counter = Counter("vllm:spec_decode_num_drafts", "Draft steps", ["engine"], registry=registry)
    accepted_counter = Counter(
        "vllm:spec_decode_num_accepted_tokens_per_pos",
        "Accepted draft tokens",
        ["engine", "position"],
        registry=registry,
    )
    for engine, count in enumerate(drafts):
        draft_counter.labels(str(engine)).inc(count)
        for position, value in enumerate(accepted[engine]):
            accepted_counter.labels(str(engine), str(position)).inc(value)
    text = generate_latest(registry).decode()
    if created is None:
        return "\n".join(line for line in text.splitlines() if "_created" not in line)
    # Fixed timestamps make this independent of wall-clock time.
    return "\n".join(
        line.rsplit(" ", 1)[0] + f" {created}" if "_created{" in line else line for line in text.splitlines()
    )


@pytest.mark.parametrize("created", [None, 1_700_000_000])
def test_analysis_metrics_sums_counter_values_only(created):
    text = _counter_metrics([10, 30], [[8, 3, 1], [12, 6, 2]], created)
    assert metrics.analysis_metrics(text, 2) == (40, [20, 9])


def test_measure_acceptance_rate_excludes_warmup(monkeypatch):
    snapshots = iter(
        [
            _counter_metrics([10, 20], [[8, 4], [10, 6]]),
            _counter_metrics([20, 50], [[16, 7], [22, 12]]),
        ]
    )
    monkeypatch.setattr(metrics, "fetch_metrics", lambda server: next(snapshots))
    baseline = metrics.capture_baseline(object(), 2)
    assert metrics.measure_acceptance_rate(object(), 2, baseline) == (0.5, [0.5, 0.225])


def test_measure_acceptance_rate_with_no_new_drafts(monkeypatch):
    monkeypatch.setattr(metrics, "fetch_metrics", lambda server: _counter_metrics([10], [[8, 4]]))
    assert metrics.measure_acceptance_rate(object(), 2, (10, [8, 4])) == (0.0, [0.0, 0.0])


@pytest.mark.parametrize(
    "baseline, drafts, accepted",
    [
        pytest.param((20, [8, 4]), [10], [[8, 4]], id="draft-counter-reset"),
        pytest.param((10, [8, 4]), [20], [[7, 5]], id="accepted-counter-reset"),
    ],
)
def test_measure_acceptance_rate_rejects_counter_resets(monkeypatch, baseline, drafts, accepted):
    monkeypatch.setattr(metrics, "fetch_metrics", lambda server: _counter_metrics(drafts, accepted))
    with pytest.raises(ValueError, match="decreased"):
        metrics.measure_acceptance_rate(object(), 2, baseline)


@pytest.mark.parametrize("accepted", [[], [1], [1, 2, 3]])
def test_measure_acceptance_rate_rejects_baseline_position_mismatch(monkeypatch, accepted):
    monkeypatch.setattr(metrics, "fetch_metrics", lambda server: _counter_metrics([20], [[10, 5]]))
    with pytest.raises(ValueError, match="position count"):
        metrics.measure_acceptance_rate(object(), 2, (10, accepted))
