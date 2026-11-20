from collections.abc import Callable

import requests
from prometheus_client.parser import text_string_to_metric_families


class MetricsEndpoint:
    """Minimal ``url_for`` compatible endpoint for a vLLM metrics server."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def url_for(self, *parts: str) -> str:
        path = "/".join(str(part).strip("/") for part in parts)
        return f"http://{self.host}:{self.port}/{path}"


def metrics_endpoints(server) -> list:
    """Normalize a single metrics server or an iterable of servers into a list."""
    if isinstance(server, (list, tuple)):
        return list(server)
    return [server]


def fetch_metrics(server) -> str:
    """Fetch /metrics endpoint content as text."""
    url = server.url_for("metrics")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def analysis_metrics(metrics_text: str, num_speculative_tokens: int) -> tuple[int, list[int]]:
    """Parse prometheus text and return (num_drafts, num_accepted_tokens_per_pos)."""
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for family in text_string_to_metric_families(metrics_text):
        if family.name == "vllm:spec_decode_num_drafts":
            for sample in family.samples:
                num_drafts += sample.value
        elif family.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for sample in family.samples:
                pos = int(sample.labels["position"])
                if 0 <= pos < num_speculative_tokens:
                    num_accepted_tokens_per_pos[pos] += sample.value
    return int(num_drafts), num_accepted_tokens_per_pos


def _sum_metrics(
    server,
    num_speculative_tokens: int,
) -> tuple[int, list[int]]:
    """Sum spec-decode counters across all endpoints in ``server``.

    External-DP deployments route one workload across several decoder ranks, so
    every participating decoder metrics endpoint contributes to the same
    sampling window.
    """
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for endpoint in metrics_endpoints(server):
        endpoint_drafts, endpoint_accepted = analysis_metrics(fetch_metrics(endpoint), num_speculative_tokens)

        # TODO: temporary smoke diagnostics, remove before merging
        print(f"[debug] endpoint={endpoint.host}:{endpoint.port} drafts={endpoint_drafts} accepted={endpoint_accepted}")

        num_drafts += endpoint_drafts
        for position, accepted in enumerate(endpoint_accepted):
            num_accepted_tokens_per_pos[position] += accepted
    return int(num_drafts), num_accepted_tokens_per_pos


def capture_baseline(
    server,
    num_speculative_tokens: int,
    warmup_fn: Callable | None = None,
) -> tuple[int, list[int]]:
    """Run warmup and capture baseline (num_drafts, num_accepted_tokens_per_pos).

    ``server`` may be a single metrics server or a list of metrics servers.
    Call this BEFORE the actual benchmark/test requests. The returned tuple
    should be passed to measure_acceptance_rate() afterwards.
    """
    if warmup_fn is not None:
        warmup_fn()
    return _sum_metrics(server, num_speculative_tokens)


def measure_acceptance_rate(
    server,
    num_speculative_tokens: int,
    baseline: tuple[int, list[int]],
) -> tuple[float, list[float]]:
    """Fetch final metrics, subtract baseline, return (pos0_rate, all_rates)."""
    base_drafts, base_accepted = baseline

    num_drafts, num_accepted_tokens_per_pos = _sum_metrics(server, num_speculative_tokens)

    # TODO: temporary smoke diagnostics, remove before merging
    print(f"[debug] total base_drafts={base_drafts} final_drafts={num_drafts} positions={num_speculative_tokens}")

    num_drafts -= base_drafts
    for i in range(len(num_accepted_tokens_per_pos)):
        if i < len(base_accepted):
            num_accepted_tokens_per_pos[i] -= base_accepted[i]

    if num_drafts <= 0:
        raise AssertionError(
            "No speculative decoding drafts were observed in the sampling window; "
            "verify the accuracy workload used the decode-side speculative configuration."
        )

    acceptance_per_pos = [v / num_drafts for v in num_accepted_tokens_per_pos]

    pos0_rate = acceptance_per_pos[0] if acceptance_per_pos else 0.0
    print("-" * 50)
    print(f"{num_drafts=}, {num_accepted_tokens_per_pos=}")
    print("acceptance rate:", acceptance_per_pos)
    print("-" * 50)
    return pos0_rate, acceptance_per_pos


def validate_acceptance_rate(actual: float, baseline: float, tolerance: float = 0.05) -> None:
    """Assert actual (pos0) is within ±tolerance of baseline."""
    lower = baseline * (1 - tolerance)
    upper = baseline * (1 + tolerance)
    assert lower <= actual <= upper, (
        f"acceptance rate pos0: {actual:.4f} not within ±{tolerance:.0%} of baseline {baseline:.4f} "
        f"(range: {lower:.4f} ~ {upper:.4f})"
    )


def validate_acceptance_rates(
    actual_rates: list[float],
    baseline_rates: float | list[float],
    tolerance: float = 0.05,
) -> None:
    """Check the minimum acceptance rate for each configured position."""
    if isinstance(baseline_rates, (int, float)):
        expected_rates = [float(baseline_rates)]
    else:
        expected_rates = [float(rate) for rate in baseline_rates]

    assert expected_rates, "acceptance rate baseline must contain at least one position"
    assert len(actual_rates) >= len(expected_rates), (
        f"acceptance rate position count is insufficient: actual={len(actual_rates)}, baseline={len(expected_rates)}"
    )
    for position, (actual, baseline) in enumerate(zip(actual_rates, expected_rates)):
        lower = baseline * (1 - tolerance)
        assert actual >= lower, (
            f"acceptance rate pos{position}: {actual:.4f} is below baseline {baseline:.4f} "
            f"with tolerance {tolerance:.0%} (minimum: {lower:.4f})"
        )
