import logging
from collections.abc import Callable
from dataclasses import dataclass

import requests
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)


@dataclass
class EndpointSnapshot:
    """Spec-decode counters, request total, and observed positions from one metrics endpoint."""

    endpoint: str
    drafts: int
    accepted_per_pos: list[int]
    requests_total: int
    observed_positions: int


@dataclass
class Baseline:
    """Counter snapshot captured before the accuracy workload starts."""

    drafts: int
    endpoints: list[EndpointSnapshot]

    @property
    def endpoint_names(self) -> list[str]:
        return [snapshot.endpoint for snapshot in self.endpoints]


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


def _parse_metrics(metrics_text: str, num_speculative_tokens: int) -> tuple[int, list[int], int, int]:
    """Return (num_drafts, num_accepted_tokens_per_pos, num_requests_total, observed_positions)."""
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    num_requests_total = 0
    observed_positions = 0
    for family in text_string_to_metric_families(metrics_text):
        if family.name == "vllm:spec_decode_num_drafts":
            for sample in family.samples:
                num_drafts += sample.value
        elif family.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for sample in family.samples:
                pos = int(sample.labels["position"])
                observed_positions = max(observed_positions, pos + 1)
                if 0 <= pos < num_speculative_tokens:
                    num_accepted_tokens_per_pos[pos] += sample.value
        elif family.name == "vllm:request_success":
            for sample in family.samples:
                num_requests_total += sample.value
    return int(num_drafts), num_accepted_tokens_per_pos, int(num_requests_total), observed_positions


def _snapshot_endpoints(server, num_speculative_tokens: int) -> list[EndpointSnapshot]:
    """Read counters from every metrics endpoint that serves the sampled workload."""
    snapshots = []
    for endpoint in metrics_endpoints(server):
        drafts, accepted, requests_total, observed_positions = _parse_metrics(
            fetch_metrics(endpoint), num_speculative_tokens
        )
        snapshots.append(
            EndpointSnapshot(
                endpoint=f"{endpoint.host}:{endpoint.port}",
                drafts=drafts,
                accepted_per_pos=accepted,
                requests_total=requests_total,
                observed_positions=observed_positions,
            )
        )
    return snapshots


def capture_baseline(
    server,
    num_speculative_tokens: int,
    warmup_fn: Callable | None = None,
) -> Baseline:
    """Run warmup and capture the counter snapshot before the benchmark workload.

    ``server`` may be a single metrics server or a list of metrics servers.
    Call this BEFORE the actual benchmark/test requests. The returned snapshot
    should be passed to measure_acceptance_rate() afterwards.
    """
    if warmup_fn is not None:
        warmup_fn()
    endpoints = _snapshot_endpoints(server, num_speculative_tokens)
    drafts = sum(snapshot.drafts for snapshot in endpoints)
    if drafts == 0:
        logger.warning(
            "spec-decode baseline captured no drafts: the warmup request did not reach the "
            "decode-side speculative path; verify the metrics endpoints belong to the decoder"
        )
    return Baseline(drafts, endpoints)


def measure_acceptance_rate(
    server,
    num_speculative_tokens: int,
    baseline: Baseline,
) -> tuple[float, list[float]]:
    """Fetch final metrics, subtract baseline, return (pos0_rate, all_rates)."""
    endpoints = _snapshot_endpoints(server, num_speculative_tokens)
    base_by_endpoint = {snapshot.endpoint: snapshot for snapshot in baseline.endpoints}
    final_names = {snapshot.endpoint for snapshot in endpoints}
    missing = [name for name in baseline.endpoint_names if name not in final_names]
    if missing:
        logger.warning("spec-decode metrics endpoints disappeared since baseline: %s", missing)

    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    num_drafts = 0
    for snapshot in endpoints:
        base = base_by_endpoint.get(snapshot.endpoint)
        draft_delta = snapshot.drafts - (base.drafts if base else 0)
        num_drafts += draft_delta
        accepted_delta = []
        for position in range(num_speculative_tokens):
            value = snapshot.accepted_per_pos[position]
            if base is not None:
                value -= base.accepted_per_pos[position]
            num_accepted_tokens_per_pos[position] += value
            accepted_delta.append(value)
        logger.info(
            "spec-decode endpoint delta: %s drafts=%s accepted_per_pos=%s requests=%s",
            snapshot.endpoint,
            draft_delta,
            accepted_delta,
            snapshot.requests_total - (base.requests_total if base else 0),
        )

    final_drafts = sum(snapshot.drafts for snapshot in endpoints)
    if num_drafts <= 0:
        logger.error(
            "spec-decode sampled no drafts: baseline_drafts=%s final_drafts=%s positions=%s endpoints=%s; "
            "verify the accuracy workload used the decode-side speculative configuration",
            baseline.drafts,
            final_drafts,
            num_speculative_tokens,
            [snapshot.endpoint for snapshot in endpoints],
        )
        raise AssertionError(
            "No speculative decoding drafts were observed in the sampling window; "
            "verify the accuracy workload used the decode-side speculative configuration."
        )

    observed_positions = max((snapshot.observed_positions for snapshot in endpoints), default=0)
    if observed_positions < num_speculative_tokens:
        logger.error(
            "spec-decode position mismatch: metrics report %s positions but sampling expects %s "
            "(endpoints=%s); the acceptance_rate baseline length does not match the server's "
            "num_speculative_tokens",
            observed_positions,
            num_speculative_tokens,
            [snapshot.endpoint for snapshot in endpoints],
        )
        raise AssertionError(
            f"Metrics endpoints report {observed_positions} speculative positions but the sampling "
            f"window expects {num_speculative_tokens}; align the acceptance_rate baseline length "
            f"with the server's num_speculative_tokens."
        )

    acceptance_per_pos = [value / num_drafts for value in num_accepted_tokens_per_pos]
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
