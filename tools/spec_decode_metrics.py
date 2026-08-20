from typing import Callable, Optional

import requests
from prometheus_client.parser import text_string_to_metric_families


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


def capture_baseline(
    server,
    num_speculative_tokens: int,
    warmup_fn: Optional[Callable] = None,
) -> tuple[int, list[int]]:
    """Run warmup and capture baseline (num_drafts, num_accepted_tokens_per_pos).

    Call this BEFORE the actual benchmark/test requests. The returned tuple
    should be passed to measure_acceptance_rate() afterwards.
    """
    if warmup_fn is not None:
        warmup_fn()
    metrics_text = fetch_metrics(server)
    return analysis_metrics(metrics_text, num_speculative_tokens)


def measure_acceptance_rate(
    server,
    num_speculative_tokens: int,
    baseline: tuple[int, list[int]],
) -> tuple[float, list[float]]:
    """Fetch final metrics, subtract baseline, return (pos0_rate, all_rates)."""
    base_drafts, base_accepted = baseline

    metrics_text = fetch_metrics(server)
    num_drafts, num_accepted_tokens_per_pos = analysis_metrics(metrics_text, num_speculative_tokens)

    num_drafts -= base_drafts
    for i in range(len(num_accepted_tokens_per_pos)):
        if i < len(base_accepted):
            num_accepted_tokens_per_pos[i] -= base_accepted[i]

    if num_drafts > 0:
        acceptance_per_pos = [
            v / num_drafts for v in num_accepted_tokens_per_pos
        ]
    else:
        acceptance_per_pos = [0.0] * num_speculative_tokens

    pos0_rate = acceptance_per_pos[0] if acceptance_per_pos else 0.0
    print("-" * 50)
    print(f"{num_drafts=}, {num_accepted_tokens_per_pos=}")
    print("acceptance rate:", acceptance_per_pos)
    print("-" * 50)
    return pos0_rate, acceptance_per_pos


def validate_acceptance_rate(
    actual: float, baseline: float, tolerance: float = 0.05
) -> None:
    """Assert actual (pos0) is within ±tolerance of baseline."""
    lower = baseline * (1 - tolerance)
    upper = baseline * (1 + tolerance)
    assert lower <= actual <= upper, (
        f"acceptance rate pos0: {actual:.4f} not within ±{tolerance:.0%} of baseline {baseline:.4f} "
        f"(range: {lower:.4f} ~ {upper:.4f})"
    )
