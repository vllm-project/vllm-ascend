# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for the synthetic Flash numerical/performance regression test.

Frozen numerical references must come from a separately recorded known-good
run, never from the candidate under test. Synthetic weights provide numerical
regression coverage, not real-model task accuracy.
"""

import math
import statistics
from collections.abc import Sequence
from typing import Any


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return float(value)


def _token_ids(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty token-ID list")
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{label} contains an invalid token ID")


def _validate_snapshot(snapshot: Any, label: str) -> None:
    if not isinstance(snapshot, list) or not snapshot:
        raise ValueError(f"{label} must contain at least one request")
    for request_index, request in enumerate(snapshot):
        prefix = f"{label} request {request_index}"
        if not isinstance(request, dict):
            raise ValueError(f"{prefix} must be a dictionary")
        _token_ids(request.get("prompt_token_ids"), f"{prefix} prompt")
        _token_ids(request.get("token_ids"), f"{prefix} completion")
        steps = request.get("logprobs")
        if not isinstance(steps, list) or len(steps) != len(request["token_ids"]):
            raise ValueError(f"{prefix} must have one logprob distribution per completion token")
        for step_index, (token, step) in enumerate(zip(request["token_ids"], steps)):
            location = f"{prefix} step {step_index}"
            if not isinstance(step, dict) or len(step) < 2:
                raise ValueError(f"{location} needs at least two candidate logprobs")
            if str(token) not in step:
                raise ValueError(f"{location} is missing the generated token's logprob")
            for candidate, value in step.items():
                if not isinstance(candidate, str) or not candidate.isdecimal() or str(int(candidate)) != candidate:
                    raise ValueError(f"{location} has an invalid candidate token ID")
                if _finite_number(value, location) > 0:
                    raise ValueError(f"{location} contains a positive logprob")
            values = list(step.values())
            if max(values) == min(values):
                raise ValueError(f"{location} has a degenerate, constant candidate distribution")
            if math.fsum(math.exp(value) for value in values) > 1.000001:
                raise ValueError(f"{location} candidate probabilities sum to more than one")


def summarize_outputs(outputs: Sequence[Any]) -> list[dict[str, Any]]:
    """Serialize and validate all returned logprobs without vLLM/torch imports."""
    snapshot = []
    for request in outputs:
        if getattr(request, "finished", None) is not True or len(getattr(request, "outputs", ())) != 1:
            raise ValueError("Each request must finish with exactly one completion")
        completion = request.outputs[0]
        returned_steps = getattr(completion, "logprobs", None)
        if returned_steps is None:
            raise ValueError("Completion logprobs were not returned")
        snapshot.append(
            {
                "prompt_token_ids": list(getattr(request, "prompt_token_ids", None) or ()),
                "token_ids": list(completion.token_ids),
                "logprobs": [
                    {str(token): value.logprob for token, value in step.items()} if step is not None else {}
                    for step in returned_steps
                ],
            }
        )
    _validate_snapshot(snapshot, "output")
    return snapshot


def compare_snapshots(actual: Any, reference: Any, *, atol: float) -> None:
    """Compare a candidate against frozen values under identical contexts.

    Context matching is intentionally strict: a different token fails, even
    if its probability is close. After token divergence, subsequent distributions
    have different contexts and are not valid numerical comparisons. Investigate
    such a failure; do not silently bless it or regenerate the reference in CI.
    """
    tolerance = _finite_number(atol, "atol")
    if tolerance < 0:
        raise ValueError("atol must be nonnegative")
    _validate_snapshot(reference, "reference")
    _validate_snapshot(actual, "actual")
    assert len(actual) == len(reference), "Request count differs from the frozen reference"
    for request_index, (candidate, expected) in enumerate(zip(actual, reference)):
        assert candidate["prompt_token_ids"] == expected["prompt_token_ids"], (
            f"Request {request_index}: prompt context differs from the frozen reference"
        )
        assert candidate["token_ids"] == expected["token_ids"], (
            f"Request {request_index}: decode token path differs; near ties are not exempt "
            "because later decode contexts would differ"
        )
        for step_index, (candidate_step, expected_step) in enumerate(zip(candidate["logprobs"], expected["logprobs"])):
            assert candidate_step.keys() == expected_step.keys(), (
                f"Request {request_index}, step {step_index}: candidate token set differs from the frozen reference"
            )
            for token, expected_logprob in expected_step.items():
                location = f"Request {request_index}, step {step_index}, token {token}"
                difference = abs(candidate_step[token] - expected_logprob)
                assert difference <= tolerance, (
                    f"{location}: logprob difference {difference:.8g} exceeds atol={tolerance:.8g}; "
                    f"actual={candidate_step[token]:.8g}, reference={expected_logprob:.8g}"
                )


def summarize_performance(elapsed_seconds: Sequence[float], *, output_tokens_per_run: int) -> dict[str, Any]:
    """Summarize completed, warm steady-state batch calls (not engine startup).

    The caller must verify every call produced ``output_tokens_per_run`` tokens,
    and exclude warmups, logprob collection and graph-replay instrumentation.
    """
    if type(output_tokens_per_run) is not int or output_tokens_per_run <= 0:
        raise ValueError("output_tokens_per_run must be a positive integer")
    if len(elapsed_seconds) < 3:
        raise ValueError("At least three measured runs are required")
    samples = [_finite_number(value, "elapsed_seconds", positive=True) for value in elapsed_seconds]
    throughputs = [
        _finite_number(output_tokens_per_run / value, "output throughput", positive=True) for value in samples
    ]
    return {
        "output_tokens_per_run": output_tokens_per_run,
        "elapsed_seconds": samples,
        "output_tokens_s": throughputs,
        "median_elapsed_s": statistics.median(samples),
        "median_output_tokens_s": statistics.median(throughputs),
    }


def assert_performance(summary: dict[str, Any], *, minimum_output_tokens_s: float) -> None:
    """Check a measured median against an independently calibrated lower bound."""
    threshold = _finite_number(minimum_output_tokens_s, "minimum_output_tokens_s", positive=True)
    measured = summarize_performance(summary["elapsed_seconds"], output_tokens_per_run=summary["output_tokens_per_run"])
    # Recompute from samples: neither NaN nor a forged summary can bypass the gate.
    for key in ("median_elapsed_s", "median_output_tokens_s"):
        if _finite_number(summary[key], key, positive=True) != measured[key]:
            raise ValueError(f"{key} does not match the recorded timing samples")
    assert measured["median_output_tokens_s"] >= threshold, (
        f"Output throughput regression: median {measured['median_output_tokens_s']:.3f} tokens/s "
        f"is below {threshold:.3f} tokens/s; measured wall times={measured['elapsed_seconds']}"
    )
