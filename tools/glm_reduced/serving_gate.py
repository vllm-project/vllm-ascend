# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation for fixed-output and serving-performance regressions.

Calibration is explicit. The nightly path never writes a baseline.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import statistics
from pathlib import Path

FORMAT = "glm-reduced-serving-v1"
CALIBRATION_STARTS = 3
CALIBRATION_REPEATS = 5
NIGHTLY_REPEATS = 3
MAX_RELATIVE_MAD = 0.03
REGRESSION_FRACTION = 0.10
METRICS = ("output_throughput", "mean_ttft_ms", "mean_tpot_ms")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def checkpoint_identity(model: str | Path) -> str:
    """Identity survives an identical rebuild in a different directory/time."""
    manifest = json.loads((Path(model) / "reduction_manifest.json").read_text(encoding="utf-8"))
    return digest(
        {
            "reduction": manifest["reduction"],
            "files": {entry["name"]: entry["sha256"] for entry in manifest["output"]["files"]},
            "tensor_sha256": manifest["output"]["tensor_sha256"],
        }
    )


def normalized_completion(response: dict, request: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Expected exactly one completion")
    choice = choices[0]
    if choice.get("index") != 0 or choice.get("finish_reason") != "length":
        raise ValueError("Missing index or truncated/early completion")
    text = choice.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty or invalid generated text")
    usage = response.get("usage", {})
    if usage.get("completion_tokens") != request["max_tokens"]:
        raise ValueError("Incomplete output token count")
    if usage.get("prompt_tokens") != len(request["prompt"]):
        raise ValueError("Input token count differs from fixed request")
    return text.replace("\r\n", "\n")


def compare_output(actual: str, expected: str) -> None:
    if actual != expected:
        diff = "\n".join(difflib.unified_diff(expected.splitlines(), actual.splitlines(), "baseline", "actual"))
        first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
        raise ValueError(f"Output regression at character {first}:\n{diff}")


def positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Missing/non-finite/non-positive {name}: {value!r}")
    return float(value)


def validate_perf(result: dict, workload: dict) -> dict[str, float]:
    count = workload["num_prompts"]
    if result.get("completed") != count:
        raise ValueError("Not all benchmark requests completed")
    for key, size in (("total_input_tokens", "random_input_len"), ("total_output_tokens", "random_output_len")):
        if result.get(key) != count * workload[size]:
            raise ValueError(f"Unexpected {key}; fixed-length workload was not executed")
    # Detailed lengths prevent one short and one long response cancelling out.
    for key, size in (("input_lens", "random_input_len"), ("output_lens", "random_output_len")):
        lengths = result.get(key)
        if not isinstance(lengths, list) or len(lengths) != count or any(n != workload[size] for n in lengths):
            raise ValueError(f"Invalid per-request {key}")
    return {key: positive_number(result.get(key), key) for key in METRICS}


def aggregate_perf(results: list[dict], workload: dict, *, repeats: int, calibrating: bool) -> dict:
    if len(results) != repeats:
        raise ValueError(f"Expected {repeats} performance rounds, got {len(results)}")
    samples = [validate_perf(result, workload) for result in results]
    medians = {key: statistics.median(row[key] for row in samples) for key in METRICS}
    relative_mad = {
        key: statistics.median(abs(row[key] - medians[key]) for row in samples) / medians[key] for key in METRICS
    }
    if calibrating and any(value > MAX_RELATIVE_MAD for value in relative_mad.values()):
        raise ValueError(f"Unstable calibration: relative MAD exceeds {MAX_RELATIVE_MAD}: {relative_mad}")
    return {"median": medians, "relative_mad": relative_mad}


def compare_perf(actual: dict, baseline: dict) -> None:
    failures = []
    for key in METRICS:
        expected = positive_number(baseline.get(key), f"baseline {key}")
        value = positive_number(actual.get(key), key)
        if key == "output_throughput":
            passed = value >= expected * (1 - REGRESSION_FRACTION)
        else:
            passed = value <= expected * (1 + REGRESSION_FRACTION)
        if not passed:
            failures.append(f"{key}: actual={value}, baseline={expected}, allowed regression=10%")
    if failures:
        raise ValueError("Performance regression: " + "; ".join(failures))


def validate_baseline(baseline: dict, identity: dict, suite: dict) -> None:
    if baseline.get("format") != FORMAT or baseline.get("identity") != identity:
        raise ValueError("Baseline format/model/hardware/engine/workload identity mismatch")
    calibration = baseline.get("calibration", {})
    if calibration.get("starts") != CALIBRATION_STARTS or calibration.get("repeats") != CALIBRATION_REPEATS:
        raise ValueError("Baseline has no complete 3-start/5-repeat calibration")
    ids = {prompt["id"] for prompt in suite["prompts"]}
    if set(baseline.get("outputs", {})) != ids:
        raise ValueError("Baseline output inventory mismatch")
    if any(not isinstance(s, str) or not s.strip() for s in baseline["outputs"].values()):
        raise ValueError("Baseline contains empty output")
    if set(baseline.get("performance", {})) != set(suite["workloads"]):
        raise ValueError("Baseline performance workload inventory mismatch")
    for name in suite["workloads"]:
        record = baseline["performance"][name]
        for key in METRICS:
            positive_number(record["median"].get(key), f"baseline {name}.{key}")
            noise = record["relative_mad"].get(key)
            if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not 0 <= noise <= MAX_RELATIVE_MAD:
                raise ValueError(f"Invalid calibration dispersion: {name}.{key}")
