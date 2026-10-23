#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Record or compare speculative-decoding refactor results on one NPU host.

Run the record mode once from the baseline checkout and once from the refactored
checkout, using the same family, model, mode, and environment. Then compare the
two JSON files. Serving TTFT/ITL results from ``vllm bench serve`` can be added
to each record with ``--serving-json`` and are compared when present.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from typing import Any

from tests.e2e.nightly.feature.test_spec_decode_refactor_parity import (
    DSPARK_CASE,
    EAGLE3_CASE,
    MTP_CASE,
    RunResult,
    SpecDecodeCase,
    _run,
)

CASES = {
    "mtp": MTP_CASE,
    "eagle3": EAGLE3_CASE,
    "dspark": DSPARK_CASE,
}
PERFORMANCE_TOLERANCE = 0.03
ACCEPTANCE_TOLERANCE = 0.05


def _serving_metrics(path: str | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metric_names = {
        "ttft_ms": ("mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms"),
        "itl_ms": ("mean_itl_ms", "median_itl_ms", "p99_itl_ms"),
        "request_throughput": ("request_throughput",),
        "output_throughput": ("output_throughput",),
    }
    result: dict[str, float] = {}
    for source_names in metric_names.values():
        for source_name in source_names:
            if source_name not in payload:
                continue
            value = payload[source_name]
            if isinstance(value, str):
                value = value.split()[0]
            result[source_name] = float(value)
    return result


def _record(args: argparse.Namespace) -> dict[str, Any]:
    case: SpecDecodeCase = CASES[args.family]
    updates: dict[str, Any] = {"name": f"{args.family}-{args.mode}"}
    if args.model:
        updates["model"] = args.model
    if args.draft_model:
        if args.family == "mtp":
            raise ValueError("--draft-model is not valid for MTP")
        updates["speculative_config"] = {
            **case.speculative_config,
            "model": args.draft_model,
        }
    case = replace(case, **updates)
    results = [_run(case, graph_mode=args.mode) for _ in range(args.repeats)]
    result = _aggregate_results(results)
    payload = asdict(result)
    payload.update(
        {
            "family": args.family,
            "model": case.model,
            "speculative_config": case.speculative_config,
            "graph_mode": args.mode,
            "repeat_count": len(results),
            "repeat_samples": [
                {
                    "acceptance_length": result.acceptance_length,
                    "elapsed_seconds": result.elapsed_seconds,
                    "hbm_in_use_bytes": result.hbm_in_use_bytes,
                    "output_tokens_per_second": result.output_tokens_per_second,
                }
                for result in results
            ],
            "serving": _serving_metrics(args.serving_json),
        }
    )
    return payload


def _aggregate_results(results: list[RunResult]) -> RunResult:
    """Use medians for short-run timing while requiring deterministic outputs."""
    baseline = results[0]
    for result in results[1:]:
        assert result.token_ids == baseline.token_ids, "repeated greedy token IDs differ"
        assert result.cache_signature == baseline.cache_signature, "repeated cache signature differs"
        assert result.family == baseline.family, "repeated speculation family differs"
        assert len(result.acceptance_per_pos) == len(baseline.acceptance_per_pos)

    elapsed_seconds = float(median(result.elapsed_seconds for result in results))
    output_token_count = sum(len(tokens) for tokens in baseline.token_ids)
    acceptance_per_pos = tuple(
        float(median(result.acceptance_per_pos[pos] for result in results))
        for pos in range(len(baseline.acceptance_per_pos))
    )
    return replace(
        baseline,
        acceptance_length=float(median(result.acceptance_length for result in results)),
        acceptance_per_pos=acceptance_per_pos,
        elapsed_seconds=elapsed_seconds,
        output_tokens_per_second=output_token_count / elapsed_seconds,
        hbm_in_use_bytes=int(median(result.hbm_in_use_bytes for result in results)),
    )


def _assert_performance_not_worse(
    baseline: float,
    candidate: float,
    *,
    name: str,
    higher_is_better: bool,
) -> None:
    if math.isclose(baseline, 0.0):
        return
    ratio = candidate / baseline
    if higher_is_better:
        assert ratio >= 1.0 - PERFORMANCE_TOLERANCE, f"{name} regressed: baseline={baseline}, candidate={candidate}"
    else:
        assert ratio <= 1.0 + PERFORMANCE_TOLERANCE, f"{name} regressed: baseline={baseline}, candidate={candidate}"


def _compare(baseline_path: str, candidate_path: str) -> None:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    for key in ("family", "model", "speculative_config", "graph_mode"):
        assert baseline[key] == candidate[key], f"{key} differs: {baseline[key]!r} vs {candidate[key]!r}"

    assert baseline["token_ids"] == candidate["token_ids"], "generated token IDs differ"
    assert baseline["cache_signature"] == candidate["cache_signature"], "target cache signature differs"
    baseline_acceptance = float(baseline["acceptance_length"])
    candidate_acceptance = float(candidate["acceptance_length"])
    acceptance_delta = abs(candidate_acceptance - baseline_acceptance) / baseline_acceptance
    assert acceptance_delta <= ACCEPTANCE_TOLERANCE, (
        f"acceptance changed: baseline={baseline_acceptance}, candidate={candidate_acceptance}"
    )
    baseline_rates = baseline["acceptance_per_pos"]
    candidate_rates = candidate["acceptance_per_pos"]
    assert len(baseline_rates) == len(candidate_rates), "per-position acceptance length differs"
    for pos, (baseline_rate, candidate_rate) in enumerate(zip(baseline_rates, candidate_rates)):
        rate_delta = abs(candidate_rate - baseline_rate) / max(abs(baseline_rate), 1e-12)
        assert rate_delta <= ACCEPTANCE_TOLERANCE, (
            f"acceptance at draft position {pos} changed: baseline={baseline_rate}, candidate={candidate_rate}"
        )

    _assert_performance_not_worse(
        float(baseline["output_tokens_per_second"]),
        float(candidate["output_tokens_per_second"]),
        name="offline output throughput",
        higher_is_better=True,
    )
    _assert_performance_not_worse(
        float(baseline["elapsed_seconds"]),
        float(candidate["elapsed_seconds"]),
        name="offline latency",
        higher_is_better=False,
    )
    _assert_performance_not_worse(
        float(baseline["hbm_in_use_bytes"]),
        float(candidate["hbm_in_use_bytes"]),
        name="HBM in use",
        higher_is_better=False,
    )

    baseline_serving = baseline.get("serving", {})
    candidate_serving = candidate.get("serving", {})
    if baseline_serving or candidate_serving:
        assert baseline_serving.keys() == candidate_serving.keys(), "serving metric keys differ"
        for metric, baseline_value in baseline_serving.items():
            candidate_value = candidate_serving[metric]
            higher_is_better = "throughput" in metric
            _assert_performance_not_worse(
                float(baseline_value),
                float(candidate_value),
                name=metric,
                higher_is_better=higher_is_better,
            )

    print(
        "parity PASS: "
        f"{baseline_path} -> {candidate_path}; "
        f"acceptance {baseline_acceptance:.4f} -> {candidate_acceptance:.4f}; "
        f"throughput {baseline['output_tokens_per_second']:.2f} -> "
        f"{candidate['output_tokens_per_second']:.2f} tok/s; "
        f"latency {baseline['elapsed_seconds']:.4f} -> {candidate['elapsed_seconds']:.4f} s; "
        f"HBM {baseline['hbm_in_use_bytes']} -> {candidate['hbm_in_use_bytes']} bytes"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--family", choices=sorted(CASES), required=True)
    record.add_argument("--mode", choices=("eager", "full_decode_only"), required=True)
    record.add_argument("--model")
    record.add_argument("--draft-model")
    record.add_argument("--serving-json", help="Output JSON from `vllm bench serve --save-result`.")
    record.add_argument("--repeats", type=int, default=3)
    record.add_argument("--output", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("baseline")
    compare.add_argument("candidate")

    args = parser.parse_args()
    if args.command == "record" and args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.command == "compare":
        _compare(args.baseline, args.candidate)
        return

    payload = _record(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {output}")


if __name__ == "__main__":
    main()
