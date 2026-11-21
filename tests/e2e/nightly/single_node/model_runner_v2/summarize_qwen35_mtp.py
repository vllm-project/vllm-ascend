# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Summarize artifacts from compare_qwen35_mtp.py without inventing thresholds."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def acceptance(record, spec_tokens):
    def counter(snapshot, suffix):
        return sum(metric.get("value", 0) for metric in snapshot if metric["name"].endswith(suffix))

    before, after = record["metrics_before"], record["metrics_after"]
    drafts = counter(after, "spec_decode_num_drafts") - counter(before, "spec_decode_num_drafts")
    proposed = counter(after, "spec_decode_num_draft_tokens") - counter(before, "spec_decode_num_draft_tokens")
    accepted = counter(after, "spec_decode_num_accepted_tokens") - counter(before, "spec_decode_num_accepted_tokens")
    vectors = []
    for snapshot in (before, after):
        vector = [0] * spec_tokens
        found = False
        for metric in snapshot:
            if metric["name"].endswith("spec_decode_num_accepted_tokens_per_pos"):
                found = True
                for index, value in enumerate(metric["values"][:spec_tokens]):
                    vector[index] += value
        vectors.append(vector if found else None)
    histogram = None
    cumulative = None
    if drafts > 0 and all(vector is not None for vector in vectors):
        cumulative = [right - left for left, right in zip(*vectors)]
        histogram = [drafts - cumulative[0]]
        histogram += [left - right for left, right in zip(cumulative, cumulative[1:])]
        histogram += [cumulative[-1]]
        if any(value < 0 for value in histogram) or sum(histogram) != drafts:
            raise ValueError("Inconsistent acceptance counters; inspect raw snapshots")
        if sum(index * value for index, value in enumerate(histogram)) != accepted:
            raise ValueError("Acceptance histogram disagrees with the accepted-token counter")
    return {
        "draft_steps": drafts,
        "draft_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "acceptance_rate": accepted / proposed if proposed > 0 else None,
        "mean_acceptance_length_including_bonus": 1 + accepted / drafts if drafts > 0 else None,
        "accepted_draft_length_histogram": histogram,
        "accepted_count_by_position": cumulative,
        "acceptance_rate_by_position": (
            [value / drafts for value in cumulative] if cumulative is not None and drafts > 0 else None
        ),
    }


def histogram_tvd(left, right):
    left_total, right_total = sum(left), sum(right)
    if len(left) != len(right) or left_total == 0 or right_total == 0:
        return None
    return 0.5 * math.fsum(
        abs(left_value / left_total - right_value / right_total) for left_value, right_value in zip(left, right)
    )


def acceptance_pair(left, right):
    rates_left = left["acceptance_rate_by_position"] or []
    rates_right = right["acceptance_rate_by_position"] or []
    rate_deltas = [abs(left_value - right_value) for left_value, right_value in zip(rates_left, rates_right)]
    return {
        "v1_draft_steps": left["draft_steps"],
        "v2_draft_steps": right["draft_steps"],
        "v1_acceptance_rate": left["acceptance_rate"],
        "v2_acceptance_rate": right["acceptance_rate"],
        "acceptance_rate_delta": right["acceptance_rate"] - left["acceptance_rate"],
        "v1_mean_acceptance_length_including_bonus": left["mean_acceptance_length_including_bonus"],
        "v2_mean_acceptance_length_including_bonus": right["mean_acceptance_length_including_bonus"],
        "acceptance_length_histogram_tvd": histogram_tvd(
            left["accepted_draft_length_histogram"], right["accepted_draft_length_histogram"]
        ),
        "max_position_acceptance_rate_delta": max(rate_deltas, default=None),
        "mean_position_acceptance_rate_delta": statistics.fmean(rate_deltas) if rate_deltas else None,
    }


def accuracy_pair(left, right):
    assert left["prompt_categories"] == right["prompt_categories"]
    assert len(left["accuracy"]) == len(right["accuracy"]) == 2
    records = []
    for case1, case2 in zip(left["accuracy"], right["accuracy"]):
        assert case1["temperature"] == case2["temperature"]
        assert len(case1["outputs"]) == len(case2["outputs"]) == len(left["prompt_categories"])
        for category, output1, output2 in zip(left["prompt_categories"], case1["outputs"], case2["outputs"]):
            overlaps = []
            for token1, token2, scores1, scores2 in zip(
                output1["token_ids"], output2["token_ids"], output1["logprobs"], output2["logprobs"]
            ):
                top1 = {token for token, value in scores1.items() if value["rank"] and value["rank"] <= 20}
                top2 = {token for token, value in scores2.items() if value["rank"] and value["rank"] <= 20}
                if top1 and top2:
                    overlaps.append(len(top1 & top2) / max(len(top1), len(top2)))
                if token1 != token2:
                    break
            records.append(
                {
                    "temperature": case1["temperature"],
                    "category": category,
                    "tokens_identical": output1["token_ids"] == output2["token_ids"],
                    "v1_tokens": len(output1["token_ids"]),
                    "v2_tokens": len(output2["token_ids"]),
                    "top20_compared_positions": len(overlaps),
                    "mean_top20_overlap_on_matching_histories": statistics.mean(overlaps) if overlaps else None,
                }
            )
    return records


def latency_and_memory(record):
    requests = [request for trial in record["trials"] for request in trial["requests"]]
    ttft = [
        request["metrics"]["first_token_latency"]
        for request in requests
        if request["metrics"] and request["metrics"]["first_token_latency"] > 0
    ]
    e2e = [
        request["metrics"]["last_token_ts"] - request["metrics"]["queued_ts"]
        for request in requests
        if request["metrics"] and request["metrics"]["last_token_ts"] > request["metrics"]["queued_ts"] > 0
    ]
    return {
        "median_ttft_seconds": statistics.median(ttft) if ttft else None,
        "median_engine_e2e_seconds": statistics.median(e2e) if e2e else None,
        "max_device_hbm_snapshot_bytes": max(item["device_hbm_used_bytes"] for item in record["memory_after_trials"]),
        "max_torch_peak_allocated_bytes": max(
            item["torch_peak_allocated_bytes"] for item in record["memory_after_trials"]
        ),
    }


def write_csv(path, records):
    if records:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.directory.glob("*.json"))]
    by_config = {(run["runner"], run["spec_tokens"], run["graph"]): run for run in runs}
    precision: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []
    acceptance_comparison: list[dict[str, Any]] = []
    for run in runs:
        for case in run.get("accuracy", []):
            acceptance_rows.append(
                {
                    "runner": run["runner"],
                    "spec_tokens": run["spec_tokens"],
                    "graph": run["graph"],
                    "phase": "accuracy",
                    "batch_size": None,
                    "temperature": case["temperature"],
                    **acceptance(case, run["spec_tokens"]),
                }
            )
        for case in run.get("performance", []):
            acceptance_rows.append(
                {
                    "runner": run["runner"],
                    "spec_tokens": run["spec_tokens"],
                    "graph": run["graph"],
                    "phase": "performance",
                    "batch_size": case["batch_size"],
                    "temperature": 0.0,
                    **acceptance(case, run["spec_tokens"]),
                }
            )
    acceptance_by_config = {
        (row["runner"], row["spec_tokens"], row["graph"], row["phase"], row["batch_size"], row["temperature"]): row
        for row in acceptance_rows
    }
    for key, left in acceptance_by_config.items():
        if key[0] != 1:
            continue
        right = acceptance_by_config.get((2, *key[1:]))
        if right is None:
            continue
        acceptance_comparison.append(
            {
                "spec_tokens": key[1],
                "graph": key[2],
                "phase": key[3],
                "batch_size": key[4],
                "temperature": key[5],
                **acceptance_pair(left, right),
            }
        )
    for (runner, spec, graph), left in by_config.items():
        if runner != 1:
            continue
        right = by_config.get((2, spec, graph))
        if right is None or left["status"] != "complete" or right["status"] != "complete":
            continue
        assert {key: value for key, value in left["arguments"].items() if key != "output"} == {
            key: value for key, value in right["arguments"].items() if key != "output"
        }
        assert len(left["performance"]) == len(right["performance"]) == len(left["arguments"]["batches"])
        precision.extend({"spec_tokens": spec, "graph": graph, **row} for row in accuracy_pair(left, right))
        for perf1, perf2 in zip(left["performance"], right["performance"]):
            assert perf1["batch_size"] == perf2["batch_size"]
            memory1, memory2 = latency_and_memory(perf1), latency_and_memory(perf2)
            assert len(perf1["trials"]) == len(perf2["trials"])
            performance.append(
                {
                    "spec_tokens": spec,
                    "graph": graph,
                    "batch_size": perf1["batch_size"],
                    "v1_output_tokens_per_second": perf1["median_output_tokens_per_second"],
                    "v2_output_tokens_per_second": perf2["median_output_tokens_per_second"],
                    "v2_over_v1_throughput": perf2["median_output_tokens_per_second"]
                    / perf1["median_output_tokens_per_second"],
                    "v1_min_trial_throughput": min(trial["output_tokens_per_second"] for trial in perf1["trials"]),
                    "v1_max_trial_throughput": max(trial["output_tokens_per_second"] for trial in perf1["trials"]),
                    "v2_min_trial_throughput": min(trial["output_tokens_per_second"] for trial in perf2["trials"]),
                    "v2_max_trial_throughput": max(trial["output_tokens_per_second"] for trial in perf2["trials"]),
                    **{f"v1_{key}": value for key, value in memory1.items()},
                    **{f"v2_{key}": value for key, value in memory2.items()},
                }
            )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "accuracy.csv", precision)
    write_csv(args.output / "performance.csv", performance)
    write_csv(args.output / "acceptance.csv", acceptance_rows)
    write_csv(args.output / "acceptance_comparison.csv", acceptance_comparison)
    summary = {
        "runs": [{key: run.get(key) for key in ("runner", "spec_tokens", "graph", "status", "error")} for run in runs],
        "accuracy": precision,
        "performance": performance,
        "acceptance": acceptance_rows,
        "acceptance_comparison": acceptance_comparison,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
