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


def histogram_js(left, right):
    left_total, right_total = sum(left), sum(right)
    if len(left) != len(right) or left_total == 0 or right_total == 0:
        return None
    probabilities_left = [value / left_total for value in left]
    probabilities_right = [value / right_total for value in right]
    midpoint = [
        (left_value + right_value) / 2 for left_value, right_value in zip(probabilities_left, probabilities_right)
    ]

    def divergence(probabilities, reference):
        return math.fsum(
            probability * math.log(probability / middle)
            for probability, middle in zip(probabilities, reference)
            if probability > 0
        )

    return (divergence(probabilities_left, midpoint) + divergence(probabilities_right, midpoint)) / 2


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
        "acceptance_length_histogram_js": histogram_js(
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
    runtime_rows: list[dict[str, Any]] = []
    for run in runs:
        for rank, runtime in enumerate(run.get("runtime_snapshot", [])):
            runtime_rows.append(
                {
                    "runner": run["runner"],
                    "spec_tokens": run["spec_tokens"],
                    "graph": run["graph"],
                    "rank": rank,
                    "use_v2_model_runner": runtime["use_v2_model_runner"],
                    "runner_class": runtime["runner_class"],
                    "speculator_class": runtime["speculator_class"],
                    "drafter_class": runtime["drafter_class"],
                    "rejection_sampler_class": runtime["rejection_sampler_class"],
                    "graph_managers": json.dumps(runtime["graph_managers"], sort_keys=True),
                    "kv_cache_group_count": len(runtime["kv_cache_groups"]),
                    "block_tables_class": runtime["block_tables_class"],
                }
            )
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
            sequence_matches = [
                request1["token_ids"] == request2["token_ids"]
                for trial1, trial2 in zip(perf1["trials"], perf2["trials"])
                for request1, request2 in zip(trial1["requests"], trial2["requests"])
            ]
            performance.append(
                {
                    "spec_tokens": spec,
                    "graph": graph,
                    "batch_size": perf1["batch_size"],
                    "greedy_sequence_matches": sum(sequence_matches),
                    "greedy_sequence_comparisons": len(sequence_matches),
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
    write_csv(args.output / "runtime.csv", runtime_rows)
    summary = {
        "runs": [{key: run.get(key) for key in ("runner", "spec_tokens", "graph", "status", "error")} for run in runs],
        "accuracy": precision,
        "performance": performance,
        "acceptance": acceptance_rows,
        "acceptance_comparison": acceptance_comparison,
        "runtime": runtime_rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Qwen3.5-27B dense MTP V1/V2 对比",
        "",
        "精度与性能原始数据来自传入测量目录的 JSON；失败或未完成的配置不纳入性能对比。",
        "",
        "采用离线引擎测量，TTFT 与端到端延迟来自 RequestStateStats，不含 HTTP 传输。",
        "top-20 仅在两端历史 token 相同的位置比较；未测量全词表 KL。",
        "HBM 为设备占用快照，Torch allocator 峰值单列；二者不等价。",
        "",
        "## 运行状态",
        "",
        "| Runner | MTP | Graph | 状态 |",
        "|---|---|---|---|",
    ]
    for run in runs:
        lines.append(f"| V{run['runner']} | {run['spec_tokens']} | {run['graph']} | {run['status']} |")
    lines += ["", "## Greedy token 一致性", "", "| MTP | Graph | 一致/总数 |", "|---|---|---|"]
    for spec, graph in sorted({(row["spec_tokens"], row["graph"]) for row in precision}):
        rows = [
            row for row in precision if (row["spec_tokens"], row["graph"]) == (spec, graph) and row["temperature"] == 0
        ]
        lines.append(f"| {spec} | {graph} | {sum(row['tokens_identical'] for row in rows)}/{len(rows)} |")
    lines += [
        "",
        "## 投机接受率与接受长度分布",
        "",
        "| MTP | Graph | 温度 | 步数 V1/V2 | V1 接受率 | V2 接受率 | V1 平均长度 | "
        "V2 平均长度 | 直方图 TVD | 直方图 JS | 最大逐位置差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    accuracy_acceptance = [row for row in acceptance_comparison if row["phase"] == "accuracy"]
    for row in accuracy_acceptance:
        lines.append(
            f"| {row['spec_tokens']} | {row['graph']} | {row['temperature']} | "
            f"{row['v1_draft_steps']}/{row['v2_draft_steps']} | "
            f"{row['v1_acceptance_rate']:.2%} | {row['v2_acceptance_rate']:.2%} | "
            f"{row['v1_mean_acceptance_length_including_bonus']:.3f} | "
            f"{row['v2_mean_acceptance_length_including_bonus']:.3f} | "
            f"{row['acceptance_length_histogram_tvd']:.4f} | "
            f"{row['acceptance_length_histogram_js']:.4f} | "
            f"{row['max_position_acceptance_rate_delta']:.2%} |"
        )
    lines += [
        "",
        "## 性能",
        "",
        "| MTP | Graph | Batch | V1 tokens/s | V2 tokens/s | V2/V1 |",
        "|---|---|---|---|---|---|",
    ]
    for row in performance:
        lines.append(
            f"| {row['spec_tokens']} | {row['graph']} | {row['batch_size']} | "
            f"{row['v1_output_tokens_per_second']:.2f} | {row['v2_output_tokens_per_second']:.2f} | "
            f"{row['v2_over_v1_throughput']:.3f} |"
        )
    lines += [
        "",
        "## 判定边界",
        "",
        "greedy token 不一致属于本次精确一致性检查失败。non-greedy 不使用逐 token 相同作为判据；",
        "其精度由接受率、逐位置接受率、接受长度直方图分布和单独 logits probe 判断。",
        "性能数值为预热后三次测量的中位数；没有预设社区验收阈值，不把任意阈值伪装成任务要求。",
        "完整精度、接受长度分布、TTFT、引擎端到端延迟、HBM 与 allocator 峰值见 CSV 和 summary.json。",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
