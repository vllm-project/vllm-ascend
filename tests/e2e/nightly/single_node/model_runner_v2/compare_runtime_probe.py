# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Validate captured V1/V2 runtime paths and KV-cache probe results."""

import argparse
import json
from pathlib import Path


def metric_value(metrics, name):
    return sum(float(metric.get("value", 0.0)) for metric in metrics if metric["name"] == name)


def metric_delta(before, after, name):
    return metric_value(after, name) - metric_value(before, name)


def load_runs(directory):
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def matrix_pair(left, right):
    left_workers = left["runtime_snapshot"]
    right_workers = right["runtime_snapshot"]
    v1 = left_workers[0]
    v2 = right_workers[0]

    def graph_evidence(snapshot):
        managers = {
            name: value
            for name, value in snapshot["graph_managers"].items()
            if "graph" in name.lower() and isinstance(value, str) and "." in value and not value.startswith("builtins.")
        }
        return bool(managers)

    return {
        "spec_tokens": left["spec_tokens"],
        "graph": left["graph"],
        "v1_runner_class": v1["runner_class"],
        "v2_runner_class": v2["runner_class"],
        "v1_drafter_class": v1["drafter_class"],
        "v2_speculator_class": v2["speculator_class"],
        "v1_rejection_sampler_class": v1["rejection_sampler_class"],
        "v2_rejection_sampler_class": v2["rejection_sampler_class"],
        "v1_graph_managers": v1["graph_managers"],
        "v2_graph_managers": v2["graph_managers"],
        "v1_kv_cache_groups": v1["kv_cache_groups"],
        "v2_kv_cache_groups": v2["kv_cache_groups"],
        "both_tp_workers_agree": left_workers[0] == left_workers[1] and right_workers[0] == right_workers[1],
        "assembly_roles_match": v1["drafter_class"] is not None
        and v1["speculator_class"] is None
        and v2["drafter_class"] is None
        and v2["speculator_class"] is not None,
        "graph_capture_present": (not left["graph"]) or (graph_evidence(v1) and graph_evidence(v2)),
    }


def kv_cache_pair(v1, v2):
    def cases(run):
        return {case["name"]: case for case in run["kv_cache_check"]["cases"]}

    v1_cases, v2_cases = cases(v1), cases(v2)
    rows = []
    for name in ("seed", "repeat", "shared_prefix"):
        left, right = v1_cases[name], v2_cases[name]
        rows.append(
            {
                "case": name,
                "v1_prefix_cache_queries": metric_delta(
                    left["metrics_before"], left["metrics_after"], "vllm:prefix_cache_queries"
                ),
                "v1_prefix_cache_hits": metric_delta(
                    left["metrics_before"], left["metrics_after"], "vllm:prefix_cache_hits"
                ),
                "v2_prefix_cache_queries": metric_delta(
                    right["metrics_before"], right["metrics_after"], "vllm:prefix_cache_queries"
                ),
                "v2_prefix_cache_hits": metric_delta(
                    right["metrics_before"], right["metrics_after"], "vllm:prefix_cache_hits"
                ),
                "v1_output": left["output"]["token_ids"],
                "v2_output": right["output"]["token_ids"],
            }
        )
    seed_v1 = v1_cases["seed"]["output"]["token_ids"]
    repeat_v1 = v1_cases["repeat"]["output"]["token_ids"]
    seed_v2 = v2_cases["seed"]["output"]["token_ids"]
    repeat_v2 = v2_cases["repeat"]["output"]["token_ids"]
    return {
        "v1_exact_repeat_stable": seed_v1 == repeat_v1,
        "v2_exact_repeat_stable": seed_v2 == repeat_v2,
        "v1_cache_hit_observed": rows[1]["v1_prefix_cache_hits"] > 0,
        "v2_cache_hit_observed": rows[1]["v2_prefix_cache_hits"] > 0,
        "cross_runner_outputs_match": all(row["v1_output"] == row["v2_output"] for row in rows),
        "cases": rows,
        "v1_runtime": v1["runtime_snapshot"][0],
        "v2_runtime": v2["runtime_snapshot"][0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--kv-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = load_runs(args.matrix_dir)
    by_config = {(run["runner"], run["spec_tokens"], run["graph"]): run for run in runs}
    matrix = []
    for (runner, spec, graph), left in by_config.items():
        if runner != 1:
            continue
        right = by_config.get((2, spec, graph))
        if right is None:
            raise ValueError(f"Missing V2 matrix result for spec={spec}, graph={graph}")
        matrix.append(matrix_pair(left, right))

    kv_runs = {run["runner"]: run for run in load_runs(args.kv_cache_dir)}
    kv_cache = kv_cache_pair(kv_runs[1], kv_runs[2])
    result = {
        "matrix": matrix,
        "kv_cache": kv_cache,
        "matrix_checks_pass": all(
            row["both_tp_workers_agree"] and row["assembly_roles_match"] and row["graph_capture_present"]
            for row in matrix
        ),
        "kv_cache_checks_pass": all(
            value
            for key, value in kv_cache.items()
            if key.endswith("_observed") or key.endswith("_stable") or key == "cross_runner_outputs_match"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
