# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Compare persisted arms using the pinned upstream numerical rule, not text alone."""

import argparse
import ast
import hashlib
import json
import math
import warnings
from pathlib import Path

import regex as re


def load_upstream_rule(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check_logprobs_close"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    namespace = {"warnings": warnings}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["check_logprobs_close"], hashlib.sha256(source.encode()).hexdigest()


def token_id(value):
    match = re.fullmatch(r"(?:token_id:)?(\d+)", str(value))
    if not match:
        raise AssertionError(f"Server did not return unambiguous token-ID logprobs: {value!r}")
    return int(match[1])


def normalize(record):
    response = record["response"]
    choice = response["choices"][0]
    ids = choice.get("token_ids")
    if not ids:
        raise AssertionError("Missing generated token IDs")
    logs = choice.get("logprobs", {}).get("top_logprobs")
    if not logs or len(logs) != len(ids):
        raise AssertionError("Generated tokens and logprobs must have matching lengths")
    result = (ids, choice["text"], [{token_id(k): v for k, v in step.items()} for step in logs])
    assert all(math.isfinite(value) for step in result[2] for value in step.values()), "Non-finite generated logprob"
    prompt_logs = choice.get("prompt_logprobs")
    if record.get("request", {}).get("prompt_logprobs") is not None:
        assert prompt_logs, "Requested fixed-history prompt logprobs are missing from the choice"
    if prompt_logs is not None:
        assert len(prompt_logs) == response["usage"]["prompt_tokens"]
        result += ([None if step is None else {int(k): v for k, v in step.items()} for step in prompt_logs],)
        assert all(
            math.isfinite(value["logprob"] if isinstance(value, dict) else value)
            for step in result[3]
            if step is not None
            for value in step.values()
        ), "Non-finite prompt logprob"
    assert len(ids) == response["usage"]["completion_tokens"]
    return result


def read_records(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len({record["key"] for record in records}) == len(records), "Duplicate result IDs"
    return {record["key"]: record for record in records}


def compare_precision(left_path, right_path, rule_path):
    left, right = read_records(left_path), read_records(right_path)
    assert left.keys() == right.keys(), "Both arms must cover exactly the same cases"
    rule, rule_sha = load_upstream_rule(rule_path)
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        assert a["input_sha256"] == b["input_sha256"], key
        assert a["request"] == b["request"], key
        aa, bb = normalize(a), normalize(b)
        row = {"key": key, "generated_token_ids_equal": aa[0] == bb[0], "text_equal": aa[1] == bb[1]}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # The pinned helper breaks after its first check. With True,
                # that is always offset zero, hiding later divergences.
                rule(outputs_0_lst=[aa], outputs_1_lst=[bb], name_0="MRV1", name_1="MRV2", always_check_logprobs=False)
            row.update(passed=True, warnings=[str(w.message) for w in caught])
        except AssertionError as error:
            row.update(passed=False, error=str(error))
        rows.append(row)
    return {
        "upstream_rule": str(rule_path),
        "upstream_rule_sha256": rule_sha,
        "case_count": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "failed": sum(not row["passed"] for row in rows),
        "rows": rows,
        "scope": (
            "Pinned upstream reciprocal top-k/first-divergence rule plus fixed-history prompt top-k checks. "
            "Not a full-dataset score."
        ),
    }


def compare_performance(left_path, right_path):
    left, right = read_records(left_path), read_records(right_path)
    assert left.keys() == right.keys(), "Incomplete performance pairing"
    rows = []
    for key in sorted(left):
        a, b = left[key], right[key]
        assert a["request"] == b["request"] and a["input_sha256"] == b["input_sha256"], key
        assert a["client_concurrency"] == b["client_concurrency"]
        assert a["request_count"] == b["request_count"]
        rows.append(
            {
                "key": key,
                "v1_output_tps": a["output_tps"],
                "v2_output_tps": b["output_tps"],
                "v2_over_v1_throughput": b["output_tps"] / a["output_tps"],
                "v1_ttft_s": a["ttft_median_s"],
                "v2_ttft_s": b["ttft_median_s"],
                "v1_tpot_s": a["tpot_median_s"],
                "v2_tpot_s": b["tpot_median_s"],
                "v1_e2e_s": a["e2e_median_s"],
                "v2_e2e_s": b["e2e_median_s"],
            }
        )
    return {
        "paired_cells": len(rows),
        "rows": rows,
        "scope": (
            "Serving metrics. Engine iteration histograms and phase counters remain in raw metrics snapshots; "
            "not pure NPU kernel timing."
        ),
    }


def compare_provenance(left_root, right_root):
    left = sorted(left_root.glob("provenance-round*.json"))
    right = sorted(right_root.glob("provenance-round*.json"))
    assert left and len(left) == len(right), "Missing per-round runtime provenance"
    reference = None
    for path in left + right:
        item = json.loads(path.read_text())
        signature = {key: item[key] for key in ("code_hashes", "ascend_head", "vllm_head")}
        signature["image"] = item["container"]["image"]
        command = list(item["container"]["command"])
        name_index = command.index("--name") + 1
        command[name_index] = "PAIRED_CONTAINER"
        selector = "VLLM_USE_V2_MODEL_RUNNER=" + ("0" if path.parent == left_root else "1")
        assert command.count(selector) == 1, f"Wrong runner selector: {path}"
        command[command.index(selector)] = "VLLM_USE_V2_MODEL_RUNNER=PAIRED_ARM"
        signature["shared_command"] = command
        assert not any("vllm_ascend/worker" in mount["Destination"] for mount in item["container"]["mounts"])
        if reference is None:
            reference = signature
        assert signature == reference, f"Runtime source/image/configuration differs: {path}"
    return {"checked_arm_rounds": len(left) + len(right), "shared": reference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--upstream-utils", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--performance-only", action="store_true", help="Report separate serving-only followup; no precision claim"
    )
    args = parser.parse_args()
    result = {
        "provenance": compare_provenance(args.v1, args.v2),
        "precision": None
        if args.performance_only
        else compare_precision(args.v1 / "precision.jsonl", args.v2 / "precision.jsonl", args.upstream_utils),
        "performance": compare_performance(args.v1 / "performance.jsonl", args.v2 / "performance.jsonl"),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "precision_passed": result["precision"]["passed"] if result["precision"] else None,
                "precision_failed": result["precision"]["failed"] if result["precision"] else None,
                "paired_performance_cells": result["performance"]["paired_cells"],
            }
        )
    )
    if result["precision"] and result["precision"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
