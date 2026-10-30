# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json

import pytest

from tests.e2e.runner_parity.compare import compare_performance, compare_precision, compare_provenance


def test_glm51_same_runtime_code(parity_root, expected_rounds):
    report = compare_provenance(parity_root / "v1", parity_root / "v2")
    assert report["checked_arm_rounds"] == 2 * expected_rounds


def test_glm51_paired_precision(parity_root, upstream_utils):
    protocol = json.loads((parity_root / "protocol.json").read_text())
    if protocol.get("stage") == "serving-only":
        pytest.skip("This capture explicitly contains serving only; validate the separate control capture")
    report = compare_precision(parity_root / "v1/precision.jsonl", parity_root / "v2/precision.jsonl", upstream_utils)
    assert report["case_count"] >= 78, "The complete bounded precision corpus must run on both arms"
    assert report["failed"] == 0, json.dumps([r for r in report["rows"] if not r["passed"]], ensure_ascii=False)


def test_glm51_paired_serving_performance(parity_root, expected_rounds):
    report = compare_performance(parity_root / "v1/performance.jsonl", parity_root / "v2/performance.jsonl")
    assert report["paired_cells"] == 9 * expected_rounds
    assert all(row["v1_output_tps"] > 0 and row["v2_output_tps"] > 0 for row in report["rows"])
    # Regression triage is explicit in the report; do not invent a bounty threshold.


def test_glm51_serving_ownership_evidence(parity_root):
    for arm in ("v1", "v2"):
        root = parity_root / arm
        assert not list(root.glob("foreign-npu-round*.json")), "Foreign workload requires a clean rerun"
        records = [json.loads(line) for path in root.glob("hbm-round*.jsonl") for line in path.read_text().splitlines()]
        assert len(records) >= 2, "Missing ownership monitoring"
        for state in records:
            assert state.get("expected_container_id"), "Explicit container ownership evidence is required"
            assert state["foreign_processes"] == [], "Foreign NPU process observed"
            assert state["healthy_devices"] == 8
