# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json

import regex as re
from compare import compare_precision
from evidence_checks import assert_control_runtime_and_ownership


def test_glm51_controlled_greedy(control_root, upstream_utils):
    report = compare_precision(
        control_root / "v1/phase-diagnostic/controlled-precision.jsonl",
        control_root / "v2/phase-diagnostic/controlled-precision.jsonl",
        upstream_utils,
    )
    assert report["case_count"] == 32
    assert report["failed"] == 0, json.dumps([row for row in report["rows"] if not row["passed"]])
    # Exact token equality and numeric deltas remain separate report fields.


def test_glm51_controlled_runtime_and_ownership(control_root):
    assert_control_runtime_and_ownership(control_root)


def test_glm51_controlled_scheduled_inputs_match(control_root):
    signatures = []
    for arm in ("v1", "v2"):
        data = json.loads((control_root / arm / "phase-diagnostic/steps.json").read_text())
        ranks = sorted(data["records"], key=lambda row: row["rank"])
        assert [row["rank"] for row in ranks] == list(range(8))
        per_rank = [
            [
                {key: step[key] for key in ("case", "scheduled", "scheduled_tokens", "scheduled_requests")}
                for step in row["steps"]
                if step["case"].startswith("precision/")
            ]
            for row in ranks
        ]
        assert per_rank[0] and all(row == per_rank[0] for row in per_rank)
        canonical = []
        for step in per_rank[0]:
            scheduled = []
            for request_id, count in step["scheduled"]:
                match = re.fullmatch(r"(\d+)-[0-9a-f]{8}", request_id)
                assert match, "Unrecognized request identity; do not guess its input position"
                scheduled.append((int(match.group(1)), count))
            # The counter identifies the input; the generated UUID suffix is
            # intentionally different across fresh LLM instances.
            canonical.append(dict(step, scheduled=scheduled))
        signatures.append(canonical)
    assert signatures[0] == signatures[1], (
        "Different request schedules invalidate an isolated runner-compute comparison"
    )
