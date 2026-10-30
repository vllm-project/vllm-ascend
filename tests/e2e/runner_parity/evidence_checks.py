# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Validate persisted controller provenance and actual NPU ownership windows."""

import json
from datetime import datetime

import regex as re


def assert_control_runtime_and_ownership(control_root, extra_source_keys=()):
    reference = None
    ownership = [json.loads(line) for line in (control_root / "ownership.jsonl").read_text().splitlines()]
    for arm, selector in (("v1", "0"), ("v2", "1")):
        root = control_root / arm / "phase-diagnostic"
        assert (root / "validated.json").is_file()
        assert (root / "cleanup.json").is_file()
        cleanup = json.loads((root / "cleanup.json").read_text())
        assert cleanup["no_process_devices"] == 8
        assert len(cleanup["hbm_mb"]) == 8 and max(cleanup["hbm_mb"]) < 8000
        meta = json.loads((root / "provenance-control.json").read_text())
        signature = {
            key: meta[key]
            for key in (
                "ascend_head",
                "vllm_head",
                "code_hashes",
                "profile_sha256",
                "common_config_sha256",
                "engine_mode",
            )
            + tuple(extra_source_keys)
        }
        signature["image"] = meta["container"]["image"]
        assert meta["container"]["runner_env"] == ["VLLM_USE_V2_MODEL_RUNNER=" + selector]
        if reference is None:
            reference = signature
        assert signature == reference
        stopped = json.loads((root / f"stopped-container-{meta['container']['id'][:12]}.json").read_text())
        started_at = datetime.fromisoformat(stopped["State"]["StartedAt"].replace("Z", "+00:00"))
        validation = json.loads((root / "validated.json").read_text())
        finished_at = datetime.fromisoformat(validation["finished_utc"])
        container_name = validation["command"][validation["command"].index("--name") + 1]
        rows = [
            row
            for row in ownership
            if row["container"] == container_name
            and started_at <= datetime.fromisoformat(row["state"]["utc"]) <= finished_at
        ]
        assert len(rows) >= 2, "Missing ownership monitoring for this successful container attempt"
        for row in rows:
            assert row["foreign"] == []
            # Older raw captures carry health in npu-smi text, before the
            # structured healthy_devices field was added. Derive, never assume.
            healthy = len(re.findall(r"\|\s*\d+\s*\|\s*Ascend950DT\s*\|\s*OK\s*\|", row["state"]["raw"]))
            assert row["state"].get("healthy_devices", healthy) == healthy == 8
    assert not (control_root / "exclusive-window-lost.json").exists()
