#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager, output_token_count_for_request
from vllm_ascend.dfx.processor import DfxProcessor
from vllm_ascend.dfx.report import dumps_report_json, sanitize_report_detail


def test_append_output_builds_cumulative_and_dedupes_suffix():
    RequestIoSnapshotManager.reset_for_tests()
    mgr = RequestIoSnapshotManager.get()
    mgr.append_output("r1", [1, 2, -1, 3])
    assert mgr.cumulative_output_ids("r1") == [1, 2, 3]
    # Same step again (spec + token_logprob) → no double append.
    mgr.append_output("r1", [1, 2, 3])
    assert mgr.cumulative_output_ids("r1") == [1, 2, 3]
    # New step may legitimately repeat the previous last token.
    mgr.append_output("r1", [3])
    assert mgr.cumulative_output_ids("r1") == [1, 2, 3, 3]
    mgr.append_output("r1", [4, 5])
    assert mgr.cumulative_output_ids("r1") == [1, 2, 3, 3, 4, 5]
    mgr.clear_req("r1")
    assert mgr.cumulative_output_ids("r1") == []


def test_snapshot_prefers_cumulative_over_placeholder_batch():
    RequestIoSnapshotManager.reset_for_tests()
    mgr = RequestIoSnapshotManager.get()
    mgr.append_output("r1", [10, 20, 30])
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_output_token_ids=[[-1, -1, -1]]),
        requests={},
    )
    snap = mgr.snapshot(runner, "r1", 0, include_token_ids=True, use_cache=False)
    assert snap.output_token_ids == [10, 20, 30]
    assert snap.output_token_count == 3
    assert output_token_count_for_request(runner, "r1", 0) == 3


def test_clear_finished_clears_cumulative():
    RequestIoSnapshotManager.reset_for_tests()
    mgr = RequestIoSnapshotManager.get()
    mgr.append_output("r1", [1, 2])
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.detectors = MagicMock()
    proc.clear_finished(["r1"])
    proc.detectors.clear_finished.assert_called_once_with("r1")
    assert mgr.cumulative_output_ids("r1") == []


def test_spec_sanitize_keeps_acceptance_metrics_drops_token_ids():
    detail = {
        "acceptance_rate": 0.1,
        "acceptance_len": 1.2,
        "accepted_sum": 3,
        "draft_sum": 30,
        "window": 10,
        "window_size": 10,
        "draft_len": 3,
        "accepted_count": 2,
        "accepted_draft_count": 1,
        "sampled_count": 4,
        "thresholds": {"low_rate": 0.3, "low_len": 1.4, "high_rate": 0.96, "high_len": 2.8},
        "window_steps": [{"accepted_draft": 1, "draft_len": 3, "sampled_count": 4, "accepted_count": 2}],
        "window_sampled_token_ids": [[1, 2, 3]],
        "current_sampled_token_ids": [1, 2],
        "current_accepted_token_ids": [1],
        "output_token_ids": [9, 8, 7],
    }
    out = sanitize_report_detail(detail, save_sensitive_info=False)
    assert out["acceptance_rate"] == 0.1
    assert out["accepted_draft_count"] == 1
    assert out["thresholds"]["low_rate"] == 0.3
    assert out["window_steps"][0]["draft_len"] == 3
    assert "window_sampled_token_ids" not in out
    assert "current_sampled_token_ids" not in out
    assert out["window_sampled_token_count"] == 3
    assert out["current_sampled_token_count"] == 2
    assert "output_token_ids" not in out
    assert out["output_token_count"] == 3


def test_dumps_report_json_keeps_int_lists_compact():
    text = dumps_report_json(
        {
            "detail": {
                "output_token_ids": [1, 2, 3, 4],
                "window_steps": [{"draft_len": 1}],
            }
        }
    )
    assert '"output_token_ids": [1, 2, 3, 4]' in text
    # Not one integer per line.
    assert "\n    1,\n" not in text
