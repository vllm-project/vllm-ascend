#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: ignore-errors
"""UT: report sanitize / truncate + ReportWriter write rollback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.observability.runtime_guard.report import (
    ReportWriter,
    sanitize_report_detail,
    truncate_token_id_fields,
)


def test_sanitize_report_detail_redacts_ids_keeps_counts():
    detail = {
        "prompt_token_ids": [1, 2, 3],
        "output_token_ids": [9, 8],
        "note": "keep-me",
    }
    out = sanitize_report_detail(detail, save_sensitive_info=False)
    assert "prompt_token_ids" not in out
    assert "output_token_ids" not in out
    assert out["prompt_token_count"] == 3
    assert out["output_token_count"] == 2
    assert out["note"] == "keep-me"


def test_truncate_token_id_fields_caps_and_flags():
    long_prompt = list(range(50))
    long_output = list(range(30))
    detail = {"prompt_token_ids": long_prompt, "output_token_ids": long_output}
    out = truncate_token_id_fields(
        detail,
        max_prompt_token_ids=10,
        max_output_token_ids=5,
    )
    assert out["prompt_token_ids"] == long_prompt[:10]
    assert out["prompt_token_ids_truncated"] is True
    assert out["prompt_token_ids_max"] == 10
    assert out["prompt_token_count"] == 50
    assert out["output_token_ids"] == long_output[:5]
    assert out["output_token_ids_truncated"] is True
    assert out["output_token_ids_max"] == 5
    assert out["output_token_count"] == 30


def test_sanitize_nested_requests_redact_and_truncate():
    nested = {
        "requests": [
            {
                "req_id": "r1",
                "prompt_token_ids": [1, 2],
                "window_token_ids": [[10, 11, 12, 13, 14]],
            }
        ],
        "meta": 1,
    }
    redacted = sanitize_report_detail(nested, save_sensitive_info=False)
    row = redacted["requests"][0]
    assert "prompt_token_ids" not in row
    assert row["prompt_token_count"] == 2
    assert "window_token_ids" not in row
    assert row["window_token_count"] == 5
    assert redacted["meta"] == 1

    truncated = truncate_token_id_fields(
        nested,
        max_prompt_token_ids=1,
        max_output_token_ids=1000,
    )
    trow = truncated["requests"][0]
    assert trow["prompt_token_ids"] == [1]
    assert trow["prompt_token_ids_truncated"] is True
    assert trow["window_token_ids"] == [[10, 11, 12, 13, 14]]


def _patch_json_write_fail(monkeypatch, *, fail: bool) -> None:
    real_open = Path.open

    def _open(self: Path, mode: str = "r", *args, **kwargs):
        if fail and mode == "w" and self.suffix == ".json":
            raise OSError("simulated write failure")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _open)


def test_report_writer_write_rolls_back_first_pair_increment(tmp_path, monkeypatch):
    w = ReportWriter(tmp_path / "report", max_per_req=3)
    pair = ("token_repeat", "r1")
    _patch_json_write_fail(monkeypatch, fail=True)
    assert w.write(incident_type="token_repeat", req_id="r1", detail={"x": 1}, dump_arm_wave=10) is None
    assert pair not in w._pair_state


def test_report_writer_write_rolls_back_repeat_pair_increment(tmp_path, monkeypatch):
    from vllm_ascend.observability.runtime_guard.report import SAME_PAIR_BACKOFF_BASE_WAVES

    w = ReportWriter(tmp_path / "report", max_per_req=3)
    pair = ("token_repeat", "r1")
    kw = dict(incident_type="token_repeat", req_id="r1", detail={"x": 1})
    assert w.write(**kw, dump_arm_wave=100) is not None
    assert w._pair_state[pair] == [100, 1]

    _patch_json_write_fail(monkeypatch, fail=True)
    wave2 = 100 + SAME_PAIR_BACKOFF_BASE_WAVES
    assert w.write(**kw, dump_arm_wave=wave2) is None
    assert w._pair_state[pair] == [100, 1]


def test_is_action_leader_rank_matches_skip_reason():
    from vllm_ascend.observability.runtime_guard.rank_gate import is_action_leader_rank

    runner = SimpleNamespace()
    with patch(
        "vllm_ascend.observability.runtime_guard.rank_gate.anomaly_check_rank_skip_reason",
        return_value=None,
    ):
        assert is_action_leader_rank(runner) is True
    with patch(
        "vllm_ascend.observability.runtime_guard.rank_gate.anomaly_check_rank_skip_reason",
        return_value="not TP0",
    ):
        assert is_action_leader_rank(runner) is False
