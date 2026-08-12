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

from __future__ import annotations

import json
from pathlib import Path

from vllm_ascend.dfx.report import DfxReportWriter, dumps_report_json, sanitize_report_detail


def test_dfx_report_writer_writes_pretty_json(tmp_path: Path):
    writer = DfxReportWriter(tmp_path / "report")
    path = writer.write(
        anomaly_type="spec_acceptance",
        req_id="req-1",
        detail={
            "acceptance_rate": 0.1,
            "window_token_ids": [1, 2, 3],
            "output_token_ids": [4, 5, 6, 7],
            "output_token_count": 4,
        },
        rank_tag="tp0",
    )
    assert path is not None
    assert path.exists()
    assert path.name.startswith("anomaly_")
    assert "_dump_" not in path.name
    assert "_pid" in path.stem
    text = path.read_text(encoding="utf-8")
    assert "\n" in text  # pretty-printed
    record = json.loads(text)
    assert record["anomaly_type"] == "spec_acceptance"
    assert record["req_id"] == "req-1"
    assert record["rank"] == "tp0"
    assert record["dump_attempted"] is False
    assert record["dump_armed"] is False
    assert record["dump_arm_wave"] is None
    assert record["dump_count"] is None
    assert record["dump_max_times"] is None
    assert record["detail"]["acceptance_rate"] == 0.1
    assert "window_token_ids" not in record["detail"]
    assert "output_token_ids" not in record["detail"]
    assert record["detail"]["window_token_count"] == 3
    assert record["detail"]["output_token_count"] == 4
    assert "save_sensitive_info" not in record
    assert "unix_ts" not in record
    assert "ts" in record


def test_dfx_report_writer_marks_dump_armed_in_filename(tmp_path: Path):
    writer = DfxReportWriter(tmp_path / "report")
    path = writer.write(
        anomaly_type="spec_acceptance",
        req_id="req-dump",
        detail={"acceptance_rate": 0.2},
        dump_attempted=True,
        dump_armed=True,
        dump_count=2,
        dump_max_times=5,
    )
    assert path is not None
    assert "_dump_pid" in path.name
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["dump_attempted"] is True
    assert record["dump_armed"] is True
    assert record["dump_arm_wave"] is None
    assert record["dump_count"] == 2
    assert record["dump_max_times"] == 5


def test_dfx_report_writer_can_save_sensitive_info(tmp_path: Path):
    writer = DfxReportWriter(tmp_path / "report", save_sensitive_info=True, decode_token_ids=False)
    path = writer.write(
        anomaly_type="token_logprob",
        req_id="r1",
        detail={"window_token_ids": [9, 8], "prompt_token_ids": [1], "output_token_ids": [2, 3]},
    )
    assert path is not None
    text = path.read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["detail"]["window_token_ids"] == [9, 8]
    assert record["detail"]["prompt_token_ids"] == [1]
    assert record["detail"]["output_token_ids"] == [2, 3]
    assert "save_sensitive_info" not in record
    # Token-id arrays stay on one line (not one int per line).
    assert '"output_token_ids": [2, 3]' in text or '"output_token_ids":[2, 3]' in text.replace(" ", "")


def test_dfx_report_truncates_and_decodes_token_ids(tmp_path: Path):
    class _Tok:
        def decode(self, ids, skip_special_tokens=False):
            return "TXT:" + ",".join(str(i) for i in ids)

    writer = DfxReportWriter(
        tmp_path / "report",
        save_sensitive_info=True,
        max_prompt_token_ids=2,
        max_output_token_ids=3,
        decode_token_ids=True,
    )
    path = writer.write(
        anomaly_type="spec_acceptance",
        req_id="r1",
        detail={
            "prompt_token_ids": [1, 2, 3, 4],
            "output_token_ids": [10, 11, 12, 13, 14],
            "prompt_token_count": 4,
            "output_token_count": 5,
            "window_token_ids": [7, 8, 9, 10],
            "window_sampled_token_ids": [[1, 2, 3, 4], [5, 6]],
            "window_accepted_token_ids": [[1, 2], [5]],
            "current_sampled_token_ids": [5, 6, 7, 8],
            "current_accepted_token_ids": [5, 6],
        },
        tokenizer=_Tok(),
    )
    assert path is not None
    record = json.loads(path.read_text(encoding="utf-8"))
    detail = record["detail"]
    assert detail["prompt_token_ids"] == [1, 2]
    assert detail["output_token_ids"] == [10, 11, 12]
    assert detail["prompt_token_count"] == 4
    assert detail["output_token_count"] == 5
    assert detail["prompt_token_ids_truncated"] is True
    assert detail["output_token_ids_truncated"] is True
    assert detail["prompt_text"] == "TXT:1,2"
    assert detail["output_text"] == "TXT:10,11,12"
    assert detail["window_token_ids"] == [7, 8, 9]
    assert detail["window_text"] == "TXT:7,8,9"
    assert detail["window_sampled_token_ids"] == [[1, 2, 3], [5, 6]]
    assert detail["window_sampled_texts"] == ["TXT:1,2,3", "TXT:5,6"]
    assert detail["window_accepted_texts"] == ["TXT:1,2", "TXT:5"]
    assert detail["current_sampled_text"] == "TXT:5,6,7"
    assert detail["current_accepted_text"] == "TXT:5,6"


def test_dfx_report_max_zero_means_unlimited(tmp_path: Path):
    """max_*=0 keeps full token-id lists (no truncation markers)."""
    ids = list(range(50))
    writer = DfxReportWriter(
        tmp_path / "report",
        save_sensitive_info=True,
        max_prompt_token_ids=0,
        max_output_token_ids=0,
        decode_token_ids=False,
    )
    path = writer.write(
        anomaly_type="output_substring",
        req_id="r1",
        detail={"prompt_token_ids": ids, "output_token_ids": ids, "prompt_token_count": 50, "output_token_count": 50},
    )
    assert path is not None
    detail = json.loads(path.read_text(encoding="utf-8"))["detail"]
    assert detail["prompt_token_ids"] == ids
    assert detail["output_token_ids"] == ids
    assert "prompt_token_ids_truncated" not in detail
    assert "output_token_ids_truncated" not in detail


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


def test_write_dump_finish_respects_save_sensitive_info(tmp_path):
    import json

    writer = DfxReportWriter(tmp_path / "report", save_sensitive_info=False, decode_token_ids=False)
    path = writer.write_dump_finish(
        req_id="req/1",
        detail={"output_token_ids": [7, 8, 9], "output_token_count": 3, "prompt_token_count": 2},
        rank_tag="dp=0 tp=0 pp=0",
        anomaly_type="token_repeat",
        source="anomaly",
        dump_arm_wave=3,
        dump_activate_wave=4,
        dump_waves_after_report=1,
        dump_count=1,
        finish_wave=10,
    )
    assert path is not None
    assert path.name.startswith("dump_finish_")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["kind"] == "dump_finish"
    assert data["req_id"] == "req/1"
    assert data["dump_arm_wave"] == 3
    assert data["dump_activate_wave"] == 4
    assert data["dump_waves_after_report"] == 1
    assert data["dump_finish_wave"] == 10
    assert "save_sensitive_info" not in data
    assert "unix_ts" not in data
    assert "output_token_ids" not in data["detail"]
    assert data["detail"]["output_token_count"] == 3

    writer_s = DfxReportWriter(tmp_path / "report2", save_sensitive_info=True, decode_token_ids=False)
    path2 = writer_s.write_dump_finish(
        req_id="r2",
        detail={"output_token_ids": [1, 2], "output_token_count": 2},
        dump_arm_wave=1,
        dump_activate_wave=1,
        dump_waves_after_report=0,
    )
    data2 = json.loads(path2.read_text(encoding="utf-8"))
    assert data2["detail"]["output_token_ids"] == [1, 2]
