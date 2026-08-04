#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from types import SimpleNamespace

from vllm_ascend.dfx.detector.output_substring import (
    OutputSubstringDetector,
    contains_token_subsequence,
    normalize_raw_patterns,
)
from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager
from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig


class _FakeTok:
    def encode(self, text: str, add_special_tokens: bool = False):
        # Deterministic: map each char to its ordinal (tests only).
        return [ord(c) for c in text]

    def decode(self, token_ids, skip_special_tokens: bool = False):
        return "".join(chr(int(t)) for t in token_ids)


def test_normalize_raw_patterns_accepts_text_and_ids():
    assert normalize_raw_patterns(["ab", [1, 2], ""]) == ["ab", [1, 2]]


def test_normalize_raw_patterns_rejects_bad_entries():
    try:
        normalize_raw_patterns([{"x": 1}])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_contains_token_subsequence():
    assert contains_token_subsequence([1, 2, 3, 4], [2, 3])
    assert not contains_token_subsequence([1, 2, 3], [2, 4])
    assert contains_token_subsequence([9], [9])


def test_output_substring_compile_and_once_per_req(tmp_path):
    RequestIoSnapshotManager.reset_for_tests()

    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    cfg._data["detector"]["output_substring"]["enabled"] = True
    cfg._data["detector"]["output_substring"]["patterns"] = ["ab", [ord("x"), ord("y")]]

    runner = SimpleNamespace(
        tp_rank=0,
        input_batch=SimpleNamespace(req_ids=["r1"]),
        requests={"r1": SimpleNamespace()},
        vllm_config=SimpleNamespace(model_config=object()),
    )
    det = OutputSubstringDetector(
        dfx_config=cfg,
        runner=runner,
        tokenizer_provider=lambda: _FakeTok(),
    )
    det.refresh_from_config()
    assert det.enabled
    assert len(det._compiled) == 2
    assert det._compiled[0].source == "text"
    assert det._compiled[0].text == "ab"
    assert list(det._compiled[0].token_ids) == [ord("a"), ord("b")]
    assert det._compiled[1].source == "token_ids"
    assert det._compiled[1].text == "xy"

    # First hit on "ab".
    alerts = det.check_all(sampled_token_ids=[[ord("a"), ord("b"), ord("c")]], req_ids=["r1"])
    assert len(alerts) == 1
    detail = alerts[0].detail
    assert detail["matched_text"] == "ab"
    assert detail["matched_token_ids"] == [ord("a"), ord("b")]
    assert detail["matched_source"] == "text"
    assert detail["matched_pattern_index"] == 0

    # Same req: no second alert even if another pattern appears later.
    alerts2 = det.check_all(sampled_token_ids=[[ord("x"), ord("y")]], req_ids=["r1"])
    assert alerts2 == []

    # Fresh req: token-ids pattern hits (IO buffer is per-req).
    alerts3 = det.check_all(sampled_token_ids=[[ord("x"), ord("y")]], req_ids=["r2"])
    assert len(alerts3) == 1
    assert alerts3[0].detail["matched_source"] == "token_ids"
    assert alerts3[0].detail["matched_text"] == "xy"

    det.clear_finished("r1")
    # After clear, r1 can alert again (still matches text pattern from cumulative IO).
    alerts4 = det.check_all(sampled_token_ids=[[ord("z")]], req_ids=["r1"])
    assert len(alerts4) == 1
    assert alerts4[0].detail["matched_source"] == "text"


def test_detectors_enabled_includes_output_substring():
    data = {
        "detector": {
            "spec_acceptance": {"enabled": False},
            "token_logprob": {"enabled": False},
            "output_substring": {"enabled": True},
        }
    }
    assert DfxRuntimeConfig.detectors_enabled_in(data) is True
