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

from tests.ut.dfx_test_utils import make_dfx_config
from vllm_ascend.dfx.detector.output_substring import (
    OutputSubstringDetector,
    contains_prefix,
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


def test_contains_prefix():
    assert contains_prefix([1, 2, 3, 4], [1, 2])
    assert not contains_prefix([1, 2, 3], [2, 3])
    assert contains_prefix([9], [9])
    assert not contains_prefix([1, 2], [1, 2, 3])


def test_output_substring_prefix_match_mode(tmp_path):
    """match_prefix=true matches only at the start of the cumulative output."""
    RequestIoSnapshotManager.reset_for_tests()

    cfg = make_dfx_config(tmp_path)
    cfg._data["detector"]["output_substring"]["enabled"] = True
    cfg._data["detector"]["output_substring"]["patterns"] = [[ord("a"), ord("b")]]
    cfg._data["detector"]["output_substring"]["match_prefix"] = True

    runner = SimpleNamespace(
        tp_rank=0,
        input_batch=SimpleNamespace(req_ids=["r1", "r2"]),
        requests={"r1": SimpleNamespace(), "r2": SimpleNamespace()},
        vllm_config=SimpleNamespace(model_config=object()),
    )
    det = OutputSubstringDetector(
        dfx_config=cfg,
        runner=runner,
        tokenizer_provider=lambda: _FakeTok(),
    )
    det.refresh_from_config()
    assert det._match_prefix is True

    # r1: output starts with [a, b, c] → prefix [a, b] matches.
    alerts = det.check_all(sampled_token_ids=[[ord("a"), ord("b"), ord("c")]], req_ids=["r1"])
    assert len(alerts) == 1
    assert alerts[0].detail["matched_token_ids"] == [ord("a"), ord("b")]
    assert alerts[0].detail["match_mode"] == "prefix"

    # r2: output [x, a, b] has [a, b] NOT at the start → prefix does not match.
    alerts2 = det.check_all(sampled_token_ids=[[ord("x"), ord("a"), ord("b")]], req_ids=["r2"])
    assert alerts2 == []


def test_output_substring_compile_and_once_per_req(tmp_path):
    RequestIoSnapshotManager.reset_for_tests()

    cfg = make_dfx_config(tmp_path)
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
    assert detail["match_mode"] == "subsequence"

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
