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

"""Synthetic UT for the RG_INJECT anomaly-injection hooks (no NPU).

Drives ``vllm_ascend.runtime_guard.inject`` entry points directly with
nested-list stand-ins for logits / sampled tokens. Live matrix + pass
criteria: ``ANOMALY_INJECTION.md``.
"""

from __future__ import annotations

import importlib
import math

import pytest

from vllm_ascend.runtime_guard import inject


def _load(monkeypatch: pytest.MonkeyPatch, value: str | None):
    if value is None:
        monkeypatch.delenv("RG_INJECT", raising=False)
    else:
        monkeypatch.setenv("RG_INJECT", value)
    mod = importlib.reload(inject)
    if value is not None:
        # Shipped builds keep _INJECT_MASTER_SWITCH False; scenario tests run
        # armed. Reload resets the constant, so arm and re-derive afterwards.
        mod._INJECT_MASTER_SWITCH = True
        mod._plan = mod._load()
        mod.ENABLED = mod._plan is not None
    return mod


def _logits(rows: int = 2, cols: int = 10) -> list[list[float]]:
    return [[float(r * 100 + c) for c in range(cols)] for r in range(rows)]


def test_env_unset_disables_and_never_mutates(monkeypatch):
    mod = _load(monkeypatch, None)
    assert mod.ENABLED is False
    lg = _logits()
    mod.inject_before_sample(lg)
    assert lg == _logits()
    mod.inject_after_sample([[1], [2]], runner=None)
    mod.inject_after_spec([3, 2])
    assert lg == _logits()


@pytest.mark.parametrize("raw", ["bogus_scenario", "nan_logits:abc", "nan_logits:0", "nan_logits:2:3:4"])
def test_invalid_env_disables(monkeypatch, raw):
    mod = _load(monkeypatch, raw)
    assert mod.ENABLED is False


def test_nan_logits_fires_once_at_step(monkeypatch):
    mod = _load(monkeypatch, "nan_logits:2")
    assert mod.ENABLED is True
    w1 = _logits()
    mod.inject_before_sample(w1)
    assert w1 == _logits()
    w2 = _logits()
    mod.inject_before_sample(w2)
    assert math.isnan(w2[0][5])
    assert [math.isnan(x) for x in w2[0]].count(True) == 1
    assert all(not math.isnan(x) for x in w2[1])
    w3 = _logits()
    mod.inject_before_sample(w3)
    assert w3 == _logits()


def test_inf_logits_fires_once_at_step(monkeypatch):
    mod = _load(monkeypatch, "inf_logits")
    w1 = _logits()
    mod.inject_before_sample(w1)  # DEFAULT_STEP=5: no fire yet
    assert w1 == _logits()
    for _ in range(3):
        mod.inject_before_sample(_logits())
    w5 = _logits()
    mod.inject_before_sample(w5)
    assert math.isinf(w5[0][3])
    w6 = _logits()
    mod.inject_before_sample(w6)
    assert w6 == _logits()


def test_forbidden_substring_cycles_raw_ids(monkeypatch):
    mod = _load(monkeypatch, "forbidden_substring:2:17,99")
    for wave in (1, 2):
        sampled = [[wave * 10], [wave * 20]]
        mod.inject_before_sample(_logits())  # tick wave counter as live flow does
        mod.inject_after_sample(sampled, runner=None)
        if wave == 1:
            assert sampled == [[10], [20]]  # below step: untouched
    seq = []
    for wave in (3, 4, 5):
        sampled = [[wave], [wave * 100]]
        mod.inject_before_sample(_logits())
        mod.inject_after_sample(sampled, runner=None)
        seq.append(sampled[0][0])
        assert sampled[1] == [wave * 100]  # row 1 untouched
    # wave 2 already consumed ids[0]=17; waves 3-5 continue the cycle.
    assert seq == [99, 17, 99]


def test_forbidden_substring_encodes_text_via_tokenizer(monkeypatch):
    mod = _load(monkeypatch, "forbidden_substring:1")

    class _Tok:
        @staticmethod
        def encode(text, add_special_tokens=False):
            assert text == "李白"
            assert add_special_tokens is False
            return [11, 22, 33]

    import vllm_ascend.runtime_guard.token_utils as token_utils_mod
    monkeypatch.setattr(token_utils_mod, "load_model_tokenizer", lambda runner: _Tok())
    seq = []
    for wave in range(1, 5):
        sampled = [[wave], [wave]]
        mod.inject_before_sample(_logits())
        mod.inject_after_sample(sampled, runner=object())
        seq.append(sampled[0][0])
    assert seq == [11, 22, 33, 11]


def test_token_loop_pins_previous_token_for_param_waves(monkeypatch):
    mod = _load(monkeypatch, "token_loop:2:3")
    sampled1 = [[10], [20]]
    mod.inject_before_sample(_logits())
    mod.inject_after_sample(sampled1, runner=None)
    assert sampled1 == [[10], [20]]  # wave 1 below step

    results = []
    for wave in range(2, 7):
        sampled = [[wave], [wave * 100]]
        mod.inject_before_sample(_logits())
        mod.inject_after_sample(sampled, runner=None)
        results.append((sampled[0][0], sampled[1][0]))
    # wave 2 anchors on its own row-0 token (2), pins waves 2-4; waves 5-6 free.
    assert results == [(2, 200), (2, 300), (2, 400), (5, 500), (6, 600)]


def test_token_loop_ticks_without_before_sample(monkeypatch):
    """Live path with logits_finite off never calls inject_before_sample."""
    mod = _load(monkeypatch, "token_loop:2:2")
    sampled1 = [[9], [1]]
    mod.inject_after_sample(sampled1, runner=None)
    assert sampled1 == [[9], [1]]  # wave 1
    sampled2 = [[7], [1]]
    mod.inject_after_sample(sampled2, runner=None)
    assert sampled2[0][0] == 7  # wave 2 anchors on 7
    sampled3 = [[3], [1]]
    mod.inject_after_sample(sampled3, runner=None)
    assert sampled3[0][0] == 7  # still pinned
    sampled4 = [[4], [1]]
    mod.inject_after_sample(sampled4, runner=None)
    assert sampled4[0][0] == 4  # loop done


def test_spec_all_reject_zeroes_once(monkeypatch):
    mod = _load(monkeypatch, "spec_all_reject:1")
    mod.inject_before_sample(_logits())
    accepted = [3, 2, 1]
    mod.inject_after_spec(accepted)
    assert accepted == [0, 0, 0]
    mod.inject_before_sample(_logits())
    accepted2 = [2, 2]
    mod.inject_after_spec(accepted2)
    assert accepted2 == [2, 2]  # one-shot


def test_spec_all_reject_without_before_sample(monkeypatch):
    mod = _load(monkeypatch, "spec_all_reject:2")
    accepted = [1, 1]
    mod.inject_after_spec(accepted)
    assert accepted == [1, 1]  # wave 1
    # Close the wave the way check_after_sample would after after_spec.
    mod.inject_after_sample([[0]], runner=None)
    accepted2 = [4, 5]
    mod.inject_after_spec(accepted2)
    assert accepted2 == [0, 0]  # wave 2
