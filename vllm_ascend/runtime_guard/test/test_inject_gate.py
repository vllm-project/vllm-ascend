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

"""Shipped-safety gate tests for the RG_INJECT double-gate.

Injection requires BOTH the source-level ``_INJECT_MASTER_SWITCH`` (shipped
False) and the ``RG_INJECT`` env var. These tests pin the shipped behavior:
an unmodified build must ignore ``RG_INJECT`` entirely.
"""

from __future__ import annotations

import importlib

import pytest

from vllm_ascend.runtime_guard import inject


def _reload(monkeypatch: pytest.MonkeyPatch, value: str | None):
    if value is None:
        monkeypatch.delenv("RG_INJECT", raising=False)
    else:
        monkeypatch.setenv("RG_INJECT", value)
    return importlib.reload(inject)


def test_shipped_switch_ignores_valid_env(monkeypatch):
    mod = _reload(monkeypatch, "nan_logits:2")
    assert mod._INJECT_MASTER_SWITCH is False
    assert mod._plan is None
    assert mod.ENABLED is False
    logits = [[1.0, 2.0], [3.0, 4.0]]
    for _ in range(6):
        mod.inject_before_sample(logits)
    assert logits == [[1.0, 2.0], [3.0, 4.0]]  # never mutated


def test_shipped_switch_ignores_invalid_env(monkeypatch):
    mod = _reload(monkeypatch, "bogus_scenario")
    assert mod.ENABLED is False


def test_armed_switch_parses_env(monkeypatch):
    mod = _reload(monkeypatch, "token_loop:2:3")
    mod._INJECT_MASTER_SWITCH = True
    mod._plan = mod._load()
    assert mod._plan is not None
    assert mod._plan.scenario == "token_loop"
    assert mod._plan.step == 2
    assert mod._plan.param == "3"


def test_armed_switch_rejects_invalid_env(monkeypatch):
    mod = _reload(monkeypatch, "nan_logits:0")
    mod._INJECT_MASTER_SWITCH = True
    assert mod._load() is None


def test_armed_switch_without_env_stays_disabled(monkeypatch):
    mod = _reload(monkeypatch, None)
    mod._INJECT_MASTER_SWITCH = True
    assert mod._load() is None
    assert mod.ENABLED is False
