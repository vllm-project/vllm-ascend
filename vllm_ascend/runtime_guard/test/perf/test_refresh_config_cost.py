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

"""Opt-in wall-clock microbenches for runtime_guard hot path (not default UT).

Run explicitly::

    pytest -m perf vllm_ascend/runtime_guard/test/perf

Shared CI runners make absolute µs bounds flaky; keep these out of correctness CI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.runtime_config.config import RuntimeConfig
from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor

pytestmark = pytest.mark.perf


def _cfg(tmp_path: Path, *, reload: float) -> RuntimeConfig:
    path = tmp_path / "runtime_config.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    return RuntimeConfig(
        config_path=path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=reload,
        sync_mode="file",
    )


def _bench(fn, n: int = 2000) -> float:
    for _ in range(50):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6  # µs / call


def _bind(cfg: RuntimeConfig) -> RuntimeGuardProcessor:
    RuntimeGuardProcessor.reset_for_tests()
    runner = MagicMock()
    runner.tp_rank = 0

    def _init(self, r):
        self.runner = r
        self.runtime_config = cfg
        self.manual_triggers = MagicMock()
        self.manual_triggers.consume_once.return_value = None
        self.action_executor = MagicMock()
        self.action_executor.apply_runtime_config = MagicMock()
        self.detectors = MagicMock()
        self.detectors.apply_runtime_config = MagicMock()
        self.report_writer = MagicMock()
        self.kv_reader = MagicMock()
        self.quota = MagicMock()

    with patch.object(RuntimeGuardProcessor, "_init_from_runner", _init):
        return RuntimeGuardProcessor.bind(runner)


def test_refresh_config_hot_reload_off_is_cheap(tmp_path: Path):
    """reload=0 → refresh_config must stay microsecond-scale on CPU."""
    cfg = _cfg(tmp_path, reload=0.0)
    proc = _bind(cfg)
    us = _bench(lambda: proc.refresh_config(allow_arm=True))
    assert us < 500.0, f"refresh_config (reload=0) too slow: {us:.1f} µs"
    RuntimeGuardProcessor.reset_for_tests()


def test_refresh_config_reload_on_detectors_off_bounded(tmp_path: Path):
    """reload>0 but detectors/dump off — still far below sampling latency."""
    cfg = _cfg(tmp_path, reload=0.05)
    proc = _bind(cfg)
    us = _bench(lambda: proc.refresh_config(allow_arm=True), n=500)
    assert us < 5000.0, f"refresh_config (reload on, idle) too slow: {us:.1f} µs"
    RuntimeGuardProcessor.reset_for_tests()


def test_isolation_refresh_vs_pure_noop(tmp_path: Path):
    """Relative bound vs empty Python call (stronger than absolute µs alone)."""
    cfg = _cfg(tmp_path, reload=0.0)
    proc = _bind(cfg)

    def noop():
        return None

    us_noop = _bench(noop, n=5000)
    us_rg = _bench(lambda: proc.refresh_config(allow_arm=True), n=2000)
    assert us_rg < us_noop + 400.0, (
        f"reload=0 path not near-noop: rg={us_rg:.1f}µs noop={us_noop:.1f}µs"
    )
    RuntimeGuardProcessor.reset_for_tests()
