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
"""Shared UT helpers for runtime_guard (avoid cross-importing test modules)."""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor


def bare_processor() -> RuntimeGuardProcessor:
    """Minimal ``RuntimeGuardProcessor`` shell for soft-fail / hook wiring UTs."""
    p = object.__new__(RuntimeGuardProcessor)
    p.detectors = MagicMock()
    p.detectors.after_sample_hot_path = MagicMock(side_effect=RuntimeError("boom"))
    p.detectors.check_before_sample = MagicMock(side_effect=RuntimeError("boom"))
    p.wave_tracker = None
    p.runner = None
    p._handle_alert = MagicMock()
    p._reap_finished_requests = MagicMock()
    p._last_input_batch = None
    p.action_executor = None
    # End-of-wave gate uses collectives; bare shells no-op it.
    p.runtime_config = MagicMock()
    p.end_of_wave_sync = MagicMock()  # type: ignore[method-assign]
    return p
