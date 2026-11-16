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

"""Detector package (lazy exports to avoid runtime_config ↔ detector import cycles)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "Incident",
    "AnomalyDetector",
    "ConfigBackedDetector",
    "DetectorManager",
    "DetectorRegistry",
    "LogitsFiniteDetector",
    "SpecAcceptanceDetector",
    "TokenRepeatDetector",
]


def __getattr__(name: str) -> Any:
    if name == "Incident":
        from vllm_ascend.observability.runtime_guard.state import Incident

        return Incident
    if name in ("AnomalyDetector", "ConfigBackedDetector", "DetectorRegistry"):
        from vllm_ascend.observability.runtime_guard.detector import base as _base

        return getattr(_base, name)
    if name == "DetectorManager":
        from vllm_ascend.observability.runtime_guard.detector.manager import DetectorManager

        return DetectorManager
    if name == "LogitsFiniteDetector":
        from vllm_ascend.observability.runtime_guard.detector.logits_finite import LogitsFiniteDetector

        return LogitsFiniteDetector
    if name == "SpecAcceptanceDetector":
        from vllm_ascend.observability.runtime_guard.detector.spec_acceptance import SpecAcceptanceDetector

        return SpecAcceptanceDetector
    if name == "TokenRepeatDetector":
        from vllm_ascend.observability.runtime_guard.detector.token_repeat import TokenRepeatDetector

        return TokenRepeatDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
