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

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from vllm_ascend.observability.runtime_guard.incident import Incident
from vllm_ascend.observability.runtime_guard.token_utils import normalize_token_ids

if TYPE_CHECKING:
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig


def resolve_batch_req_ids(runner: Any, req_ids: list[str] | None) -> list[str]:
    """Pass-through when the caller provides ids; else read the runner batch.

    ``check_after_sample`` hooks may be called with explicit ids (processor
    knows the step's rows) or ``None`` (fallback to ``runner.input_batch``).
    """
    if req_ids is not None:
        return req_ids
    input_batch = getattr(runner, "input_batch", None) if runner is not None else None
    return list(getattr(input_batch, "req_ids", None) or [])


class AnomalyDetector:
    """Base detector: returns ``Incident`` values; processor runs actions."""

    incident_type: str = "unknown"

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig | None = None,
        runner: Any | None = None,
        enabled: bool = True,
    ) -> None:
        self._runtime_config = runtime_config
        self._runner = runner
        self._enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def refresh_from_config(self) -> None:
        """Pull live knobs from ``RuntimeConfig`` (default: no-op)."""
        return

    def clear_finished(self, req_id: str) -> None:
        """Drop per-request state when a request finishes."""

    def on_alert_armed(self, alert: Incident) -> None:
        """Optional hook after dump arm or detect-only alert handling."""

    def _precheck(self) -> bool:
        """Refresh thresholds then return whether this detector is enabled."""
        self.refresh_from_config()
        return self._enabled

    @staticmethod
    def _normalize_token_ids(token_ids: Any) -> list[int]:
        return normalize_token_ids(token_ids)


class ConfigBackedDetector(AnomalyDetector):
    """Detector whose enable flag and thresholds live in ``detector.<section>``."""

    section_key: str = ""
    enable_key: str = "enabled"

    def refresh_from_config(self) -> None:
        if self._runtime_config is None:
            return
        if self.section_key:
            section = self._runtime_config.detector_section(self.section_key)
        else:
            section = self._runtime_config.detector
        self._apply_detector_section(section)

    def _apply_detector_section(self, section: dict[str, Any]) -> None:
        if self.enable_key:
            self._enabled = bool(section.get(self.enable_key, self._enabled))
        self._apply_detector_values(section.get)

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement _apply_detector_values")


class DetectorRegistry:
    """Ordered registry keyed by ``AnomalyDetector.incident_type``."""

    def __init__(self) -> None:
        self._items: list[AnomalyDetector] = []
        self._by_type: dict[str, AnomalyDetector] = {}

    def register(self, detector: AnomalyDetector) -> AnomalyDetector:
        key = str(getattr(detector, "incident_type", "") or type(detector).__name__)
        if key in self._by_type:
            raise ValueError(f"detector incident_type already registered: {key}")
        self._items.append(detector)
        self._by_type[key] = detector
        return detector

    def get(self, incident_type: str) -> AnomalyDetector | None:
        return self._by_type.get(incident_type)

    def __iter__(self) -> Iterator[AnomalyDetector]:
        return iter(self._items)
