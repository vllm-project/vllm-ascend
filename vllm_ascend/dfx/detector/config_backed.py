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

"""Config-backed anomaly detectors (knobs under ``dfx_config.detector.<section>``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm_ascend.dfx.detector.base import AnomalyDetector


def detector_section_getter(section: Any) -> Callable[[str, Any], Any]:
    """``dict.get``-like accessor for a detector config section or object."""
    if isinstance(section, dict):
        return section.get
    return lambda key, default=None: getattr(section, key, default)


class ConfigBackedDetector(AnomalyDetector):
    """Detector whose enable flag and thresholds live in ``detector.<section>``.

    Subclasses set :attr:`section_key` (and optionally :attr:`enable_key`) and
    implement :meth:`_apply_detector_values`. Call :meth:`refresh_from_config`
    after instance defaults are set (typically at the end of ``__init__`` when
    ``dfx_config`` is present).

    Manual / one-shot triggers should keep subclassing :class:`AnomalyDetector`
    directly — they are not config-threshold sensors.
    """

    # Nested object name under ``detector``, e.g. ``spec_acceptance``.
    section_key: str = ""
    # Bool enable switch inside that section. Empty → do not touch ``_enabled``.
    enable_key: str = "enabled"

    def refresh_from_config(self) -> None:
        """Pull enable + knobs from live ``dfx_config.detector.<section_key>``."""
        if self._dfx_config is None:
            return
        if self.section_key:
            section = self._dfx_config.detector_section(self.section_key)
        else:
            section = self._dfx_config.detector
        self._apply_detector_section(section)

    def _apply_detector_section(self, section: Any) -> None:
        getter = detector_section_getter(section)
        if self.enable_key:
            self._enabled = bool(getter(self.enable_key, self._enabled))
        self._apply_detector_values(getter)

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        """Apply thresholds / windows from ``getter(key, default)``. Override."""
        raise NotImplementedError(f"{type(self).__name__} must implement _apply_detector_values")
