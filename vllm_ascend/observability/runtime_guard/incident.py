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

"""Incident types emitted by runtime_guard detectors.

``ILL_TYPE_*`` codes align with legacy msprobe ILLDetector category ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Align with msprobe response_anomaly ILLDetector ill_type codes.
ILL_TYPE_NONE = 0
ILL_TYPE_RARE = 1
ILL_TYPE_GARBLED = 2
ILL_TYPE_REPEAT = 3
ILL_TYPE_NAN = 4

ILL_TYPE_NAME: dict[int, str] = {
    ILL_TYPE_NONE: "none",
    ILL_TYPE_RARE: "rare",
    ILL_TYPE_GARBLED: "garbled",
    ILL_TYPE_REPEAT: "repetition",
    ILL_TYPE_NAN: "nan",
}


@dataclass(slots=True)
class Incident:
    """One runtime finding handed to the action executor."""

    incident_type: str
    req_id: str
    is_ill: bool = True
    ill_type: int = ILL_TYPE_NONE
    req_idx: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    consume_quota: bool = True
    block_ids: list[int] = field(default_factory=list)
    wave: int | None = None
    log_context: dict[str, Any] = field(default_factory=dict)

    @property
    def ill_type_name(self) -> str:
        return ILL_TYPE_NAME.get(self.ill_type, f"unknown({self.ill_type})")

    def to_report_detail(self) -> dict[str, Any]:
        out = dict(self.detail)
        if self.ill_type != ILL_TYPE_NONE:
            out.setdefault("ill_type", self.ill_type)
            out.setdefault("ill_type_name", self.ill_type_name)
        out.setdefault("is_ill", self.is_ill)
        # ``block_ids`` is emitted solely by RuntimeGuardProcessor.
        # _enrich_detail_with_block_meta, which gates it on
        # report.include_block_ids. Emitting it here bypassed that flag.
        return out
