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

"""Anomaly alert types returned by DFX detectors.

Shape mirrors msprobe ``ILLDetector.detector(...)`` result fields
(``is_ill`` / ``ill_type``) and adds dump/report routing metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_ascend.dfx.dfx_types import ILL_TYPE_NAME, ILL_TYPE_NONE


@dataclass(slots=True)
class AnomalyAlert:
    """One anomaly finding for the model runner to hand to ``Dumper``.

    Compatible with msprobe ILLDetector output:
    - ``is_ill``: whether an anomaly was detected
    - ``ill_type``: category code (0 if not token/logprob ILL)

    Extra fields drive dump / report behavior.
    """

    anomaly_type: str
    req_id: str
    is_ill: bool = True
    ill_type: int = ILL_TYPE_NONE
    req_idx: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    skip_related_check: bool = False
    # False for manual dump_once: do not bump max_times / cooldown.
    consume_quota: bool = True
    # Optional context for post-arm logging (e.g. spec token dumps).
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
        return out

    @classmethod
    def from_ill_result(
        cls,
        *,
        req_id: str,
        result: Any,
        req_idx: int | None = None,
        detail: dict[str, Any] | None = None,
        skip_related_check: bool = True,
    ) -> AnomalyAlert | None:
        """Build an alert from msprobe ``detector(...)`` return value."""
        is_ill = bool(getattr(result, "is_ill", False))
        if not is_ill:
            return None
        ill_type = int(getattr(result, "ill_type", 0) or 0)
        return cls(
            anomaly_type="token_logprob",
            req_id=req_id,
            req_idx=req_idx,
            is_ill=True,
            ill_type=ill_type,
            detail=detail or {},
            skip_related_check=skip_related_check,
        )
