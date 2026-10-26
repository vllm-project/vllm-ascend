#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Tier selection for Agent Hint context management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Tier = Literal["npu_retain", "cpu", "disk"]

_SHORT_MAX_S = 60
_MEDIUM_MAX_S = 600


@dataclass(frozen=True)
class TierDecision:
    tier: Tier
    reason: str
    fallback: bool = False


class TierPolicy:
    """Map ``expected_resume_s`` to a storage tier.

    The policy is capability-aware: a requested tier that has no data-plane
    implementation is reported as a CPU fallback instead of being claimed.
    """

    def __init__(
        self,
        *,
        short_max_s: int = _SHORT_MAX_S,
        medium_max_s: int = _MEDIUM_MAX_S,
        npu_retain_enabled: bool = True,
        disk_enabled: bool = False,
    ) -> None:
        if short_max_s < 0 or medium_max_s < short_max_s:
            raise ValueError("invalid tier thresholds")
        self.short_max_s = short_max_s
        self.medium_max_s = medium_max_s
        self.npu_retain_enabled = npu_retain_enabled
        self.disk_enabled = disk_enabled

    def decide(self, expected_resume_s: int | None) -> TierDecision:
        if expected_resume_s is None:
            return TierDecision("cpu", "no resume prediction")
        if expected_resume_s <= self.short_max_s:
            if self.npu_retain_enabled:
                return TierDecision(
                    "npu_retain",
                    f"short pause ({expected_resume_s}s): retain on NPU",
                )
            return TierDecision(
                "cpu",
                f"short pause ({expected_resume_s}s): NPU retain unavailable",
            )
        if expected_resume_s <= self.medium_max_s:
            return TierDecision(
                "cpu",
                f"medium pause ({expected_resume_s}s): offload to CPU",
            )
        if self.disk_enabled:
            return TierDecision(
                "disk",
                f"long pause ({expected_resume_s}s): offload to disk",
            )
        return TierDecision(
            "cpu",
            f"long pause ({expected_resume_s}s): disk unavailable",
            fallback=True,
        )
