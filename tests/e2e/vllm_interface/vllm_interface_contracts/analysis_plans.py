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
"""Fixed execution plan for the upstream vLLM interface CI check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ANALYSIS_PLAN_VERSION = 3
VLLM_INTERFACE_SCENARIO = "vllm-interface"


@dataclass(frozen=True)
class AnalysisPlan:
    """The only supported analysis scope in this CI package."""

    scenario: str = VLLM_INTERFACE_SCENARIO

    @property
    def relation_types(self) -> frozenset[str]:
        return frozenset({"override"})

    def capabilities(self) -> dict[str, dict[str, Any]]:
        return {
            "inheritance_mro": {
                "state": "prerequisite",
                "produces_findings": False,
            },
            "override": {
                "state": "analyzed",
                "produces_findings": True,
            },
            "monkey_patch": {
                "state": "skipped",
                "produces_findings": False,
            },
            "direct_import": {
                "state": "analyzed",
                "produces_findings": True,
            },
            "direct_call": {
                "state": "analyzed",
                "produces_findings": True,
            },
            "generator_findings": {
                "state": "skipped",
                "produces_findings": False,
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "plan_version": ANALYSIS_PLAN_VERSION,
            "report_style": "upstream-pr-introduced-only",
            "capabilities": self.capabilities(),
        }


VLLM_INTERFACE_PLAN = AnalysisPlan()


def resolve_analysis_plan() -> AnalysisPlan:
    return VLLM_INTERFACE_PLAN
