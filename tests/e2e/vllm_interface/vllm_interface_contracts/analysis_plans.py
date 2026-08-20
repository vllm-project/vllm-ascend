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
"""Fixed execution plans for interface-contract analysis scenarios.

The plans select existing engine capabilities.  They deliberately do not
expose independent low-level switches because combinations must preserve the
dependencies between inheritance/MRO discovery and override resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ANALYSIS_PLAN_VERSION = 2
MAIN2MAIN_SCENARIO = "main2main"
VLLM_INTERFACE_SCENARIO = "vllm-interface"
SCENARIOS = (MAIN2MAIN_SCENARIO, VLLM_INTERFACE_SCENARIO)


@dataclass(frozen=True)
class AnalysisPlan:
    """One reviewed combination of engine phases and report semantics."""

    scenario: str
    collect_inheritance: bool
    analyze_inheritance: bool
    collect_overrides: bool
    collect_monkey_patches: bool
    analyze_direct_imports: bool
    analyze_direct_calls: bool
    include_generator_findings: bool
    report_style: str

    @property
    def relation_types(self) -> frozenset[str]:
        enabled: set[str] = set()
        if self.analyze_inheritance:
            enabled.add("inheritance")
        if self.collect_overrides:
            enabled.add("override")
        if self.collect_monkey_patches:
            enabled.add("monkey_patch")
        return frozenset(enabled)

    def capabilities(self) -> dict[str, dict[str, Any]]:
        inheritance_state = (
            "analyzed" if self.analyze_inheritance else "prerequisite" if self.collect_inheritance else "skipped"
        )
        return {
            "inheritance_mro": {
                "state": inheritance_state,
                "produces_findings": self.analyze_inheritance,
            },
            "override": {
                "state": "analyzed" if self.collect_overrides else "skipped",
                "produces_findings": self.collect_overrides,
            },
            "monkey_patch": {
                "state": "analyzed" if self.collect_monkey_patches else "skipped",
                "produces_findings": self.collect_monkey_patches,
            },
            "direct_import": {
                "state": "analyzed" if self.analyze_direct_imports else "skipped",
                "produces_findings": self.analyze_direct_imports,
            },
            "direct_call": {
                "state": "analyzed" if self.analyze_direct_calls else "skipped",
                "produces_findings": self.analyze_direct_calls,
            },
            "generator_findings": {
                "state": "included" if self.include_generator_findings else "skipped",
                "produces_findings": self.include_generator_findings,
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "plan_version": ANALYSIS_PLAN_VERSION,
            "report_style": self.report_style,
            "capabilities": self.capabilities(),
        }


MAIN2MAIN_PLAN = AnalysisPlan(
    scenario=MAIN2MAIN_SCENARIO,
    collect_inheritance=True,
    analyze_inheritance=True,
    collect_overrides=True,
    collect_monkey_patches=True,
    analyze_direct_imports=True,
    analyze_direct_calls=True,
    include_generator_findings=True,
    report_style="main2main-full",
)

VLLM_INTERFACE_PLAN = AnalysisPlan(
    scenario=VLLM_INTERFACE_SCENARIO,
    collect_inheritance=True,
    analyze_inheritance=False,
    collect_overrides=True,
    collect_monkey_patches=False,
    analyze_direct_imports=True,
    analyze_direct_calls=True,
    include_generator_findings=False,
    report_style="upstream-pr-introduced-only",
)

_PLANS = {
    MAIN2MAIN_PLAN.scenario: MAIN2MAIN_PLAN,
    VLLM_INTERFACE_PLAN.scenario: VLLM_INTERFACE_PLAN,
}


def resolve_analysis_plan(scenario: str = MAIN2MAIN_SCENARIO) -> AnalysisPlan:
    try:
        return _PLANS[scenario]
    except KeyError as error:
        choices = ", ".join(SCENARIOS)
        raise ValueError(f"unsupported scenario {scenario!r}; choose one of: {choices}") from error
