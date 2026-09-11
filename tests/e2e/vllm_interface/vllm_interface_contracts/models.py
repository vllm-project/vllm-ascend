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
"""Public data models for range-level interface compatibility reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceEndpoint:
    file: str | None
    owner: str | None
    name: str | None
    line: int | None = None
    signature: list[object] | None = field(default=None, compare=False)
    descriptor: str | None = None
    symbol_kind: str | None = None
    signature_status: str | None = None
    analysis_fingerprint: str | None = field(default=None, compare=False)
    return_contract: dict[str, Any] | None = field(default=None, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompatibilityState:
    exists: bool | None
    compatible: bool | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RangeFinding:
    finding_id: str
    classification: str
    relation: str
    priority: str
    action: str
    confidence: str
    upstream_old: SourceEndpoint
    upstream_new: SourceEndpoint
    downstream: SourceEndpoint
    old_state: CompatibilityState
    new_state: CompatibilityState
    change: str
    evidence: list[dict[str, Any]]
    gates: dict[str, bool]
    suggestion: str
    source: str = "dynamic_relation_graph"
    contract_kind: str = "call_arguments"
    direction: str = "upstream_contract_to_downstream_implementation"
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.finding_id,
            "classification": self.classification,
            "relation": self.relation,
            "priority": self.priority,
            "action": self.action,
            "confidence": self.confidence,
            "upstream": {
                "old": self.upstream_old.as_dict(),
                "new": self.upstream_new.as_dict(),
            },
            "downstream": self.downstream.as_dict(),
            "compatibility": {
                "old": self.old_state.as_dict(),
                "new": self.new_state.as_dict(),
            },
            "change": self.change,
            "evidence": self.evidence,
            "gates": self.gates,
            "suggestion": self.suggestion,
            "source": self.source,
            "contract_kind": self.contract_kind,
            "direction": self.direction,
            "details": self.details,
        }
