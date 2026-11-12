# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Typed intermediate representation shared by all analyzer stages."""

from dataclasses import dataclass

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


class BuildSyntaxError(ValueError):
    """A Dockerfile or RUN command cannot be parsed reliably."""


@dataclass(frozen=True)
class DockerToken:
    source: str
    line: int
    text: str
    line_map: tuple[int, ...] = ()


@dataclass(frozen=True)
class Instruction:
    source: str
    line: int
    kind: str
    body: str
    line_map: tuple[int, ...] = ()


@dataclass(frozen=True)
class ShellDependency:
    """A preceding command whose shell state is inherited by an operation."""

    kind: str
    value: str


@dataclass(frozen=True)
class Operation:
    source: str
    line: int
    kind: str
    value: str
    condition: str | None = None
    unresolved: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    shell_dependencies: tuple[ShellDependency, ...] = ()
    outputs: tuple[str, ...] = ()
    owner_index: int | None = None
    owner_key: tuple[str, str] | None = None
    resource_reads: tuple[str, ...] = ()
    resource_writes: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()

    @property
    def semantic_key(self) -> tuple[object, ...]:
        # Source names identify build layers; line numbers are provenance.
        value: object = self.value
        if self.kind == "requirement":
            try:
                requirement = Requirement(self.value)
                value = (
                    canonicalize_name(requirement.name),
                    tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
                    str(requirement.specifier),
                    requirement.url,
                    str(requirement.marker) if requirement.marker else None,
                )
            except InvalidRequirement:
                pass
        context = (
            ()
            if self.kind in {"source", "export", "cd", "assign", "if-select", "case-select", "pip-cache"}
            else tuple((dependency.kind, dependency.value) for dependency in self.shell_dependencies)
        )
        return (
            self.source,
            self.kind,
            value,
            self.condition,
            self.unresolved,
            context,
            self.outputs,
            self.owner_key,
            self.resource_reads,
            self.resource_writes,
        )


@dataclass(frozen=True)
class PackageEffect:
    """A normalized package requirement or pip operation in the build chain."""

    name: str
    action: str
    specification: str
    source: str
    line: int


@dataclass(frozen=True)
class PackageOutcome:
    """Last pip action or attached requirement in Docker build order."""

    name: str
    action: str
    specification: str
    source: str
    line: int


@dataclass(frozen=True)
class PackageDelta:
    """A difference between final package effects of two build chains."""

    name: str
    before: PackageOutcome | None
    after: PackageOutcome | None


@dataclass(frozen=True)
class BuildAnalysis:
    instructions: tuple[Instruction, ...]
    operations: tuple[Operation, ...]
    diagnostics: tuple[str, ...]
    package_effects: tuple[PackageEffect, ...] = ()
    package_outcomes: tuple[PackageOutcome, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
