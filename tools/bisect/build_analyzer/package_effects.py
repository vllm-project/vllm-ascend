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

"""Normalize package identities across requirements and Docker pip commands."""

import shlex
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename

from tools.bisect.build_analyzer.ir import Operation, PackageEffect, PackageOutcome

_OPTIONS_WITH_VALUES = frozenset(
    {
        "-r",
        "--requirement",
        "-e",
        "--editable",
        "-c",
        "--constraint",
        "-i",
        "--index-url",
        "--extra-index-url",
        "--find-links",
        "--target",
        "--prefix",
        "--root",
        "--no-binary",
        "--only-binary",
    }
)


def _package_specification(token: str) -> tuple[str, str] | None:
    filename = urlsplit(token).path.rsplit("/", 1)[-1]
    if filename.endswith(".whl"):
        try:
            name, version, _, _ = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            return None
        return canonicalize_name(name), f"=={version}"
    try:
        requirement = Requirement(token)
    except InvalidRequirement:
        return None
    return canonicalize_name(requirement.name), str(requirement.specifier)


def has_requirement_marker(operation: Operation) -> bool:
    """A marker depends on the target Python environment, not the analyzer host."""
    if operation.kind != "requirement":
        return False
    try:
        return Requirement(operation.value).marker is not None
    except InvalidRequirement:
        return False


def package_effects(operations: tuple[Operation, ...]) -> tuple[PackageEffect, ...]:
    effects: list[PackageEffect] = []
    for operation in operations:
        if operation.kind == "requirement":
            parsed = _package_specification(operation.value)
            if parsed is not None:
                name, specification = parsed
                effects.append(PackageEffect(name, "require", specification, operation.source, operation.line))
            continue
        if operation.kind not in ("pip-install", "pip-uninstall"):
            continue
        try:
            words = shlex.split(operation.value)
        except ValueError:
            continue
        action = "install" if operation.kind == "pip-install" else "uninstall"
        if "pip" not in words:
            continue
        index = words.index("pip") + 1
        if index >= len(words) or words[index] != action:
            continue
        index += 1
        while index < len(words):
            token = words[index]
            if token in _OPTIONS_WITH_VALUES:
                index += 2
                continue
            if token.startswith("-"):
                index += 1
                continue
            parsed = _package_specification(token)
            if parsed is not None:
                name, specification = parsed
                effects.append(PackageEffect(name, action, specification, operation.source, operation.line))
            index += 1
    return tuple(effects)


def package_outcomes(operations: tuple[Operation, ...]) -> tuple[PackageOutcome, ...]:
    """Resolve pip actions and their attached requirements in build order."""
    outcomes: dict[str, PackageOutcome] = {}
    for operation in operations:
        if operation.kind not in ("pip-install", "pip-uninstall", "requirement"):
            continue
        for effect in package_effects((operation,)):
            action = (
                "unknown"
                if operation.condition or operation.unresolved or has_requirement_marker(operation)
                else effect.action
            )
            if action == "require":
                action = "install"
            outcomes[effect.name] = PackageOutcome(
                effect.name, action, effect.specification, operation.source, operation.line
            )
    return tuple(outcomes.values())
