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

"""Lower semantic IR changes into an ordered, reviewable transition plan.

This module does not execute commands. Unknown effects and irreversible
changes block automatic replay rather than being silently skipped.
"""

import posixpath
import re
import shlex
from dataclasses import dataclass
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from tools.bisect.build_analyzer.diff import (
    compare_builds,
    downloaded_artifact,
    local_wheels,
    package_state_changes,
)
from tools.bisect.build_analyzer.ir import BuildAnalysis, Operation
from tools.bisect.build_analyzer.operation_rules import explicit_rule_for
from tools.bisect.build_analyzer.package_effects import (
    has_requirement_marker,
    package_effects,
    package_outcomes,
)


@dataclass(frozen=True)
class PlanAction:
    id: int
    kind: str
    command: str
    source: str
    line: int
    requires: tuple[int, ...] = ()
    package: str | None = None
    artifact: str | None = None
    optional: bool = False
    shell_context: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransitionPlan:
    status: str
    actions: tuple[PlanAction, ...]
    blockers: tuple[str, ...]
    skipped: tuple[str, ...]
    preflight: tuple[str, ...]
    postflight: tuple[str, ...]
    notes: tuple[str, ...]


_DECLARATIONS = frozenset({"arg", "assign", "if-select", "case-select"})


def _requirement_package(operation: Operation) -> str | None:
    if operation.kind != "requirement":
        return None
    try:
        return canonicalize_name(Requirement(operation.value).name)
    except InvalidRequirement:
        return None


def _removed_download_branch(operation: Operation) -> bool:
    """A removed branch selecting only a download route has no lasting effect."""
    if operation.kind != "branch":
        return False
    words = _words(operation)
    return "wget" in words and not any(word in words for word in ("pip", "mfcli", "apt-get", "dpkg", "git"))


def _declaration_name(operation: Operation) -> str | None:
    if operation.kind not in ("arg", "assign"):
        return None
    return operation.value.partition("=")[0]


def _consumers(analysis: BuildAnalysis, declaration: Operation) -> tuple[Operation, ...]:
    """Follow variable assignments to effects within the same Docker build layer."""
    name = _declaration_name(declaration)
    if name is None:
        return ()
    tainted = {name}
    consumers: list[Operation] = []
    start = next((index for index, item in enumerate(analysis.operations) if item is declaration), None)
    if start is None:
        return ()
    for operation in analysis.operations[start + 1 :]:
        if operation.source != declaration.source:
            continue
        assigned = _declaration_name(operation)
        if assigned:
            if tainted.intersection(operation.inputs):
                tainted.add(assigned)
            else:
                tainted.discard(assigned)
        elif tainted.intersection(operation.inputs) and operation.kind not in _DECLARATIONS:
            consumers.append(operation)
    return tuple(consumers)


def _words(operation: Operation) -> list[str]:
    try:
        return shlex.split(operation.value)
    except ValueError:
        return []


def _package_from_install(words: list[str]) -> str | None:
    if "install" not in words:
        return None
    index = words.index("install") + 1
    argument_options = frozenset(
        {
            "-r",
            "--requirement",
            "-e",
            "--editable",
            "-c",
            "--constraint",
            "--index-url",
            "--extra-index-url",
            "--find-links",
            "--target",
            "--prefix",
            "--root",
        }
    )
    while index < len(words):
        token = words[index]
        if token in argument_options:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        filename = urlsplit(token).path.rsplit("/", 1)[-1]
        if filename.endswith(".whl"):
            parts = filename.split("-", 2)
            return canonicalize_name(parts[0]) if len(parts) >= 2 else None
        match = re.match(r"([A-Za-z0-9_.-]+)(?:[<=>!~].*)?$", token)
        return canonicalize_name(match.group(1)) if match else None
    return None


def _explicit_wheel_package(operation: Operation) -> str | None:
    """Identify a package only when the removed install names a wheel."""
    if operation.kind != "pip-install":
        return None
    words = _words(operation)
    wheel = next((word for word in words if word.endswith(".whl")), None)
    if wheel is None:
        return None
    filename = urlsplit(wheel).path.rsplit("/", 1)[-1]
    parts = filename.split("-", 2)
    return canonicalize_name(parts[0]) if len(parts) >= 3 else None


def _explicit_wheel_version(operation: Operation) -> str | None:
    if operation.kind != "pip-install":
        return None
    wheel = next((word for word in _words(operation) if word.endswith(".whl")), None)
    if wheel is None:
        return None
    parts = urlsplit(wheel).path.rsplit("/", 1)[-1].split("-", 2)
    return parts[1] if len(parts) >= 3 else None


def _vllm_clone(operation: Operation) -> tuple[str, str, str] | None:
    """Read a versioned clone of the known vLLM source checkout."""
    if operation.kind != "git-clone":
        return None
    words = _words(operation)
    if words[:2] != ["git", "clone"] or "-b" not in words:
        return None
    branch_index = words.index("-b")
    if branch_index + 1 >= len(words) or len(words) < 4:
        return None
    repository, directory = words[-2:]
    if directory != "/vllm-workspace/vllm":
        return None
    return repository, directory, words[branch_index + 1]


def _vllm_editable_install(analysis: BuildAnalysis) -> Operation | None:
    return next(
        (
            operation
            for operation in analysis.operations
            if operation.kind == "pip-install"
            and "/vllm-workspace/vllm/" in operation.value
            and " -e " in operation.value
        ),
        None,
    )


def _wheel_follows_project_install(analysis: BuildAnalysis, wheel: Operation) -> bool:
    """Check that a wheel install runs after the editable project install."""
    source_order = {instruction.source: index for index, instruction in enumerate(analysis.instructions)}
    project_installs = (
        operation
        for operation in analysis.operations
        if operation.kind == "pip-install"
        and "/vllm-workspace/vllm-ascend" in operation.value
        and " -e " in operation.value
    )
    wheel_position = source_order.get(wheel.source, -1), wheel.line
    return any((source_order.get(install.source, -1), install.line) < wheel_position for install in project_installs)


def _classify(operation: Operation) -> tuple[str, str | None, str | None, bool] | None:
    words = _words(operation)
    if not words:
        return None
    if operation.kind == "pip-install":
        return "install", _package_from_install(words), None, False
    if operation.kind == "pip-uninstall":
        effects = package_effects((operation,))
        return "uninstall", effects[0].name if len(effects) == 1 else None, None, False
    if rule := explicit_rule_for(operation):
        return rule.forward_action_kind, None, None, False
    if operation.kind == "source":
        return "load-environment", None, None, False
    if operation.kind == "export":
        return "set-environment", None, None, False
    if operation.kind == "download":
        artifact = operation.outputs[0] if operation.outputs else None
        if artifact is None:
            for option in ("-O", "-o", "--output"):
                if option in words and words.index(option) + 1 < len(words):
                    artifact = words[words.index(option) + 1]
                    break
            if artifact is None and "-P" in words and words.index("-P") + 1 < len(words):
                destination = words[words.index("-P") + 1]
                url = next((word for word in words if word.startswith(("http://", "https://"))), None)
                if url:
                    artifact = posixpath.join(destination, posixpath.basename(urlsplit(url).path))
        return "download", None, artifact, False
    if words[0] == "tar" and any(flag.startswith("-x") for flag in words[1:]):
        artifact = words[words.index("-C") + 1] if "-C" in words and words.index("-C") + 1 < len(words) else None
        return "extract", None, artifact, False
    if words[:2] == ["mkdir", "-p"]:
        return "prepare-directory", None, words[-1], False
    if words[:2] == ["rm", "-rf"]:
        return "cleanup", None, None, True
    if operation.kind == "pip-cache" and words[-2:] == ["cache", "purge"]:
        return "clear-cache", None, None, True
    return None


def _shell_context(analysis: BuildAnalysis, operation: Operation) -> tuple[str, ...]:
    return tuple(dependency.value for dependency in operation.shell_dependencies)


def _requirement_install_source(
    analysis: BuildAnalysis, operation: Operation
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Inherit pip indexes and shell state from the install that read a requirement."""
    index = operation.owner_index
    if index is None or not 0 <= index < len(analysis.operations):
        return None
    owner = analysis.operations[index]
    if owner.kind != "pip-install" or (owner.source, owner.value) != operation.owner_key:
        return None
    words = _words(owner)
    flags: list[str] = []
    source_options = {"-i", "--index-url", "--extra-index-url", "-f", "--find-links", "--trusted-host"}
    for position, word in enumerate(words):
        if word in source_options:
            if position + 1 >= len(words):
                return None
            flags.extend((word, words[position + 1]))
        elif word == "--no-index" or any(
            word.startswith(option + "=") for option in source_options if option.startswith("--")
        ):
            flags.append(word)
    return tuple(flags), _shell_context(analysis, owner)


def _ephemeral_checkout_lines(analysis: BuildAnalysis) -> set[tuple[str, int]]:
    """Find Git checkout steps whose temporary tree is deleted in the same RUN."""
    transient: set[tuple[str, int]] = set()
    for instruction in analysis.instructions:
        if instruction.kind != "RUN":
            continue
        run_operations = [
            operation
            for operation in analysis.operations
            if operation.source == instruction.source and operation.line in instruction.line_map
        ]
        cleanups = [operation for operation in run_operations if _words(operation)[:2] == ["rm", "-rf"]]
        for clone in run_operations:
            if clone.kind != "git-clone":
                continue
            words = _words(clone)
            if not words:
                continue
            checkout_path = words[-1]
            if not checkout_path.startswith("/tmp/") or not any(
                cleanup.line > clone.line and checkout_path in _words(cleanup)[2:] for cleanup in cleanups
            ):
                continue
            for operation in run_operations:
                if operation.line >= clone.line and operation.kind in (
                    "git-clone",
                    "git-checkout",
                    "git-submodule",
                    "cd",
                ):
                    transient.add((operation.source, operation.line))
    return transient


def plan_transition(before: BuildAnalysis, after: BuildAnalysis) -> TransitionPlan:
    """Plan first-to-second replay of persistent build effects."""
    changes = compare_builds(before, after)
    additions = [operation for kind, _, operation in changes if kind in ("added", "replay") and operation is not None]
    removed = [operation for kind, operation, _ in changes if kind == "removed" and operation is not None]
    target_package_effects = after.package_effects or package_effects(after.operations)
    target_outcomes = {
        outcome.name: outcome for outcome in (after.package_outcomes or package_outcomes(after.operations))
    }
    operation_positions = {id(operation): index for index, operation in enumerate(after.operations)}
    additions.sort(key=lambda operation: operation_positions[id(operation)])
    selected_installs = {id(operation) for operation in additions if operation.kind == "pip-install"}
    covered_requirements: set[int] = set()
    active_install = False
    for operation in after.operations:
        if operation.kind == "pip-install":
            active_install = id(operation) in selected_installs
        elif operation.kind in ("source-digest", "requirement"):
            if active_install and operation.kind == "requirement":
                covered_requirements.add(id(operation))
        else:
            active_install = False
    replayed_installs = {
        effect.name
        for operation in after.operations
        if id(operation) in selected_installs | covered_requirements
        for effect in package_effects((operation,))
        if effect.action in ("install", "require")
    }
    target_providers = {effect.name for effect in target_package_effects if effect.action in ("require", "install")}
    target_providers.difference_update(name for name, outcome in target_outcomes.items() if outcome.action != "install")
    target_uninstalls = {name for name, outcome in target_outcomes.items() if outcome.action == "uninstall"}
    unknown_packages = {name for name, outcome in target_outcomes.items() if outcome.action == "unknown"}
    added_wheels = {
        package: operation for operation in additions if (package := _explicit_wheel_package(operation)) is not None
    }
    added_packages = {
        package
        for operation in additions
        if operation.kind == "pip-install" and (package := _package_from_install(_words(operation)))
    }
    added_effects = {
        (kind, package or artifact)
        for operation in additions
        if (classified := _classify(operation)) is not None
        for kind, package, artifact, _ in (classified,)
    }
    old_vllm_clones = {clone[1]: operation for operation in removed if (clone := _vllm_clone(operation)) is not None}
    vllm_switches: dict[Operation, tuple[Operation, tuple[str, str, str], Operation]] = {}
    vllm_install = _vllm_editable_install(after)
    for operation in additions:
        clone = _vllm_clone(operation)
        if clone is None or vllm_install is None:
            continue
        previous = old_vllm_clones.get(clone[1])
        if previous is not None and (old_clone := _vllm_clone(previous)) is not None and old_clone[0] == clone[0]:
            vllm_switches[operation] = previous, clone, vllm_install
    handled_old_clones = {previous for previous, _, _ in vllm_switches.values()}
    blockers: list[str] = []
    skipped: list[str] = []
    actions: list[PlanAction] = []
    planned_uninstalls: set[str] = set()
    for delta in package_state_changes(before, after):
        if (
            delta.after is not None
            and delta.after.action == "unknown"
            and any(
                has_requirement_marker(operation)
                and any(effect.name == delta.name for effect in package_effects((operation,)))
                for operation in after.operations
            )
        ):
            blockers.append(
                f"{delta.after.source}:{delta.after.line}: conditional requirement for "
                f"{delta.name} needs target environment evaluation"
            )
        if delta.before is None or delta.before.action != "install" or delta.after is not None:
            continue
        action_id = len(actions) + 1
        actions.append(
            PlanAction(
                id=action_id,
                kind="uninstall",
                command=f"python3 -m pip uninstall -y {delta.name}",
                source=delta.before.source,
                line=delta.before.line,
                requires=(action_id - 1,) if action_id > 1 else (),
                package=delta.name,
            )
        )
        planned_uninstalls.add(delta.name)
    selected_positions = {id(operation): operation_positions[id(operation)] for operation in additions}
    for operation in additions:
        for artifact in local_wheels(operation):
            if not any(
                downloaded_artifact(producer) == artifact
                and selected_positions[id(producer)] < selected_positions[id(operation)]
                for producer in additions
                if producer.kind == "download"
            ):
                blockers.append(
                    f"{operation.source}:{operation.line}: local wheel {artifact} has no "
                    "replayed producer; its availability cannot be established"
                )
    transient_branch_lines = {
        (operation.source, operation.line) for operation in removed if _removed_download_branch(operation)
    }
    transient_checkout_lines = _ephemeral_checkout_lines(before)
    for operation in removed:
        if operation in handled_old_clones:
            continue
        if operation.kind in _DECLARATIONS:
            continue
        if operation.kind == "source-digest" and operation.source.endswith(".txt"):
            continue
        requirement_package = _requirement_package(operation)
        if requirement_package is not None:
            if has_requirement_marker(operation):
                blockers.append(
                    f"{operation.source}:{operation.line}: conditional requirement for "
                    f"{requirement_package} needs target environment evaluation"
                )
                continue
            if requirement_package in unknown_packages:
                blockers.append(
                    f"{operation.source}:{operation.line}: target package state for "
                    f"{requirement_package} depends on an unresolved or conditional pip action"
                )
                continue
            if requirement_package in target_providers:
                continue
            if requirement_package in target_uninstalls:
                continue
            if requirement_package in planned_uninstalls:
                continue
            action_id = len(actions) + 1
            actions.append(
                PlanAction(
                    id=action_id,
                    kind="uninstall",
                    command=f"python3 -m pip uninstall -y {requirement_package}",
                    source=operation.source,
                    line=operation.line,
                    requires=(action_id - 1,) if action_id > 1 else (),
                    package=requirement_package,
                )
            )
            planned_uninstalls.add(requirement_package)
            continue
        if operation.kind in ("source", "export", "cd"):
            continue
        if (operation.source, operation.line) in transient_checkout_lines:
            continue
        if (operation.source, operation.line) in transient_branch_lines and operation.kind in (
            "branch",
            "unresolved",
        ):
            continue
        if operation.kind in ("source-digest", "requirement"):
            blockers.append(
                f"{operation.source}:{operation.line}: dependency source changed; "
                "the required reinstall or rebuild has not been inferred"
            )
            continue
        if operation.kind == "pip-uninstall":
            old_packages = {effect.name for effect in package_effects((operation,))}
            if old_packages and all(
                package in replayed_installs
                if package in target_providers
                else package in target_uninstalls or package not in target_outcomes
                for package in old_packages
            ):
                continue
        package = _package_from_install(_words(operation)) if operation.kind == "pip-install" else None
        if operation.kind == "pip-install":
            old_packages = {effect.name for effect in package_effects((operation,))}
            if old_packages and old_packages <= (target_providers | target_uninstalls | planned_uninstalls):
                continue
        if package and package in added_packages:
            continue
        wheel_package = _explicit_wheel_package(operation)
        if wheel_package in unknown_packages:
            blockers.append(
                f"{operation.source}:{operation.line}: target package state for "
                f"{wheel_package} depends on an unresolved or conditional pip action"
            )
            continue
        if wheel_package and wheel_package in target_providers:
            continue
        if wheel_package:
            if wheel_package in target_uninstalls:
                continue
            if wheel_package in planned_uninstalls:
                continue
            action_id = len(actions) + 1
            actions.append(
                PlanAction(
                    id=action_id,
                    kind="uninstall",
                    command=f"python3 -m pip uninstall -y {wheel_package}",
                    source=operation.source,
                    line=operation.line,
                    requires=(action_id - 1,) if action_id > 1 else (),
                    package=wheel_package,
                )
            )
            planned_uninstalls.add(wheel_package)
            continue
        inverse_rule = explicit_rule_for(operation)
        if inverse_rule is not None and not any(
            target.kind == operation.kind and target.value == operation.value for target in after.operations
        ):
            inverse_command = inverse_rule.inverse_command(operation)
            if inverse_command is None:
                blockers.append(f"{operation.source}:{operation.line}: inverse rule {inverse_rule.name} did not match")
                continue
            action_id = len(actions) + 1
            actions.append(
                PlanAction(
                    id=action_id,
                    kind=inverse_rule.inverse_action_kind,
                    command=inverse_command,
                    source=operation.source,
                    line=operation.line,
                    requires=(action_id - 1,) if action_id > 1 else (),
                )
            )
            continue
        old_effect = _classify(operation)
        if old_effect and (old_effect[0], old_effect[1] or old_effect[2]) in added_effects:
            continue
        if old_effect and old_effect[0] in ("download", "extract", "prepare-directory", "cleanup", "clear-cache"):
            continue
        if operation.kind == "shell" and _words(operation)[:1] == ["echo"]:
            continue
        if old_effect and old_effect[3]:
            continue
        blockers.append(
            f"{operation.source}:{operation.line}: removed {operation.kind}; no inverse operation can be inferred"
        )
    for operation in additions:
        if operation in vllm_switches:
            _, (repository, directory, reference), install = vllm_switches[operation]
            commands = (
                ("git-fetch", shlex.join(["git", "-C", directory, "fetch", "--depth", "1", repository, reference])),
                ("git-checkout", shlex.join(["git", "-C", directory, "checkout", "FETCH_HEAD"])),
                ("install", install.value),
            )
            for kind, command in commands:
                action_id = len(actions) + 1
                actions.append(
                    PlanAction(
                        id=action_id,
                        kind=kind,
                        command=command,
                        source=operation.source,
                        line=operation.line,
                        requires=(action_id - 1,) if action_id > 1 else (),
                        package="vllm" if kind == "install" else None,
                        shell_context=_shell_context(after, install) if kind == "install" else (),
                    )
                )
            continue
        if operation.kind in _DECLARATIONS or operation.kind in ("source", "export", "cd"):
            continue
        if operation.kind == "source-digest" and operation.source.endswith(".txt"):
            continue
        requirement_package = _requirement_package(operation)
        if requirement_package is not None:
            if id(operation) in covered_requirements:
                continue
            install_source = _requirement_install_source(after, operation)
            if install_source is None:
                blockers.append(
                    f"{operation.source}:{operation.line}: requirement install source could not be resolved"
                )
                continue
            if requirement_package in added_wheels:
                wheel = added_wheels[requirement_package]
                version = _explicit_wheel_version(wheel) or "<unknown>"
                if _wheel_follows_project_install(after, wheel):
                    skipped.append(
                        f"{operation.source}:{operation.line}: {operation.value} is superseded "
                        f"by {wheel.source}:{wheel.line} wheel version {version}"
                    )
                else:
                    blockers.append(
                        f"{operation.source}:{operation.line}: install order relative to "
                        f"{wheel.source}:{wheel.line} wheel version {version} is unknown"
                    )
                continue
            action_id = len(actions) + 1
            source_flags, shell_context = install_source
            actions.append(
                PlanAction(
                    id=action_id,
                    kind="install",
                    command=shlex.join(["python3", "-m", "pip", "install", operation.value, *source_flags]),
                    source=operation.source,
                    line=operation.line,
                    requires=(action_id - 1,) if action_id > 1 else (),
                    package=requirement_package,
                    shell_context=shell_context,
                )
            )
            continue
        if operation.kind in ("source-digest", "requirement"):
            blockers.append(
                f"{operation.source}:{operation.line}: dependency source changed; "
                "the required reinstall or rebuild has not been inferred"
            )
            continue
        if operation.unresolved:
            blockers.append(
                f"{operation.source}:{operation.line}: {operation.kind} has unresolved inputs: "
                f"{', '.join(operation.unresolved)}"
            )
            continue
        if operation.kind == "shell" and _words(operation)[:1] == ["echo"]:
            skipped.append(f"{operation.source}:{operation.line}: informational echo")
            continue
        classified = _classify(operation)
        if classified is None:
            blockers.append(f"{operation.source}:{operation.line}: cannot replay {operation.kind}: {operation.value}")
            continue
        kind, package, artifact, optional = classified
        action_id = len(actions) + 1
        actions.append(
            PlanAction(
                id=action_id,
                kind=kind,
                command=operation.value,
                source=operation.source,
                line=operation.line,
                requires=(action_id - 1,) if action_id > 1 else (),
                package=package,
                artifact=artifact,
                optional=optional,
                shell_context=_shell_context(after, operation),
            )
        )
    old_declarations = {
        (operation.source, operation.kind, name): operation
        for operation in removed
        if (name := _declaration_name(operation)) is not None
    }
    for operation in additions:
        name = _declaration_name(operation)
        if name is None:
            continue
        old = old_declarations.get((operation.source, operation.kind, name))
        before_value = old.value.partition("=")[2] if old else "<absent>"
        after_value = operation.value.partition("=")[2]
        consumers = _consumers(after, operation)
        if consumers and not any(
            consumer in additions and (_classify(consumer) or consumer in vllm_switches) for consumer in consumers
        ):
            blockers.append(
                f"{operation.source}:{operation.line}: {name} affects an active operation "
                "but no replay action was inferred"
            )
        elif not consumers:
            skipped.append(
                f"{operation.source}:{operation.line}: {name} {before_value} -> {after_value}; "
                "no active consumer in this build layer"
            )
    unique_blockers = tuple(dict.fromkeys(blockers))
    status = "blocked" if unique_blockers else ("reviewable" if actions else "no-op")
    contexts = tuple(sorted({context for action in actions for context in action.shell_context}))
    preflight = ["python3 -m pip list --format=freeze"]
    preflight.extend(
        f"git -C {shlex.quote(directory)} rev-parse --is-inside-work-tree"
        for directory in sorted({clone[1] for _, clone, _ in vllm_switches.values()})
    )
    preflight.extend(
        f"test -f {words[1]}"
        for context in contexts
        if (words := _words(Operation("", 0, "source", context)))[:1] == ["source"] and len(words) == 2
    )
    installed_packages = {action.package for action in actions if action.kind == "install" and action.package}
    installed_commands = {action.command for action in actions if action.kind == "install"}
    installed_packages.update(
        effect.name
        for operation in additions
        if operation.kind == "pip-install" and operation.value in installed_commands
        for effect in package_effects((operation,))
        if effect.action == "install"
    )
    postflight = [f"python3 -m pip show {package}" for package in sorted(installed_packages)]
    postflight.append("python3 -m pip check")
    notes = (
        "Plan only; commands have not been executed",
        "Run shell_context in the same shell as each action",
        "Review target package compatibility and installed versions before replay",
    )
    return TransitionPlan(
        status,
        tuple(actions),
        unique_blockers,
        tuple(skipped),
        tuple(preflight),
        tuple(postflight),
        notes,
    )
