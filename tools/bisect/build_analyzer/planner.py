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
import shlex
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

import regex as re
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from tools.bisect.build_analyzer.base_image import BaseImageDelta
from tools.bisect.build_analyzer.diff import (
    compare_builds,
    local_wheels,
    package_state_changes,
    produces_artifact,
)
from tools.bisect.build_analyzer.git_commands import parse_git_command
from tools.bisect.build_analyzer.image_history import target_stage_plan
from tools.bisect.build_analyzer.ir import BuildAnalysis, Operation
from tools.bisect.build_analyzer.operation_rules import (
    explicit_rule_for,
    warning_only_download,
    warning_only_repository,
)
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
    content_image: str | None = None
    destination: str | None = None
    environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TransitionPlan:
    status: str
    actions: tuple[PlanAction, ...]
    blockers: tuple[str, ...]
    skipped: tuple[str, ...]
    preflight: tuple[str, ...]
    postflight: tuple[str, ...]
    notes: tuple[str, ...]
    previous_environment: tuple[tuple[str, str], ...] = ()
    target_environment: tuple[tuple[str, str], ...] | None = None


def plan_target_image_transition(
    before: BuildAnalysis, after: BuildAnalysis, reference: str, history: list[dict], *, rebuild_project: bool = False
) -> TransitionPlan:
    """Replay the target base image, then the complete target build chain."""
    replay = target_stage_plan(history, reference)
    base = plan_transition(
        before,
        after,
        rebuild_project=rebuild_project,
        replay_all=True,
    )
    actions = []
    for step in replay["actions"]:
        identifier = len(actions) + 1
        actions.append(
            PlanAction(
                identifier,
                f"image-{step.kind}",
                step.command,
                "<target-image-history>",
                identifier,
                requires=(identifier - 1,) if identifier > 1 else (),
                artifact=step.content_path,
                content_image=step.content_image,
                destination=step.destination,
            )
        )
    offset = len(actions)
    actions.extend(
        replace(
            action,
            id=action.id + offset,
            requires=tuple(item + offset for item in action.requires) or ((offset,) if offset else ()),
        )
        for action in base.actions
    )
    blockers = tuple(
        dict.fromkeys(
            (
                *replay["blockers"],
                *base.blockers,
            )
        )
    )
    return replace(
        base,
        status="blocked" if blockers else "reviewable",
        actions=tuple(actions),
        blockers=blockers,
        notes=base.notes
        + (
            f"Target image content source: {reference}",
            "Replay target image including its Python, then all target build operations; no downstream diff selection",
            *replay.get("warnings", ()),
        ),
    )


_DECLARATIONS = frozenset({"arg", "assign", "if-select", "case-select"})
_APT_MUTATIONS = frozenset({"install", "remove", "purge"})
_APT_VALUE_OPTIONS = frozenset({"-o", "--option", "-c", "--config-file", "-t", "--target-release"})


def _apt_operation(operation: Operation) -> tuple[str, tuple[str, ...]] | None:
    """Read explicit apt-get package effects without treating options as packages."""
    words = _words(operation)
    if len(words) < 2 or words[0] != "apt-get":
        return None
    command = words[1]
    if command not in _APT_MUTATIONS:
        return command, ()
    packages: list[str] = []
    index = 2
    while index < len(words):
        word = words[index]
        if word in _APT_VALUE_OPTIONS:
            index += 2
            continue
        if word.startswith("-"):
            index += 1
            continue
        packages.append(word)
        index += 1
    return command, tuple(packages)


def _apt_package_name(specification: str) -> str:
    return specification.split("=", 1)[0].rstrip("+-")


def _apt_outcomes(analysis: BuildAnalysis) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for operation in analysis.operations:
        parsed = _apt_operation(operation)
        if parsed is None or parsed[0] not in _APT_MUTATIONS:
            continue
        command, packages = parsed
        for specification in packages:
            outcomes[_apt_package_name(specification)] = command
    return outcomes


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


def _repository_clone(operation: Operation) -> tuple[str, str, str | None] | None:
    """Read a clone with an explicit destination and optional branch/tag."""
    if operation.kind != "git-clone":
        return None
    git = parse_git_command(_words(operation))
    if git is None or git.subcommand != "clone":
        return None
    words = list(git.arguments)
    if len(words) < 2:
        return None
    if any(
        word in {"--bare", "--mirror", "--separate-git-dir"} or word.startswith("--separate-git-dir=") for word in words
    ):
        return None
    reference = None
    for index, word in enumerate(words[:-2]):
        if word in {"-b", "--branch"}:
            if index + 1 >= len(words) - 2:
                return None
            reference = words[index + 1]
        elif word.startswith("--branch="):
            reference = word.partition("=")[2]
        elif word.startswith("-b"):
            reference = word[2:]
    repository, directory = words[-2:]
    if (
        not directory.startswith("/")
        or posixpath.normpath(directory) == "/"
        or ".." in directory.split("/")
        or not repository.startswith(("https://", "http://", "ssh://", "git@"))
    ):
        return None
    return repository, directory, reference


def _repository_sync_command(operation: Operation, repository: str, directory: str, reference: str | None) -> str:
    """Replay clone itself so reference discovery and metadata match the recipe."""
    clone_words = _words(operation)
    clone_words[-1] = "__BISECT_CLONE_DESTINATION__"
    clone_command = shlex.join(clone_words).replace("__BISECT_CLONE_DESTINATION__", '"$bisect_stage/repository"')
    return "\n".join(
        (
            f"bisect_directory={shlex.quote(directory)}",
            f"bisect_repository={shlex.quote(repository)}",
            f"bisect_reference={shlex.quote(reference or 'HEAD')}",
            'printf "[git-sync] target=%s ref=%s directory=%s\\n" '
            '"$bisect_repository" "$bisect_reference" "$bisect_directory"',
            '  mkdir -p "$(dirname "$bisect_directory")"',
            '  bisect_stage=$(mktemp -d "${bisect_directory}.bisect-XXXXXX")',
            f'  {clone_command} || {{ echo "[git-sync] clone failed; original directory retained" >&2; exit 1; }}',
            '  if [ -e "$bisect_directory" ] || [ -L "$bisect_directory" ]; then',
            '    mv "$bisect_directory" "$bisect_stage/previous"',
            '    printf "[git-sync] previous directory retained at %s/previous\\n" "$bisect_stage"',
            "  fi",
            '  if ! mv "$bisect_stage/repository" "$bisect_directory"; then',
            '    if [ -e "$bisect_stage/previous" ] || [ -L "$bisect_stage/previous" ]; then',
            '      mv "$bisect_stage/previous" "$bisect_directory"',
            "    fi",
            "    exit 1",
            "  fi",
        )
    )


def _mark_outputs_available(operation: Operation, unavailable: set[str]) -> None:
    unavailable.difference_update(
        resource
        for resource in tuple(unavailable)
        if any(
            resource == output or (output.endswith("/") and resource.startswith(output))
            for output in operation.resource_writes
        )
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
    if operation.kind == "env":
        return "set-build-environment", None, None, False
    if operation.kind == "file-write":
        return "write-file", None, None, False
    if operation.kind in {"git-init", "git-fetch", "git-checkout", "git-switch", "git-reset", "git-submodule"}:
        return operation.kind, None, None, False
    if operation.kind == "pip-install":
        return "install", _package_from_install(words), None, False
    if operation.kind == "pip-uninstall":
        effects = package_effects((operation,))
        return "uninstall", effects[0].name if len(effects) == 1 else None, None, False
    if apt := _apt_operation(operation):
        command, _ = apt
        if command in {"update", "install", "remove", "purge", "clean", "autoclean", "autoremove"}:
            return f"apt-{command}", None, None, command in {"clean", "autoclean"}
    if rule := explicit_rule_for(operation):
        return rule.forward_action_kind, None, None, False
    if operation.kind == "source":
        return "load-environment", None, None, False
    if operation.kind == "bash" and len(words) >= 2:
        return "build-script", None, None, False
    if operation.kind == "pipeline":
        return "pipeline", None, None, False
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
        return "download", None, artifact, warning_only_download(artifact)
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


def plan_transition(
    before: BuildAnalysis,
    after: BuildAnalysis,
    base_delta: BaseImageDelta | None = None,
    *,
    base_environment_changed: bool = False,
    rebuild_project: bool = False,
    replay_all: bool = False,
) -> TransitionPlan:
    """Plan incremental effects, or replay the whole target after an image change."""
    project_installs = tuple(
        operation
        for operation in after.operations
        if operation.kind == "pip-install"
        and any(word.rstrip("/") == "/vllm-workspace/vllm-ascend" for word in _words(operation))
    )
    changes = (
        tuple(("replay", None, operation) for operation in after.operations)
        if replay_all
        else compare_builds(
            before,
            after,
            base_delta,
            base_environment_changed=base_environment_changed,
            replay_operations=project_installs if rebuild_project else (),
        )
    )
    additions = [operation for kind, _, operation in changes if kind in ("added", "replay") and operation is not None]
    removed = [operation for kind, operation, _ in changes if kind == "removed" and operation is not None]
    old_apt = {} if replay_all else _apt_outcomes(before)
    target_apt = _apt_outcomes(after)
    if base_delta is not None:
        additions = [operation for operation in additions if operation.kind != "from"]
        removed = [operation for operation in removed if operation.kind != "from"]
    target_package_effects = after.package_effects or package_effects(after.operations)
    target_outcomes = {
        outcome.name: outcome for outcome in (after.package_outcomes or package_outcomes(after.operations))
    }
    operation_positions = {id(operation): index for index, operation in enumerate(after.operations)}
    workdirs: dict[tuple[str, int], str] = {}
    workdir = "/"
    if replay_all:
        for operation in after.operations:
            if operation.kind == "workdir":
                workdir = posixpath.normpath(posixpath.join(workdir, operation.value))
            workdirs[operation.source, operation.line] = workdir
    additions.sort(key=lambda operation: operation_positions[id(operation)])
    if any(_apt_operation(operation) and _apt_operation(operation)[0] in _APT_MUTATIONS for operation in additions):
        first_mutation = min(
            operation_positions[id(operation)]
            for operation in additions
            if _apt_operation(operation) and _apt_operation(operation)[0] in _APT_MUTATIONS
        )
        updates = [
            operation
            for operation in after.operations[:first_mutation]
            if _apt_operation(operation) and _apt_operation(operation)[0] == "update"
        ]
        if updates and updates[-1] not in additions:
            additions.insert(0, updates[-1])
    selected_installs = {id(operation) for operation in additions if operation.kind == "pip-install"}
    selected_install_keys = {
        (operation.source, operation.value) for operation in additions if operation.kind == "pip-install"
    }
    covered_sources = {
        id(operation)
        for operation in after.operations
        if operation.kind in {"source-digest", "requirement", "documentation"}
        and operation.owner_index is not None
        and 0 <= operation.owner_index < len(after.operations)
        and id(after.operations[operation.owner_index]) in selected_installs
        and operation.owner_key
        == (after.operations[operation.owner_index].source, after.operations[operation.owner_index].value)
    }
    covered_requirements = {
        id(operation)
        for operation in after.operations
        if operation.kind == "requirement" and id(operation) in covered_sources
    }
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
    blockers: list[str] = []
    if rebuild_project and not project_installs:
        blockers.append("Target build has no project installation to replay")
    skipped: list[str] = []
    actions: list[PlanAction] = []
    removed_environment = set(dict(before.environment)) - set(dict(after.environment))
    if removed_environment:
        actions.append(
            PlanAction(1, "restore-build-environment", " ".join(sorted(removed_environment)), "<environment>", 0)
        )
    if base_delta is not None and (base_delta.changed_files or base_delta.changed_config):
        source = next((operation for operation in after.operations if operation.kind == "from"), None)
        actions.append(
            PlanAction(
                id=len(actions) + 1,
                kind="apply-base-image",
                command=f"{base_delta.after.reference} (digest {base_delta.after.digest})",
                source=source.source if source else "<base-image>",
                line=source.line if source else 0,
                requires=(actions[-1].id,) if actions else (),
            )
        )
    planned_uninstalls: set[str] = set()
    if base_delta is None:
        for package in sorted(old_apt):
            if old_apt[package] != "install" or package in target_apt:
                continue
            action_id = len(actions) + 1
            actions.append(
                PlanAction(
                    id=action_id,
                    kind="apt-remove",
                    command=shlex.join(["apt-get", "remove", "-y", package]),
                    source="<apt-state>",
                    line=0,
                    requires=(action_id - 1,) if action_id > 1 else (),
                    package=package,
                )
            )
    for delta in () if replay_all else package_state_changes(before, after):
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
                produces_artifact(producer, artifact)
                and selected_positions[id(producer)] < selected_positions[id(operation)]
                for producer in additions
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
        if operation.kind == "env":
            continue
        if operation.kind == "unresolved":
            skipped.append(f"{operation.source}:{operation.line}: removed diagnostic: {operation.value}")
            continue
        if operation.kind == "source-digest" and operation.owner_key in selected_install_keys:
            continue
        if operation.kind.startswith("git-") and any(
            target.kind.startswith("git-")
            and any(
                old == new or (new.endswith("/") and old.startswith(new))
                for old in operation.resource_writes
                for new in target.resource_writes
            )
            for target in after.operations
        ):
            continue
        old_clone = _repository_clone(operation)
        if old_clone and any(
            (clone := _repository_clone(target)) is not None and clone[:2] == old_clone[:2] for target in additions
        ):
            continue
        if operation.kind in _DECLARATIONS:
            continue
        if (old_apt_operation := _apt_operation(operation)) is not None:
            command, packages = old_apt_operation
            if command in {"remove", "purge"} and any(
                _apt_package_name(package) not in target_apt for package in packages
            ):
                blockers.append(
                    f"{operation.source}:{operation.line}: removed apt-get {command}; "
                    "the previous package versions cannot be inferred"
                )
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
    unavailable_resources: set[str] = set()
    for operation in additions:
        if replay_all and operation.kind in {"from", "workdir", "label", "cmd", "entrypoint"}:
            skipped.append(f"{operation.source}:{operation.line}: {operation.kind} is represented by replay context")
            continue
        if (
            replay_all
            and operation.kind == "copy"
            and _words(operation)
            in (
                [".", "/vllm-workspace/vllm-ascend/"],
                [".", "/vllm-workspace/vllm-ascend"],
            )
        ):
            skipped.append(f"{operation.source}:{operation.line}: project COPY supplied by target Git checkout")
            continue
        unavailable = next(
            (
                output
                for output in unavailable_resources
                if any(
                    read == output
                    or (output.endswith("/") and (read.startswith(output) or read.rstrip("/") == output.rstrip("/")))
                    for read in operation.resource_reads
                )
            ),
            None,
        )
        if unavailable is not None:
            skipped.append(f"{operation.source}:{operation.line}: dependency unavailable: {unavailable}")
            unavailable_resources.update(operation.resource_writes)
            continue
        if id(operation) in covered_sources:
            continue
        if operation.kind == "documentation":
            skipped.append(f"{operation.source}:{operation.line}: documentation has no standalone replay action")
            continue
        clone = _repository_clone(operation)
        if clone:
            repository, directory, reference = clone
            command = _repository_sync_command(operation, repository, directory, reference)
            identifier = len(actions) + 1
            actions.append(
                PlanAction(
                    identifier,
                    "git-sync",
                    command,
                    operation.source,
                    operation.line,
                    requires=(identifier - 1,) if identifier > 1 else (),
                    artifact=directory,
                    optional=warning_only_repository(repository),
                )
            )
            _mark_outputs_available(operation, unavailable_resources)
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
        if classified is None and replay_all and operation.kind == "shell":
            # A fully parsed concrete shell command belongs to the target
            # recipe even when it has no incremental inverse/classification.
            classified = "shell", None, None, False
        if classified is None:
            blockers.append(f"{operation.source}:{operation.line}: cannot replay {operation.kind}: {operation.value}")
            unavailable_resources.update(operation.resource_writes)
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
        _mark_outputs_available(operation, unavailable_resources)
    old_declarations = {
        (operation.source, operation.kind, name): operation
        for operation in removed
        if (name := _declaration_name(operation)) is not None
    }
    for operation in () if replay_all else additions:
        name = _declaration_name(operation)
        if name is None:
            continue
        old = old_declarations.get((operation.source, operation.kind, name))
        before_value = old.value.partition("=")[2] if old else "<absent>"
        after_value = operation.value.partition("=")[2]
        consumers = _consumers(after, operation)
        if consumers and not any(
            consumer in additions and (_classify(consumer) or _repository_clone(consumer)) for consumer in consumers
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
    if base_delta is not None and (base_delta.changed_files or base_delta.changed_config):
        notes += (
            "Base image application requires live-container validation; "
            "Native rebuilds are represented by target project installation actions",
        )
    return TransitionPlan(
        status,
        tuple(
            replace(
                action,
                environment=next(
                    (
                        operation.environment
                        for operation in after.operations
                        if (operation.source, operation.line) == (action.source, action.line)
                    ),
                    (),
                ),
                shell_context=(
                    (f"cd {shlex.quote(workdirs[action.source, action.line])}", *action.shell_context)
                    if replay_all and (action.source, action.line) in workdirs
                    else action.shell_context
                ),
            )
            for action in actions
        ),
        unique_blockers,
        tuple(skipped),
        tuple(preflight),
        tuple(postflight),
        notes,
        before.environment,
        after.environment,
    )
