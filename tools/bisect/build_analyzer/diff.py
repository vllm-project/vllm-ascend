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

"""Compare ordered semantic IR; source positions are provenance only."""

import difflib
import posixpath
import shlex
from urllib.parse import urlsplit

from tools.bisect.build_analyzer.base_image import BaseImageDelta
from tools.bisect.build_analyzer.ir import BuildAnalysis, Operation, PackageDelta
from tools.bisect.build_analyzer.package_effects import (
    has_requirement_marker,
    package_effects,
    package_outcomes,
)

_FULLY_MODELED_PACKAGE_OPERATIONS = frozenset({"pip-uninstall", "requirement"})


def _final_package_keys(analysis: BuildAnalysis) -> dict[str, tuple[object, ...]]:
    """Represent known final package values and the writer that produced them."""
    keys: dict[str, tuple[object, ...]] = {}
    for operation in analysis.operations:
        for effect in package_effects((operation,)):
            action = "install" if effect.action == "require" else effect.action
            if operation.condition or operation.unresolved or has_requirement_marker(operation):
                action = "unknown"
            keys[effect.name] = (action, effect.specification, operation.semantic_key)
    return keys


def _neutral_package_change(
    operation: Operation,
    old_state: dict[str, tuple[object, ...]],
    new_state: dict[str, tuple[object, ...]],
) -> bool:
    """Discard a fully modeled intermediate write erased by a later writer."""
    if operation.kind not in _FULLY_MODELED_PACKAGE_OPERATIONS:
        return False
    effects = package_effects((operation,))
    return bool(effects) and all(
        name in old_state and old_state[name] == new_state.get(name) and old_state[name][0] != "unknown"
        for name in (effect.name for effect in effects)
    )


def package_state_changes(before: BuildAnalysis, after: BuildAnalysis) -> tuple[PackageDelta, ...]:
    """Compare final package effects, including the identity of their writers."""
    old_outcomes = {
        outcome.name: outcome for outcome in (before.package_outcomes or package_outcomes(before.operations))
    }
    new_outcomes = {outcome.name: outcome for outcome in (after.package_outcomes or package_outcomes(after.operations))}
    old_keys = _final_package_keys(before)
    new_keys = _final_package_keys(after)
    return tuple(
        PackageDelta(name, old_outcomes.get(name), new_outcomes.get(name))
        for name in sorted(old_outcomes.keys() | new_outcomes.keys())
        if old_keys.get(name) != new_keys.get(name)
    )


def local_wheels(operation: Operation) -> tuple[str, ...]:
    """Local wheel paths read by a pip install operation."""
    if operation.kind != "pip-install":
        return ()
    try:
        words = shlex.split(operation.value)
    except ValueError:
        return ()
    return tuple(word for word in words if word.startswith("/") and word.endswith(".whl"))


def downloaded_artifact(operation: Operation) -> str | None:
    """Output path of a simple wget download, if statically known."""
    if operation.kind != "download":
        return None
    if operation.outputs:
        return operation.outputs[0]
    try:
        words = shlex.split(operation.value)
    except ValueError:
        return None
    for flag in ("-O", "-o", "--output"):
        if flag in words and words.index(flag) + 1 < len(words):
            return words[words.index(flag) + 1]
    if "-P" in words and words.index("-P") + 1 < len(words):
        destination = words[words.index("-P") + 1]
        url = next((word for word in words if word.startswith(("http://", "https://"))), None)
        if url:
            return posixpath.join(destination, posixpath.basename(urlsplit(url).path))
    return None


def produces_artifact(operation: Operation, artifact: str) -> bool:
    """Match an exact download or an extraction-directory output."""
    return (
        (
            operation.kind.startswith("git-")
            and any(
                artifact == output.rstrip("/") or artifact.startswith(output) for output in operation.resource_writes
            )
        )
        or downloaded_artifact(operation) == artifact
        or (
            operation.value.startswith("tar ")
            and any(output.endswith("/") and artifact.startswith(output) for output in operation.resource_writes)
        )
    )


def _apt_cache_cleanup(operation: Operation) -> bool:
    """Recognize transient APT cache cleanup in an unchanged RUN command."""
    try:
        words = shlex.split(operation.value)
    except ValueError:
        return False
    return (
        words[:2] == ["rm", "-rf"]
        and len(words) > 2
        and all(path.startswith(("/var/cache/apt/", "/var/lib/apt/lists/")) for path in words[2:])
    )


def _apt_base_inputs_changed(delta: BaseImageDelta) -> bool:
    """Whether the new base changes files consumed by APT operations."""
    prefixes = (
        "etc/apt/",
        "var/lib/apt/",
        "var/lib/dpkg/",
        "var/cache/apt/",
        "usr/bin/apt",
        "usr/bin/dpkg",
        "usr/lib/apt/",
        "usr/lib/dpkg/",
        "usr/lib/aarch64-linux-gnu/libapt",
        "usr/lib/x86_64-linux-gnu/libapt",
    )
    return any(path.startswith(prefixes) for path in delta.changed_files)


def _install_owner(operations: tuple[Operation, ...], index: int) -> Operation | None:
    """Read the install provenance established by call-chain analysis."""
    owner_index = operations[index].owner_index
    if owner_index is not None:
        if 0 <= owner_index < index and operations[owner_index].kind == "pip-install":
            owner = operations[owner_index]
            if operations[index].owner_key == (owner.source, owner.value):
                return owner
    return None


def _required_replays(
    before: BuildAnalysis,
    after: BuildAnalysis,
    changes: list[tuple[str, Operation | None, Operation | None]],
) -> list[tuple[str, Operation | None, Operation | None]]:
    """Close changed package resources over their target build writers."""
    operations = after.operations
    writes_by_owner: dict[int, set[str]] = {}
    final_writers: dict[str, tuple[int, Operation]] = {}
    for index, operation in enumerate(operations):
        effects = package_effects((operation,))
        if not effects:
            continue
        owner = _install_owner(operations, index) if operation.kind == "requirement" else operation
        if owner is None:
            continue
        writes_by_owner.setdefault(id(owner), set()).update(effect.name for effect in effects)
        for effect in effects:
            final_writers[effect.name] = index, owner

    selected = {id(new) for kind, _, new in changes if kind in {"added", "replay"} and new is not None}
    removed_resources = {
        effect.name
        for kind, old, _ in changes
        if kind == "removed" and old is not None
        for effect in package_effects((old,))
    }
    old_outcomes = {
        item.name: (item.action, item.specification)
        for item in (before.package_outcomes or package_outcomes(before.operations))
    }
    new_outcomes = {
        item.name: (item.action, item.specification)
        for item in (after.package_outcomes or package_outcomes(operations))
    }
    replays: list[tuple[str, Operation | None, Operation | None]] = []
    positions = {id(operation): index for index, operation in enumerate(operations)}
    installs = {
        (operation.source, operation.value): operation for operation in operations if operation.kind == "pip-install"
    }
    for kind, old, new in changes:
        dependency = new if new is not None else old
        if dependency is None or dependency.kind != "source-digest" or dependency.source.endswith(".txt"):
            continue
        owner = (
            _install_owner(operations, positions[id(new)]) if new is not None else installs.get(dependency.owner_key)
        )
        if owner is not None and id(owner) not in selected:
            selected.add(id(owner))
            replays.append(("replay", None, owner))
    while True:
        changed_resources: set[str] = set()
        for operation in operations:
            if id(operation) not in selected and any(
                read == write
                or (write.endswith("/") and (read.startswith(write) or read.rstrip("/") == write.rstrip("/")))
                for read in operation.resource_reads
                for write in changed_resources
            ):
                selected.add(id(operation))
                replays.append(("replay", None, operation))
            if id(operation) in selected:
                changed_resources.update(operation.resource_writes)
        artifact_producer = None
        for index, operation in enumerate(operations):
            if id(operation) not in selected:
                continue
            for artifact in (*local_wheels(operation), *operation.resource_reads):
                producer = next(
                    (candidate for candidate in reversed(operations[:index]) if produces_artifact(candidate, artifact)),
                    None,
                )
                if producer is not None and id(producer) not in selected:
                    artifact_producer = producer
                    break
            if artifact_producer is not None:
                break
        if artifact_producer is not None:
            selected.add(id(artifact_producer))
            replays.append(("replay", None, artifact_producer))
            continue
        selected_positions = {
            name: max(
                (
                    position
                    for position, operation in enumerate(operations)
                    if id(operation) in selected
                    and (
                        name in writes_by_owner.get(id(operation), ())
                        or any(effect.name == name for effect in package_effects((operation,)))
                    )
                ),
                default=-1,
            )
            for name in final_writers
        }
        needed = set(removed_resources)
        needed.update(name for owner in selected for name in writes_by_owner.get(owner, ()))
        needed.update(
            effect.name
            for operation in operations
            if id(operation) in selected
            for effect in package_effects((operation,))
        )
        next_owner = None
        for name in sorted(needed):
            writer = final_writers.get(name)
            if writer is None:
                continue
            writer_position, owner = writer
            if id(owner) in selected:
                continue
            changed_final_state = name in removed_resources and old_outcomes.get(name) != new_outcomes.get(name)
            if (changed_final_state and selected_positions[name] < writer_position) or (
                selected_positions[name] >= 0 and writer_position > selected_positions[name]
            ):
                next_owner = owner
                break
        if next_owner is None:
            break
        selected.add(id(next_owner))
        replays.append(("replay", None, next_owner))
    return replays


def _live_environment_operations(analysis: BuildAnalysis) -> set[int]:
    writers: dict[str, Operation] = {}
    live: set[int] = set()
    source = None
    for operation in analysis.operations:
        if operation.kind == "from":
            if operation.source == source:
                writers.clear()
            source = operation.source
        if operation.kind == "env":
            for output in operation.resource_writes:
                writers[output.removeprefix("env:")] = operation
        else:
            live.update(id(writers[name]) for name, _ in operation.environment if name in writers)
    live.update(id(operation) for name, operation in writers.items() if name in dict(analysis.environment))
    return live


def compare_builds(
    before: BuildAnalysis,
    after: BuildAnalysis,
    base_delta: BaseImageDelta | None = None,
    *,
    base_environment_changed: bool = False,
    replay_operations: tuple[Operation, ...] = (),
) -> tuple[tuple[str, Operation | None, Operation | None], ...]:
    """Align effects by semantics; provenance line moves do not create a delta."""
    old = [operation.semantic_key for operation in before.operations]
    new = [operation.semantic_key for operation in after.operations]
    changes: list[tuple[str, Operation | None, Operation | None]] = []
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    old_state = _final_package_keys(before)
    new_state = _final_package_keys(after)
    old_environment = _live_environment_operations(before)
    new_environment = _live_environment_operations(after)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        for index in range(old_start, old_end):
            operation = before.operations[index]
            if operation.kind == "env" and id(operation) not in old_environment:
                continue
            if not _neutral_package_change(operation, old_state, new_state):
                changes.append(("removed", operation, None))
        for index in range(new_start, new_end):
            operation = after.operations[index]
            if operation.kind == "env" and id(operation) not in new_environment:
                continue
            if not _neutral_package_change(operation, old_state, new_state):
                changes.append(("added", None, operation))
    selected = {id(operation) for _, _, operation in changes if operation is not None}
    for operation in replay_operations:
        if id(operation) not in selected:
            changes.append(("replay", None, operation))
            selected.add(id(operation))
    if base_delta is not None and (base_delta.changed_files or base_delta.changed_config):
        # Only replay operations whose inputs can be identified in the base
        # delta. The build manager separately rebuilds native extensions after
        # a base change; replaying every unchanged Dockerfile command would
        # repeat source clones, image metadata directives and transient setup.
        selected = {id(new) for kind, _, new in changes if new is not None and kind in ("added", "replay")}
        replay_apt = _apt_base_inputs_changed(base_delta) or any(
            operation.kind.startswith("apt-get-") and id(operation) in selected for operation in after.operations
        )
        if replay_apt:
            for operation in after.operations:
                if not (operation.kind.startswith("apt-get-") or _apt_cache_cleanup(operation)):
                    continue
                if id(operation) not in selected:
                    changes.append(("replay", None, operation))
                    selected.add(id(operation))
    if base_environment_changed:
        changed_resources = {"base:environment"}
        selected = {id(new) for _, _, new in changes if new is not None}
        while True:
            previous = len(selected), len(changed_resources)
            for operation in after.operations:
                if not any(
                    read == resource or (resource.endswith("/") and read.startswith(resource))
                    for read in operation.resource_reads
                    for resource in changed_resources
                ):
                    continue
                if id(operation) not in selected:
                    changes.append(("replay", None, operation))
                    selected.add(id(operation))
                changed_resources.update(operation.resource_writes)
            if (len(selected), len(changed_resources)) == previous:
                break
    changes.extend(_required_replays(before, after, changes))
    return tuple(changes)
