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

"""Select the hardware build chain and resolve copied local dependencies."""

import ast
import hashlib
import json
import posixpath
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import regex as re
import tomllib

from tools.bisect.build_analyzer.ir import BuildAnalysis, BuildSyntaxError, Operation
from tools.bisect.build_analyzer.package_effects import package_effects, package_outcomes
from tools.bisect.build_analyzer.semantics import analyze_build
from tools.bisect.build_recipe import NIGHTLY_DOCKERFILE_DIR, ROOT_DOCKERFILES, normalize_build_soc


def _read_text(repo: Path, commit: str, path: str) -> str | None:
    result = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, check=False)
    if result.returncode:
        return None
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BuildSyntaxError(f"{path}: build source is not UTF-8") from exc


def _reachable_python_nodes(node: ast.AST):
    """Walk syntax while excluding branches with a literal false condition."""
    yield node
    if isinstance(node, ast.If) and isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
        children = node.body if node.test.value else node.orelse
    else:
        children = ast.iter_child_nodes(node)
    for child in children:
        yield from _reachable_python_nodes(child)


def _analyze_source(
    read_text: Callable[[str], str | None],
    label: str,
    soc: str,
    architecture: str,
    os_variant: str = "ubuntu",
    build_args: dict[str, str] | None = None,
) -> BuildAnalysis:
    """Select root Dockerfile by hardware/OS, then parse its nightly layer.

    Architecture is an explicit input. The build's TARGETPLATFORM default is
    deliberately not inferred from the machine running the bisect tool.
    """
    target = normalize_build_soc(soc)
    if architecture not in ("aarch64", "x86_64"):
        raise ValueError(f"unsupported architecture {architecture!r}")
    root = ROOT_DOCKERFILES[target]
    if "openeuler" in os_variant.lower():
        root += ".openEuler"
    nightly = f"{NIGHTLY_DOCKERFILE_DIR}/Dockerfile.nightly.{target}"
    layers: list[tuple[str, str]] = []
    for path in (root, nightly):
        content = read_text(path)
        if content is None:
            raise BuildSyntaxError(f"{label}: missing build source {path}")
        layers.append((path, content))
    overrides = {"TARGETPLATFORM": "linux/arm64" if architecture == "aarch64" else "linux/amd64"}
    overrides.update(build_args or {})
    analysis = analyze_build(tuple(layers), overrides)
    operations = list(analysis.operations)
    diagnostics = list(analysis.diagnostics)
    if build_args is None:
        diagnostics.append("Build-arg overrides were not supplied; Dockerfile defaults were assumed")

    # Resolve only files copied from this repository's build context. A git
    # clone or an inherited base image is external, never guessed from HEAD.
    copied_roots: list[str] = []
    external_roots: list[str] = []
    cwd = "/"
    run_cwd = cwd
    current_run: tuple[str, int] | None = None
    source_files: list[tuple[str, Operation, bool]] = []
    for instruction in analysis.instructions:
        if instruction.kind == "COPY":
            try:
                words = shlex.split(instruction.body)
            except ValueError:
                continue
            if len(words) == 2 and words[0] == ".":
                copied_roots.append(posixpath.normpath(words[1]))

    def context_path(container_path: str) -> str | None:
        full = posixpath.normpath(container_path)
        if any(full == root or full.startswith(root + "/") for root in external_roots):
            return None
        for root in copied_roots:
            if full == root or full.startswith(root + "/"):
                return full[len(root) :].lstrip("/")
        return None

    for operation in analysis.operations:
        instruction = next(
            (
                item
                for item in reversed(analysis.instructions)
                if item.source == operation.source and item.line <= operation.line
            ),
            None,
        )
        if instruction is None:
            continue
        if instruction.kind == "WORKDIR":
            target = instruction.body if instruction.body.startswith("/") else posixpath.join(cwd, instruction.body)
            cwd = posixpath.normpath(target)
            run_cwd = cwd
            continue
        if instruction.kind != "RUN":
            continue
        key = instruction.source, instruction.line
        if key != current_run:
            current_run = key
            run_cwd = cwd
        if operation.kind == "cd":
            try:
                words = shlex.split(operation.value)
                target = words[1] if words[1].startswith("/") else posixpath.join(run_cwd, words[1])
                run_cwd = posixpath.normpath(target)
            except (ValueError, IndexError):
                diagnostics.append(f"{operation.source}:{operation.line}: unresolved working directory")
        if operation.kind == "git-clone":
            try:
                words = shlex.split(operation.value)
                external_roots.append(posixpath.normpath(words[-1]))
            except (ValueError, IndexError):
                diagnostics.append(f"{operation.source}:{operation.line}: unresolved clone destination")
        if operation.kind != "pip-install":
            continue
        try:
            words = shlex.split(operation.value.replace("\\\n", " "))
        except ValueError:
            diagnostics.append(f"{operation.source}:{operation.line}: unresolved pip command")
            continue
        install_dependencies = "--no-deps" not in words
        for index, word in enumerate(words[:-1]):
            if word not in ("-r", "--requirement", "-e", "--editable"):
                continue
            target_path = words[index + 1].split("[", 1)[0].rstrip("/")
            if not target_path.startswith("/"):
                target_path = posixpath.join(run_cwd, target_path)
            local_path = context_path(target_path)
            if local_path is None:
                diagnostics.append(f"{operation.source}:{operation.line}: external pip source {target_path}")
                continue
            if word in ("-e", "--editable"):
                local_path = posixpath.join(local_path, "pyproject.toml")
                if local_path == "pyproject.toml":
                    source_files.append(("setup.py", operation, install_dependencies))
            source_files.append((local_path, operation, install_dependencies or word in ("-r", "--requirement")))

    visited: set[tuple[int, str, bool]] = set()
    dependencies_by_install: dict[int, list[Operation]] = {}

    def add_source(
        path: str,
        install: Operation,
        include_requirements: bool,
        ancestry: frozenset[str] = frozenset(),
    ) -> None:
        if path in ancestry:
            diagnostics.append(f"requirements include cycle: {path}")
            return
        key = id(install), path, include_requirements
        if key in visited:
            return
        visited.add(key)
        dependency_operations = dependencies_by_install.setdefault(id(install), [])
        content = read_text(path)
        if content is None:
            diagnostics.append(f"missing build dependency source: {path}")
            return
        if posixpath.basename(path).casefold() in {"readme.md", "readme.rst", "readme.txt", "readme"}:
            # Packaging descriptions remain visible in the call chain, but
            # their text does not change the runtime dependency environment.
            dependency_operations.append(Operation(path, 1, "documentation", path))
            return
        if path.endswith(".py"):
            try:
                parsed_python = ast.parse(content, filename=path)
            except SyntaxError as exc:
                raise BuildSyntaxError(f"{path}:{exc.lineno}: invalid Python source") from exc
            semantic_content = ast.dump(parsed_python, include_attributes=False)
            digest = hashlib.sha256(semantic_content.encode("utf-8")).hexdigest()
            dependency_operations.append(Operation(path, 1, "source-digest", digest))
            for node in _reachable_python_nodes(parsed_python):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id == "_read_requirements" and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        add_source(posixpath.normpath(argument.value), install, include_requirements, ancestry | {path})
                if node.func.id == "get_path" and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        add_source(posixpath.normpath(argument.value), install, include_requirements, ancestry | {path})
                if node.func.id == "load_module_from_path" and len(node.args) >= 2:
                    location = node.args[1]
                    if not isinstance(location, ast.Call) or not isinstance(location.func, ast.Attribute):
                        continue
                    if location.func.attr != "join" or not location.args:
                        continue
                    if not isinstance(location.args[0], ast.Name) or location.args[0].id != "ROOT_DIR":
                        continue
                    parts = [arg.value for arg in location.args[1:] if isinstance(arg, ast.Constant)]
                    if len(parts) == len(location.args) - 1 and all(isinstance(part, str) for part in parts):
                        add_source(posixpath.join(*parts), install, include_requirements, ancestry | {path})
            return
        if path.endswith(".toml"):
            try:
                parsed = tomllib.loads(content)
            except tomllib.TOMLDecodeError as exc:
                raise BuildSyntaxError(f"{path}: invalid TOML") from exc
            semantic_content = json.dumps(parsed, sort_keys=True, default=str)
            digest = hashlib.sha256(semantic_content.encode("utf-8")).hexdigest()
            dependency_operations.append(Operation(path, 1, "source-digest", digest))
            return
        if not path.endswith(".txt"):
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            dependency_operations.append(Operation(path, 1, "source-digest", digest))
            return
        entries: list[tuple[int, str]] = []
        for number, raw in enumerate(content.splitlines(), 1):
            line = raw.split(" #", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            entries.append((number, line))
        semantic_content = "\n".join(line for _, line in entries)
        digest = hashlib.sha256(semantic_content.encode("utf-8")).hexdigest()
        dependency_operations.append(Operation(path, 1, "source-digest", digest))
        if not include_requirements:
            return
        for number, line in entries:
            dependency_operations.append(Operation(path, number, "requirement", line))
            match = re.match(r"^(?:-r|--requirement)\s+(.+)$", line)
            if match:
                child = posixpath.normpath(posixpath.join(posixpath.dirname(path), match.group(1)))
                add_source(child, install, include_requirements, ancestry | {path})

    for source_file, install, include_requirements in source_files:
        add_source(source_file, install, include_requirements)
    ordered_operations: list[Operation] = []
    for operation in operations:
        ordered_operations.append(operation)
        owner_index = len(ordered_operations) - 1
        ordered_operations.extend(
            replace(
                dependency,
                owner_index=owner_index,
                owner_key=(operation.source, operation.value),
                environment=operation.environment,
            )
            for dependency in dependencies_by_install.get(id(operation), ())
        )
    operations = ordered_operations
    for diagnostic in diagnostics:
        match = re.match(r"^(.*?):(\d+):\s*(.*)$", diagnostic)
        if match:
            operations.append(Operation(match.group(1), int(match.group(2)), "unresolved", match.group(3)))
        else:
            operations.append(Operation("<build-input>", 0, "unresolved", diagnostic))
    parsed_operations = tuple(operations)
    return BuildAnalysis(
        analysis.instructions,
        parsed_operations,
        tuple(diagnostics),
        package_effects(parsed_operations),
        package_outcomes(parsed_operations),
        analysis.environment,
    )


def analyze_commit(
    repo: Path,
    commit: str,
    soc: str,
    architecture: str,
    os_variant: str = "ubuntu",
    build_args: dict[str, str] | None = None,
) -> BuildAnalysis:
    """Analyze build inputs from one Git commit."""
    return _analyze_source(
        lambda path: _read_text(repo, commit, path),
        commit[:12],
        soc,
        architecture,
        os_variant,
        build_args,
    )


def analyze_directory(
    directory: Path,
    soc: str,
    architecture: str,
    os_variant: str = "ubuntu",
    build_args: dict[str, str] | None = None,
) -> BuildAnalysis:
    """Analyze an editable copy of the build inputs without a Git commit."""
    root = directory.resolve()

    def read_text(path: str) -> str | None:
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise BuildSyntaxError(f"build source escapes analysis directory: {path}")
        try:
            return target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    return _analyze_source(read_text, str(root), soc, architecture, os_variant, build_args)
