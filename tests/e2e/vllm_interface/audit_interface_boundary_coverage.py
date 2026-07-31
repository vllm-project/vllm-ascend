# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independently audit source-candidate coverage of an interface mapping.

This module deliberately does not import ``generate_interface_boundaries``.
It builds a second, smaller AST index and asks a narrow question: did every
statically visible patch, direct inheritance, and verified-override candidate
receive exactly one disposition in the generated JSONL?

The audit is a coverage backstop, not another mapping generator.  A verified
relationship can still have an incompatible signature; the interface boundary
test owns that contract comparison.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AUDIT_VERSION = "0.4.0"
RELATIONS = frozenset({"inheritance", "monkey_patch", "override"})
STATUSES = frozenset({"verified", "risk", "expected", "excluded", "review"})
STDLIB_STRUCTURAL_BASES: dict[str, tuple[str, ...]] = {
    "abc.ABC": (),
    "typing.Generic": (),
    "typing.Protocol": ("typing.Generic",),
}


def _expression_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _expression_name(node.value)
        return f"{owner}.{node.attr}" if owner else None
    if isinstance(node, ast.Subscript):
        return _expression_name(node.value)
    return None


def _normalized_expression(node: ast.AST | None) -> str | None:
    name = _expression_name(node)
    if name:
        return name
    if node is None:
        return None
    return " ".join(ast.unparse(node).split())


def _module_name(package_name: str, package_root: Path, path: Path) -> tuple[str, bool]:
    relative = path.relative_to(package_root).with_suffix("")
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    suffix = ".".join(parts)
    return (f"{package_name}.{suffix}" if suffix else package_name, is_package)


def _relative_import_module(module: str, is_package: bool, level: int, imported: str | None) -> str:
    if level == 0:
        return imported or ""
    package_parts = module.split(".") if is_package else module.split(".")[:-1]
    parents = level - 1
    if parents:
        package_parts = package_parts[:-parents]
    if imported:
        package_parts.extend(imported.split("."))
    return ".".join(package_parts)


def _scope_name(scope: tuple[str, ...]) -> str | None:
    return ".".join(scope) if scope else None


def _string_values(node: ast.AST | None, state: FlowState) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name):
        return set(state.strings.get(node.id, ()))
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return {value for element in node.elts for value in _string_values(element, state)}
    if isinstance(node, ast.IfExp):
        return {
            *_string_values(node.body, state),
            *_string_values(node.orelse, state),
        }
    return set()


def _main_condition_value(node: ast.AST, state: FlowState) -> bool | None:
    if isinstance(node, ast.Call):
        function = _expression_name(node.func)
        if function and function.rsplit(".", 1)[-1] == "vllm_version_is":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                return False
    if isinstance(node, ast.Name):
        values = state.booleans.get(node.id, set())
        if len(values) == 1:
            return next(iter(values))
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _main_condition_value(node.operand, state)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [_main_condition_value(value, state) for value in node.values]
        if any(value is None for value in values):
            return None
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
    return None


@dataclass
class FlowState:
    bindings: dict[str, set[str]] = field(default_factory=dict)
    strings: dict[str, set[str]] = field(default_factory=dict)
    booleans: dict[str, set[bool]] = field(default_factory=dict)

    def clone(self) -> FlowState:
        return FlowState(
            bindings={name: set(values) for name, values in self.bindings.items()},
            strings={name: set(values) for name, values in self.strings.items()},
            booleans={name: set(values) for name, values in self.booleans.items()},
        )

    @classmethod
    def merged(cls, states: Iterable[FlowState]) -> FlowState:
        result = cls()
        for state in states:
            for name, values in state.bindings.items():
                result.bindings.setdefault(name, set()).update(values)
            for name, values in state.strings.items():
                result.strings.setdefault(name, set()).update(values)
            for name, values in state.booleans.items():
                result.booleans.setdefault(name, set()).update(values)
        return result


@dataclass(frozen=True)
class MethodSlot:
    name: str
    line: int
    kind: str
    value_expression: str | None = None
    value_targets: tuple[str, ...] = ()
    calls_same_method_on_super: bool = False


@dataclass(frozen=True)
class BaseReference:
    raw_expression: str
    targets: tuple[str, ...]


@dataclass(frozen=True)
class ClassRecord:
    qualified_name: str
    module: str
    file: str
    line: int
    bases: tuple[BaseReference, ...]
    methods: tuple[MethodSlot, ...]

    def method(self, name: str) -> MethodSlot | None:
        return next((method for method in self.methods if method.name == name), None)


@dataclass(frozen=True)
class MroResult:
    owners: tuple[str, ...]
    complete: bool
    reason: str | None = None


@dataclass(frozen=True)
class CandidateSite:
    relation: str
    file: str
    line: int
    scope: str | None
    targets: tuple[str, ...]
    kinds: tuple[str, ...]

    @property
    def site_key(self) -> tuple[str, str, int, str]:
        return (self.relation, self.file, self.line, self.scope or "")

    @property
    def candidate_id(self) -> str:
        identity = json.dumps(
            [self.relation, self.file, self.line, self.scope, self.targets],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode()).hexdigest()[:20]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "relation": self.relation,
            "file": self.file,
            "line": self.line,
            "scope": self.scope,
            "targets": list(self.targets),
            "kinds": list(self.kinds),
        }


@dataclass(frozen=True)
class Disposition:
    relation: str
    file: str
    line: int
    scope: str | None
    status: str
    origin: str
    target: str | None = None
    reason_code: str | None = None
    generator_issue: bool = False

    @property
    def site_key(self) -> tuple[str, str, int, str]:
        return (self.relation, self.file, self.line, self.scope or "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "file": self.file,
            "line": self.line,
            "scope": self.scope,
            "status": self.status,
            "origin": self.origin,
            "target": self.target,
            "reason_code": self.reason_code,
            "generator_issue": self.generator_issue,
        }


class IndependentCandidateScanner:
    """High-recall AST scanner with no dependency on the mapping generator."""

    def __init__(
        self,
        vllm_root: Path,
        ascend_root: Path,
        *,
        external_roots: dict[str, Path] | None = None,
    ):
        self.vllm_root = vllm_root.resolve()
        self.ascend_root = ascend_root.resolve()
        self.external_roots = {package: root.resolve() for package, root in (external_roots or {}).items()}
        invalid_packages = {
            package
            for package in self.external_roots
            if not package or any(not part.isidentifier() for part in package.split("."))
        }
        if invalid_packages:
            raise ValueError("invalid external package name(s): " + ", ".join(sorted(invalid_packages)))
        reserved = {"vllm", "vllm_ascend"} & set(self.external_roots)
        reserved.update(
            package
            for package in self.external_roots
            if package in {name.split(".", 1)[0] for name in STDLIB_STRUCTURAL_BASES}
        )
        if reserved:
            raise ValueError("external package name conflicts with an owned package: " + ", ".join(sorted(reserved)))
        self._package_roles = {
            "vllm": "upstream",
            "vllm_ascend": "downstream",
            **{package: "external" for package in self.external_roots},
        }
        self._candidate_parts: dict[tuple[str, str, int, str], dict[str, set[str]]] = {}
        self._classes: dict[str, ClassRecord] = {
            qualified_name: ClassRecord(
                qualified_name=qualified_name,
                module=qualified_name.rsplit(".", 1)[0],
                file="<stdlib-structural>",
                line=0,
                bases=tuple(BaseReference(raw_expression=base, targets=(base,)) for base in bases),
                methods=(),
            )
            for qualified_name, bases in STDLIB_STRUCTURAL_BASES.items()
        }
        self._callables: set[str] = set()
        self._aliases: dict[str, set[str]] = {}
        self._star_imports: dict[str, set[str]] = {}
        self._parse_errors: list[str] = []
        self._mro_cache: dict[str, MroResult] = {}

    def scan(self) -> list[CandidateSite]:
        for package, root in sorted(self.external_roots.items()):
            self._scan_repository(root, package, collect_candidates=False)
        self._scan_repository(self.vllm_root, "vllm", collect_candidates=False)
        self._scan_repository(self.ascend_root, "vllm_ascend", collect_candidates=True)
        if self._parse_errors:
            raise ValueError("Python source parsing failed: " + "; ".join(self._parse_errors))
        self._drop_non_upstream_patch_candidates()
        self._collect_inheritance_candidates()
        self._collect_override_candidates()
        return [
            CandidateSite(
                relation=key[0],
                file=key[1],
                line=key[2],
                scope=key[3] or None,
                targets=tuple(sorted(parts["targets"])),
                kinds=tuple(sorted(parts["kinds"])),
            )
            for key, parts in sorted(self._candidate_parts.items())
        ]

    def _scan_repository(self, root: Path, package_name: str, *, collect_candidates: bool) -> None:
        package_root = root.joinpath(*package_name.split("."))
        if not package_root.is_dir():
            raise ValueError(f"package directory not found: {package_root}")
        for path in sorted(package_root.rglob("*.py")):
            relative_file = path.relative_to(root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError) as error:
                self._parse_errors.append(f"{relative_file}: {type(error).__name__}: {error}")
                continue
            module, is_package = _module_name(package_name, package_root, path)
            state = FlowState()
            final_state = self._scan_statements(
                module=module,
                is_package=is_package,
                relative_file=relative_file,
                statements=tree.body,
                state=state,
                scope=(),
                collect_candidates=collect_candidates,
                module_scope=True,
            )
            for local_name, targets in final_state.bindings.items():
                self._aliases[f"{module}.{local_name}"] = set(targets)

    def _scan_statements(
        self,
        *,
        module: str,
        is_package: bool,
        relative_file: str,
        statements: Sequence[ast.stmt],
        state: FlowState,
        scope: tuple[str, ...],
        collect_candidates: bool,
        module_scope: bool,
    ) -> FlowState:
        current = state.clone()
        for node in statements:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._update_imports(module, is_package, node, current)
                continue

            if isinstance(node, ast.If):
                selected = _main_condition_value(node.test, current)
                if selected is not None:
                    branch = node.body if selected else node.orelse
                    current = self._scan_statements(
                        module=module,
                        is_package=is_package,
                        relative_file=relative_file,
                        statements=branch,
                        state=current,
                        scope=scope,
                        collect_candidates=collect_candidates,
                        module_scope=module_scope,
                    )
                else:
                    body = self._scan_statements(
                        module=module,
                        is_package=is_package,
                        relative_file=relative_file,
                        statements=node.body,
                        state=current,
                        scope=scope,
                        collect_candidates=collect_candidates,
                        module_scope=module_scope,
                    )
                    otherwise = self._scan_statements(
                        module=module,
                        is_package=is_package,
                        relative_file=relative_file,
                        statements=node.orelse,
                        state=current,
                        scope=scope,
                        collect_candidates=collect_candidates,
                        module_scope=module_scope,
                    )
                    current = FlowState.merged([body, otherwise])
                continue

            if isinstance(node, ast.Try):
                branches = [
                    self._scan_statements(
                        module=module,
                        is_package=is_package,
                        relative_file=relative_file,
                        statements=[*node.body, *node.orelse],
                        state=current,
                        scope=scope,
                        collect_candidates=collect_candidates,
                        module_scope=module_scope,
                    )
                ]
                branches.extend(
                    self._scan_statements(
                        module=module,
                        is_package=is_package,
                        relative_file=relative_file,
                        statements=handler.body,
                        state=current,
                        scope=scope,
                        collect_candidates=collect_candidates,
                        module_scope=module_scope,
                    )
                    for handler in node.handlers
                )
                current = FlowState.merged(branches)
                current = self._scan_statements(
                    module=module,
                    is_package=is_package,
                    relative_file=relative_file,
                    statements=node.finalbody,
                    state=current,
                    scope=scope,
                    collect_candidates=collect_candidates,
                    module_scope=module_scope,
                )
                continue

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{module}.{'.'.join((*scope, node.name))}"
                current.bindings[node.name] = {qualified}
                self._callables.add(qualified)
                if collect_candidates:
                    self._scan_statements(
                        module=module,
                        is_package=is_package,
                        relative_file=relative_file,
                        statements=node.body,
                        state=current,
                        scope=(*scope, node.name),
                        collect_candidates=True,
                        module_scope=False,
                    )
                continue

            if isinstance(node, ast.ClassDef):
                resolved_bases = []
                for base in node.bases:
                    raw_base = _expression_name(base) or _normalized_expression(base) or "<dynamic>"
                    resolved_bases.append(
                        BaseReference(
                            raw_expression=raw_base,
                            targets=tuple(
                                sorted(
                                    self._resolve_expression(
                                        module,
                                        raw_base,
                                        current,
                                    )
                                )
                            ),
                        )
                    )

                qualified = f"{module}.{'.'.join((*scope, node.name))}"
                methods = self._class_methods(module, node, current)
                self._classes[qualified] = ClassRecord(
                    qualified_name=qualified,
                    module=module,
                    file=relative_file,
                    line=node.lineno,
                    bases=tuple(resolved_bases),
                    methods=tuple(methods),
                )
                self._callables.add(qualified)
                self._callables.update(f"{qualified}.{method.name}" for method in methods)
                current.bindings[node.name] = {qualified}
                if collect_candidates:
                    self._scan_statements(
                        module=module,
                        is_package=is_package,
                        relative_file=relative_file,
                        statements=node.body,
                        state=current,
                        scope=(*scope, node.name),
                        collect_candidates=True,
                        module_scope=False,
                    )
                continue

            if isinstance(node, (ast.For, ast.AsyncFor)):
                loop_state = current.clone()
                if isinstance(node.target, ast.Name):
                    values = _string_values(node.iter, current)
                    if values:
                        loop_state.strings[node.target.id] = values
                body = self._scan_statements(
                    module=module,
                    is_package=is_package,
                    relative_file=relative_file,
                    statements=node.body,
                    state=loop_state,
                    scope=scope,
                    collect_candidates=collect_candidates,
                    module_scope=module_scope,
                )
                otherwise = self._scan_statements(
                    module=module,
                    is_package=is_package,
                    relative_file=relative_file,
                    statements=node.orelse,
                    state=current,
                    scope=scope,
                    collect_candidates=collect_candidates,
                    module_scope=module_scope,
                )
                current = FlowState.merged([current, body, otherwise])
                continue

            if isinstance(node, ast.While):
                body = self._scan_statements(
                    module=module,
                    is_package=is_package,
                    relative_file=relative_file,
                    statements=node.body,
                    state=current,
                    scope=scope,
                    collect_candidates=collect_candidates,
                    module_scope=module_scope,
                )
                otherwise = self._scan_statements(
                    module=module,
                    is_package=is_package,
                    relative_file=relative_file,
                    statements=node.orelse,
                    state=current,
                    scope=scope,
                    collect_candidates=collect_candidates,
                    module_scope=module_scope,
                )
                current = FlowState.merged([current, body, otherwise])
                continue

            if isinstance(node, (ast.With, ast.AsyncWith)):
                body = self._scan_statements(
                    module=module,
                    is_package=is_package,
                    relative_file=relative_file,
                    statements=node.body,
                    state=current,
                    scope=scope,
                    collect_candidates=collect_candidates,
                    module_scope=module_scope,
                )
                current = FlowState.merged([current, body])
                continue

            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if collect_candidates:
                    for target in targets:
                        if not isinstance(target, ast.Attribute):
                            continue
                        raw_target = _expression_name(target)
                        upstream_targets = {
                            reference
                            for reference in self._resolve_expression(module, raw_target, current)
                            if reference.startswith("vllm.")
                        }
                        if upstream_targets:
                            self._add_candidate(
                                "monkey_patch",
                                relative_file,
                                node.lineno,
                                _scope_name(scope),
                                {raw_target or "<dynamic>", *upstream_targets},
                                "assignment",
                            )
                self._update_assignments(module, targets, node.value, current)
                continue

            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if collect_candidates and _expression_name(call.func) == "setattr" and len(call.args) >= 3:
                    owner = _expression_name(call.args[0])
                    owner_targets = {
                        target
                        for target in self._resolve_expression(module, owner, current)
                        if target.startswith("vllm.")
                    }
                    if owner_targets:
                        attributes = _string_values(call.args[1], current)
                        raw_targets = {f"{owner}.{attribute}" for attribute in attributes} or {f"{owner}.<dynamic>"}
                        resolved_targets = {
                            f"{target}.{attribute}" for target in owner_targets for attribute in attributes
                        } or owner_targets
                        self._add_candidate(
                            "monkey_patch",
                            relative_file,
                            node.lineno,
                            _scope_name(scope),
                            {*raw_targets, *resolved_targets},
                            "setattr",
                        )

        if module_scope:
            for name, targets in current.bindings.items():
                self._aliases[f"{module}.{name}"] = set(targets)
        return current

    def _update_imports(
        self,
        module: str,
        is_package: bool,
        node: ast.Import | ast.ImportFrom,
        state: FlowState,
    ) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                state.bindings[local_name] = {alias.name if alias.asname else alias.name.split(".", 1)[0]}
            return
        source = _relative_import_module(module, is_package, node.level, node.module)
        for alias in node.names:
            if alias.name == "*":
                if source:
                    self._star_imports.setdefault(module, set()).add(source)
                continue
            local_name = alias.asname or alias.name
            state.bindings[local_name] = {f"{source}.{alias.name}" if source else alias.name}

    def _update_assignments(
        self,
        module: str,
        targets: Sequence[ast.AST],
        value: ast.AST | None,
        state: FlowState,
    ) -> None:
        raw_value = _expression_name(value)
        references = self._resolve_expression(module, raw_value, state) if raw_value else set()
        strings = _string_values(value, state)
        boolean = _main_condition_value(value, state) if value is not None else None
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if references:
                state.bindings[target.id] = references
            else:
                state.bindings.pop(target.id, None)
            if strings:
                state.strings[target.id] = strings
            else:
                state.strings.pop(target.id, None)
            if boolean is not None:
                state.booleans[target.id] = {boolean}
            else:
                state.booleans.pop(target.id, None)

    def _resolve_expression(self, module: str, expression: str | None, state: FlowState) -> set[str]:
        if not expression:
            return set()
        if expression.startswith(("vllm.", "vllm_ascend.")):
            return {expression}
        head, *tail = expression.split(".")
        suffix = f".{'.'.join(tail)}" if tail else ""
        if head in state.bindings:
            return {f"{target}{suffix}" for target in state.bindings[head]}
        return {f"{module}.{expression}"}

    def _class_methods(self, module: str, node: ast.ClassDef, state: FlowState) -> list[MethodSlot]:
        methods: list[MethodSlot] = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(
                    MethodSlot(
                        child.name,
                        child.lineno,
                        "definition",
                        calls_same_method_on_super=self._calls_same_method_on_super(
                            child,
                            child.name,
                        ),
                    )
                )
                continue
            if not isinstance(child, (ast.Assign, ast.AnnAssign)):
                continue
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            value = child.value
            value_node = value
            if isinstance(value, ast.Call):
                wrapper = _expression_name(value.func)
                if wrapper in {"classmethod", "staticmethod", "property"} and value.args:
                    value_node = value.args[0]
            value_expression = _expression_name(value_node)
            if not value_expression and not isinstance(value_node, ast.Lambda):
                continue
            value_targets = tuple(sorted(self._resolve_expression(module, value_expression, state)))
            for target in targets:
                if isinstance(target, ast.Name):
                    methods.append(
                        MethodSlot(
                            target.id,
                            child.lineno,
                            "callable_alias",
                            value_expression or "<lambda>",
                            value_targets,
                        )
                    )
        return methods

    def _calls_same_method_on_super(
        self,
        method_node: ast.FunctionDef | ast.AsyncFunctionDef,
        method_name: str,
    ) -> bool:
        """Return whether a live method path directly calls ``super().name``.

        Only zero-argument ``super()`` and the current method's own name are
        accepted.  Deferred nested scopes are deliberately not traversed, and
        statically dead branches or statements after an unconditional exit do
        not prove a runtime dependency.
        """

        state = FlowState()

        def condition_value(node: ast.AST) -> bool | None:
            if isinstance(node, ast.BoolOp):
                values = [condition_value(value) for value in node.values]
                if isinstance(node.op, ast.And):
                    if False in values:
                        return False
                    return True if all(value is True for value in values) else None
                if isinstance(node.op, ast.Or):
                    if True in values:
                        return True
                    return False if all(value is False for value in values) else None
            return _main_condition_value(node, state)

        def expression_has_call(node: ast.AST | None) -> bool:
            if node is None or isinstance(
                node,
                (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef, ast.Lambda),
            ):
                return False
            if isinstance(node, ast.BoolOp):
                for value in node.values:
                    if expression_has_call(value):
                        return True
                    value_condition = condition_value(value)
                    if isinstance(node.op, ast.And) and value_condition is False:
                        break
                    if isinstance(node.op, ast.Or) and value_condition is True:
                        break
                return False
            if isinstance(node, ast.IfExp):
                if expression_has_call(node.test):
                    return True
                selected = condition_value(node.test)
                if selected is True:
                    return expression_has_call(node.body)
                if selected is False:
                    return expression_has_call(node.orelse)
                return expression_has_call(node.body) or expression_has_call(node.orelse)
            if isinstance(node, ast.Call):
                function = node.func
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr == method_name
                    and isinstance(function.value, ast.Call)
                ):
                    super_call = function.value
                    if (
                        isinstance(super_call.func, ast.Name)
                        and super_call.func.id == "super"
                        and not super_call.args
                        and not super_call.keywords
                    ):
                        return True
            return any(expression_has_call(child) for child in ast.iter_child_nodes(node))

        def scan_block(statements: Sequence[ast.stmt]) -> tuple[bool, bool]:
            """Return ``(found, may_complete_normally)`` for one live block."""

            for statement in statements:
                if isinstance(
                    statement,
                    (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef),
                ):
                    continue
                if isinstance(statement, ast.If):
                    if expression_has_call(statement.test):
                        return True, True
                    selected = condition_value(statement.test)
                    if selected is True:
                        found, live = scan_block(statement.body)
                    elif selected is False:
                        found, live = scan_block(statement.orelse)
                    else:
                        body_found, body_live = scan_block(statement.body)
                        else_found, else_live = scan_block(statement.orelse)
                        found = body_found or else_found
                        live = body_live or else_live
                    if found:
                        return True, live
                    if not live:
                        return False, False
                    continue
                if isinstance(statement, ast.Try):
                    branches = [scan_block(statement.body)]
                    branches.extend(scan_block(handler.body) for handler in statement.handlers)
                    branches.append(scan_block(statement.orelse))
                    if any(found for found, _ in branches):
                        return True, True
                    final_found, final_live = scan_block(statement.finalbody)
                    if final_found:
                        return True, final_live
                    if not final_live:
                        return False, False
                    continue
                if isinstance(statement, ast.While):
                    if expression_has_call(statement.test):
                        return True, True
                    selected = condition_value(statement.test)
                    if selected is not False:
                        body_found, _ = scan_block(statement.body)
                        if body_found:
                            return True, True
                    else_found, _ = scan_block(statement.orelse)
                    if else_found:
                        return True, True
                    continue
                if isinstance(statement, (ast.For, ast.AsyncFor)):
                    if expression_has_call(statement.iter):
                        return True, True
                    body_found, _ = scan_block(statement.body)
                    else_found, _ = scan_block(statement.orelse)
                    if body_found or else_found:
                        return True, True
                    continue
                if isinstance(statement, (ast.With, ast.AsyncWith)):
                    if any(expression_has_call(item.context_expr) for item in statement.items):
                        return True, True
                    found, live = scan_block(statement.body)
                    if found:
                        return True, live
                    if not live:
                        return False, False
                    continue

                expressions: tuple[ast.AST | None, ...] = ()
                if isinstance(
                    statement,
                    (ast.AnnAssign, ast.Assign, ast.AugAssign, ast.Expr, ast.Return),
                ):
                    expressions = (statement.value,)
                elif isinstance(statement, ast.Raise):
                    expressions = (statement.exc, statement.cause)
                elif isinstance(statement, ast.Assert):
                    expressions = (statement.test, statement.msg)
                if any(expression_has_call(expression) for expression in expressions):
                    return True, True
                if isinstance(statement, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
                    return False, False
            return False, True

        found, _ = scan_block(method_node.body)
        return found

    def _expand_alias(self, qualified_name: str, *, limit: int = 24) -> set[str]:
        pending = deque([qualified_name])
        results: set[str] = set()
        seen: set[str] = set()
        while pending and len(seen) < 128:
            current = pending.popleft()
            if current in seen:
                continue
            seen.add(current)
            matching = [alias for alias in self._aliases if current == alias or current.startswith(f"{alias}.")]
            if not matching:
                star_modules = [module for module in self._star_imports if current.startswith(f"{module}.")]
                if star_modules:
                    longest_module = max(star_modules, key=len)
                    replacements = {
                        f"{source}{current[len(longest_module) :]}" for source in self._star_imports[longest_module]
                    }
                    if replacements != {current}:
                        pending.extend(replacements)
                        continue
            if not matching or limit <= 0:
                results.add(current)
                continue
            longest = max(map(len, matching))
            replacements = {
                f"{target}{current[len(alias) :]}"
                for alias in matching
                if len(alias) == longest
                for target in self._aliases[alias]
            }
            if not replacements or replacements == {current}:
                results.add(current)
            else:
                pending.extend(replacements)
        return results or {qualified_name}

    def _class_candidates(self, qualified_name: str) -> list[ClassRecord]:
        return [self._classes[target] for target in self._expand_alias(qualified_name) if target in self._classes]

    def _package_name(self, qualified_name: str) -> str | None:
        matches = [
            package
            for package in self._package_roles
            if qualified_name == package or qualified_name.startswith(f"{package}.")
        ]
        return max(matches, key=len) if matches else None

    def _package_role(self, qualified_name: str) -> str | None:
        package = self._package_name(qualified_name)
        return self._package_roles.get(package) if package else None

    def _drop_non_upstream_patch_candidates(self) -> None:
        """Drop writes whose canonical owner is only an external package.

        A symbol imported through a vLLM module is not automatically owned by
        vLLM.  Following the alias to its defining package prevents re-exported
        objects such as a third-party module from becoming vLLM patch edges.
        """

        for key in list(self._candidate_parts):
            if key[0] != "monkey_patch":
                continue
            targets = self._candidate_parts[key]["targets"]
            canonical_targets = {canonical for target in targets for canonical in self._expand_alias(target)}
            if not any(self._package_role(target) == "upstream" for target in canonical_targets):
                del self._candidate_parts[key]

    def _resolved_base_name(
        self,
        base: BaseReference,
    ) -> tuple[str | None, str | None]:
        expanded = {canonical for target in base.targets for canonical in self._expand_alias(target)}
        if len(expanded) != 1:
            return None, (f"base {base.raw_expression!r} resolves to {', '.join(sorted(expanded)) or '<nothing>'}")
        target = next(iter(expanded))
        if target not in self._classes:
            return None, f"base {target} is not indexed"
        return target, None

    def _strict_mro(
        self,
        qualified_name: str,
        stack: tuple[str, ...] = (),
    ) -> MroResult:
        if qualified_name in self._mro_cache:
            return self._mro_cache[qualified_name]
        if qualified_name in stack:
            return MroResult(
                owners=(qualified_name,),
                complete=False,
                reason=f"inheritance cycle at {qualified_name}",
            )
        record = self._classes.get(qualified_name)
        if record is None:
            return MroResult(
                owners=(qualified_name,),
                complete=False,
                reason=f"class {qualified_name} is not indexed",
            )

        base_names: list[str] = []
        for base in record.bases:
            base_name, reason = self._resolved_base_name(base)
            if base_name is None:
                result = MroResult(
                    owners=(qualified_name,),
                    complete=False,
                    reason=reason,
                )
                self._mro_cache[qualified_name] = result
                return result
            base_names.append(base_name)

        base_results = [self._strict_mro(base, (*stack, qualified_name)) for base in base_names]
        incomplete = next(
            (result for result in base_results if not result.complete),
            None,
        )
        if incomplete is not None:
            result = MroResult(
                owners=(qualified_name,),
                complete=False,
                reason=incomplete.reason,
            )
            self._mro_cache[qualified_name] = result
            return result

        sequences = [list(result.owners) for result in base_results]
        sequences.append(base_names.copy())
        owners = [qualified_name]
        while any(sequences):
            sequences = [sequence for sequence in sequences if sequence]
            candidate = next(
                (sequence[0] for sequence in sequences if not any(sequence[0] in other[1:] for other in sequences)),
                None,
            )
            if candidate is None:
                result = MroResult(
                    owners=(qualified_name,),
                    complete=False,
                    reason=f"C3 merge failed at {qualified_name}",
                )
                self._mro_cache[qualified_name] = result
                return result
            owners.append(candidate)
            for sequence in sequences:
                if sequence and sequence[0] == candidate:
                    sequence.pop(0)

        result = MroResult(owners=tuple(owners), complete=True)
        self._mro_cache[qualified_name] = result
        return result

    def _possible_known_ancestors(self, record: ClassRecord) -> set[str]:
        result: set[str] = set()
        pending = deque(record.bases)
        while pending:
            base = pending.popleft()
            for target in base.targets:
                for canonical in self._expand_alias(target):
                    if canonical in result:
                        continue
                    owner = self._classes.get(canonical)
                    if owner is None:
                        continue
                    result.add(canonical)
                    pending.extend(owner.bases)
        return result

    def _alias_slot_is_callable(self, slot: MethodSlot) -> bool:
        if slot.kind != "callable_alias":
            return True
        if slot.value_expression == "<lambda>":
            return True
        return any(
            target in self._callables or any(expanded in self._callables for expanded in self._expand_alias(target))
            for target in slot.value_targets
        )

    def _collect_inheritance_candidates(self) -> None:
        for record in self._classes.values():
            if not record.qualified_name.startswith("vllm_ascend."):
                continue
            upstream_bases = {
                target
                for base in record.bases
                for reference in base.targets
                for target in self._expand_alias(reference)
                if self._package_role(target) == "upstream"
            }
            if upstream_bases:
                self._add_candidate(
                    "inheritance",
                    record.file,
                    record.line,
                    None,
                    upstream_bases,
                    "class_base",
                )

    def _collect_override_candidates(self) -> None:
        for record in self._classes.values():
            if not record.qualified_name.startswith("vllm_ascend."):
                continue
            possible_ancestors = self._possible_known_ancestors(record)
            if not any(self._package_role(owner) == "upstream" for owner in possible_ancestors):
                continue
            mro = self._strict_mro(record.qualified_name)
            for method in record.methods:
                if not self._alias_slot_is_callable(method):
                    continue
                if mro.complete:
                    effective_owner = next(
                        (owner for owner in mro.owners[1:] if self._classes[owner].method(method.name)),
                        None,
                    )
                    if effective_owner is None:
                        missing_super_owner = (
                            next(
                                (owner for owner in mro.owners[1:] if self._package_role(owner) == "upstream"),
                                None,
                            )
                            if method.calls_same_method_on_super and not hasattr(object, method.name)
                            else None
                        )
                        if missing_super_owner is not None:
                            self._add_candidate(
                                "override",
                                record.file,
                                method.line,
                                None,
                                {f"{missing_super_owner}.{method.name}"},
                                "missing_upstream_super_target",
                            )
                        continue
                    role = self._package_role(effective_owner)
                    if role not in {"upstream", "external"}:
                        continue
                    self._add_candidate(
                        "override",
                        record.file,
                        method.line,
                        None,
                        {f"{effective_owner}.{method.name}"},
                        ("external_override" if role == "external" else method.kind),
                    )
                    continue

                possible_owners = {
                    owner
                    for owner in possible_ancestors
                    if self._classes[owner].method(method.name)
                    and self._package_role(owner) in {"upstream", "external"}
                }
                if possible_owners:
                    self._add_candidate(
                        "override",
                        record.file,
                        method.line,
                        None,
                        {f"{owner}.{method.name}" for owner in possible_owners},
                        "incomplete_mro",
                    )

    def _add_candidate(
        self,
        relation: str,
        file: str,
        line: int,
        scope: str | None,
        targets: Iterable[str],
        kind: str,
    ) -> None:
        key = (relation, file, line, scope or "")
        parts = self._candidate_parts.setdefault(
            key,
            {"targets": set(), "kinds": set()},
        )
        parts["targets"].update(target for target in targets if target)
        parts["kinds"].add(kind)


def _load_dispositions(path: Path) -> tuple[dict[str, Any], list[Disposition]]:
    meta: dict[str, Any] = {}
    dispositions: list[Disposition] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if "_meta" in payload:
            meta = payload["_meta"]
            continue
        if "f" in payload:
            finding = payload["f"]
            relation = finding.get("relation")
            status = finding.get("status")
            if relation not in RELATIONS or status not in STATUSES:
                continue
            evidence = finding.get("evidence", {})
            dispositions.append(
                Disposition(
                    relation=relation,
                    file=evidence.get("file") or finding["downstream"]["file"],
                    line=int(evidence.get("line", 0)),
                    scope=evidence.get("scope"),
                    status=status,
                    origin=f"finding:{line_number}",
                    target=finding.get("target_expression"),
                    reason_code=finding.get("reason_code"),
                    generator_issue=bool(finding.get("generator_issue", False)),
                )
            )
            continue
        if "u" not in payload:
            continue
        for evidence_record in payload.get("e", []):
            consumer = evidence_record.get("consumer", [])
            if not consumer or consumer[0] not in RELATIONS:
                continue
            relation = consumer[0]
            for occurrence in evidence_record.get("occurrences", []):
                dispositions.append(
                    Disposition(
                        relation=relation,
                        file=occurrence["file"],
                        line=int(occurrence["line"]),
                        scope=occurrence.get("scope"),
                        status="verified",
                        origin=f"relation:{line_number}",
                        target=occurrence.get("target_expression"),
                    )
                )
    return meta, dispositions


def _git_checkout_head(root: Path) -> str | None:
    """Return HEAD only when ``root`` is the checkout's exact top level."""

    resolved_root = root.resolve()
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(resolved_root),
                "rev-parse",
                "--show-toplevel",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            encoding="utf-8",
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = result.stdout.splitlines()
    if len(lines) != 2 or Path(lines[0]).resolve() != resolved_root:
        return None
    return lines[1].strip()


def _snapshot_source_sha(root: Path, package: str) -> str:
    manifest_path = root.resolve() / ".interface-source.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"external source is neither an exact Git checkout nor a valid snapshot: {package}={root}"
        ) from error

    if manifest.get("schema") != 1 or manifest.get("package") != package:
        raise ValueError(f"invalid external source manifest identity: {manifest_path}")
    commit = manifest.get("commit")
    expected_files = manifest.get("files")
    if not isinstance(commit, str) or not commit or not isinstance(expected_files, dict):
        raise ValueError(f"invalid external source manifest: {manifest_path}")

    package_root = root.resolve().joinpath(*package.split("."))
    actual_files = {path.relative_to(root.resolve()).as_posix() for path in package_root.rglob("*.py")}
    expected_names: set[str] = set()
    package_prefix = tuple(package.split("."))
    for relative_path, expected_digest in expected_files.items():
        if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
            raise ValueError(f"invalid external source file record: {manifest_path}")
        parts = tuple(relative_path.split("/"))
        if (
            not relative_path.endswith(".py")
            or not parts
            or any(part in {"", ".", ".."} for part in parts)
            or parts[: len(package_prefix)] != package_prefix
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected_digest)
        ):
            raise ValueError(f"invalid external source file record: {manifest_path}")
        expected_names.add(relative_path)

    if actual_files != expected_names:
        missing = sorted(expected_names - actual_files)
        extra = sorted(actual_files - expected_names)
        raise ValueError(f"external source snapshot file set changed for {package}: missing={missing}, extra={extra}")
    for relative_path, expected_digest in sorted(expected_files.items()):
        actual_digest = hashlib.sha256((root.resolve() / relative_path).read_bytes()).hexdigest()
        if actual_digest != expected_digest.lower():
            raise ValueError(
                f"external source snapshot digest mismatch: {relative_path}; "
                f"expected {expected_digest}, found {actual_digest}"
            )
    return commit


def _verify_source_inputs(
    vllm_root: Path,
    ascend_root: Path,
    external_roots: dict[str, Path],
    *,
    expect_vllm_sha: str | None,
    expect_ascend_sha: str | None,
    expect_external_shas: dict[str, str],
) -> dict[str, Any]:
    if set(external_roots) != set(expect_external_shas):
        raise ValueError("--external-root and --expect-external-sha must name the same packages")

    verified: dict[str, Any] = {"external_sources": {}}
    for label, root, expected in (
        ("vLLM", vllm_root, expect_vllm_sha),
        ("vllm-ascend", ascend_root, expect_ascend_sha),
    ):
        if expected is None:
            continue
        actual = _git_checkout_head(root)
        if actual is None:
            raise ValueError(f"{label} source root is not an exact Git checkout: {root.resolve()}")
        if actual != expected:
            raise ValueError(f"{label} source SHA mismatch: expected {expected}, found {actual}")
        verified["vllm" if label == "vLLM" else "vllm_ascend"] = actual

    for package, root in sorted(external_roots.items()):
        actual = _git_checkout_head(root)
        if actual is None:
            actual = _snapshot_source_sha(root, package)
        expected = expect_external_shas[package]
        if actual != expected:
            raise ValueError(f"external package {package} SHA mismatch: expected {expected}, found {actual}")
        verified["external_sources"][package] = actual
    return verified


def _verify_mapping_sources(
    mapping_meta: dict[str, Any],
    *,
    expect_vllm_sha: str | None,
    expect_ascend_sha: str | None,
    expect_external_shas: dict[str, str],
) -> None:
    if expect_vllm_sha and mapping_meta.get("vllm") != expect_vllm_sha:
        raise ValueError(f"mapping vLLM SHA mismatch: expected {expect_vllm_sha}, got {mapping_meta.get('vllm')}")
    if expect_ascend_sha and mapping_meta.get("vllm_ascend") != expect_ascend_sha:
        raise ValueError(
            f"mapping vllm-ascend SHA mismatch: expected {expect_ascend_sha}, got {mapping_meta.get('vllm_ascend')}"
        )
    actual_external = mapping_meta.get("external_sources", {})
    if actual_external != expect_external_shas:
        raise ValueError(
            "mapping external source SHA mismatch: "
            f"expected {dict(sorted(expect_external_shas.items()))}, got {actual_external}"
        )


def audit_mapping_coverage(
    vllm_root: Path,
    ascend_root: Path,
    mapping_path: Path,
    *,
    external_roots: dict[str, Path] | None = None,
    expect_vllm_sha: str | None = None,
    expect_ascend_sha: str | None = None,
    expect_external_shas: dict[str, str] | None = None,
) -> dict[str, Any]:
    external_roots = external_roots or {}
    expect_external_shas = expect_external_shas or {}
    mapping_meta, dispositions = _load_dispositions(mapping_path)
    _verify_mapping_sources(
        mapping_meta,
        expect_vllm_sha=expect_vllm_sha,
        expect_ascend_sha=expect_ascend_sha,
        expect_external_shas=expect_external_shas,
    )
    verified_sources = _verify_source_inputs(
        vllm_root,
        ascend_root,
        external_roots,
        expect_vllm_sha=expect_vllm_sha,
        expect_ascend_sha=expect_ascend_sha,
        expect_external_shas=expect_external_shas,
    )
    candidates = IndependentCandidateScanner(
        vllm_root,
        ascend_root,
        external_roots=external_roots,
    ).scan()

    candidates_by_site = {candidate.site_key: candidate for candidate in candidates}
    dispositions_by_site: dict[tuple[str, str, int, str], list[Disposition]] = defaultdict(list)
    for disposition in dispositions:
        dispositions_by_site[disposition.site_key].append(disposition)

    missing = [candidate.as_dict() for key, candidate in candidates_by_site.items() if key not in dispositions_by_site]
    conflicting = []
    for key in sorted(candidates_by_site.keys() & dispositions_by_site.keys()):
        records = dispositions_by_site[key]
        statuses = sorted({record.status for record in records})
        if len(statuses) > 1:
            conflicting.append(
                {
                    "candidate": candidates_by_site[key].as_dict(),
                    "statuses": statuses,
                    "dispositions": [record.as_dict() for record in records],
                }
            )
    orphan = [
        {
            "site": {
                "relation": key[0],
                "file": key[1],
                "line": key[2],
                "scope": key[3] or None,
            },
            "dispositions": [record.as_dict() for record in records],
        }
        for key, records in sorted(dispositions_by_site.items())
        if key not in candidates_by_site
    ]
    generator_issue_review = [
        disposition.as_dict()
        for disposition in dispositions
        if disposition.status == "review" and disposition.generator_issue
    ]
    classified = sum(
        key in dispositions_by_site and len({record.status for record in dispositions_by_site[key]}) == 1
        for key in candidates_by_site
    )
    return {
        "_meta": {
            "audit_version": AUDIT_VERSION,
            "mapping": str(mapping_path),
            "mapping_meta": mapping_meta,
            "external_roots": {package: str(root.resolve()) for package, root in sorted(external_roots.items())},
            "verified_sources": verified_sources,
        },
        "summary": {
            "candidates": len(candidates),
            "classified": classified,
            "missing": len(missing),
            "conflicting": len(conflicting),
            "orphan": len(orphan),
            "generator_issue_review": len(generator_issue_review),
        },
        "counts_by_relation": dict(
            sorted(
                (relation, sum(candidate.relation == relation for candidate in candidates)) for relation in RELATIONS
            )
        ),
        "missing": missing,
        "conflicting": conflicting,
        "orphan": orphan,
        "generator_issue_review": generator_issue_review,
    }


def _parse_named_values(values: Sequence[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        package, separator, item = value.partition("=")
        if not separator or not package or not item or any(not part.isidentifier() for part in package.split(".")):
            raise ValueError(f"invalid {option} {value!r}; expected PACKAGE=VALUE")
        if package in parsed:
            raise ValueError(f"duplicate {option} package: {package}")
        parsed[package] = item
    return parsed


def _parse_external_roots(values: Sequence[str]) -> dict[str, Path]:
    return {package: Path(path_text) for package, path_text in _parse_named_values(values, "--external-root").items()}


def _parse_external_shas(values: Sequence[str]) -> dict[str, str]:
    return _parse_named_values(values, "--expect-external-sha")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--ascend-root", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument(
        "--external-root",
        action="append",
        default=[],
        metavar="PACKAGE=PATH",
        help="indexed external package source; may be repeated",
    )
    parser.add_argument(
        "--expect-external-sha",
        action="append",
        default=[],
        metavar="PACKAGE=SHA",
        help="required identity for each external source; may be repeated",
    )
    parser.add_argument("--expect-vllm-sha")
    parser.add_argument("--expect-ascend-sha")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    external_roots = _parse_external_roots(args.external_root)
    external_shas = _parse_external_shas(args.expect_external_sha)
    report = audit_mapping_coverage(
        args.vllm_root,
        args.ascend_root,
        args.mapping,
        external_roots=external_roots,
        expect_vllm_sha=args.expect_vllm_sha,
        expect_ascend_sha=args.expect_ascend_sha,
        expect_external_shas=external_shas,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    summary = report["summary"]
    if any(summary[name] for name in ("missing", "conflicting", "orphan", "generator_issue_review")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
