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

"""Generate exact vLLM interface relations from the current source pair.

The collector is intentionally consumer-first: it records patch and inheritance
intent from vllm-ascend before resolving the target in vLLM. A missing upstream
target is therefore kept as an explicit risk finding instead of silently
disappearing.

The first implementation covers:

* explicit monkey patches (assignment and literal-name ``setattr``);
* direct inheritance from an imported vLLM class;
* verified overrides whose effective owner is found in the combined MRO.

It targets vLLM main: an exact ``vllm_version_is("<tag>")`` branch is treated
as release-only, and the opposite branch is scanned. An incomplete MRO is
reported instead of being guessed.

It does not import either package and does not require an NPU.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
GENERATOR_VERSION = "0.5.0"
SUPPORTED_RELATIONS = frozenset({"inheritance", "monkey_patch", "override"})
FINDING_STATUSES = frozenset({"expected", "excluded", "review", "risk"})


def _jsonable_signature(node: ast.AST | None) -> list[object] | None:
    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda)):
        return None

    arguments = node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    required_count = len(positional) - len(arguments.defaults)
    return [
        "async" if isinstance(node, ast.AsyncFunctionDef) else "sync",
        [
            [argument.arg, index < required_count]
            for index, argument in enumerate(arguments.posonlyargs)
        ],
        [
            [
                argument.arg,
                index + len(arguments.posonlyargs) < required_count,
            ]
            for index, argument in enumerate(arguments.args)
        ],
        arguments.vararg.arg if arguments.vararg else None,
        [
            [argument.arg, default is None]
            for argument, default in zip(
                arguments.kwonlyargs,
                arguments.kw_defaults,
            )
        ],
        arguments.kwarg.arg if arguments.kwarg else None,
    ]


def _expression_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return _expression_name(node.value)
    return None


def _module_name(package_name: str, package_root: Path, path: Path) -> tuple[str, bool]:
    relative = path.relative_to(package_root)
    parts = list(relative.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    suffix = ".".join(parts)
    return (f"{package_name}.{suffix}" if suffix else package_name), is_package


def _relative_import_module(
    current_module: str,
    is_package: bool,
    level: int,
    imported_module: str | None,
) -> str:
    if level == 0:
        return imported_module or ""

    package_parts = current_module.split(".") if is_package else current_module.split(".")[:-1]
    keep = len(package_parts) - (level - 1)
    if keep < 0:
        return imported_module or ""
    result = package_parts[:keep]
    if imported_module:
        result.extend(imported_module.split("."))
    return ".".join(result)


def _method_nodes(node: ast.ClassDef) -> dict[str, ast.AST]:
    return {
        child.name: child
        for child in node.body
        if isinstance(child, (ast.AsyncFunctionDef, ast.FunctionDef))
    }


def _is_exact_tag_check(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and _expression_name(node.func) == "vllm_version_is"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    )


def _tag_guard_names(statements: Sequence[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for node in statements:
        if isinstance(node, ast.Assign) and _is_exact_tag_check(node.value):
            names.update(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            )
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and _is_exact_tag_check(node.value)
        ):
            names.add(node.target.id)
    return names


def _main_condition_value(
    node: ast.AST,
    tag_guard_names: set[str],
) -> bool | None:
    if _is_exact_tag_check(node):
        return False
    if isinstance(node, ast.Name) and node.id in tag_guard_names:
        return False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _main_condition_value(node.operand, tag_guard_names)
        return None if value is None else not value
    if isinstance(node, ast.BoolOp):
        values = [
            _main_condition_value(value, tag_guard_names)
            for value in node.values
        ]
        if isinstance(node.op, ast.And):
            if False in values:
                return False
            return True if all(value is True for value in values) else None
        if isinstance(node.op, ast.Or):
            if True in values:
                return True
            return False if all(value is False for value in values) else None
    return None


def _main_module_statements(
    statements: Sequence[ast.stmt],
    tag_guard_names: set[str],
) -> Iterable[ast.stmt]:
    for node in statements:
        if isinstance(node, ast.If):
            condition = _main_condition_value(
                node.test,
                tag_guard_names,
            )
            if condition is True:
                yield from _main_module_statements(
                    node.body,
                    tag_guard_names,
                )
            elif condition is False:
                yield from _main_module_statements(
                    node.orelse,
                    tag_guard_names,
                )
            else:
                yield from _main_module_statements(
                    node.body,
                    tag_guard_names,
                )
                yield from _main_module_statements(
                    node.orelse,
                    tag_guard_names,
                )
            continue
        if isinstance(node, ast.Try):
            yield from _main_module_statements(
                node.body,
                tag_guard_names,
            )
            for handler in node.handlers:
                yield from _main_module_statements(
                    handler.body,
                    tag_guard_names,
                )
            yield from _main_module_statements(
                node.orelse,
                tag_guard_names,
            )
            yield from _main_module_statements(
                node.finalbody,
                tag_guard_names,
            )
            continue
        yield node


def _main_ast_walk(tree: ast.AST) -> Iterable[ast.AST]:
    statements = tree.body if isinstance(tree, ast.Module) else ()
    tag_guard_names = _tag_guard_names(statements)

    def walk(node: ast.AST) -> Iterable[ast.AST]:
        yield node
        if isinstance(node, ast.If):
            condition = _main_condition_value(
                node.test,
                tag_guard_names,
            )
            branches: Sequence[ast.stmt]
            if condition is True:
                branches = node.body
            elif condition is False:
                branches = node.orelse
            else:
                branches = (*node.body, *node.orelse)
            for child in branches:
                yield from walk(child)
            return
        for child in ast.iter_child_nodes(node):
            yield from walk(child)

    yield from walk(tree)


def _string_assignment(
    node: ast.AST,
) -> tuple[str, str] | None:
    target: ast.AST | None = None
    value: ast.AST | None = None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        value = node.value
    elif isinstance(node, ast.AnnAssign):
        target = node.target
        value = node.value
    if (
        isinstance(target, ast.Name)
        and isinstance(value, ast.Constant)
        and isinstance(value.value, str)
    ):
        return target.id, value.value
    return None


def _resolve_bound_reference(
    module: str,
    expression: str,
    imports: dict[str, str],
    local_names: set[str],
) -> str:
    parts = expression.split(".")
    if parts[0] in imports:
        return ".".join([imports[parts[0]], *parts[1:]])
    if parts[0] in local_names:
        return f"{module}.{expression}"
    if expression.startswith(("vllm.", "vllm_ascend.")):
        return expression
    return f"{module}.{expression}"


@dataclass(frozen=True)
class ClassInfo:
    qualified_name: str
    module: str
    file: str
    name: str
    bases: tuple[str, ...]
    resolved_bases: tuple[str, ...]
    methods: dict[str, ast.AST] = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class CallableInfo:
    qualified_name: str
    module: str
    file: str
    owner: str | None
    name: str
    node: ast.AST | None = field(compare=False, hash=False, repr=False)
    binding_line: int | None = None
    origin_kind: str = "definition"

    @property
    def signature(self) -> list[object] | None:
        return _jsonable_signature(self.node)


@dataclass
class ModuleInfo:
    name: str
    file: str
    is_package: bool
    tree: ast.Module
    imports: dict[str, str]
    classes: dict[str, ClassInfo]
    functions: dict[str, CallableInfo]
    loose_functions: dict[str, list[CallableInfo]]
    string_constants: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class MroResult:
    owners: tuple[str, ...]
    complete: bool
    reason: str | None = None


@dataclass(frozen=True)
class RelationEvidence:
    file: str
    line: int
    scope: str | None = None
    guards: tuple[str, ...] = ()
    patch_kind: str | None = None
    definition_line: int | None = None
    binding_line: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "file": self.file,
            "line": self.line,
        }
        if self.scope:
            payload["scope"] = self.scope
        if self.guards:
            payload["guards"] = list(self.guards)
        if self.patch_kind:
            payload["patch_kind"] = self.patch_kind
        if self.definition_line is not None:
            payload["definition_line"] = self.definition_line
        if self.binding_line is not None:
            payload["binding_line"] = self.binding_line
        return payload


@dataclass(frozen=True)
class Relation:
    relation: str
    upstream_file: str
    upstream_owner: str | None
    upstream_name: str
    upstream_signature: list[object] | None = field(compare=False, hash=False)
    downstream_file: str
    downstream_owner: str | None
    downstream_name: str
    downstream_signature: list[object] | None = field(compare=False, hash=False)
    evidence_file: str
    evidence_line: int
    evidence: tuple[RelationEvidence, ...] = field(
        default=(),
        compare=False,
        hash=False,
    )

    def upstream_key(self) -> tuple[str, str, str]:
        return (
            self.upstream_file,
            self.upstream_owner or "",
            self.upstream_name,
        )

    def downstream_key(self) -> tuple[str, str, str, str]:
        return (
            self.relation,
            self.downstream_file,
            self.downstream_owner or "",
            self.downstream_name,
        )

    def exact_key(self) -> tuple[str, ...]:
        return (*self.downstream_key(), *self.upstream_key())

    def comparison_downstream_keys(
        self,
    ) -> tuple[tuple[str, str, str, str], ...]:
        keys = {self.downstream_key()}
        if self.relation == "monkey_patch":
            keys.update(
                (
                    self.relation,
                    evidence.file,
                    self.downstream_owner or "",
                    self.downstream_name,
                )
                for evidence in self.evidence
            )
        return tuple(sorted(keys))

    def comparison_exact_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            (*downstream_key, *self.upstream_key())
            for downstream_key in self.comparison_downstream_keys()
        )


@dataclass(frozen=True)
class CandidateFinding:
    relation: str
    downstream_file: str
    downstream_owner: str | None
    downstream_name: str
    target_expression: str
    evidence_line: int
    reason: str
    status: str = "review"
    reason_code: str = "analysis_gap"
    generator_issue: bool = True
    evidence_scope: str | None = None
    evidence_guards: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        if self.status not in FINDING_STATUSES:
            raise ValueError(f"unsupported finding status: {self.status}")
        return {
            "relation": self.relation,
            "downstream": {
                "file": self.downstream_file,
                "owner": self.downstream_owner,
                "name": self.downstream_name,
            },
            "target_expression": self.target_expression,
            "evidence": {
                "file": self.downstream_file,
                "line": self.evidence_line,
                **(
                    {"scope": self.evidence_scope}
                    if self.evidence_scope
                    else {}
                ),
                **(
                    {"guards": list(self.evidence_guards)}
                    if self.evidence_guards
                    else {}
                ),
            },
            "status": self.status,
            "reason_code": self.reason_code,
            "generator_issue": self.generator_issue,
            "reason": self.reason,
        }


# Kept as a source-compatible alias for callers of the v0.3 POC.
UnresolvedRelation = CandidateFinding


@dataclass
class PatchScanContext:
    bindings: dict[str, set[str]] = field(default_factory=dict)
    strings: dict[str, set[str]] = field(default_factory=dict)
    local_callables: dict[str, list[CallableInfo]] = field(default_factory=dict)
    scope: tuple[str, ...] = ()
    guards: tuple[str, ...] = ()

    def clone(
        self,
        *,
        scope: tuple[str, ...] | None = None,
        guards: tuple[str, ...] | None = None,
    ) -> PatchScanContext:
        return PatchScanContext(
            bindings={name: set(values) for name, values in self.bindings.items()},
            strings={name: set(values) for name, values in self.strings.items()},
            local_callables={
                name: list(values)
                for name, values in self.local_callables.items()
            },
            scope=self.scope if scope is None else scope,
            guards=self.guards if guards is None else guards,
        )

    def merge(self, contexts: Sequence[PatchScanContext]) -> None:
        if not contexts:
            return
        self.bindings = _merge_candidate_maps(
            context.bindings for context in contexts
        )
        self.strings = _merge_candidate_maps(
            context.strings for context in contexts
        )
        callable_names = {
            name
            for context in contexts
            for name in context.local_callables
        }
        merged_callables: dict[str, list[CallableInfo]] = {}
        for name in callable_names:
            candidates: dict[tuple[str, str | None, str, int], CallableInfo] = {}
            for context in contexts:
                for candidate in context.local_callables.get(name, []):
                    key = (
                        candidate.file,
                        candidate.owner,
                        candidate.name,
                        getattr(candidate.node, "lineno", 0),
                    )
                    candidates[key] = candidate
            merged_callables[name] = list(candidates.values())
        self.local_callables = merged_callables


@dataclass(frozen=True)
class PatchReplacement:
    info: CallableInfo | None
    kind: str
    reason: str | None = None
    is_restore: bool = False


def _merge_candidate_maps(
    mappings: Iterable[dict[str, set[str]]],
) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for mapping in mappings:
        for name, values in mapping.items():
            merged[name].update(values)
    return dict(merged)


class RepositoryIndex:
    """AST-only symbol and import index for one Python package."""

    def __init__(self, repo_root: Path, package_name: str):
        self.repo_root = repo_root.resolve()
        self.package_name = package_name
        self.package_root = self.repo_root / package_name
        if not self.package_root.is_dir():
            raise ValueError(f"package directory not found: {self.package_root}")

        self.modules: dict[str, ModuleInfo] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.callables: dict[str, CallableInfo] = {}
        self.aliases: dict[str, str] = {}
        self._pending_method_aliases: list[
            tuple[str, str, str, str, int]
        ] = []
        self.parse_errors: list[dict[str, str]] = []
        self._parse()

    def _parse(self) -> None:
        for path in sorted(self.package_root.rglob("*.py")):
            relative_file = path.relative_to(self.repo_root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError) as error:
                self.parse_errors.append(
                    {
                        "file": relative_file,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue

            module, is_package = _module_name(self.package_name, self.package_root, path)
            imports: dict[str, str] = {}
            classes: dict[str, ClassInfo] = {}
            functions: dict[str, CallableInfo] = {}
            loose_functions: dict[str, list[CallableInfo]] = defaultdict(list)
            string_constants: dict[str, set[str]] = defaultdict(set)
            tag_guard_names = _tag_guard_names(tree.body)
            module_statements = list(
                _main_module_statements(
                    tree.body,
                    tag_guard_names,
                )
            )

            for node in module_statements:
                string_assignment = _string_assignment(node)
                if string_assignment:
                    name, value = string_assignment
                    string_constants[name].add(value)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        local_name = alias.asname or alias.name.split(".", 1)[0]
                        imports[local_name] = alias.name if alias.asname else local_name
                elif isinstance(node, ast.ImportFrom):
                    source_module = _relative_import_module(
                        module,
                        is_package,
                        node.level,
                        node.module,
                    )
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        local_name = alias.asname or alias.name
                        imports[local_name] = (
                            f"{source_module}.{alias.name}" if source_module else alias.name
                        )
                elif isinstance(node, ast.ClassDef):
                    bases = tuple(
                        name
                        for name in (
                            _expression_name(base)
                            for base in node.bases
                        )
                        if name
                    )
                    resolved_bases = tuple(
                        _resolve_bound_reference(
                            module,
                            base,
                            imports,
                            {*classes, *functions},
                        )
                        for base in bases
                    )
                    imports.pop(node.name, None)
                    qualified_name = f"{module}.{node.name}"
                    info = ClassInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        name=node.name,
                        bases=bases,
                        resolved_bases=resolved_bases,
                        methods=_method_nodes(node),
                    )
                    classes[node.name] = info
                    self.classes[qualified_name] = info
                    self.callables[qualified_name] = CallableInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        owner=None,
                        name=node.name,
                        node=node,
                    )
                    for method_name, method_node in info.methods.items():
                        method_qualified_name = f"{qualified_name}.{method_name}"
                        self.callables[method_qualified_name] = CallableInfo(
                            qualified_name=method_qualified_name,
                            module=module,
                            file=relative_file,
                            owner=node.name,
                            name=method_name,
                            node=method_node,
                        )
                    self._collect_class_callable_aliases(
                        node,
                        module,
                        qualified_name,
                        imports,
                        {*classes, *functions},
                    )
                elif isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    imports.pop(node.name, None)
                    qualified_name = f"{module}.{node.name}"
                    info = CallableInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        owner=None,
                        name=node.name,
                        node=node,
                    )
                    functions[node.name] = info
                    self.callables[qualified_name] = info

            for node in _main_ast_walk(tree):
                if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    continue
                qualified_name = f"{module}.{node.name}"
                loose_functions[node.name].append(
                    CallableInfo(
                        qualified_name=qualified_name,
                        module=module,
                        file=relative_file,
                        owner=None,
                        name=node.name,
                        node=node,
                    )
                )

            module_info = ModuleInfo(
                name=module,
                file=relative_file,
                is_package=is_package,
                tree=tree,
                imports=imports,
                classes=classes,
                functions=functions,
                loose_functions=dict(loose_functions),
                string_constants={
                    name: tuple(sorted(values))
                    for name, values in string_constants.items()
                },
            )
            self.modules[module] = module_info
            for local_name, target in imports.items():
                self.aliases[f"{module}.{local_name}"] = target

        self._materialize_class_callable_aliases()

    def _collect_class_callable_aliases(
        self,
        node: ast.ClassDef,
        module: str,
        class_name: str,
        imports: dict[str, str],
        local_names: set[str],
    ) -> None:
        explicit_methods = _method_nodes(node)
        for statement in node.body:
            value: ast.AST | None = None
            targets: Sequence[ast.AST] = ()
            if isinstance(statement, ast.Assign):
                value = statement.value
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign):
                value = statement.value
                targets = (statement.target,)
            else:
                continue

            kind = "callable_alias"
            if isinstance(value, ast.Call):
                wrapper = _expression_name(value.func)
                if wrapper not in {"classmethod", "property", "staticmethod"}:
                    continue
                if len(value.args) != 1:
                    continue
                kind = wrapper
                value = value.args[0]
            expression = _expression_name(value)
            if expression is None:
                continue
            resolved = _resolve_bound_reference(
                module,
                expression,
                imports,
                local_names,
            )
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id in explicit_methods:
                    continue
                self._pending_method_aliases.append(
                    (
                        class_name,
                        target.id,
                        resolved,
                        kind,
                        getattr(statement, "lineno", 0),
                    )
                )

    def _materialize_class_callable_aliases(self) -> None:
        for class_name, member_name, target, kind, line in (
            self._pending_method_aliases
        ):
            source = self.find_callable(target)
            if source is None or not isinstance(
                source.node,
                (ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda),
            ):
                continue
            class_info = self.classes[class_name]
            if member_name in class_info.methods:
                continue
            qualified_name = f"{class_name}.{member_name}"
            alias = CallableInfo(
                qualified_name=qualified_name,
                module=class_info.module,
                file=class_info.file,
                owner=class_info.name,
                name=member_name,
                node=source.node,
                binding_line=line,
                origin_kind=kind,
            )
            class_info.methods[member_name] = source.node
            self.callables[qualified_name] = alias

    def resolve_reference(self, module: str, expression: str) -> str:
        parts = expression.split(".")
        module_info = self.modules[module]
        if parts[0] in module_info.imports:
            target = module_info.imports[parts[0]]
            return ".".join([target, *parts[1:]])
        if parts[0] in module_info.classes or parts[0] in module_info.functions:
            return f"{module}.{expression}"
        if expression.startswith((f"{self.package_name}.", "vllm.", "vllm_ascend.")):
            return expression
        return f"{module}.{expression}"

    def canonical_name(self, qualified_name: str) -> str:
        result = qualified_name
        visited: set[str] = set()
        while result not in visited:
            visited.add(result)
            replacement = None
            for alias in sorted(self.aliases, key=len, reverse=True):
                if result == alias or result.startswith(f"{alias}."):
                    replacement = f"{self.aliases[alias]}{result[len(alias):]}"
                    break
            if replacement is None or replacement == result:
                break
            result = replacement
        return result

    def find_class(self, qualified_name: str) -> ClassInfo | None:
        canonical = self.canonical_name(qualified_name)
        return self.classes.get(canonical)

    def find_callable(self, qualified_name: str) -> CallableInfo | None:
        canonical = self.canonical_name(qualified_name)
        return self.callables.get(canonical)

    def find_loose_function(self, module: str, name: str) -> CallableInfo | None:
        candidates = self.modules[module].loose_functions.get(name, [])
        return candidates[0] if len(candidates) == 1 else None


class InterfaceBoundaryGenerator:
    def __init__(self, vllm_root: Path, ascend_root: Path):
        self.upstream = RepositoryIndex(vllm_root, "vllm")
        self.downstream = RepositoryIndex(ascend_root, "vllm_ascend")
        parse_errors = [
            ("vLLM", error)
            for error in self.upstream.parse_errors
        ] + [
            ("vllm-ascend", error)
            for error in self.downstream.parse_errors
        ]
        if parse_errors:
            details = "; ".join(
                f"{repository}:{error['file']}: {error['error']}"
                for repository, error in parse_errors
            )
            raise ValueError(f"Python source parsing failed: {details}")
        self.relations: list[Relation] = []
        self.findings: list[CandidateFinding] = []
        self._mro_cache: dict[str, MroResult] = {}

    def generate(self) -> tuple[list[Relation], list[CandidateFinding]]:
        self._collect_inheritance()
        self._collect_verified_overrides()
        self._collect_monkey_patches()
        grouped: dict[tuple[str, ...], list[Relation]] = defaultdict(list)
        for relation in self.relations:
            grouped[relation.exact_key()].append(relation)
        deduplicated = {}
        for key, occurrences in grouped.items():
            first = min(
                occurrences,
                key=lambda item: (
                    item.evidence_file,
                    item.evidence_line,
                ),
            )
            evidence = {
                item
                for relation in occurrences
                for item in (
                    relation.evidence
                    or (
                        RelationEvidence(
                            file=relation.evidence_file,
                            line=relation.evidence_line,
                        ),
                    )
                )
            }
            deduplicated[key] = replace(
                first,
                evidence=tuple(
                    sorted(
                        evidence,
                        key=lambda item: (
                            item.file,
                            item.line,
                            item.scope or "",
                            item.guards,
                            item.patch_kind or "",
                        ),
                    )
                ),
            )
        self.relations = sorted(
            deduplicated.values(),
            key=lambda relation: (
                relation.upstream_key(),
                relation.downstream_key(),
            ),
        )
        self.findings = sorted(
            set(self.findings),
            key=lambda relation: (
                relation.status,
                relation.reason_code,
                relation.relation,
                relation.downstream_file,
                relation.downstream_owner or "",
                relation.downstream_name,
                relation.target_expression,
                relation.evidence_line,
                relation.evidence_scope or "",
                relation.evidence_guards,
                relation.reason,
            ),
        )
        return self.relations, self.findings

    def _resolve_downstream_reference(self, module: str, expression: str) -> str:
        qualified = self.downstream.resolve_reference(module, expression)
        if qualified.startswith("vllm."):
            return self.upstream.canonical_name(qualified)
        return self.downstream.canonical_name(qualified)

    def _class_info(self, qualified_name: str) -> ClassInfo | None:
        if qualified_name.startswith("vllm_ascend."):
            return self.downstream.find_class(qualified_name)
        if qualified_name.startswith("vllm."):
            return self.upstream.find_class(qualified_name)
        return None

    def _class_bases(
        self,
        qualified_name: str,
    ) -> tuple[list[str], list[str]]:
        info = self._class_info(qualified_name)
        if info is None:
            return [], [qualified_name]
        bases: list[str] = []
        missing: list[str] = []
        normalized_bases: list[str] = []
        for candidate in info.resolved_bases:
            if candidate.startswith("vllm."):
                candidate = self.upstream.canonical_name(candidate)
            elif candidate.startswith("vllm_ascend."):
                candidate = self.downstream.canonical_name(candidate)
            normalized_bases.append(candidate)

        known_owned = [
            self._class_info(candidate) is not None
            for candidate in normalized_bases
        ]
        for index, candidate in enumerate(normalized_bases):
            if self._class_info(candidate):
                bases.append(candidate)
            elif candidate.startswith(("vllm.", "vllm_ascend.")):
                missing.append(candidate)
                break
            elif any(known_owned[index + 1 :]):
                missing.append(
                    f"opaque base before owned base: {candidate}"
                )
                break
        return bases, missing

    def _linearized_mro(
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

        bases, missing = self._class_bases(qualified_name)
        if not bases:
            if missing:
                result = MroResult(
                    owners=(qualified_name,),
                    complete=False,
                    reason=(
                        "unresolved base(s): "
                        f"{', '.join(sorted(missing))}"
                    ),
                )
                self._mro_cache[qualified_name] = result
                return result
            result = MroResult(
                owners=(qualified_name,),
                complete=True,
            )
            self._mro_cache[qualified_name] = result
            return result

        base_results = [
            self._linearized_mro(base, (*stack, qualified_name))
            for base in bases
        ]
        incomplete = next(
            (result for result in base_results if not result.complete),
            None,
        )
        if missing or incomplete is not None:
            prefix: tuple[str, ...] = (qualified_name,)
            if len(base_results) == 1:
                prefix = (*prefix, *base_results[0].owners)
            reason_parts = []
            if missing:
                reason_parts.append(
                    f"unresolved base(s): {', '.join(sorted(missing))}"
                )
            if incomplete is not None and incomplete.reason:
                reason_parts.append(incomplete.reason)
            result = MroResult(
                owners=prefix,
                complete=False,
                reason="; ".join(reason_parts),
            )
            self._mro_cache[qualified_name] = result
            return result

        sequences = [
            list(result.owners)
            for result in base_results
        ]
        sequences.append(bases.copy())
        result = [qualified_name]
        while any(sequences):
            sequences = [sequence for sequence in sequences if sequence]
            candidate = next(
                (
                    sequence[0]
                    for sequence in sequences
                    if not any(sequence[0] in other[1:] for other in sequences)
                ),
                None,
            )
            if candidate is None:
                incomplete_result = MroResult(
                    owners=tuple(result),
                    complete=False,
                    reason=f"invalid or ambiguous MRO at {qualified_name}",
                )
                self._mro_cache[qualified_name] = incomplete_result
                return incomplete_result
            result.append(candidate)
            for sequence in sequences:
                if sequence and sequence[0] == candidate:
                    sequence.pop(0)

        complete_result = MroResult(
            owners=tuple(result),
            complete=True,
        )
        self._mro_cache[qualified_name] = complete_result
        return complete_result

    def _collect_inheritance(self) -> None:
        for class_info in self.downstream.classes.values():
            for base_expression, resolved in zip(
                class_info.bases,
                class_info.resolved_bases,
            ):
                if resolved.startswith("vllm."):
                    resolved = self.upstream.canonical_name(resolved)
                elif resolved.startswith("vllm_ascend."):
                    resolved = self.downstream.canonical_name(resolved)
                if not resolved.startswith("vllm."):
                    continue
                upstream_class = self.upstream.find_class(resolved)
                if upstream_class is None:
                    self.findings.append(
                        CandidateFinding(
                            relation="inheritance",
                            downstream_file=class_info.file,
                            downstream_owner=class_info.name,
                            downstream_name=class_info.name,
                            target_expression=resolved,
                            evidence_line=self._class_line(class_info),
                            reason="upstream base class was not found",
                            status="risk",
                            reason_code="missing_upstream_base",
                            generator_issue=False,
                        )
                    )
                    continue
                self.relations.append(
                    Relation(
                        relation="inheritance",
                        upstream_file=upstream_class.file,
                        upstream_owner=None,
                        upstream_name=upstream_class.name,
                        upstream_signature=None,
                        downstream_file=class_info.file,
                        downstream_owner=class_info.name,
                        downstream_name=base_expression.rsplit(".", 1)[-1],
                        downstream_signature=None,
                        evidence_file=class_info.file,
                        evidence_line=self._class_line(class_info),
                    )
                )

    def _collect_verified_overrides(self) -> None:
        for class_info in self.downstream.classes.values():
            mro_result = self._linearized_mro(
                class_info.qualified_name
            )
            mro = mro_result.owners
            if (
                mro_result.complete
                and not any(
                    owner.startswith("vllm.")
                    for owner in mro[1:]
                )
            ):
                continue
            for method_name, method_node in class_info.methods.items():
                effective_owner = self._effective_method_owner(mro[1:], method_name)
                if effective_owner is None:
                    candidates = (
                        self._candidate_upstream_method_owners(
                            class_info.qualified_name,
                            method_name,
                        )
                        if not mro_result.complete
                        else ()
                    )
                    if candidates:
                        self.findings.append(
                            CandidateFinding(
                                relation="override",
                                downstream_file=class_info.file,
                                downstream_owner=class_info.name,
                                downstream_name=method_name,
                                target_expression=", ".join(
                                    candidates
                                ),
                                evidence_line=getattr(
                                    method_node,
                                    "lineno",
                                    0,
                                ),
                                reason=(
                                    f"incomplete MRO ({mro_result.reason}); "
                                    "candidate upstream owner was not "
                                    "selected"
                                ),
                                status="review",
                                reason_code="ambiguous_mro",
                                generator_issue=False,
                            )
                        )
                    continue
                if not effective_owner.startswith("vllm."):
                    continue
                upstream_callable = self.upstream.find_callable(
                    f"{effective_owner}.{method_name}"
                )
                if upstream_callable is None:
                    continue
                downstream_callable = self.downstream.find_callable(
                    f"{class_info.qualified_name}.{method_name}"
                )
                evidence_line = (
                    downstream_callable.binding_line
                    if downstream_callable
                    and downstream_callable.binding_line is not None
                    else getattr(method_node, "lineno", 0)
                )
                self.relations.append(
                    Relation(
                        relation="override",
                        upstream_file=upstream_callable.file,
                        upstream_owner=upstream_callable.owner,
                        upstream_name=upstream_callable.name,
                        upstream_signature=upstream_callable.signature,
                        downstream_file=class_info.file,
                        downstream_owner=class_info.name,
                        downstream_name=method_name,
                        downstream_signature=(
                            downstream_callable.signature
                            if downstream_callable
                            else _jsonable_signature(method_node)
                        ),
                        evidence_file=class_info.file,
                        evidence_line=evidence_line,
                    )
                )

    def _effective_method_owner(
        self,
        mro: Sequence[str],
        method_name: str,
    ) -> str | None:
        for owner in mro:
            class_info = self._class_info(owner)
            if class_info and method_name in class_info.methods:
                return owner
        return None

    def _candidate_upstream_method_owners(
        self,
        qualified_name: str,
        method_name: str,
        seen: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        if qualified_name in seen:
            return ()
        class_info = self._class_info(qualified_name)
        if class_info is None:
            return ()

        candidates: set[str] = set()
        next_seen = (*seen, qualified_name)
        for base in class_info.resolved_bases:
            if base.startswith("vllm."):
                base = self.upstream.canonical_name(base)
            elif base.startswith("vllm_ascend."):
                base = self.downstream.canonical_name(base)
            base_info = self._class_info(base)
            if base_info is None:
                continue
            if (
                base.startswith("vllm.")
                and method_name in base_info.methods
            ):
                candidates.add(base)
            candidates.update(
                self._candidate_upstream_method_owners(
                    base,
                    method_name,
                    frozenset(next_seen),
                )
            )
        return tuple(sorted(candidates))

    def _collect_monkey_patches(self) -> None:
        for module_info in self.downstream.modules.values():
            context = PatchScanContext()
            self._scan_patch_statements(
                module_info,
                module_info.tree.body,
                context,
                _tag_guard_names(module_info.tree.body),
            )

    def _scan_patch_statements(
        self,
        module_info: ModuleInfo,
        statements: Sequence[ast.stmt],
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> None:
        for node in statements:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._update_import_bindings(module_info, node, context)
                continue

            if isinstance(node, ast.If):
                self._scan_patch_if(
                    module_info,
                    node,
                    context,
                    tag_guard_names,
                )
                continue

            if isinstance(node, ast.Try):
                self._scan_patch_try(
                    module_info,
                    node,
                    context,
                    tag_guard_names,
                )
                continue

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                callable_info = CallableInfo(
                    qualified_name=(
                        f"{module_info.name}."
                        f"{'.'.join((*context.scope, node.name))}"
                    ),
                    module=module_info.name,
                    file=module_info.file,
                    owner=None,
                    name=node.name,
                    node=node,
                )
                context.bindings.pop(node.name, None)
                context.local_callables[node.name] = [callable_info]
                child = context.clone(
                    scope=(*context.scope, node.name),
                )
                self._scan_patch_statements(
                    module_info,
                    node.body,
                    child,
                    tag_guard_names,
                )
                continue

            if isinstance(node, ast.ClassDef):
                qualified_name = f"{module_info.name}.{node.name}"
                context.bindings[node.name] = {qualified_name}
                context.local_callables[node.name] = [
                    CallableInfo(
                        qualified_name=qualified_name,
                        module=module_info.name,
                        file=module_info.file,
                        owner=None,
                        name=node.name,
                        node=node,
                    )
                ]
                child = context.clone(
                    scope=(*context.scope, node.name),
                )
                self._scan_patch_statements(
                    module_info,
                    node.body,
                    child,
                    tag_guard_names,
                )
                continue

            if isinstance(node, (ast.For, ast.AsyncFor)):
                self._scan_patch_for(
                    module_info,
                    node,
                    context,
                    tag_guard_names,
                )
                continue

            if isinstance(node, ast.While):
                guard = self._guard_text(node.test)
                body = context.clone(
                    guards=(*context.guards, guard),
                )
                self._scan_patch_statements(
                    module_info,
                    node.body,
                    body,
                    tag_guard_names,
                )
                empty = context.clone()
                context.merge([body, empty])
                if node.orelse:
                    self._scan_patch_statements(
                        module_info,
                        node.orelse,
                        context,
                        tag_guard_names,
                    )
                continue

            if isinstance(node, (ast.With, ast.AsyncWith)):
                child = context.clone(
                    guards=(*context.guards, "with-context"),
                )
                self._scan_patch_statements(
                    module_info,
                    node.body,
                    child,
                    tag_guard_names,
                )
                context.merge([context.clone(), child])
                continue

            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Attribute):
                        self._record_patch_node(
                            module_info,
                            target,
                            value,
                            context,
                            getattr(node, "lineno", 0),
                        )
                self._update_assignment_bindings(
                    module_info,
                    targets,
                    value,
                    context,
                )
                continue

            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and _expression_name(node.value.func) == "setattr"
                and len(node.value.args) >= 3
            ):
                self._record_setattr_patch(
                    module_info,
                    node.value,
                    context,
                    getattr(node, "lineno", 0),
                )

    def _scan_patch_if(
        self,
        module_info: ModuleInfo,
        node: ast.If,
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> None:
        condition = _main_condition_value(node.test, tag_guard_names)
        if condition is not None:
            selected = node.body if condition else node.orelse
            branch = context.clone()
            self._scan_patch_statements(
                module_info,
                selected,
                branch,
                tag_guard_names,
            )
            context.merge([branch])
            return

        guard = self._guard_text(node.test)
        body = context.clone(guards=(*context.guards, guard))
        self._scan_patch_statements(
            module_info,
            node.body,
            body,
            tag_guard_names,
        )
        otherwise = context.clone(
            guards=(*context.guards, f"not ({guard})"),
        )
        self._scan_patch_statements(
            module_info,
            node.orelse,
            otherwise,
            tag_guard_names,
        )
        context.merge([body, otherwise])

    def _scan_patch_try(
        self,
        module_info: ModuleInfo,
        node: ast.Try,
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> None:
        success = context.clone(guards=(*context.guards, "try-success"))
        self._scan_patch_statements(
            module_info,
            node.body,
            success,
            tag_guard_names,
        )
        self._scan_patch_statements(
            module_info,
            node.orelse,
            success,
            tag_guard_names,
        )
        branches = [success]
        for handler in node.handlers:
            exception_name = _expression_name(handler.type) or "Exception"
            branch = context.clone(
                guards=(*context.guards, f"except {exception_name}"),
            )
            self._scan_patch_statements(
                module_info,
                handler.body,
                branch,
                tag_guard_names,
            )
            branches.append(branch)
        context.merge(branches)
        self._scan_patch_statements(
            module_info,
            node.finalbody,
            context,
            tag_guard_names,
        )

    def _scan_patch_for(
        self,
        module_info: ModuleInfo,
        node: ast.For | ast.AsyncFor,
        context: PatchScanContext,
        tag_guard_names: set[str],
    ) -> None:
        values = self._string_values(node.iter, context)
        branches = [context.clone()]
        if isinstance(node.target, ast.Name) and values:
            for value in sorted(values):
                branch = context.clone(
                    guards=(
                        *context.guards,
                        f"for {node.target.id}={value!r}",
                    ),
                )
                branch.strings[node.target.id] = {value}
                self._scan_patch_statements(
                    module_info,
                    node.body,
                    branch,
                    tag_guard_names,
                )
                branches.append(branch)
        else:
            branch = context.clone(
                guards=(*context.guards, "for-loop"),
            )
            self._scan_patch_statements(
                module_info,
                node.body,
                branch,
                tag_guard_names,
            )
            branches.append(branch)
        context.merge(branches)
        self._scan_patch_statements(
            module_info,
            node.orelse,
            context,
            tag_guard_names,
        )

    def _update_import_bindings(
        self,
        module_info: ModuleInfo,
        node: ast.Import | ast.ImportFrom,
        context: PatchScanContext,
    ) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                target = alias.name if alias.asname else local_name
                context.bindings[local_name] = {target}
            return

        source_module = _relative_import_module(
            module_info.name,
            module_info.is_package,
            node.level,
            node.module,
        )
        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            target = (
                f"{source_module}.{alias.name}"
                if source_module
                else alias.name
            )
            context.bindings[local_name] = {target}

    def _update_assignment_bindings(
        self,
        module_info: ModuleInfo,
        targets: Sequence[ast.AST],
        value: ast.AST | None,
        context: PatchScanContext,
    ) -> None:
        string_values = self._string_values(value, context)
        expression = _expression_name(value)
        references = (
            self._resolve_patch_references(
                module_info,
                expression,
                context,
            )
            if expression
            else set()
        )
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if string_values:
                context.strings[target.id] = set(string_values)
            else:
                context.strings.pop(target.id, None)
            if references:
                context.bindings[target.id] = set(references)
            else:
                context.bindings.pop(target.id, None)

    def _resolve_patch_references(
        self,
        module_info: ModuleInfo,
        expression: str,
        context: PatchScanContext,
    ) -> set[str]:
        parts = expression.split(".")
        if parts[0] in context.bindings:
            candidates = {
                ".".join([candidate, *parts[1:]])
                for candidate in context.bindings[parts[0]]
            }
        elif expression.startswith(("vllm.", "vllm_ascend.")):
            candidates = {expression}
        else:
            candidates = {f"{module_info.name}.{expression}"}

        resolved = set()
        for candidate in candidates:
            if candidate.startswith("vllm."):
                candidate = self.upstream.canonical_name(candidate)
            elif candidate.startswith("vllm_ascend."):
                candidate = self.downstream.canonical_name(candidate)
            resolved.add(candidate)
        return resolved

    def _string_values(
        self,
        node: ast.AST | None,
        context: PatchScanContext,
    ) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return set(context.strings.get(node.id, ()))
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            return {
                value
                for element in node.elts
                for value in self._string_values(element, context)
            }
        if isinstance(node, ast.IfExp):
            return {
                *self._string_values(node.body, context),
                *self._string_values(node.orelse, context),
            }
        return set()

    def _record_setattr_patch(
        self,
        module_info: ModuleInfo,
        call: ast.Call,
        context: PatchScanContext,
        line: int,
    ) -> None:
        owner = _expression_name(call.args[0])
        attributes = self._string_values(call.args[1], context)
        if not owner:
            return
        owner_targets = self._resolve_patch_references(
            module_info,
            owner,
            context,
        )
        upstream_owners = sorted(
            target for target in owner_targets if target.startswith("vllm.")
        )
        if not upstream_owners:
            return
        if not attributes:
            self._append_unresolved_patch(
                module_info,
                context,
                ", ".join(upstream_owners),
                call.args[2],
                line,
                "dynamic setattr attribute name",
            )
            return
        target_expressions = {
            f"{owner_target}.{attribute}"
            for owner_target in upstream_owners
            for attribute in attributes
        }
        live_targets = {
            target
            for target in target_expressions
            if self._find_upstream_patch_target(target) is not None
        }
        selected = live_targets or target_expressions
        if len(selected) != 1:
            self._append_unresolved_patch(
                module_info,
                context,
                ", ".join(sorted(selected)),
                call.args[2],
                line,
                "ambiguous setattr patch target",
            )
            return
        self._record_resolved_patch(
            module_info,
            next(iter(selected)),
            call.args[2],
            context,
            line,
        )

    def _record_patch_node(
        self,
        module_info: ModuleInfo,
        target_node: ast.Attribute,
        replacement_node: ast.AST | None,
        context: PatchScanContext,
        line: int,
    ) -> None:
        expression = _expression_name(target_node)
        if not expression:
            return
        targets = sorted(
            target
            for target in self._resolve_patch_references(
                module_info,
                expression,
                context,
            )
            if target.startswith("vllm.")
        )
        if not targets:
            return
        if len(targets) != 1:
            self._append_unresolved_patch(
                module_info,
                context,
                ", ".join(targets),
                replacement_node,
                line,
                "ambiguous patch target alias",
            )
            return
        self._record_resolved_patch(
            module_info,
            targets[0],
            replacement_node,
            context,
            line,
        )

    def _record_resolved_patch(
        self,
        module_info: ModuleInfo,
        target: str,
        replacement_node: ast.AST | None,
        context: PatchScanContext,
        line: int,
    ) -> None:
        replacement = self._resolve_patch_replacement(
            module_info,
            replacement_node,
            context,
            target,
            line,
        )
        if replacement.is_restore:
            return
        if replacement.info is None:
            self._append_unresolved_patch(
                module_info,
                context,
                target,
                replacement_node,
                line,
                replacement.reason or "replacement callable was not resolved",
            )
            return

        upstream_callable = self._find_upstream_patch_target(target)
        if upstream_callable is None:
            status, reason_code, generator_issue = (
                self._missing_patch_target_classification(
                    target,
                    context,
                )
            )
            self.findings.append(
                CandidateFinding(
                    relation="monkey_patch",
                    downstream_file=module_info.file,
                    downstream_owner=replacement.info.owner,
                    downstream_name=replacement.info.name,
                    target_expression=target,
                    evidence_line=line,
                    reason="upstream patch target was not found",
                    status=status,
                    reason_code=reason_code,
                    generator_issue=generator_issue,
                    evidence_scope=self._scope_name(context),
                    evidence_guards=context.guards,
                )
            )
            return

        definition_line = getattr(replacement.info.node, "lineno", None)
        evidence = RelationEvidence(
            file=module_info.file,
            line=line,
            scope=self._scope_name(context),
            guards=context.guards,
            patch_kind=replacement.kind,
            definition_line=definition_line,
            binding_line=replacement.info.binding_line,
        )
        self.relations.append(
            Relation(
                relation="monkey_patch",
                upstream_file=upstream_callable.file,
                upstream_owner=upstream_callable.owner,
                upstream_name=upstream_callable.name,
                upstream_signature=upstream_callable.signature,
                downstream_file=replacement.info.file,
                downstream_owner=replacement.info.owner,
                downstream_name=replacement.info.name,
                downstream_signature=replacement.info.signature,
                evidence_file=module_info.file,
                evidence_line=line,
                evidence=(evidence,),
            )
        )

    def _resolve_patch_replacement(
        self,
        module_info: ModuleInfo,
        node: ast.AST | None,
        context: PatchScanContext,
        target: str,
        line: int,
    ) -> PatchReplacement:
        kind = "replacement"
        if isinstance(node, ast.Call):
            wrapper = _expression_name(node.func)
            if wrapper in {"classmethod", "staticmethod"} and len(node.args) == 1:
                kind = wrapper
                node = node.args[0]
            elif wrapper == "property" and node.args:
                kind = "property"
                node = node.args[0]
            else:
                return PatchReplacement(
                    info=None,
                    kind="wrapper",
                    reason="patch replacement is produced by a call or wrapper",
                )

        if isinstance(node, ast.Lambda):
            definition_line = getattr(node, "lineno", line)
            return PatchReplacement(
                info=CallableInfo(
                    qualified_name=(
                        f"{module_info.name}.<lambda>@{definition_line}"
                    ),
                    module=module_info.name,
                    file=module_info.file,
                    owner=None,
                    name=f"<lambda>@{definition_line}",
                    node=node,
                ),
                kind="lambda",
            )

        expression = _expression_name(node)
        if not expression:
            return PatchReplacement(
                info=None,
                kind=kind,
                reason="unsupported patch replacement expression",
            )

        if "." not in expression:
            local_candidates = context.local_callables.get(expression, [])
            if len(local_candidates) == 1:
                return PatchReplacement(
                    info=local_candidates[0],
                    kind=kind,
                )
            if len(local_candidates) > 1:
                return PatchReplacement(
                    info=None,
                    kind=kind,
                    reason="ambiguous local replacement callable",
                )

        references = self._resolve_patch_references(
            module_info,
            expression,
            context,
        )
        if references == {target}:
            return PatchReplacement(
                info=None,
                kind="restore_original",
                is_restore=True,
            )
        if any(reference.startswith("vllm.") for reference in references):
            return PatchReplacement(
                info=None,
                kind="alias_rebind",
                reason="replacement is another upstream callable",
            )

        candidates: dict[tuple[str, str | None, str], CallableInfo] = {}
        for reference in references:
            candidate = self._find_downstream_patch_replacement(reference)
            if candidate is None and reference.startswith(f"{module_info.name}."):
                candidate = self.downstream.find_loose_function(
                    module_info.name,
                    reference.rsplit(".", 1)[-1],
                )
            if candidate:
                candidates[(candidate.file, candidate.owner, candidate.name)] = candidate
        if len(candidates) == 1:
            return PatchReplacement(
                info=next(iter(candidates.values())),
                kind=kind,
            )
        return PatchReplacement(
            info=None,
            kind=kind,
            reason=(
                "ambiguous replacement callable"
                if candidates
                else "replacement callable was not found"
            ),
        )

    def _find_downstream_patch_replacement(
        self,
        qualified_name: str,
    ) -> CallableInfo | None:
        direct = self.downstream.find_callable(qualified_name)
        if direct is not None:
            return direct
        if "." not in qualified_name:
            return None

        owner_name, method_name = qualified_name.rsplit(".", 1)
        owner = self.downstream.find_class(owner_name)
        if owner is None:
            return None
        mro_result = self._linearized_mro(owner.qualified_name)
        effective_owner = self._effective_method_owner(
            mro_result.owners[1:],
            method_name,
        )
        if effective_owner is None:
            return None
        if effective_owner.startswith("vllm_ascend."):
            return self.downstream.find_callable(
                f"{effective_owner}.{method_name}"
            )
        return None

    def _append_unresolved_patch(
        self,
        module_info: ModuleInfo,
        context: PatchScanContext,
        target_expression: str,
        replacement_node: ast.AST | None,
        line: int,
        reason: str,
        *,
        status: str = "review",
        reason_code: str | None = None,
        generator_issue: bool = True,
    ) -> None:
        replacement_name = _expression_name(replacement_node)
        if replacement_name is None and isinstance(replacement_node, ast.Lambda):
            replacement_name = f"<lambda>@{line}"
        codes = {
            "ambiguous local replacement callable": "ambiguous_replacement_callable",
            "ambiguous patch target alias": "ambiguous_patch_target",
            "ambiguous replacement callable": "ambiguous_replacement_callable",
            "ambiguous setattr patch target": "ambiguous_patch_target",
            "dynamic setattr attribute name": "dynamic_setattr_name",
            "patch replacement is produced by a call or wrapper": "wrapper_factory",
            "replacement callable was not found": "missing_replacement_callable",
            "replacement is another upstream callable": "upstream_alias_rebind",
            "unsupported patch replacement expression": "unsupported_replacement_expression",
        }
        self.findings.append(
            CandidateFinding(
                relation="monkey_patch",
                downstream_file=module_info.file,
                downstream_owner=None,
                downstream_name=replacement_name or "<unknown>",
                target_expression=target_expression,
                evidence_line=line,
                reason=reason,
                status=status,
                reason_code=reason_code or codes.get(reason, "analysis_gap"),
                generator_issue=generator_issue,
                evidence_scope=self._scope_name(context),
                evidence_guards=context.guards,
            )
        )

    def _missing_patch_target_classification(
        self,
        target: str,
        context: PatchScanContext,
    ) -> tuple[str, str, bool]:
        guards = " ".join(context.guards)
        if "not hasattr(" in guards:
            return "expected", "inject_missing_member", False
        if "hasattr(" in guards:
            return "excluded", "inactive_guard", False

        owner_name = target.rsplit(".", 1)[0]
        owner_exists = (
            self.upstream.find_class(owner_name) is not None
            or owner_name in self.upstream.modules
        )
        if owner_exists:
            return "risk", "missing_upstream_member", False
        return "review", "unresolved_patch_owner", True

    def _scope_name(self, context: PatchScanContext) -> str | None:
        return ".".join(context.scope) if context.scope else None

    def _guard_text(self, node: ast.AST) -> str:
        return " ".join(ast.unparse(node).split())

    def _find_upstream_patch_target(
        self,
        qualified_name: str,
    ) -> CallableInfo | None:
        direct = self.upstream.find_callable(qualified_name)
        if direct is not None:
            return direct
        if "." not in qualified_name:
            return None

        owner_name, method_name = qualified_name.rsplit(".", 1)
        owner = self.upstream.find_class(owner_name)
        if owner is None:
            return None
        mro_result = self._linearized_mro(owner.qualified_name)
        effective_owner = self._effective_method_owner(
            mro_result.owners[1:],
            method_name,
        )
        if effective_owner is None:
            return None
        return self.upstream.find_callable(
            f"{effective_owner}.{method_name}"
        )

    def _class_line(self, class_info: ClassInfo) -> int:
        node = self.downstream.find_callable(class_info.qualified_name)
        return getattr(node.node, "lineno", 0) if node else 0


def _relation_payloads(
    relations: Iterable[Relation],
    *,
    vllm_sha: str,
    ascend_sha: str,
    findings: Iterable[CandidateFinding] = (),
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str | None, str, str],
        list[Relation],
    ] = defaultdict(list)
    for relation in relations:
        signature_key = json.dumps(
            relation.upstream_signature,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        grouped[
            (
                relation.upstream_file,
                relation.upstream_owner,
                relation.upstream_name,
                signature_key,
            )
        ].append(relation)

    payloads: list[dict[str, Any]] = []
    relation_count = 0
    for key in sorted(
        grouped,
        key=lambda item: (item[0], item[1] or "", item[2], item[3]),
    ):
        upstream_file, owner, name, signature_key = key
        consumers = []
        evidence_records = []
        for relation in sorted(
            grouped[key],
            key=lambda item: (
                item.relation,
                item.downstream_file,
                item.downstream_owner or "",
                item.downstream_name,
            ),
        ):
            consumers.append(
                [
                    relation.relation,
                    relation.downstream_file,
                    relation.downstream_owner,
                    relation.downstream_name,
                    relation.downstream_signature,
                ]
            )
            evidence_records.append(
                {
                    "consumer": [
                        relation.relation,
                        relation.downstream_file,
                        relation.downstream_owner,
                        relation.downstream_name,
                    ],
                    "occurrences": [
                        evidence.as_dict()
                        for evidence in relation.evidence
                    ],
                }
            )
            relation_count += 1
        payloads.append(
            {
                "u": [
                    upstream_file,
                    owner,
                    name,
                    json.loads(signature_key),
                ],
                "c": consumers,
                "e": evidence_records,
            }
        )

    finding_payloads = [
        {"f": finding.as_dict()}
        for finding in sorted(
            findings,
            key=lambda item: (
                item.status,
                item.reason_code,
                item.relation,
                item.downstream_file,
                item.evidence_line,
                item.target_expression,
            ),
        )
    ]
    finding_statuses = Counter(
        payload["f"]["status"]
        for payload in finding_payloads
    )
    meta = {
        "_meta": {
            "schema": SCHEMA_VERSION,
            "generator": GENERATOR_VERSION,
            "vllm": vllm_sha,
            "vllm_ascend": ascend_sha,
            "contracts": len(payloads),
            "relations": relation_count,
            "findings": len(finding_payloads),
            "findings_by_status": dict(sorted(finding_statuses.items())),
            "scope": sorted(SUPPORTED_RELATIONS),
        }
    }
    return [meta, *payloads, *finding_payloads]


def _write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    text = "\n".join(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        for payload in payloads
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _load_compact_relations(path: Path) -> list[Relation]:
    relations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if "_meta" in payload or "f" in payload:
            continue
        upstream_file, upstream_owner, upstream_name, upstream_signature = payload["u"]
        for consumer in payload["c"]:
            relation, downstream_file, downstream_owner, downstream_name, downstream_signature = consumer
            if relation not in SUPPORTED_RELATIONS:
                continue
            relations.append(
                Relation(
                    relation=relation,
                    upstream_file=upstream_file,
                    upstream_owner=upstream_owner,
                    upstream_name=upstream_name,
                    upstream_signature=upstream_signature,
                    downstream_file=downstream_file,
                    downstream_owner=downstream_owner,
                    downstream_name=downstream_name,
                    downstream_signature=downstream_signature,
                    evidence_file=downstream_file,
                    evidence_line=0,
                )
            )
    return relations


def _relation_label(relation: Relation) -> dict[str, Any]:
    return {
        "relation": relation.relation,
        "upstream": {
            "file": relation.upstream_file,
            "owner": relation.upstream_owner,
            "name": relation.upstream_name,
        },
        "downstream": {
            "file": relation.downstream_file,
            "owner": relation.downstream_owner,
            "name": relation.downstream_name,
        },
    }


def _downstream_label(
    key: tuple[str, str, str, str],
) -> dict[str, Any]:
    return {
        "relation": key[0],
        "file": key[1],
        "owner": key[2] or None,
        "name": key[3],
    }


def compare_relations(
    generated: Sequence[Relation],
    baseline: Sequence[Relation],
    findings: Sequence[CandidateFinding],
) -> dict[str, Any]:
    finding_statuses = Counter(finding.status for finding in findings)
    generated_exact = {
        relation.exact_key(): relation
        for relation in generated
    }
    baseline_exact = {
        relation.exact_key(): relation
        for relation in baseline
    }
    generated_exact_aliases = {
        key: relation
        for relation in generated
        for key in relation.comparison_exact_keys()
    }
    baseline_exact_aliases = {
        key: relation
        for relation in baseline
        for key in relation.comparison_exact_keys()
    }
    generated_downstream: dict[
        tuple[str, str, str, str],
        list[Relation],
    ] = defaultdict(list)
    baseline_downstream: dict[
        tuple[str, str, str, str],
        list[Relation],
    ] = defaultdict(list)
    for relation in generated:
        for key in relation.comparison_downstream_keys():
            generated_downstream[key].append(relation)
    for relation in baseline:
        for key in relation.comparison_downstream_keys():
            baseline_downstream[key].append(relation)

    exact_matches = {
        key
        for key in baseline_exact
        if any(
            alias in generated_exact_aliases
            for alias in baseline_exact[key].comparison_exact_keys()
        )
    }
    different_upstream = []
    baseline_downstream_keys = {
        relation.downstream_key()
        for relation in baseline
    }
    for key in sorted(baseline_downstream_keys & set(generated_downstream)):
        generated_targets = sorted(
            relation.upstream_key()
            for relation in generated_downstream[key]
        )
        baseline_targets = sorted(
            relation.upstream_key()
            for relation in baseline_downstream[key]
        )
        if generated_targets != baseline_targets:
            different_upstream.append(
                {
                    "downstream": {
                        "relation": key[0],
                        "file": key[1],
                        "owner": key[2] or None,
                        "name": key[3],
                    },
                    "baseline_upstream": baseline_targets,
                    "generated_upstream": generated_targets,
                }
            )

    old_only_keys = set(baseline_exact) - exact_matches
    new_only_keys = {
        key
        for key, relation in generated_exact.items()
        if not any(
            alias in baseline_exact_aliases
            for alias in relation.comparison_exact_keys()
        )
    }
    generated_downstream_keys = {
        relation.downstream_key()
        for relation in generated
    }
    covered_downstream_keys = {
        key
        for key in baseline_downstream_keys
        if key in generated_downstream
    }
    missing_downstream_keys = baseline_downstream_keys - covered_downstream_keys
    new_downstream_keys = {
        key
        for key in generated_downstream_keys
        if not any(
            alias in baseline_downstream
            for relation in generated
            if relation.downstream_key() == key
            for alias in relation.comparison_downstream_keys()
        )
    }
    downstream_coverage = (
        len(covered_downstream_keys) / len(baseline_downstream_keys) * 100
        if baseline_downstream_keys
        else 100.0
    )
    report = {
        "summary": {
            "generated_relations": len(generated),
            "baseline_relations": len(baseline),
            "exact_matches": len(exact_matches),
            "same_downstream_different_upstream": len(different_upstream),
            "old_only": len(old_only_keys),
            "new_only": len(new_only_keys),
            "findings": len(findings),
            "unresolved": finding_statuses["review"],
            "upstream_risks": finding_statuses["risk"],
            "expected": finding_statuses["expected"],
            "excluded": finding_statuses["excluded"],
            "generator_issues": sum(
                finding.generator_issue
                for finding in findings
            ),
            "generated_downstream_endpoints": len(
                generated_downstream_keys
            ),
            "baseline_downstream_endpoints": len(
                baseline_downstream_keys
            ),
            "covered_downstream_endpoints": len(
                covered_downstream_keys
            ),
            "missing_downstream_endpoints": len(
                missing_downstream_keys
            ),
            "new_downstream_endpoints": len(new_downstream_keys),
            "downstream_coverage_percent": round(
                downstream_coverage,
                2,
            ),
            "generated_by_relation": dict(
                sorted(Counter(relation.relation for relation in generated).items())
            ),
            "baseline_by_relation": dict(
                sorted(Counter(relation.relation for relation in baseline).items())
            ),
        },
        "same_downstream_different_upstream": different_upstream,
        "old_only": [
            _relation_label(baseline_exact[key])
            for key in sorted(old_only_keys)
        ],
        "new_only": [
            _relation_label(generated_exact[key])
            for key in sorted(new_only_keys)
        ],
        "missing_downstream": [
            _downstream_label(key)
            for key in sorted(missing_downstream_keys)
        ],
        "new_downstream": [
            _downstream_label(key)
            for key in sorted(new_downstream_keys)
        ],
        "findings": [finding.as_dict() for finding in findings],
    }
    return report


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_sha(label: str, actual: str, expected: str | None) -> None:
    if expected and actual != expected:
        raise SystemExit(
            f"{label} SHA mismatch: expected {expected}, found {actual}"
        )


def _canonical_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--ascend-root", type=Path, required=True)
    parser.add_argument("--expect-vllm-sha")
    parser.add_argument("--expect-ascend-sha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unresolved-output", type=Path)
    parser.add_argument("--compare-with", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    vllm_sha = _git_head(args.vllm_root)
    ascend_sha = _git_head(args.ascend_root)
    _verify_sha("vLLM", vllm_sha, args.expect_vllm_sha)
    _verify_sha("vllm-ascend", ascend_sha, args.expect_ascend_sha)

    generator = InterfaceBoundaryGenerator(args.vllm_root, args.ascend_root)
    relations, findings = generator.generate()
    _write_jsonl(
        args.output,
        _relation_payloads(
            relations,
            vllm_sha=vllm_sha,
            ascend_sha=ascend_sha,
            findings=findings,
        ),
    )

    if args.unresolved_output:
        _write_jsonl(
            args.unresolved_output,
            (finding.as_dict() for finding in findings),
        )

    finding_statuses = Counter(finding.status for finding in findings)
    report: dict[str, Any] = {
        "inputs": {
            "vllm_sha": vllm_sha,
            "vllm_ascend_sha": ascend_sha,
            "generator_version": GENERATOR_VERSION,
        },
        "generated": {
            "relations": len(relations),
            "findings": len(findings),
            "unresolved": finding_statuses["review"],
            "upstream_risks": finding_statuses["risk"],
            "expected": finding_statuses["expected"],
            "excluded": finding_statuses["excluded"],
            "generator_issues": sum(
                finding.generator_issue
                for finding in findings
            ),
            "findings_by_status": dict(sorted(finding_statuses.items())),
            "by_relation": dict(
                sorted(Counter(relation.relation for relation in relations).items())
            ),
            "sha256": _canonical_digest(args.output),
        },
    }
    if args.compare_with:
        baseline = _load_compact_relations(args.compare_with)
        report["comparison"] = compare_relations(relations, baseline, findings)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )
    console_report = {
        "inputs": report["inputs"],
        "generated": report["generated"],
    }
    if "comparison" in report:
        console_report["comparison"] = report["comparison"]["summary"]
    print(json.dumps(console_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
