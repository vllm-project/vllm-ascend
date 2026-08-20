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
"""Analyze an exact vLLM commit range against live vllm-ascend dependencies.

The dependency graph is generated consumer-first from the selected source pair.
The range analyzer then resolves every dependency at both upstream endpoints so
historical incompatibilities are separated from breaks introduced by the range.
"""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis_plans import (
    MAIN2MAIN_SCENARIO,
    AnalysisPlan,
    resolve_analysis_plan,
)
from .call_contracts import (
    DirectCallDependency,
    DirectCallDetector,
    ReturnContract,
    ReturnShape,
    bind_call_shape,
    infer_return_contract,
    replacement_return_compatible,
    return_contract_from_dict,
    return_use_compatible,
)
from .generator import (
    _TRITON_JIT_DECORATOR,
    _TRITON_KERNEL_PROTOCOL,
    GENERATOR_VERSION,
    STDLIB_STRUCTURAL_BASES,
    HistoricalOverrideCandidate,
    InterfaceBoundaryGenerator,
    Relation,
    RelationEvidence,
    SignatureContract,
    _accepts_signature_contract,
    _expression_name,
    _import_binding_reference,
    _jsonable_signature,
    _scope_final_bindings,
    _tag_guard_names,
)
from .models import (
    CompatibilityState,
    RangeFinding,
    SourceEndpoint,
)

RANGE_SCHEMA_VERSION = 7
RANGE_ANALYZER_VERSION = "1.7.0"
CLASSIFICATIONS = (
    "introduced_break",
    "compatibility_warning",
    "preexisting",
    "fixed",
    "analysis_unresolved",
)


def _diagnostic_timing(
    label: str,
    started: float,
    timings: dict[str, float | None] | None = None,
) -> float:
    now = time.perf_counter()
    elapsed = round(now - started, 6)
    if timings is not None:
        timings[label] = elapsed
    if os.environ.get("VLLM_INTERFACE_TIMINGS") == "1":
        print(f"[vllm-interface] {label}: {elapsed:.3f}s", file=sys.stderr, flush=True)
    return now


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def git_head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def resolve_commit(repo: Path, revision: str) -> str:
    try:
        return _git(repo, "rev-parse", f"{revision}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Git commit does not exist: {revision}") from error


def verify_range(vllm_root: Path, old: str, new: str) -> tuple[str, str]:
    old_sha = resolve_commit(vllm_root, old)
    new_sha = resolve_commit(vllm_root, new)
    ancestor = subprocess.run(
        ["git", "-C", str(vllm_root), "merge-base", "--is-ancestor", old_sha, new_sha],
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError(f"old is not an ancestor of new: {old_sha} -> {new_sha}")
    return old_sha, new_sha


def verify_head(label: str, root: Path, expected: str) -> str:
    actual = git_head(root)
    resolved = resolve_commit(root, expected)
    if actual != resolved:
        raise ValueError(f"{label} checkout mismatch: expected {resolved}, got {actual}")
    return actual


def _module_file(module: str) -> tuple[str, str]:
    stem = module.replace(".", "/")
    return f"{stem}.py", f"{stem}/__init__.py"


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _decorator_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _descriptor(node: ast.AST, resolver: Any | None = None) -> str | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    names = {
        (resolver(raw) if resolver is not None and (raw := _decorator_name(item)) else _decorator_name(item))
        for item in node.decorator_list
    }
    for candidate in ("property", "classmethod", "staticmethod"):
        if f"builtins.{candidate}" in names or (resolver is None and candidate in names):
            return candidate
    if any((name or "").rsplit(".", 1)[-1] in {"property", "classmethod", "staticmethod"} for name in names):
        return "unknown"
    return "ordinary"


def _signature_status(
    node: ast.AST | None,
    resolver: Any | None = None,
) -> str | None:
    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return None
    known = {
        "abc.abstractmethod",
        "contextlib.asynccontextmanager",
        "contextlib.contextmanager",
        "typing.override",
        "typing_extensions.override",
    }
    builtin_descriptors = {"builtins.classmethod", "builtins.property", "builtins.staticmethod"}
    for item in node.decorator_list:
        raw = _decorator_name(item)
        resolved = resolver(raw) if resolver is not None and raw else raw
        if (
            resolved in builtin_descriptors
            or (resolver is None and raw in {"classmethod", "property", "staticmethod"})
            or resolved in known
        ):
            continue
        return "unknown"
    return "exact"


def _invocation_signature_status(
    node: ast.AST | None,
    resolver: Any | None,
    invocation_kind: str,
) -> str | None:
    if invocation_kind != _TRITON_KERNEL_PROTOCOL:
        return _signature_status(node, resolver)
    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return None
    references: list[str | None] = []
    for item in node.decorator_list:
        raw = _decorator_name(item)
        references.append(resolver(raw) if resolver is not None and raw else raw or None)
    return "exact" if references == [_TRITON_JIT_DECORATOR] else "unknown"


def _class_nodes(tree: ast.Module) -> Iterator[tuple[tuple[str, ...], ast.ClassDef]]:
    def visit(body: list[ast.stmt], parents: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], ast.ClassDef]]:
        for item in body:
            if isinstance(item, ast.ClassDef):
                path = (*parents, item.name)
                yield path, item
                yield from visit(item.body, path)

    yield from visit(tree.body, ())


def _owner_node(tree: ast.Module, owner: str | None) -> ast.ClassDef | None:
    if not owner:
        return None
    expected = tuple(owner.split("."))
    matches = [node for path, node in _class_nodes(tree) if path == expected or path[-len(expected) :] == expected]
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class _NamedBinding:
    node: ast.AST | None
    status: str
    fingerprint: str | None = None


@dataclass(frozen=True)
class _QualifiedBinding:
    file: str
    owner: str | None
    name: str
    node: ast.AST | None
    status: str
    fingerprint: str | None = None


@dataclass(frozen=True)
class _ResolvedCallBinding:
    binding: _QualifiedBinding
    dispatch_kind: str
    receiver_class: str | None = None


def _body_named_binding(body: list[ast.stmt], name: str) -> _NamedBinding:
    """Return one final runtime namespace binding, or fail closed.

    The shared scope-flow interpreter handles overload stubs followed by a
    concrete implementation, conditional definitions, rebinding, and delete.
    A path-dependent final binding is ``unknown`` rather than ``missing``.
    """

    alternatives = _scope_final_bindings(body, _tag_guard_names(body)).get(name, ())
    if not alternatives:
        return _NamedBinding(None, "missing")
    fingerprint = hashlib.sha256(
        json.dumps(
            [
                {
                    "kind": binding.kind,
                    "node": (ast.dump(binding.node, include_attributes=False) if binding.node is not None else None),
                }
                for binding in alternatives
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if len(alternatives) != 1:
        return _NamedBinding(None, "unknown", fingerprint)
    binding = alternatives[0]
    if binding.kind == "unbound":
        return _NamedBinding(None, "missing", fingerprint)
    if binding.kind in {"function", "class"} and binding.node is not None:
        return _NamedBinding(binding.node, "exact", fingerprint)
    if binding.kind in {"alias", "value"}:
        return _NamedBinding(binding.node, "non_callable", fingerprint)
    return _NamedBinding(None, "unknown", fingerprint)


def _named_binding(tree: ast.Module, owner: str | None, name: str) -> _NamedBinding:
    if owner:
        class_node = _owner_node(tree, owner)
        if class_node is None:
            return _NamedBinding(None, "missing")
        return _body_named_binding(class_node.body, name)
    return _body_named_binding(tree.body, name)


def _named_node(tree: ast.Module, owner: str | None, name: str) -> ast.AST | None:
    binding = _named_binding(tree, owner, name)
    return binding.node if binding.status == "exact" else None


def _node_fingerprint(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _definition_fingerprint(node: ast.AST | None) -> str | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    normalized = copy.deepcopy(node)
    normalized.name = "__interface_callable__"
    normalized.decorator_list = []
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()


def _file_module(file_name: str) -> tuple[str, bool]:
    normalized = file_name.replace("\\", "/")
    stem = normalized[:-3] if normalized.endswith(".py") else normalized
    parts = stem.split("/")
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _bound_signature(
    signature: list[object] | None,
    *,
    descriptor: str | None,
    access_kind: str,
) -> list[object] | None:
    if signature is None:
        return None
    result = copy.deepcopy(signature)
    binds_receiver = access_kind in {"constructor", "instance"} or (
        access_kind == "class_attribute" and descriptor == "classmethod"
    )
    if not binds_receiver or descriptor == "staticmethod":
        return result
    if descriptor not in {"classmethod", "ordinary", None} and access_kind != "constructor":
        return None
    positional_only = result[1]
    positional_or_keyword = result[2]
    if positional_only:
        positional_only.pop(0)
    elif positional_or_keyword:
        positional_or_keyword.pop(0)
    elif result[3] is None:
        return None
    return result


def _signature_parameters(signature: list[object] | None) -> tuple[dict[str, object], ...] | None:
    if not isinstance(signature, list) or len(signature) != 6:
        return None
    parameters: list[dict[str, object]] = []
    for group_index, kind in ((1, "positional_only"), (2, "positional_or_keyword"), (4, "keyword_only")):
        group = signature[group_index]
        if not isinstance(group, list):
            return None
        for position, raw in enumerate(group):
            if (
                not isinstance(raw, list)
                or len(raw) != 2
                or not isinstance(raw[0], str)
                or not isinstance(raw[1], bool)
            ):
                return None
            parameters.append(
                {
                    "name": raw[0],
                    "kind": kind,
                    "required": raw[1],
                    "position": position,
                }
            )
    names = [str(item["name"]) for item in parameters]
    return tuple(parameters) if len(names) == len(set(names)) else None


def _signature_delta(
    old_signature: list[object] | None,
    new_signature: list[object] | None,
) -> dict[str, object] | None:
    old_parameters = _signature_parameters(old_signature)
    new_parameters = _signature_parameters(new_signature)
    if old_parameters is None or new_parameters is None:
        return None
    old_by_name = {str(item["name"]): item for item in old_parameters}
    new_by_name = {str(item["name"]): item for item in new_parameters}
    old_names = [str(item["name"]) for item in old_parameters]
    new_names = [str(item["name"]) for item in new_parameters]
    shared_names = set(old_names) & set(new_names)
    changed = [
        {
            "name": name,
            "old_kind": old_by_name[name]["kind"],
            "new_kind": new_by_name[name]["kind"],
            "old_required": old_by_name[name]["required"],
            "new_required": new_by_name[name]["required"],
        }
        for name in old_names
        if name in shared_names
        and (
            old_by_name[name]["kind"] != new_by_name[name]["kind"]
            or old_by_name[name]["required"] != new_by_name[name]["required"]
        )
    ]
    return {
        "added": [dict(item) for item in new_parameters if str(item["name"]) not in old_by_name],
        "removed": [dict(item) for item in old_parameters if str(item["name"]) not in new_by_name],
        "changed": changed,
        "shared_order_changed": (
            [name for name in old_names if name in shared_names] != [name for name in new_names if name in shared_names]
        ),
        "async_changed": old_signature[0] != new_signature[0],
        "vararg_changed": old_signature[3] != new_signature[3],
        "kwarg_changed": old_signature[5] != new_signature[5],
    }


def _signature_delta_changed(delta: dict[str, object] | None) -> bool:
    if delta is None:
        return False
    return bool(
        delta["added"]
        or delta["removed"]
        or delta["changed"]
        or delta["shared_order_changed"]
        or delta["async_changed"]
        or delta["vararg_changed"]
        or delta["kwarg_changed"]
    )


def _optional_only_signature_additions(delta: dict[str, object] | None) -> tuple[str, ...]:
    if delta is None or not delta["added"]:
        return ()
    if (
        delta["removed"]
        or delta["changed"]
        or delta["shared_order_changed"]
        or delta["async_changed"]
        or delta["vararg_changed"]
        or delta["kwarg_changed"]
    ):
        return ()
    added = delta["added"]
    if not isinstance(added, list) or any(
        item.get("required") is not False or item.get("kind") not in {"positional_or_keyword", "keyword_only"}
        for item in added
        if isinstance(item, dict)
    ):
        return ()
    if any(not isinstance(item, dict) for item in added):
        return ()
    return tuple(str(item["name"]) for item in added)


def _relation_symbol_presence(endpoint: SourceEndpoint) -> bool | None:
    """Return proven symbol presence without conflating ambiguity with deletion."""

    if endpoint.file is None or endpoint.symbol_kind == "missing":
        return False
    if endpoint.symbol_kind in {None, "unknown"}:
        return None
    return True


def _snapshot_signature_contract(endpoint: SourceEndpoint) -> SignatureContract | None:
    """Build the provable runtime-signature view available from one Git snapshot."""

    if endpoint.symbol_kind != "callable":
        return None
    status = endpoint.signature_status or "unknown"
    runtime_signature = endpoint.signature if status == "exact" else None
    binding_descriptor = "ordinary" if endpoint.descriptor == "property" else endpoint.descriptor
    bound_signature = (
        _bound_signature(
            runtime_signature,
            descriptor=binding_descriptor,
            access_kind="instance" if endpoint.owner is not None else "module",
        )
        if runtime_signature is not None
        else None
    )
    if status == "exact" and bound_signature is None:
        status = "unknown"
    return SignatureContract(
        definition_signature=endpoint.signature,
        runtime_entry_signature=runtime_signature,
        reported_signature=runtime_signature,
        bound_call_signature=bound_signature,
        protocol="property_access" if endpoint.descriptor == "property" else "python_call",
        status=status,
        provenance=("git_snapshot",),
    )


def _signature_contract_semantics(contract: SignatureContract | None) -> object:
    if contract is None:
        return None
    return (
        contract.definition_signature,
        contract.runtime_entry_signature,
        contract.reported_signature,
        contract.bound_call_signature,
        contract.forwarded_targets,
        contract.protocol,
        contract.status,
    )


def _runtime_signature_changed(
    old: SourceEndpoint,
    new: SourceEndpoint,
) -> bool:
    """Compare runtime contracts when both snapshot definitions are exact."""

    if old.signature_status != new.signature_status:
        return True
    if old.signature_status != "exact" or new.signature_status != "exact":
        return False
    old_contract = _snapshot_signature_contract(old)
    new_contract = _snapshot_signature_contract(new)
    return _signature_contract_semantics(old_contract) != _signature_contract_semantics(new_contract)


def _ambiguous_binding_changed(old: SourceEndpoint, new: SourceEndpoint) -> bool:
    if old.analysis_fingerprint == new.analysis_fingerprint:
        return False
    return (
        old.symbol_kind in {None, "unknown"}
        or new.symbol_kind in {None, "unknown"}
        or old.signature_status == "unknown"
        or new.signature_status == "unknown"
    )


class GitSnapshot:
    def __init__(self, root: Path, revision: str):
        self.root = root
        self.revision = revision
        self._files: set[str] | None = None
        self._source: dict[str, str | None] = {}
        self._trees: dict[str, ast.Module | None] = {}
        self._bindings: dict[str, dict[str, str]] = {}
        self._keyword_call_candidates: dict[
            tuple[tuple[str, ...], str],
            list[tuple[str, ast.Call, str | None, str]],
        ] = {}
        self._keyword_call_resolutions: dict[
            tuple[tuple[str, ...], str, int, int],
            _ResolvedCallBinding | None,
        ] = {}

    @property
    def files(self) -> set[str]:
        if self._files is None:
            output = _git(self.root, "ls-tree", "-r", "--name-only", self.revision)
            self._files = {line.strip() for line in output.splitlines() if line.strip()}
        return self._files

    def source(self, file_name: str) -> str | None:
        normalized = file_name.replace("\\", "/")
        if normalized not in self._source:
            if normalized not in self.files:
                self._source[normalized] = None
            else:
                raw = subprocess.run(
                    ["git", "-C", str(self.root), "show", f"{self.revision}:{normalized}"],
                    check=True,
                    capture_output=True,
                ).stdout
                self._source[normalized] = raw.decode("utf-8", errors="replace")
        return self._source[normalized]

    def tree(self, file_name: str) -> ast.Module | None:
        normalized = file_name.replace("\\", "/")
        if normalized not in self._trees:
            source = self.source(normalized)
            if source is None:
                self._trees[normalized] = None
            else:
                try:
                    self._trees[normalized] = ast.parse(source, filename=normalized)
                except SyntaxError:
                    self._trees[normalized] = None
        return self._trees[normalized]

    def resolve_module(self, module: str) -> str | None:
        return next((candidate for candidate in _module_file(module) if candidate in self.files), None)

    def _module_bindings(self, file_name: str) -> dict[str, str]:
        normalized = file_name.replace("\\", "/")
        if normalized in self._bindings:
            return self._bindings[normalized]
        tree = self.tree(normalized)
        module, is_package = _file_module(normalized)
        bindings: dict[str, str] = {}
        pending: dict[str, str] = {}
        if tree is not None:
            final = _scope_final_bindings(tree.body, _tag_guard_names(tree.body))
            for name, alternatives in final.items():
                if len(alternatives) != 1:
                    continue
                binding = alternatives[0]
                node = binding.node
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    reference = _import_binding_reference(
                        node,
                        name,
                        module=module,
                        is_package=is_package,
                    )
                    if reference is not None:
                        bindings[name] = reference
                elif binding.kind in {"class", "function"} and isinstance(
                    node,
                    (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef),
                ):
                    if node.name != name:
                        bindings[name] = f"{module}.{node.name}"
                elif binding.kind == "alias" and isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    reference = _expression_name(value)
                    if reference is not None:
                        pending[name] = reference

            changed = True
            while changed:
                changed = False
                for name, reference in pending.items():
                    if name in bindings:
                        continue
                    root, separator, remainder = reference.partition(".")
                    if root in bindings:
                        bindings[name] = f"{bindings[root]}.{remainder}" if separator else bindings[root]
                        changed = True
                    elif reference.startswith("vllm."):
                        bindings[name] = reference
                        changed = True
                    elif _body_named_binding(tree.body, root).status == "exact":
                        bindings[name] = f"{module}.{reference}"
                        changed = True
        self._bindings[normalized] = bindings
        return bindings

    def _resolve_qualified_node(
        self,
        expression: str,
        seen: frozenset[str] = frozenset(),
    ) -> _QualifiedBinding | None:
        if expression in seen or not expression.startswith("vllm"):
            return None
        parts = expression.split(".")
        for split in range(len(parts), 0, -1):
            module = ".".join(parts[:split])
            file_name = self.resolve_module(module)
            if file_name is None:
                continue
            suffix = parts[split:]
            if not suffix:
                return None
            bindings = self._module_bindings(file_name)
            if suffix[0] in bindings:
                target = ".".join([bindings[suffix[0]], *suffix[1:]])
                if not target.startswith("vllm."):
                    target = f"{module}.{target}"
                return self._resolve_qualified_node(target, frozenset((*seen, expression)))
            owner = ".".join(suffix[:-1]) or None
            tree = self.tree(file_name)
            if tree is None:
                return _QualifiedBinding(file_name, owner, suffix[-1], None, "unknown")
            binding = _named_binding(tree, owner, suffix[-1])
            return _QualifiedBinding(
                file_name,
                owner,
                suffix[-1],
                binding.node,
                binding.status,
                binding.fingerprint,
            )
        return None

    def _base_reference(self, file_name: str, node: ast.expr) -> str | None:
        expression_node = node.value if isinstance(node, ast.Subscript) else node
        expression = _expression_name(expression_node)
        if expression is None:
            return None
        if expression in {"object", "builtins.object"}:
            return "builtins.object"
        return self._return_resolver(file_name)(expression)

    def _effective_member(
        self,
        receiver_type: str,
        member: str,
        seen: frozenset[str] = frozenset(),
    ) -> _QualifiedBinding | None:
        """Resolve a member through a provable single-inheritance chain.

        Multiple inheritance requires a complete C3 index.  The range layer
        deliberately returns ``unknown`` for that case instead of borrowing
        the checked-out new endpoint's owner or guessing a DFS order.
        """

        if receiver_type in seen:
            return _QualifiedBinding("", None, member, None, "unknown")
        resolved = self._resolve_qualified_node(receiver_type)
        if resolved is None:
            return None
        if resolved.status != "exact":
            return _QualifiedBinding(
                resolved.file,
                resolved.owner,
                member,
                None,
                resolved.status,
                resolved.fingerprint,
            )
        if not isinstance(resolved.node, ast.ClassDef):
            return _QualifiedBinding(
                resolved.file,
                resolved.owner,
                member,
                resolved.node,
                "non_callable",
                resolved.fingerprint,
            )
        class_node = resolved.node
        actual_owner = ".".join(item for item in (resolved.owner, class_node.name) if item)
        if class_node.decorator_list:
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                None,
                "unknown",
                _node_fingerprint(class_node),
            )
        direct = _body_named_binding(class_node.body, member)
        if direct.status != "missing":
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                direct.node,
                direct.status,
                direct.fingerprint,
            )
        if not class_node.bases:
            return _QualifiedBinding(resolved.file, actual_owner, member, None, "missing")
        if len(class_node.bases) != 1:
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                None,
                "unknown",
                _node_fingerprint(class_node),
            )
        base = self._base_reference(resolved.file, class_node.bases[0])
        if base == "builtins.object":
            return _QualifiedBinding(resolved.file, actual_owner, member, None, "missing")
        if base in STDLIB_STRUCTURAL_BASES:
            # The generator models these stdlib marker bases as structural
            # MRO nodes with no interface members.  Mirror that exact model in
            # snapshot lookup so an otherwise complete vLLM chain does not
            # become unknown merely because it terminates at ``abc.ABC``.
            return _QualifiedBinding(resolved.file, actual_owner, member, None, "missing")
        if base is None or not base.startswith("vllm."):
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                None,
                "unknown",
                _node_fingerprint(class_node),
            )
        return self._effective_member(base, member, frozenset((*seen, receiver_type)))

    def _constructor_class_safe(
        self,
        class_reference: str,
        seen: frozenset[str] = frozenset(),
    ) -> bool:
        """Prove that no class in a single-inheritance chain changes ``type.__call__``."""

        if class_reference in seen:
            return False
        resolved = self._resolve_qualified_node(class_reference)
        if resolved is None or resolved.status != "exact" or not isinstance(resolved.node, ast.ClassDef):
            return False
        node = resolved.node
        if node.decorator_list or node.keywords:
            return False
        if not node.bases:
            return True
        if len(node.bases) != 1:
            return False
        base = self._base_reference(resolved.file, node.bases[0])
        if base == "builtins.object":
            return True
        if base is None or not base.startswith("vllm."):
            return False
        return self._constructor_class_safe(base, frozenset((*seen, class_reference)))

    def _return_resolver(self, file_name: str) -> Any:
        module, _ = _file_module(file_name)
        bindings = self._module_bindings(file_name)

        def resolve(expression: str) -> str | None:
            root, separator, remainder = expression.partition(".")
            if root in bindings:
                return f"{bindings[root]}.{remainder}" if separator else bindings[root]
            if expression.startswith("vllm."):
                return expression
            if (
                expression in {"classmethod", "property", "staticmethod"}
                and (tree := self.tree(file_name)) is not None
                and _body_named_binding(tree.body, expression).status == "missing"
            ):
                return f"builtins.{expression}"
            return f"{module}.{expression}"

        return resolve

    @staticmethod
    def _call_binding_key(
        binding: _QualifiedBinding | None,
    ) -> tuple[str, str | None, str] | None:
        if binding is None or binding.status != "exact":
            return None
        if isinstance(binding.node, ast.ClassDef):
            owner = ".".join(item for item in (binding.owner, binding.node.name) if item)
            return binding.file, owner, "__init__"
        if isinstance(binding.node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return binding.file, binding.owner, binding.name
        return None

    def _call_binding(
        self,
        file_name: str,
        node: ast.Call,
        class_reference: str | None,
    ) -> _ResolvedCallBinding | None:
        function = node.func
        if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
            if function.value.id in {"self", "cls"} and class_reference is not None:
                binding = self._effective_member(class_reference, function.attr)
                return _ResolvedCallBinding(binding, "self_member", class_reference) if binding is not None else None
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Call)
            and isinstance(function.value.func, ast.Name)
            and function.value.func.id == "super"
            and not function.value.args
            and not function.value.keywords
            and class_reference is not None
        ):
            current = self._resolve_qualified_node(class_reference)
            if (
                current is None
                or current.status != "exact"
                or not isinstance(current.node, ast.ClassDef)
                or len(current.node.bases) != 1
            ):
                return None
            base = self._base_reference(current.file, current.node.bases[0])
            if base is None or not base.startswith("vllm."):
                return None
            binding = self._effective_member(base, function.attr)
            return _ResolvedCallBinding(binding, "super_member", class_reference) if binding is not None else None
        expression = _expression_name(function)
        if expression is None:
            return None
        target = self._return_resolver(file_name)(expression)
        if target is None or not target.startswith("vllm."):
            return None
        binding = self._resolve_qualified_node(target)
        if binding is None:
            return None
        return _ResolvedCallBinding(
            binding,
            "direct_constructor" if isinstance(binding.node, ast.ClassDef) else "direct_callable",
        )

    def _keyword_call_candidate_index(
        self,
        files: tuple[str, ...],
        parameters: set[str],
    ) -> dict[str, list[tuple[str, ast.Call, str | None, str]]]:
        normalized_files = tuple(sorted(file_name.replace("\\", "/") for file_name in files))
        index: dict[str, list[tuple[str, ast.Call, str | None, str]]] = {}

        class KeywordCallVisitor(ast.NodeVisitor):
            def __init__(
                self,
                file_name: str,
                parameter: str,
                candidates: list[tuple[str, ast.Call, str | None, str]],
            ):
                self.file_name = file_name
                self.parameter = parameter
                self.candidates = candidates
                self.module, _ = _file_module(file_name)
                self.class_path: list[str] = []
                self.scope_path: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.class_path.append(node.name)
                self.scope_path.append(node.name)
                self.generic_visit(node)
                self.scope_path.pop()
                self.class_path.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.scope_path.append(node.name)
                self.generic_visit(node)
                self.scope_path.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.scope_path.append(node.name)
                self.generic_visit(node)
                self.scope_path.pop()

            def visit_Call(self, node: ast.Call) -> None:
                keywords = sorted(keyword.arg for keyword in node.keywords if keyword.arg is not None)
                if self.parameter in keywords:
                    class_reference = ".".join((self.module, *self.class_path)) if self.class_path else None
                    scope = ".".join((self.module, *self.scope_path))
                    self.candidates.append((self.file_name, node, class_reference, scope))
                self.generic_visit(node)

        changed_file_set = set(normalized_files)
        revision_prefix = f"{self.revision}:"
        for parameter in sorted(parameters):
            cache_key = (normalized_files, parameter)
            if cache_key not in self._keyword_call_candidates:
                grep_output = _git(
                    self.root,
                    "grep",
                    "-l",
                    "-F",
                    "-e",
                    parameter,
                    self.revision,
                    "--",
                    ":(glob)**/*.py",
                    check=False,
                )
                matched_files = {
                    line.removeprefix(revision_prefix).replace("\\", "/")
                    for line in grep_output.splitlines()
                    if line.strip()
                }
                candidates: list[tuple[str, ast.Call, str | None, str]] = []
                for file_name in sorted(matched_files & changed_file_set):
                    if file_name not in self.files:
                        continue
                    tree = self.tree(file_name)
                    if tree is not None:
                        KeywordCallVisitor(file_name, parameter, candidates).visit(tree)
                self._keyword_call_candidates[cache_key] = candidates
            index[parameter] = self._keyword_call_candidates[cache_key]
        return index

    def exact_keyword_call_evidence(
        self,
        endpoint: SourceEndpoint,
        parameter_names: Iterable[str],
        changed_files: tuple[str, ...],
    ) -> list[dict[str, object]]:
        if endpoint.file is None or endpoint.owner is None or endpoint.name is None:
            return []
        expected = (endpoint.file.replace("\\", "/"), endpoint.owner, endpoint.name)
        parameters = set(parameter_names)
        evidence: list[dict[str, object]] = []
        normalized_files = tuple(sorted(file_name.replace("\\", "/") for file_name in changed_files))
        candidate_index = self._keyword_call_candidate_index(normalized_files, parameters)
        candidates = {
            (file_name, node.lineno, node.col_offset): (file_name, node, class_reference, scope)
            for parameter in parameters
            for file_name, node, class_reference, scope in candidate_index.get(parameter, [])
        }
        for candidate_key, (file_name, node, class_reference, scope) in sorted(candidates.items()):
            cache_key = (normalized_files, *candidate_key)
            if cache_key not in self._keyword_call_resolutions:
                self._keyword_call_resolutions[cache_key] = self._call_binding(
                    file_name,
                    node,
                    class_reference,
                )
            resolved_call = self._keyword_call_resolutions[cache_key]
            if resolved_call is None:
                continue
            key = self._call_binding_key(resolved_call.binding)
            if key != expected:
                continue
            keywords = sorted(keyword.arg for keyword in node.keywords if keyword.arg is not None)
            matched = sorted(parameters & set(keywords))
            if not matched:
                continue
            evidence.append(
                {
                    "file": file_name,
                    "line": node.lineno,
                    "column": node.col_offset,
                    "scope": scope,
                    "target": ".".join(item for item in (key[1], key[2]) if item),
                    "keywords": keywords,
                    "matched_parameters": matched,
                    "dispatch_kind": resolved_call.dispatch_kind,
                    "receiver_class": resolved_call.receiver_class,
                }
            )
        return evidence

    def endpoint(self, file_name: str, owner: str | None, name: str) -> SourceEndpoint:
        tree = self.tree(file_name)
        binding = _named_binding(tree, owner, name) if tree is not None else _NamedBinding(None, "unknown")
        node = binding.node if binding.status == "exact" else None
        return_contract = infer_return_contract(
            node,
            resolver=self._return_resolver(file_name),
        )
        if binding.status == "exact":
            symbol_kind = (
                "callable"
                if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
                else "class"
                if isinstance(node, ast.ClassDef)
                else "unknown"
            )
        elif binding.status == "non_callable":
            symbol_kind = "non_callable"
        else:
            symbol_kind = binding.status
        return SourceEndpoint(
            file=file_name if file_name in self.files else None,
            owner=owner,
            name=name,
            line=getattr(node, "lineno", None),
            signature=_jsonable_signature(node),
            descriptor=_descriptor(node, self._return_resolver(file_name)) if node is not None else None,
            symbol_kind=symbol_kind,
            signature_status=(
                "unknown" if binding.status == "unknown" else _signature_status(node, self._return_resolver(file_name))
            ),
            analysis_fingerprint=binding.fingerprint,
            return_contract=return_contract.as_dict() if return_contract is not None else None,
        )

    def call_endpoint(
        self,
        expression: str,
        access_kind: str,
        *,
        receiver_type: str | None = None,
        member: str | None = None,
        invocation_kind: str = "python_call",
    ) -> SourceEndpoint:
        effective_access_kind = access_kind
        resolved: _QualifiedBinding | None = None
        if access_kind == "instance" and receiver_type is not None and member is not None:
            if receiver_type.startswith("vllm."):
                resolved = self._effective_member(receiver_type, member)
            else:
                # ``self``/``super`` receiver classes live downstream and are
                # not present in this upstream snapshot.  The detector's
                # exact effective owner is safe while it still exists at this
                # endpoint; if it moved or disappeared, report unknown rather
                # than treating the new-side owner as an old-side deletion.
                candidate = self._resolve_qualified_node(expression)
                if candidate is None or candidate.status != "exact":
                    return SourceEndpoint(
                        file=None,
                        owner=None,
                        name=member,
                        symbol_kind="unknown",
                        signature_status="unknown",
                    )
                resolved = candidate
        elif access_kind == "direct" and member is not None and expression.endswith(f".{member}"):
            receiver = expression[: -(len(member) + 1)]
            receiver_binding = self._resolve_qualified_node(receiver)
            if (
                receiver_binding is not None
                and receiver_binding.status == "exact"
                and isinstance(receiver_binding.node, ast.ClassDef)
            ):
                resolved = self._effective_member(receiver, member)
                effective_access_kind = "class_attribute"
        if resolved is None:
            resolved = self._resolve_qualified_node(expression)
        if resolved is None:
            return SourceEndpoint(None, None, expression, symbol_kind="missing")
        file_name, owner, name, node = (
            resolved.file,
            resolved.owner,
            resolved.name,
            resolved.node,
        )
        if resolved.status in {"missing", "unknown"}:
            return SourceEndpoint(
                file=file_name or None,
                owner=owner,
                name=name,
                symbol_kind=resolved.status,
                signature_status="unknown" if resolved.status == "unknown" else None,
                analysis_fingerprint=resolved.fingerprint,
            )
        if resolved.status == "non_callable":
            return SourceEndpoint(
                file=file_name,
                owner=owner,
                name=name,
                line=getattr(node, "lineno", None),
                symbol_kind="non_callable",
                analysis_fingerprint=resolved.fingerprint,
            )
        if isinstance(node, ast.ClassDef):
            initializer_binding = self._effective_member(expression, "__init__")
            new_binding = self._effective_member(expression, "__new__")
            constructor_fingerprint = hashlib.sha256(
                json.dumps(
                    [
                        resolved.fingerprint,
                        initializer_binding.fingerprint if initializer_binding is not None else None,
                        new_binding.fingerprint if new_binding is not None else None,
                    ],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            constructor_unknown = (
                bool(node.decorator_list)
                or bool(node.keywords)
                or not self._constructor_class_safe(expression)
                or initializer_binding is None
                or initializer_binding.status in {"non_callable", "unknown"}
                or new_binding is None
                or new_binding.status != "missing"
            )
            initializer = (
                initializer_binding.node
                if initializer_binding is not None and initializer_binding.status == "exact"
                else None
            )
            if (
                initializer is None
                and initializer_binding is not None
                and initializer_binding.status == "missing"
                and not constructor_unknown
            ):
                initializer = ast.parse("def __init__(self): pass").body[0]
            signature = _bound_signature(
                _jsonable_signature(initializer),
                descriptor="ordinary",
                access_kind="constructor",
            )
            contract = ReturnContract(
                protocol="value",
                variants=(
                    # Constructor calls expose the created object, not
                    # ``__init__``'s mandatory None return.
                    ReturnShape("object", type_ref=expression),
                ),
                status="exact",
                provenance=("class_constructor",),
            )
            return SourceEndpoint(
                file=file_name,
                owner=owner,
                name=name,
                line=node.lineno,
                signature=signature,
                descriptor=None,
                symbol_kind="constructor",
                signature_status=(
                    "unknown"
                    if constructor_unknown or initializer is None
                    else _signature_status(
                        initializer,
                        self._return_resolver(initializer_binding.file)
                        if initializer_binding is not None and initializer_binding.file
                        else None,
                    )
                ),
                analysis_fingerprint=constructor_fingerprint,
                return_contract=(
                    ReturnContract(
                        protocol=contract.protocol,
                        variants=contract.variants,
                        status="unknown",
                        provenance=(*contract.provenance, "unknown_constructor_protocol"),
                    ).as_dict()
                    if constructor_unknown
                    else contract.as_dict()
                ),
            )
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return SourceEndpoint(
                file=file_name,
                owner=owner,
                name=name,
                line=getattr(node, "lineno", None),
                symbol_kind="non_callable",
                analysis_fingerprint=resolved.fingerprint,
            )
        descriptor = _descriptor(node, self._return_resolver(file_name))
        if access_kind == "direct":
            effective_access_kind = "class_attribute" if owner is not None else "module"
        signature = _bound_signature(
            _jsonable_signature(node),
            descriptor=descriptor,
            access_kind=effective_access_kind,
        )
        contract = infer_return_contract(node, resolver=self._return_resolver(file_name))
        return SourceEndpoint(
            file=file_name,
            owner=owner,
            name=name,
            line=node.lineno,
            signature=signature,
            descriptor=descriptor,
            symbol_kind="callable",
            signature_status=_invocation_signature_status(
                node,
                self._return_resolver(file_name),
                invocation_kind,
            ),
            analysis_fingerprint=resolved.fingerprint,
            return_contract=contract.as_dict() if contract is not None else None,
        )

    def expression_endpoint(self, expression: str) -> SourceEndpoint | None:
        parts = expression.strip().split(".")
        if not parts or parts[0] != "vllm":
            return None
        for split in range(len(parts), 0, -1):
            module = ".".join(parts[:split])
            file_name = self.resolve_module(module)
            if file_name is None:
                continue
            suffix = parts[split:]
            if not suffix:
                return SourceEndpoint(file=file_name, owner=None, name=None)
            owner = ".".join(suffix[:-1]) or None
            return self.endpoint(file_name, owner, suffix[-1])
        return None

    def unique_rename(
        self,
        file_name: str,
        owner: str | None,
        old_name: str,
        fingerprint: str | None,
    ) -> SourceEndpoint | None:
        if fingerprint is None:
            return None
        tree = self.tree(file_name)
        if tree is None:
            return None
        owner_node = _owner_node(tree, owner) if owner else None
        if owner and owner_node is None:
            return None
        body = owner_node.body if owner_node is not None else tree.body
        matches = [
            item
            for item in body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name != old_name
            and _definition_fingerprint(item) == fingerprint
        ]
        if len(matches) != 1:
            return None
        node = matches[0]
        return SourceEndpoint(
            file=file_name,
            owner=owner,
            name=node.name,
            line=node.lineno,
            signature=_jsonable_signature(node),
            descriptor=_descriptor(node, self._return_resolver(file_name)),
            symbol_kind="callable",
            signature_status=_signature_status(node, self._return_resolver(file_name)),
            analysis_fingerprint=_node_fingerprint(node),
            return_contract=(
                contract.as_dict()
                if (contract := infer_return_contract(node, resolver=self._return_resolver(file_name))) is not None
                else None
            ),
        )


def _rename_maps(root: Path, old: str, new: str) -> tuple[dict[str, str], dict[str, str]]:
    output = _git(root, "diff", "--name-status", "--find-renames", old, new)
    old_to_new: dict[str, str] = {}
    new_to_old: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].startswith("R"):
            old_path, new_path = parts[1], parts[2]
            old_to_new[old_path] = new_path
            new_to_old[new_path] = old_path
    return old_to_new, new_to_old


def _changed_python_files(root: Path, old: str, new: str) -> tuple[str, ...]:
    output = _git(
        root,
        "diff",
        "--name-only",
        "--diff-filter=ACMRT",
        old,
        new,
    )
    return tuple(
        sorted(line.strip().replace("\\", "/") for line in output.splitlines() if line.strip().endswith(".py"))
    )


def _state(
    upstream: SourceEndpoint,
    downstream_signature: list[object] | None,
    relation: str,
    installed_descriptor: str | None = None,
    upstream_contract: SignatureContract | None = None,
) -> CompatibilityState:
    presence = _relation_symbol_presence(upstream)
    if presence is False:
        return CompatibilityState(False, False, "upstream target does not exist")
    if presence is None:
        return CompatibilityState(None, None, "upstream target binding could not be proven")
    if relation == "inheritance":
        return (
            CompatibilityState(True, True, "upstream base class exists")
            if upstream.symbol_kind == "class"
            else CompatibilityState(True, False, "upstream base target is no longer a class")
        )
    if upstream.symbol_kind != "callable":
        return CompatibilityState(True, False, "upstream target is no longer callable")
    if (
        upstream.owner is not None
        and upstream.descriptor is not None
        and installed_descriptor is not None
        and upstream.descriptor != installed_descriptor
    ):
        return CompatibilityState(
            True,
            False,
            "installed descriptor does not preserve the upstream access protocol",
        )
    if upstream_contract is not None:
        if upstream_contract.status != "exact":
            return CompatibilityState(
                True,
                None,
                "upstream runtime signature transform could not be proven",
            )
        upstream_signature = upstream_contract.bound_call_signature
    else:
        if upstream.signature_status == "unknown":
            return CompatibilityState(
                True,
                None,
                "upstream runtime signature transform could not be proven",
            )
        upstream_signature = _bound_signature(
            upstream.signature,
            descriptor=upstream.descriptor,
            access_kind="instance" if upstream.owner is not None else "module",
        )
    if upstream_signature is None or downstream_signature is None:
        return CompatibilityState(True, None, "callable signature could not be compared")
    compatible = _accepts_signature_contract(upstream_signature, downstream_signature)
    return CompatibilityState(
        True,
        compatible,
        (
            "downstream accepts the upstream call contract"
            if compatible
            else "downstream does not accept the upstream call contract"
        ),
    )


def _direct_call_state(upstream: SourceEndpoint, dependency: DirectCallDependency) -> CompatibilityState:
    if upstream.symbol_kind in {None, "unknown"}:
        return CompatibilityState(None, None, "upstream call target binding could not be proven")
    if upstream.file is None or upstream.symbol_kind == "missing":
        return CompatibilityState(False, False, "upstream call target does not exist")
    if upstream.symbol_kind not in {"callable", "constructor"}:
        return CompatibilityState(True, False, "upstream target is no longer callable")
    if upstream.signature_status == "unknown":
        return CompatibilityState(True, None, "upstream runtime signature transform could not be proven")
    compatible, reason = bind_call_shape(upstream.signature, dependency.call_shape)
    return CompatibilityState(True, compatible, reason)


def _replacement_return_state(
    upstream: SourceEndpoint,
    downstream: SourceEndpoint,
) -> CompatibilityState:
    presence = _relation_symbol_presence(upstream)
    if presence is False:
        return CompatibilityState(False, False, "upstream target does not exist")
    if presence is None:
        return CompatibilityState(None, None, "upstream target binding could not be proven")
    if upstream.symbol_kind != "callable":
        return CompatibilityState(True, False, "upstream target is no longer callable")
    compatible, reason = replacement_return_compatible(
        return_contract_from_dict(upstream.return_contract),
        return_contract_from_dict(downstream.return_contract),
    )
    return CompatibilityState(True, compatible, reason)


def _return_use_state(
    upstream: SourceEndpoint,
    dependency: DirectCallDependency,
) -> CompatibilityState:
    if upstream.symbol_kind in {None, "unknown"}:
        return CompatibilityState(None, None, "upstream call target binding could not be proven")
    if upstream.file is None or upstream.symbol_kind == "missing":
        return CompatibilityState(False, False, "upstream call target does not exist")
    if upstream.symbol_kind not in {"callable", "constructor"}:
        return CompatibilityState(True, False, "upstream target is no longer callable")
    compatible, reason = return_use_compatible(
        return_contract_from_dict(upstream.return_contract),
        dependency.return_use,
    )
    return CompatibilityState(True, compatible, reason)


def _classify(
    old_state: CompatibilityState,
    new_state: CompatibilityState,
    contract_changed: bool,
    *,
    newly_introduced_contract: bool = False,
) -> str:
    if newly_introduced_contract:
        if new_state.compatible is False:
            return "introduced_break"
        if new_state.compatible is True:
            return "compatibility_warning"
        return "analysis_unresolved"
    if old_state.compatible is True and new_state.compatible is False:
        return "introduced_break"
    if old_state.compatible is False and new_state.compatible is True:
        return "fixed"
    if old_state.compatible is False and new_state.compatible is False:
        return "preexisting"
    if contract_changed and old_state.compatible is True and new_state.compatible is True:
        return "compatibility_warning"
    return "analysis_unresolved"


def _change_text(
    old: SourceEndpoint,
    new: SourceEndpoint,
    contract_kind: str = "call_arguments",
    *,
    runtime_signature_changed: bool = False,
) -> str:
    if old.file is None and new.file is not None:
        return "upstream target was added"
    if old.file is not None and new.file is None:
        return "upstream target was removed"
    if old.symbol_kind == "missing" and new.symbol_kind != "missing":
        return "upstream symbol was added"
    if old.symbol_kind != "missing" and new.symbol_kind == "missing":
        return "upstream symbol was removed"
    if old.symbol_kind != new.symbol_kind:
        return f"upstream symbol binding changed: {old.symbol_kind} -> {new.symbol_kind}"
    if old.file != new.file:
        return f"upstream target moved: {old.file} -> {new.file}"
    if old.name != new.name:
        return f"upstream callable renamed: {old.name} -> {new.name}"
    if old.descriptor != new.descriptor:
        return f"descriptor changed: {old.descriptor} -> {new.descriptor}"
    if runtime_signature_changed:
        return "callable runtime signature contract changed"
    if _ambiguous_binding_changed(old, new):
        return "ambiguous callable binding changed and requires review"
    if contract_kind == "return_usage" or contract_kind == "replacement_return":
        if old.return_contract != new.return_contract:
            return "callable return contract changed"
    elif old.signature != new.signature:
        return "callable parameter contract changed"
    return "no exact callable contract delta"


def _suggestion(relation: str, classification: str, old: SourceEndpoint, new: SourceEndpoint) -> str:
    if classification == "preexisting":
        return "作为历史问题单独处理，不归因于本次上游升级。"
    if classification == "fixed":
        return "上游已经恢复兼容，确认下游兼容代码是否仍需保留。"
    if new.file is None:
        return "更新下游依赖目标；若上游已删除该能力，需要移除 patch/继承并补充替代实现。"
    if old.name != new.name:
        return f"把下游依赖从 {old.name} 更新到 {new.name}，并重新核对参数转发。"
    if relation == "monkey_patch":
        return "调整 replacement 签名，使其接受上游新调用方式，并确认 patch 安装路径仍生效。"
    if relation == "override":
        return "同步 override 参数并检查 super() 调用和关键字转发。"
    if relation == "inheritance":
        return "核对新基类路径和 MRO；不要在继承链不完整时猜测替代类。"
    if relation == "direct_import":
        return "更新 import 模块或符号路径，并补充导入边界测试。"
    if relation == "direct_call":
        return "同步下游调用参数或返回值消费方式，并为该调用点补充接口级回归测试。"
    return "根据上下游精确契约差异调整依赖，并补充接口级回归测试。"


def _finding_id(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _relation_endpoints(
    relation: Relation,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    new_to_old: dict[str, str],
) -> tuple[SourceEndpoint, SourceEndpoint]:
    old_file = new_to_old.get(relation.upstream_file, relation.upstream_file)
    old_endpoint = old_snapshot.endpoint(old_file, relation.upstream_owner, relation.upstream_name)
    new_endpoint = new_snapshot.endpoint(
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
    )
    if _relation_symbol_presence(new_endpoint) is False:
        old_tree = old_snapshot.tree(old_file)
        old_node = _named_node(old_tree, relation.upstream_owner, relation.upstream_name) if old_tree else None
        renamed = new_snapshot.unique_rename(
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            _definition_fingerprint(old_node),
        )
        if renamed is not None:
            new_endpoint = renamed
    return old_endpoint, new_endpoint


def _relation_downstream_endpoint(
    relation: Relation,
    engine: InterfaceBoundaryGenerator,
) -> SourceEndpoint:
    module, _ = _file_module(relation.downstream_file)
    qualified_name = ".".join(item for item in (module, relation.downstream_owner, relation.downstream_name) if item)
    callable_info = engine.downstream.find_callable(qualified_name)
    node = callable_info.node if callable_info is not None else None

    def resolve(expression: str) -> str | None:
        if (
            expression in {"classmethod", "property", "staticmethod"}
            and relation.installed_descriptor_kind == expression
        ):
            # Descriptor discovery has already proven this bare decorator is
            # the corresponding builtin.  Preserve that proof for return
            # inference instead of treating it as an unknown runtime transform.
            return f"builtins.{expression}"
        return engine.downstream.resolve_reference(module, expression)

    return_contract = infer_return_contract(
        node,
        resolver=resolve,
        forward_name=relation.upstream_name if relation.relation == "override" else None,
    )
    installed_contract = relation.installed_signature_contract
    if return_contract is not None and installed_contract is not None and installed_contract.status != "exact":
        return_contract = ReturnContract(
            protocol=return_contract.protocol,
            variants=return_contract.variants,
            status="unknown",
            provenance=(*return_contract.provenance, "unknown_runtime_wrapper"),
        )
    installed_signature = (
        installed_contract.bound_call_signature
        if installed_contract is not None and installed_contract.status == "exact"
        else None
    )
    if installed_contract is None:
        installed_signature = _bound_signature(
            relation.downstream_signature,
            descriptor=relation.installed_descriptor_kind,
            access_kind="instance" if relation.upstream_owner is not None else "module",
        )
    downstream = SourceEndpoint(
        file=relation.downstream_file,
        owner=relation.downstream_owner,
        name=relation.downstream_name,
        line=relation.evidence_line,
        signature=installed_signature,
        descriptor=relation.installed_descriptor_kind,
        symbol_kind="callable",
        signature_status=(installed_contract.status if installed_contract is not None else "exact"),
        return_contract=return_contract.as_dict() if return_contract is not None else None,
    )
    return downstream


def _finding_action(classification: str, gates: dict[str, bool]) -> str:
    if classification == "introduced_break" and all(gates.values()):
        return "modify"
    return "dismiss" if classification in {"preexisting", "fixed"} else "review"


def _override_details(relation: Relation) -> dict[str, Any]:
    if relation.relation != "override" or not relation.override_paths:
        return {}
    paths = [list(path) for path in relation.override_paths]
    return {
        "override_paths": paths,
        "override_depth": max(len(path) - 1 for path in relation.override_paths),
        "impact_kind": (
            "transitive_subclass_override"
            if any(len(path) > 2 for path in relation.override_paths)
            else "direct_override"
        ),
    }


def _function_body_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    nodes: list[ast.AST] = []

    class BodyVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def generic_visit(self, node: ast.AST) -> None:
            nodes.append(node)
            super().generic_visit(node)

    BodyVisitor().visit(function)
    return nodes


def _resolved_local_expression(
    expression: ast.AST,
    bindings: dict[str, str],
    *,
    module: str,
    module_symbols: set[str],
) -> str | None:
    raw = _expression_name(expression)
    if raw is None:
        return None
    root, separator, remainder = raw.partition(".")
    if root in bindings:
        return f"{bindings[root]}.{remainder}" if separator else bindings[root]
    if root in module_symbols:
        return f"{module}.{raw}"
    if raw.startswith(("vllm.", "vllm_ascend.")):
        return raw
    return None


def _registered_oot_overrides(
    engine: InterfaceBoundaryGenerator,
) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Prove exact ``CustomOp.register_oot`` name-to-class registrations.

    This is intentionally narrow.  A dict literal is accepted only when the
    same function iterates that dict and forwards its key/value variables to
    ``CustomOp.register_oot(name=..., _decorated_op_cls=...)``.
    """

    registrations: dict[tuple[str, str], list[dict[str, object]]] = {}
    for module_info in engine.downstream.modules.values():
        module_symbols = {*module_info.classes, *module_info.functions}
        for function in (
            node for node in ast.walk(module_info.tree) if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        ):
            body_nodes = _function_body_nodes(function)
            local_bindings = dict(module_info.imports)
            for node in body_nodes:
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                for alias in node.names:
                    local_name = alias.asname or (
                        alias.name.split(".", 1)[0] if isinstance(node, ast.Import) else alias.name
                    )
                    target = _import_binding_reference(
                        node,
                        local_name,
                        module=module_info.name,
                        is_package=module_info.is_package,
                    )
                    if target is not None:
                        local_bindings[local_name] = target

            registered_dicts: set[str] = set()
            registration_lines: dict[str, int] = {}
            for node in body_nodes:
                if not isinstance(node, ast.For):
                    continue
                if not (
                    isinstance(node.target, (ast.List, ast.Tuple))
                    and len(node.target.elts) == 2
                    and all(isinstance(item, ast.Name) for item in node.target.elts)
                    and isinstance(node.iter, ast.Call)
                    and not node.iter.args
                    and not node.iter.keywords
                    and isinstance(node.iter.func, ast.Attribute)
                    and node.iter.func.attr == "items"
                    and isinstance(node.iter.func.value, ast.Name)
                ):
                    continue
                key_name = node.target.elts[0].id
                value_name = node.target.elts[1].id
                registry_name = node.iter.func.value.id
                for call in (
                    child for statement in node.body for child in ast.walk(statement) if isinstance(child, ast.Call)
                ):
                    if not (
                        isinstance(call.func, ast.Attribute)
                        and call.func.attr == "register_oot"
                        and _resolved_local_expression(
                            call.func.value,
                            local_bindings,
                            module=module_info.name,
                            module_symbols=module_symbols,
                        )
                        == "vllm.model_executor.custom_op.CustomOp"
                    ):
                        continue
                    keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}
                    if (
                        isinstance(keywords.get("name"), ast.Name)
                        and keywords["name"].id == key_name
                        and isinstance(keywords.get("_decorated_op_cls"), ast.Name)
                        and keywords["_decorated_op_cls"].id == value_name
                    ):
                        registered_dicts.add(registry_name)
                        registration_lines[registry_name] = call.lineno

            if not registered_dicts:
                continue
            for node in body_nodes:
                dictionary: ast.Dict | None = None
                target_name: str | None = None
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Dict)
                ):
                    target_name = node.targets[0].id
                    dictionary = node.value
                elif (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and isinstance(node.value, ast.Dict)
                ):
                    target_name = node.target.id
                    dictionary = node.value
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "update"
                    and isinstance(node.func.value, ast.Name)
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Dict)
                    and not node.keywords
                ):
                    target_name = node.func.value.id
                    dictionary = node.args[0]
                if target_name not in registered_dicts or dictionary is None:
                    continue
                for key, value in zip(dictionary.keys, dictionary.values, strict=True):
                    if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                        continue
                    downstream_target = _resolved_local_expression(
                        value,
                        local_bindings,
                        module=module_info.name,
                        module_symbols=module_symbols,
                    )
                    if downstream_target is None:
                        continue
                    evidence = {
                        "file": module_info.file,
                        "line": getattr(value, "lineno", node.lineno),
                        "scope": f"{module_info.name}.{function.name}",
                        "registry": target_name,
                        "registration_line": registration_lines[target_name],
                        "upstream_class_name": key.value,
                        "downstream_target": downstream_target,
                    }
                    registrations.setdefault((key.value, downstream_target), []).append(evidence)
    return registrations


def _optional_override_dispatch_evidence(
    relation: Relation,
    endpoint: SourceEndpoint,
    candidates: list[dict[str, object]],
    registered_overrides: dict[tuple[str, str], list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Keep only calls whose dispatch can reach this exact downstream override."""

    if endpoint.file is None or endpoint.owner is None or endpoint.name is None:
        return []
    if endpoint.name != "__init__":
        module, _ = _file_module(endpoint.file)
        defining_class = f"{module}.{endpoint.owner}"
        return [
            item
            for item in candidates
            if item.get("dispatch_kind") == "self_member" and item.get("receiver_class") == defining_class
        ]

    downstream_module, _ = _file_module(relation.downstream_file)
    downstream_target = ".".join(item for item in (downstream_module, relation.downstream_owner) if item)
    registration_evidence = registered_overrides.get(
        (endpoint.owner.rsplit(".", 1)[-1], downstream_target),
        [],
    )
    if not registration_evidence:
        return []
    return [
        {**item, "dispatch_proof": registration_evidence}
        for item in candidates
        if item.get("dispatch_kind") == "direct_constructor"
    ]


def _relation_findings(
    relation: Relation,
    engine: InterfaceBoundaryGenerator,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    new_to_old: dict[str, str],
    changed_upstream_files: tuple[str, ...],
    registered_overrides: dict[tuple[str, str], list[dict[str, object]]],
) -> list[RangeFinding]:
    """Compare one verified replacement relation across the selected range."""
    old_endpoint, new_endpoint = _relation_endpoints(
        relation,
        old_snapshot,
        new_snapshot,
        new_to_old,
    )
    downstream = _relation_downstream_endpoint(relation, engine)
    old_exists = _relation_symbol_presence(old_endpoint)
    new_exists = _relation_symbol_presence(new_endpoint)
    runtime_signature_changed = _runtime_signature_changed(
        old_endpoint,
        new_endpoint,
    )
    contract_changed = (
        old_exists != new_exists
        or old_endpoint.file != new_endpoint.file
        or old_endpoint.name != new_endpoint.name
        or old_endpoint.signature != new_endpoint.signature
        or old_endpoint.descriptor != new_endpoint.descriptor
        or old_endpoint.signature_status != new_endpoint.signature_status
        or _ambiguous_binding_changed(old_endpoint, new_endpoint)
        or runtime_signature_changed
    )
    evidence = [item.as_dict() for item in relation.evidence] or [
        {"file": relation.evidence_file, "line": relation.evidence_line}
    ]
    findings: list[RangeFinding] = []
    if contract_changed:
        contract_kind = "base_presence" if relation.relation == "inheritance" else "call_arguments"
        old_state = _state(
            old_endpoint,
            downstream.signature,
            relation.relation,
            downstream.descriptor,
        )
        new_state = _state(
            new_endpoint,
            downstream.signature,
            relation.relation,
            downstream.descriptor,
            _snapshot_signature_contract(new_endpoint),
        )
        classification = _classify(
            old_state,
            new_state,
            contract_changed,
            newly_introduced_contract=(
                relation.relation in {"monkey_patch", "override"} and old_exists is False and new_exists is True
            ),
        )
        parameter_delta = (
            _signature_delta(old_endpoint.signature, new_endpoint.signature)
            if old_endpoint.signature_status == "exact" and new_endpoint.signature_status == "exact"
            else None
        )
        optional_parameters = (
            _optional_only_signature_additions(parameter_delta)
            if relation.relation == "override"
            and classification == "introduced_break"
            and old_state.compatible is True
            and new_state.compatible is False
            else ()
        )
        candidate_upstream_call_evidence = (
            new_snapshot.exact_keyword_call_evidence(
                new_endpoint,
                optional_parameters,
                changed_upstream_files,
            )
            if optional_parameters
            else []
        )
        upstream_call_evidence = _optional_override_dispatch_evidence(
            relation,
            new_endpoint,
            candidate_upstream_call_evidence,
            registered_overrides,
        )
        optional_contract_review = bool(optional_parameters and not upstream_call_evidence)
        masked_preexisting_delta = bool(
            relation.relation == "override"
            and classification == "preexisting"
            and _signature_delta_changed(parameter_delta)
        )
        gates = {
            "relationship_verified": True,
            "contract_changed": contract_changed,
            "runtime_reachable": True,
            "version_lane_matches": True,
        }
        action = (
            "review" if optional_contract_review or masked_preexisting_delta else _finding_action(classification, gates)
        )
        if optional_contract_review:
            suggestion = (
                "暂不判定为运行时不兼容。请确认实际调用路径；如果运行时会使用该下游实现，请同步方法签名并处理新增参数。"
            )
        elif masked_preexisting_delta:
            suggestion = (
                "本次上游又引入了新的精确参数差异，但下游在 old 已存在其他不兼容；"
                "作为独立 review 核对本次增量，不直接归入升级修复清单。"
            )
        else:
            suggestion = _suggestion(
                relation.relation,
                classification,
                old_endpoint,
                new_endpoint,
            )
        review_details: dict[str, object] = {}
        if optional_parameters:
            review_details.update(
                {
                    "optional_contract_only": True,
                    "new_optional_parameters": list(optional_parameters),
                    "upstream_call_evidence": upstream_call_evidence,
                    "candidate_upstream_call_evidence": candidate_upstream_call_evidence,
                    "actionability_reason": (
                        "exact_upstream_call_and_dispatch_proof_pass_new_optional_parameter"
                        if upstream_call_evidence
                        else "strict_optional_contract_without_proven_downstream_dispatch"
                    ),
                    "parameter_delta": parameter_delta,
                }
            )
        if masked_preexisting_delta:
            review_details.update(
                {
                    "new_delta_on_preexisting_break": True,
                    "actionability_reason": "new_delta_masked_by_preexisting_incompatibility",
                    "parameter_delta": parameter_delta,
                }
            )
        findings.append(
            RangeFinding(
                finding_id=_finding_id(
                    relation.exact_key(),
                    contract_kind,
                    old_snapshot.revision,
                    new_snapshot.revision,
                ),
                classification=classification,
                relation=relation.relation,
                priority=(
                    "P0"
                    if relation.relation == "monkey_patch" and action == "modify"
                    else ("P1" if action == "modify" else "P2")
                ),
                action=action,
                confidence="high" if classification != "analysis_unresolved" else "medium",
                upstream_old=old_endpoint,
                upstream_new=new_endpoint,
                downstream=downstream,
                old_state=old_state,
                new_state=new_state,
                change=_change_text(
                    old_endpoint,
                    new_endpoint,
                    runtime_signature_changed=runtime_signature_changed,
                ),
                evidence=evidence,
                gates=gates,
                suggestion=suggestion,
                contract_kind=contract_kind,
                direction="upstream_contract_to_downstream_implementation",
                details={
                    "installed_signature": downstream.signature,
                    "installed_descriptor": downstream.descriptor,
                    **review_details,
                    **_override_details(relation),
                },
            )
        )

    return_changed = new_exists is True and (
        old_exists is not True or old_endpoint.return_contract != new_endpoint.return_contract
    )
    if relation.relation in {"monkey_patch", "override"} and return_changed:
        old_state = _replacement_return_state(old_endpoint, downstream)
        new_state = _replacement_return_state(new_endpoint, downstream)
        classification = _classify(
            old_state,
            new_state,
            return_changed,
            newly_introduced_contract=old_exists is False and new_exists is True,
        )
        gates = {
            "relationship_verified": True,
            "contract_changed": return_changed,
            "runtime_reachable": True,
            "version_lane_matches": True,
        }
        action = _finding_action(classification, gates)
        findings.append(
            RangeFinding(
                finding_id=_finding_id(
                    relation.exact_key(),
                    "replacement_return",
                    old_snapshot.revision,
                    new_snapshot.revision,
                ),
                classification=classification,
                relation=relation.relation,
                priority="P0"
                if relation.relation == "monkey_patch" and action == "modify"
                else ("P1" if action == "modify" else "P2"),
                action=action,
                confidence="high" if classification != "analysis_unresolved" else "medium",
                upstream_old=old_endpoint,
                upstream_new=new_endpoint,
                downstream=downstream,
                old_state=old_state,
                new_state=new_state,
                change=_change_text(old_endpoint, new_endpoint, "replacement_return"),
                evidence=evidence,
                gates=gates,
                suggestion="同步 patch/override 的返回协议，使其满足上游新接口约定，并补充返回值回归测试。",
                contract_kind="replacement_return",
                direction="upstream_contract_to_downstream_implementation",
                details={
                    "upstream_old_return": old_endpoint.return_contract,
                    "upstream_new_return": new_endpoint.return_contract,
                    "downstream_return": downstream.return_contract,
                    **_override_details(relation),
                },
            )
        )
    return findings


@dataclass(frozen=True)
class ImportReference:
    module: str
    symbol: str | None
    file: str
    line: int


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.references: list[ImportReference] = []
        self._version_guard_depth = 0
        self._vllm_roots: set[str] = set()
        self._parents: dict[int, ast.AST] = {}

    def visit(self, node: ast.AST) -> Any:
        for child in ast.iter_child_nodes(node):
            self._parents[id(child)] = node
        return super().visit(node)

    @staticmethod
    def _is_version_guard(node: ast.AST) -> bool:
        return any(
            isinstance(item, ast.Call)
            and (
                isinstance(item.func, ast.Name)
                and item.func.id == "vllm_version_is"
                or isinstance(item.func, ast.Attribute)
                and item.func.attr == "vllm_version_is"
            )
            for item in ast.walk(node)
        )

    def visit_If(self, node: ast.If) -> None:
        guarded = self._is_version_guard(node.test)
        self._version_guard_depth += int(guarded)
        for item in (*node.body, *node.orelse):
            self.visit(item)
        self._version_guard_depth -= int(guarded)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._version_guard_depth == 0 and node.module and node.module.startswith("vllm"):
            for alias in node.names:
                if alias.name != "*":
                    self.references.append(ImportReference(node.module, alias.name, self.file_name, node.lineno))

    def visit_Import(self, node: ast.Import) -> None:
        if self._version_guard_depth:
            return
        for alias in node.names:
            if not alias.name.startswith("vllm"):
                continue
            self.references.append(ImportReference(alias.name, None, self.file_name, node.lineno))
            if alias.name == "vllm":
                self._vllm_roots.add(alias.asname or "vllm")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._version_guard_depth:
            return
        parent = self._parents.get(id(node))
        if isinstance(parent, ast.Attribute) and parent.value is node:
            return
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name) and current.id in self._vllm_roots:
            chain = ["vllm", *reversed(parts)]
            self.references.append(ImportReference(".".join(chain), None, self.file_name, node.lineno))
        self.generic_visit(node)


def discover_imports(ascend_root: Path) -> list[ImportReference]:
    references: list[ImportReference] = []
    for path in sorted((ascend_root / "vllm_ascend").rglob("*.py")):
        relative = path.relative_to(ascend_root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError, UnicodeError):
            continue
        visitor = _ImportVisitor(relative)
        visitor.visit(tree)
        references.extend(visitor.references)
    unique = {(item.module, item.symbol, item.file, item.line): item for item in references}
    ordered = sorted(
        unique,
        key=lambda item: (item[0], item[1] or "", item[2], item[3]),
    )
    return [unique[key] for key in ordered]


def _top_level_symbol(snapshot: GitSnapshot, file_name: str, name: str) -> SourceEndpoint:
    endpoint = snapshot.endpoint(file_name, None, name)
    if endpoint.line is not None:
        return endpoint
    tree = snapshot.tree(file_name)
    if tree is not None:
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name)]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.asname or alias.name.rsplit(".", 1)[-1] for alias in node.names]
            if name in names:
                return SourceEndpoint(file=file_name, owner=None, name=name, line=node.lineno)
    return SourceEndpoint(file=None, owner=None, name=name)


def _import_findings(
    ascend_root: Path,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    old_to_new: dict[str, str],
) -> list[RangeFinding]:
    findings: list[RangeFinding] = []
    for reference in discover_imports(ascend_root):
        old_file = old_snapshot.resolve_module(reference.module)
        new_file = new_snapshot.resolve_module(reference.module)
        # For ``import vllm; vllm.a.b.symbol`` resolve the longest module prefix.
        symbol = reference.symbol
        if old_file is None and reference.symbol is None:
            parts = reference.module.split(".")
            for split in range(len(parts) - 1, 0, -1):
                old_file = old_snapshot.resolve_module(".".join(parts[:split]))
                if old_file is not None:
                    symbol = ".".join(parts[split:]) or None
                    new_file = new_snapshot.resolve_module(".".join(parts[:split]))
                    break
        if old_file is None:
            continue
        imported_submodule: str | None = None
        if symbol and "." not in symbol:
            imported_submodule = old_snapshot.resolve_module(f"{reference.module}.{symbol}")
            if imported_submodule is not None:
                old_file = imported_submodule
                new_file = new_snapshot.resolve_module(f"{reference.module}.{symbol}")
        moved_file = old_to_new.get(old_file)
        if imported_submodule is not None:
            old_endpoint = SourceEndpoint(file=old_file, owner=None, name=None)
            new_endpoint = SourceEndpoint(file=new_file, owner=None, name=None)
        elif symbol and "." not in symbol:
            old_endpoint = _top_level_symbol(old_snapshot, old_file, symbol)
            new_endpoint = (
                _top_level_symbol(new_snapshot, new_file, symbol)
                if new_file is not None
                else SourceEndpoint(file=None, owner=None, name=symbol)
            )
        else:
            old_endpoint = SourceEndpoint(file=old_file, owner=None, name=symbol)
            new_endpoint = SourceEndpoint(file=new_file, owner=None, name=symbol)
        # An import can only be attributed to this upgrade when its exact old
        # module or exported symbol was proven to resolve at the old endpoint.
        if old_endpoint.file is None:
            continue
        if new_endpoint.file is not None:
            continue
        relocated = moved_file is not None and moved_file in new_snapshot.files
        if relocated:
            new_endpoint = SourceEndpoint(file=moved_file, owner=None, name=symbol)
        old_state = CompatibilityState(True, True, "import target exists at old")
        new_state = CompatibilityState(False, False, "old import path no longer resolves at new")
        gates = {
            "relationship_verified": True,
            "contract_changed": True,
            "runtime_reachable": True,
            "version_lane_matches": True,
        }
        target = ".".join(value for value in (reference.module, reference.symbol) if value)
        root_upstream = old_endpoint
        if reference.symbol and "." not in reference.symbol:
            resolved_root = old_snapshot.call_endpoint(target, "direct")
            if resolved_root.file is not None:
                root_upstream = resolved_root
        findings.append(
            RangeFinding(
                finding_id=_finding_id("import", reference, old_snapshot.revision, new_snapshot.revision),
                classification="introduced_break",
                relation="direct_import",
                priority="P1",
                action="modify",
                confidence="high",
                upstream_old=old_endpoint,
                upstream_new=new_endpoint,
                downstream=SourceEndpoint(reference.file, None, reference.symbol or reference.module, reference.line),
                old_state=old_state,
                new_state=new_state,
                change=(
                    f"import module moved: {old_file} -> {moved_file}"
                    if relocated
                    else "import module or symbol was removed"
                ),
                evidence=[
                    {
                        "file": reference.file,
                        "line": reference.line,
                        "import": reference.module,
                        "symbol": reference.symbol,
                    }
                ],
                gates=gates,
                suggestion=_suggestion("direct_import", "introduced_break", old_endpoint, new_endpoint),
                source="direct_import_detector",
                contract_kind="symbol_presence",
                direction="downstream_import_to_upstream",
                details={
                    "target": target,
                    "root_upstream": root_upstream.as_dict(),
                },
            )
        )
    return findings


def _direct_call_findings(
    dependencies: Iterable[DirectCallDependency],
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
) -> tuple[list[RangeFinding], list[DirectCallDependency]]:
    """Compare exact downstream call and return-use contracts at both SHAs."""
    findings: list[RangeFinding] = []
    exact_dependencies: list[DirectCallDependency] = []
    for dependency in dependencies:
        endpoint_receiver = dependency.lookup_root or dependency.receiver_type
        old_endpoint = old_snapshot.call_endpoint(
            dependency.target,
            dependency.access_kind,
            receiver_type=endpoint_receiver,
            member=dependency.member,
            invocation_kind=dependency.invocation_kind,
        )
        new_endpoint = new_snapshot.call_endpoint(
            dependency.target,
            dependency.access_kind,
            receiver_type=endpoint_receiver,
            member=dependency.member,
            invocation_kind=dependency.invocation_kind,
        )
        callable_kinds = {"callable", "constructor"}
        exact_dependencies.append(dependency)
        downstream = SourceEndpoint(
            file=dependency.file,
            owner=dependency.owner,
            name=dependency.callee,
            line=dependency.line,
            symbol_kind="callsite",
        )
        parameter_changed = (
            old_endpoint.file != new_endpoint.file
            or old_endpoint.owner != new_endpoint.owner
            or old_endpoint.name != new_endpoint.name
            or old_endpoint.signature != new_endpoint.signature
            or old_endpoint.descriptor != new_endpoint.descriptor
            or old_endpoint.symbol_kind != new_endpoint.symbol_kind
            or old_endpoint.signature_status != new_endpoint.signature_status
            or _ambiguous_binding_changed(old_endpoint, new_endpoint)
        )
        if parameter_changed:
            contract_kind = (
                "call_target_presence"
                if old_endpoint.symbol_kind not in callable_kinds or new_endpoint.symbol_kind not in callable_kinds
                else "call_arguments"
            )
            old_state = _direct_call_state(old_endpoint, dependency)
            new_state = _direct_call_state(new_endpoint, dependency)
            classification = _classify(old_state, new_state, parameter_changed)
            gates = {
                "relationship_verified": True,
                "contract_changed": parameter_changed,
                "runtime_reachable": True,
                "version_lane_matches": True,
            }
            action = _finding_action(classification, gates)
            findings.append(
                RangeFinding(
                    finding_id=_finding_id(
                        "direct_call",
                        contract_kind,
                        dependency.file,
                        dependency.line,
                        dependency.column,
                        dependency.target,
                        old_snapshot.revision,
                        new_snapshot.revision,
                    ),
                    classification=classification,
                    relation="direct_call",
                    priority="P1" if action == "modify" else "P2",
                    action=action,
                    confidence="high" if classification != "analysis_unresolved" else "medium",
                    upstream_old=old_endpoint,
                    upstream_new=new_endpoint,
                    downstream=downstream,
                    old_state=old_state,
                    new_state=new_state,
                    change=_change_text(
                        old_endpoint,
                        new_endpoint,
                        runtime_signature_changed=(old_endpoint.signature_status != new_endpoint.signature_status),
                    ),
                    evidence=[dependency.as_dict()],
                    gates=gates,
                    suggestion=_suggestion("direct_call", classification, old_endpoint, new_endpoint),
                    source="direct_call_detector",
                    contract_kind=contract_kind,
                    direction="downstream_call_to_upstream",
                    details={
                        "target": dependency.target,
                        "access_kind": dependency.access_kind,
                        "receiver_type": dependency.receiver_type,
                        "member": dependency.member,
                        "invocation_kind": dependency.invocation_kind,
                        "lookup_root": dependency.lookup_root,
                        "resolution_basis": dependency.resolution_basis,
                        "call_shape": dependency.call_shape.as_dict(),
                        "scope": dependency.scope,
                    },
                )
            )

        return_changed = (
            old_endpoint.owner != new_endpoint.owner
            or old_endpoint.name != new_endpoint.name
            or old_endpoint.return_contract != new_endpoint.return_contract
        )
        if (
            old_endpoint.symbol_kind not in callable_kinds
            or new_endpoint.symbol_kind not in callable_kinds
            or not dependency.return_use.constrains_return
            or not return_changed
        ):
            continue
        old_state = _return_use_state(old_endpoint, dependency)
        new_state = _return_use_state(new_endpoint, dependency)
        classification = _classify(old_state, new_state, return_changed)
        gates = {
            "relationship_verified": True,
            "contract_changed": return_changed,
            "runtime_reachable": True,
            "version_lane_matches": True,
        }
        action = _finding_action(classification, gates)
        findings.append(
            RangeFinding(
                finding_id=_finding_id(
                    "direct_call",
                    "return_usage",
                    dependency.file,
                    dependency.line,
                    dependency.column,
                    dependency.target,
                    old_snapshot.revision,
                    new_snapshot.revision,
                ),
                classification=classification,
                relation="direct_call",
                priority="P1" if action == "modify" else "P2",
                action=action,
                confidence="high" if classification != "analysis_unresolved" else "medium",
                upstream_old=old_endpoint,
                upstream_new=new_endpoint,
                downstream=downstream,
                old_state=old_state,
                new_state=new_state,
                change=_change_text(old_endpoint, new_endpoint, "return_usage"),
                evidence=[dependency.as_dict()],
                gates=gates,
                suggestion="同步下游对上游返回值的解包、下标或协议使用，并补充该调用点回归测试。",
                source="direct_call_detector",
                contract_kind="return_usage",
                direction="downstream_call_to_upstream",
                details={
                    "target": dependency.target,
                    "return_use": dependency.return_use.as_dict(),
                    "upstream_old_return": old_endpoint.return_contract,
                    "upstream_new_return": new_endpoint.return_contract,
                    "scope": dependency.scope,
                },
            )
        )
    return findings, exact_dependencies


def _verified_historical_direct_calls(
    candidates: Iterable[DirectCallDependency],
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
) -> list[DirectCallDependency]:
    """Promote only old-proven/new-missing self or super call candidates.

    The checked-out downstream tree proves the callsite and a complete new MRO
    proves that the member is absent.  The range still needs old-side evidence
    before this is a dependency: otherwise a dynamic downstream ``self.foo()``
    could be mistaken for a deleted upstream method merely because it shares a
    class with some vLLM base.
    """

    verified: list[DirectCallDependency] = []
    for candidate in candidates:
        if candidate.lookup_root is None or candidate.member is None:
            continue
        old_endpoint = old_snapshot.call_endpoint(
            candidate.target,
            candidate.access_kind,
            receiver_type=candidate.lookup_root,
            member=candidate.member,
            invocation_kind=candidate.invocation_kind,
        )
        new_endpoint = new_snapshot.call_endpoint(
            candidate.target,
            candidate.access_kind,
            receiver_type=candidate.lookup_root,
            member=candidate.member,
            invocation_kind=candidate.invocation_kind,
        )
        if old_endpoint.symbol_kind == "callable" and new_endpoint.symbol_kind == "missing":
            verified.append(candidate)
    return verified


def _verified_historical_override_relations(
    candidates: Iterable[HistoricalOverrideCandidate],
    engine: InterfaceBoundaryGenerator,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    old_to_new: dict[str, str],
) -> list[Relation]:
    """Promote old-proven override targets that are absent from the new MRO."""

    relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (
            candidate.downstream_qualified_owner,
            candidate.downstream_name,
            candidate.lookup_root,
        )
        if key in seen:
            continue
        seen.add(key)
        target = f"{candidate.lookup_root}.{candidate.downstream_name}"
        old_endpoint = old_snapshot.call_endpoint(
            target,
            "instance",
            receiver_type=candidate.lookup_root,
            member=candidate.downstream_name,
        )
        new_endpoint = new_snapshot.call_endpoint(
            target,
            "instance",
            receiver_type=candidate.lookup_root,
            member=candidate.downstream_name,
        )
        if (
            old_endpoint.symbol_kind != "callable"
            or old_endpoint.file is None
            or old_endpoint.owner is None
            or old_endpoint.name is None
            or new_endpoint.symbol_kind != "missing"
        ):
            continue
        downstream_name = f"{candidate.downstream_qualified_owner}.{candidate.downstream_name}"
        downstream_callable = engine.downstream.find_callable(downstream_name)
        if downstream_callable is None:
            continue
        downstream_descriptor = downstream_callable.descriptor_kind
        downstream_contract = engine._signature_contract(
            downstream_callable,
            descriptor_kind=downstream_descriptor,
        )
        old_module, _ = _file_module(old_endpoint.file)
        old_qualified_owner = ".".join(item for item in (old_module, old_endpoint.owner) if item)
        old_qualified_name = f"{old_qualified_owner}.{old_endpoint.name}"
        relations.append(
            Relation(
                relation="override",
                upstream_file=old_to_new.get(old_endpoint.file, old_endpoint.file),
                upstream_owner=old_endpoint.owner,
                upstream_name=old_endpoint.name,
                upstream_signature=old_endpoint.signature,
                downstream_file=candidate.downstream_file,
                downstream_owner=candidate.downstream_owner,
                downstream_name=candidate.downstream_name,
                downstream_signature=downstream_callable.signature,
                evidence_file=candidate.downstream_file,
                evidence_line=candidate.evidence_line,
                evidence=(
                    RelationEvidence(
                        file=candidate.downstream_file,
                        line=candidate.evidence_line,
                        target_expression=old_qualified_name,
                        installed_descriptor_kind=downstream_descriptor,
                    ),
                ),
                upstream_descriptor_kind=old_endpoint.descriptor,
                downstream_descriptor_kind=downstream_descriptor,
                installed_descriptor_kind=downstream_descriptor,
                downstream_property_accessors=downstream_callable.property_accessors,
                installed_property_accessors=downstream_callable.property_accessors,
                downstream_signature_contract=downstream_contract,
                installed_signature_contract=downstream_contract,
                override_paths=((downstream_name, old_qualified_name),),
            )
        )
    return relations


def validate_current_contracts(
    engine: InterfaceBoundaryGenerator,
    relations: Iterable[Relation],
    snapshot: GitSnapshot,
    plan: AnalysisPlan | None = None,
) -> tuple[list[DirectCallDependency], list[dict[str, Any]]]:
    """Validate exact call/return contracts for one checked-out source pair."""

    plan = plan or resolve_analysis_plan()
    discovered_dependencies = DirectCallDetector(engine).discover() if plan.analyze_direct_calls else []
    dependencies: list[DirectCallDependency] = []
    findings: list[dict[str, Any]] = []
    for dependency in discovered_dependencies:
        upstream = snapshot.call_endpoint(
            dependency.target,
            dependency.access_kind,
            receiver_type=dependency.receiver_type,
            member=dependency.member,
            invocation_kind=dependency.invocation_kind,
        )
        dependencies.append(dependency)
        argument_state = _direct_call_state(upstream, dependency)
        if argument_state.compatible is not True:
            findings.append(
                {
                    "relation": "direct_call",
                    "contract_kind": "call_arguments",
                    "status": "risk" if argument_state.compatible is False else "review",
                    "upstream": upstream.as_dict(),
                    "downstream": {
                        "file": dependency.file,
                        "owner": dependency.owner,
                        "name": dependency.callee,
                        "line": dependency.line,
                    },
                    "reason": argument_state.reason,
                    "evidence": dependency.as_dict(),
                }
            )
        if dependency.return_use.constrains_return:
            return_state = _return_use_state(upstream, dependency)
            if return_state.compatible is not True:
                findings.append(
                    {
                        "relation": "direct_call",
                        "contract_kind": "return_usage",
                        "status": "risk" if return_state.compatible is False else "review",
                        "upstream": upstream.as_dict(),
                        "downstream": {
                            "file": dependency.file,
                            "owner": dependency.owner,
                            "name": dependency.callee,
                            "line": dependency.line,
                        },
                        "reason": return_state.reason,
                        "evidence": dependency.as_dict(),
                    }
                )

    for relation in relations:
        if (
            relation.upstream_package != "vllm"
            or relation.relation not in {"monkey_patch", "override"}
            or relation.relation not in plan.relation_types
        ):
            continue
        upstream = snapshot.endpoint(
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
        )
        downstream = _relation_downstream_endpoint(relation, engine)
        upstream_contract = return_contract_from_dict(upstream.return_contract)
        downstream_contract = return_contract_from_dict(downstream.return_contract)
        if upstream_contract is None or downstream_contract is None or upstream_contract.status == "bottom":
            continue
        state = _replacement_return_state(upstream, downstream)
        if state.compatible is True:
            continue
        findings.append(
            {
                "relation": relation.relation,
                "contract_kind": "replacement_return",
                "status": "risk" if state.compatible is False else "review",
                "upstream": upstream.as_dict(),
                "downstream": downstream.as_dict(),
                "reason": state.reason,
                "evidence": [item.as_dict() for item in relation.evidence]
                or [{"file": relation.evidence_file, "line": relation.evidence_line}],
            }
        )

    ordered = sorted(
        findings,
        key=lambda item: (
            item["status"],
            item["relation"],
            item["contract_kind"],
            item["downstream"]["file"] or "",
            item["downstream"]["line"] or 0,
        ),
    )
    return dependencies, ordered


def analyze_range(
    *,
    vllm_root: Path,
    ascend_root: Path,
    old: str,
    new: str,
    expect_ascend_sha: str,
    external_roots: dict[str, Path] | None = None,
    external_shas: dict[str, str] | None = None,
    profile: str = "exact-contracts",
    scenario: str = MAIN2MAIN_SCENARIO,
) -> dict[str, Any]:
    """Run the selected source-analysis plan for an exact vLLM range."""
    analysis_started = time.perf_counter()
    phase_started = time.perf_counter()
    timings: dict[str, float | None] = {}
    plan = resolve_analysis_plan(scenario)
    if profile not in {"exact-contracts", "expanded"}:
        raise ValueError(f"unsupported profile: {profile}")
    if scenario != MAIN2MAIN_SCENARIO and profile != "exact-contracts":
        raise ValueError("vllm-interface scenario supports only the exact-contracts profile")
    old_sha, new_sha = verify_range(vllm_root, old, new)
    verify_head("vLLM new", vllm_root, new_sha)
    ascend_sha = verify_head("vllm-ascend", ascend_root, expect_ascend_sha)
    external_roots = external_roots or {}
    external_shas = external_shas or {}
    if set(external_roots) != set(external_shas):
        raise ValueError("external roots and SHAs must name the same packages")
    for package, root in external_roots.items():
        verify_head(f"external {package}", root, external_shas[package])
    phase_started = _diagnostic_timing("input_verification", phase_started, timings)

    generator = InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        external_roots,
        source_versions={"vllm": new_sha, "vllm_ascend": ascend_sha, **external_shas},
    )
    phase_started = _diagnostic_timing("repository_indexing", phase_started, timings)
    relations, generator_findings = generator.generate(plan)
    timings.update({f"relation_generation.{name}": duration for name, duration in generator.phase_timings.items()})
    phase_started = time.perf_counter()
    old_snapshot = GitSnapshot(vllm_root, old_sha)
    new_snapshot = GitSnapshot(vllm_root, new_sha)
    old_to_new, new_to_old = _rename_maps(vllm_root, old_sha, new_sha)
    changed_upstream_files = _changed_python_files(vllm_root, old_sha, new_sha)
    registered_overrides = _registered_oot_overrides(generator)
    relations.extend(
        _verified_historical_override_relations(
            generator.historical_override_candidates,
            generator,
            old_snapshot,
            new_snapshot,
            old_to_new,
        )
    )

    findings = [
        finding
        for relation in relations
        if relation.upstream_package == "vllm" and relation.relation in plan.relation_types
        for finding in _relation_findings(
            relation,
            generator,
            old_snapshot,
            new_snapshot,
            new_to_old,
            changed_upstream_files,
            registered_overrides,
        )
    ]
    phase_started = _diagnostic_timing("relation_comparison", phase_started, timings)
    if plan.analyze_direct_imports:
        import_started = time.perf_counter()
        findings.extend(_import_findings(ascend_root, old_snapshot, new_snapshot, old_to_new))
        _diagnostic_timing("direct_import_analysis", import_started, timings)
    else:
        timings["direct_import_analysis"] = None

    direct_call_dependencies: list[DirectCallDependency] = []
    if plan.analyze_direct_calls:
        direct_started = time.perf_counter()
        direct_call_detector = DirectCallDetector(generator)
        discovered_direct_calls = direct_call_detector.discover()
        discovered_direct_calls.extend(
            _verified_historical_direct_calls(
                direct_call_detector.historical_candidates,
                old_snapshot,
                new_snapshot,
            )
        )
        comparison_started = _diagnostic_timing(
            "direct_call_discovery",
            direct_started,
            timings,
        )
        direct_call_findings, direct_call_dependencies = _direct_call_findings(
            discovered_direct_calls,
            old_snapshot,
            new_snapshot,
        )
        findings.extend(direct_call_findings)
        _diagnostic_timing(
            "direct_call_comparison",
            comparison_started,
            timings,
        )
    else:
        timings["direct_call_discovery"] = None
        timings["direct_call_comparison"] = None

    generator_finding_started = time.perf_counter()
    for candidate in generator_findings if plan.include_generator_findings else []:
        if candidate.status not in {"review", "risk"}:
            continue
        old_endpoint = old_snapshot.expression_endpoint(candidate.target_expression)
        new_endpoint = new_snapshot.expression_endpoint(candidate.target_expression)
        old_endpoint = old_endpoint or SourceEndpoint(None, None, candidate.target_expression)
        new_endpoint = new_endpoint or SourceEndpoint(None, None, candidate.target_expression)
        old_exists = old_endpoint.file is not None and (old_endpoint.name is None or old_endpoint.line is not None)
        new_exists = new_endpoint.file is not None and (new_endpoint.name is None or new_endpoint.line is not None)
        verified_removal = (
            old_exists and not new_exists and candidate.status == "risk" and not candidate.generator_issue
        )
        classification = "introduced_break" if verified_removal else "analysis_unresolved"
        old_state = CompatibilityState(
            old_exists,
            True if verified_removal else None,
            "upstream target exists at old" if verified_removal else candidate.reason,
        )
        new_state = CompatibilityState(
            new_exists,
            False if verified_removal else None,
            "upstream target was removed at new" if verified_removal else candidate.reason,
        )
        gates = {
            "relationship_verified": not candidate.generator_issue,
            "contract_changed": verified_removal,
            "runtime_reachable": verified_removal,
            "version_lane_matches": True,
        }
        findings.append(
            RangeFinding(
                finding_id=_finding_id(candidate.as_dict(), old_sha, new_sha),
                classification=classification,
                relation=candidate.relation,
                priority=(
                    "P0"
                    if verified_removal and candidate.relation == "monkey_patch"
                    else "P1"
                    if verified_removal
                    else "P2"
                ),
                action="modify" if verified_removal else "review",
                confidence=("high" if verified_removal else "low" if candidate.generator_issue else "medium"),
                upstream_old=old_endpoint,
                upstream_new=new_endpoint,
                downstream=SourceEndpoint(
                    candidate.downstream_file,
                    candidate.downstream_owner,
                    candidate.downstream_name,
                    candidate.evidence_line,
                ),
                old_state=old_state,
                new_state=new_state,
                change=(
                    "upstream target existed at old and was removed at new" if verified_removal else candidate.reason
                ),
                evidence=[candidate.as_dict()["evidence"]],
                gates=gates,
                suggestion=(
                    _suggestion(
                        candidate.relation,
                        classification,
                        old_endpoint,
                        new_endpoint,
                    )
                    if verified_removal
                    else "静态证据不足，先人工确认目标绑定或补充分析规则，不要直接修改下游代码。"
                ),
                source="generator_finding",
                contract_kind="target_presence" if verified_removal else "analysis_evidence",
                direction="upstream_contract_to_downstream_implementation",
            )
        )

    if plan.include_generator_findings:
        _diagnostic_timing(
            "generator_finding_conversion",
            generator_finding_started,
            timings,
        )
    else:
        timings["generator_finding_conversion"] = None

    deduplicated = {item.finding_id: item for item in findings}
    ordered = sorted(
        deduplicated.values(),
        key=lambda item: (
            CLASSIFICATIONS.index(item.classification),
            item.relation,
            item.downstream.file or "",
            item.downstream.line or 0,
            item.finding_id,
        ),
    )
    counts = Counter(item.classification for item in ordered)
    action_counts = Counter(item.action for item in ordered)
    relation_counts = Counter(item.relation for item in ordered)
    contract_counts = Counter(item.contract_kind for item in ordered)
    analyzed_relation_count = sum(
        relation.upstream_package == "vllm" and relation.relation in plan.relation_types for relation in relations
    )
    timings["total"] = round(time.perf_counter() - analysis_started, 6)
    return {
        "schema_version": RANGE_SCHEMA_VERSION,
        "metadata": {
            "range_analyzer_version": RANGE_ANALYZER_VERSION,
            "generator_version": GENERATOR_VERSION,
            "profile": profile,
            "scenario": plan.scenario,
            "analysis_plan": plan.as_dict(),
            "vllm_old_sha": old_sha,
            "vllm_new_sha": new_sha,
            "vllm_ascend_sha": ascend_sha,
            "external_sources": dict(sorted(external_shas.items())),
            "timings_seconds": timings,
        },
        "summary": {
            "relations": analyzed_relation_count,
            "relations_collected": len(relations),
            "direct_call_dependencies": len(direct_call_dependencies),
            "generator_findings": (len(generator_findings) if plan.include_generator_findings else 0),
            "total": len(ordered),
            "by_relation": dict(sorted(relation_counts.items())),
            "by_contract": dict(sorted(contract_counts.items())),
            "actionable_introduced_break": sum(
                item.classification == "introduced_break" and item.action == "modify" for item in ordered
            ),
            "by_action": dict(sorted(action_counts.items())),
            **{name: counts[name] for name in CLASSIFICATIONS},
        },
        "findings": [item.as_dict() for item in ordered],
    }


def _csv_rows(findings: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for item in findings:
        old = item["upstream"]["old"]
        new = item["upstream"]["new"]
        downstream = item["downstream"]
        yield {
            "classification": item["classification"],
            "priority": item["priority"],
            "action": item["action"],
            "relation": item["relation"],
            "contract_kind": item.get("contract_kind", ""),
            "direction": item.get("direction", ""),
            "upstream_old": ":".join(str(value or "") for value in (old["file"], old["owner"], old["name"])),
            "upstream_new": ":".join(str(value or "") for value in (new["file"], new["owner"], new["name"])),
            "downstream": ":".join(
                str(value or "") for value in (downstream["file"], downstream["owner"], downstream["name"])
            ),
            "downstream_line": downstream["line"],
            "change": item["change"],
            "old_compatible": item["compatibility"]["old"]["compatible"],
            "new_compatible": item["compatibility"]["new"]["compatible"],
            "confidence": item["confidence"],
            "suggestion": item["suggestion"],
            "call_shape": json.dumps(item.get("details", {}).get("call_shape"), ensure_ascii=False),
            "return_use": json.dumps(item.get("details", {}).get("return_use"), ensure_ascii=False),
            "override_paths": json.dumps(item.get("details", {}).get("override_paths"), ensure_ascii=False),
            "upstream_old_return": json.dumps(old.get("return_contract"), ensure_ascii=False),
            "upstream_new_return": json.dumps(new.get("return_contract"), ensure_ascii=False),
            "downstream_return": json.dumps(downstream.get("return_contract"), ensure_ascii=False),
        }


CSV_FIELDS = [
    "classification",
    "priority",
    "action",
    "relation",
    "contract_kind",
    "direction",
    "upstream_old",
    "upstream_new",
    "downstream",
    "downstream_line",
    "change",
    "old_compatible",
    "new_compatible",
    "confidence",
    "suggestion",
    "call_shape",
    "return_use",
    "override_paths",
    "upstream_old_return",
    "upstream_new_return",
    "downstream_return",
]


def _write_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    rows = list(_csv_rows(findings))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: dict[str, Any]) -> str:
    meta = report["metadata"]
    summary = report["summary"]
    lines = [
        "# vLLM main2main 接口兼容报告",
        "",
        f"- vLLM 区间：`{meta['vllm_old_sha']}` → `{meta['vllm_new_sha']}`",
        f"- vllm-ascend 基线：`{meta['vllm_ascend_sha']}`",
        f"- 本次升级需修改：{summary['actionable_introduced_break']}",
        f"- 严格契约不兼容（含 review）：{summary['introduced_break']}",
        f"- 兼容性提醒：{summary['compatibility_warning']}",
        f"- 历史问题：{summary['preexisting']}",
        f"- 无法静态确认：{summary['analysis_unresolved']}",
        "",
        "## 本次升级需要处理",
        "",
    ]
    introduced = [
        item
        for item in report["findings"]
        if item["classification"] == "introduced_break" and item["action"] == "modify"
    ]
    if not introduced:
        lines.append("没有发现能够确认由本次区间引入的接口 break。")
    for item in introduced:
        downstream = item["downstream"]
        contract_labels = {
            "call_arguments": "调用参数",
            "call_target_presence": "调用目标",
            "return_usage": "返回值消费",
            "replacement_return": "替代实现返回协议",
            "symbol_presence": "导入符号",
            "target_presence": "上游目标",
            "base_presence": "继承目标",
        }
        contract_label = contract_labels.get(item.get("contract_kind"), item.get("contract_kind", "接口契约"))
        lines.extend(
            [
                f"### {item['relation']} / {contract_label}：{downstream['file']}:{downstream['line'] or ''}",
                "",
                f"- 变化：{item['change']}",
                f"- 下游接口：`{downstream['owner'] or ''}.{downstream['name'] or ''}`",
                f"- 建议：{item['suggestion']}",
                "",
            ]
        )
    reviews = [
        item
        for item in report["findings"]
        if item["action"] == "review"
        and (
            item.get("details", {}).get("optional_contract_only")
            or item.get("details", {}).get("new_delta_on_preexisting_break")
        )
    ]
    lines.extend(["## 本次升级需人工审视", ""])
    if not reviews:
        lines.append("没有发现可证明的可选契约或历史不兼容遮蔽增量。")
    for item in reviews:
        details = item.get("details", {})
        downstream = item["downstream"]
        if details.get("optional_contract_only"):
            optional_parameters = details.get("new_optional_parameters") or []
            parameter_names = "、".join(f"`{name}`" for name in optional_parameters)
            parameter_reference = "该参数" if len(optional_parameters) == 1 else "这些参数"
            reason = (
                f"下游重写的方法未接收上游新增的可选参数 {parameter_names}，"
                f"但目前没有证据表明运行时会把{parameter_reference}传给该下游实现"
            )
        else:
            reason = "下游在 old 已不兼容，但本区间又新增了精确参数差异"
        lines.extend(
            [
                f"### {item['relation']} / review：{downstream['file']}:{downstream['line'] or ''}",
                "",
                f"- 原因：{reason}",
                f"- 建议：{item['suggestion']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 说明",
            "",
            "`preexisting` 表示 old 和 new 都不兼容，不归因于这次升级；"
            "`analysis_unresolved` 表示证据不足，脚本没有猜测。",
            "",
        ]
    )
    return "\n".join(lines)


def _upstream_pr_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in report["findings"]
        if item["classification"] == "introduced_break"
        and item["action"] == "modify"
        and item["relation"] in {"override", "direct_call", "direct_import"}
    ]


def _upstream_pr_review_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in report["findings"]
        if item["action"] == "review"
        and item["relation"] in {"override", "direct_call", "direct_import"}
        and (
            item.get("details", {}).get("optional_contract_only")
            or item.get("details", {}).get("new_delta_on_preexisting_break")
        )
    ]


def _root_cause_key(item: dict[str, Any]) -> tuple[object, ...]:
    details = item.get("details", {})
    upstream = details.get("root_upstream") or item["upstream"]["old"]
    fingerprint = upstream.get("analysis_fingerprint")
    if fingerprint:
        return ("upstream_fingerprint", fingerprint, upstream.get("name"))
    target = details.get("target")
    if target:
        return ("upstream_target", target, item.get("change"))
    return (
        "upstream_endpoint",
        upstream.get("file"),
        upstream.get("owner"),
        upstream.get("name"),
        item.get("change"),
    )


def _upstream_pr_payload(report: dict[str, Any]) -> dict[str, Any]:
    findings = _upstream_pr_findings(report)
    review_findings = _upstream_pr_review_findings(report)
    relation_counts = Counter(item["relation"] for item in findings)
    contract_counts = Counter(item.get("contract_kind", "") for item in findings)
    review_reason_counts = Counter(item.get("details", {}).get("actionability_reason", "") for item in review_findings)
    return {
        "schema_version": report["schema_version"],
        "metadata": report["metadata"],
        "summary": {
            "introduced_breaks": len(findings),
            "root_causes": len({_root_cause_key(item) for item in findings}),
            "review_findings": len(review_findings),
            "review_root_causes": len({_root_cause_key(item) for item in review_findings}),
            "by_relation": dict(sorted(relation_counts.items())),
            "by_contract": dict(sorted(contract_counts.items())),
            "review_by_reason": dict(sorted(review_reason_counts.items())),
        },
        "findings": findings,
        "review_findings": review_findings,
    }


def _upstream_pr_finding_lines(
    item: dict[str, Any],
    index: int,
    *,
    review: bool,
) -> list[str]:
    details = item.get("details", {})
    upstream = details.get("root_upstream") or item["upstream"]["new"]
    if not upstream.get("file"):
        upstream = item["upstream"]["old"]
    downstream = item["downstream"]
    override_paths = details.get("override_paths") or []
    path_lines = [f"- Override path: `{' -> '.join(path)}`" for path in override_paths if len(path) > 2]
    call_lines = [
        "- Upstream call evidence: "
        f"`{evidence['file']}:{evidence['line']}` passes "
        f"`{', '.join(evidence['matched_parameters'])}`"
        for evidence in details.get("upstream_call_evidence", [])
    ]
    upstream_name = ".".join(value for value in (upstream.get("owner"), upstream.get("name")) if value)
    reason_lines: list[str] = []
    if review:
        if details.get("optional_contract_only"):
            optional_parameters = details.get("new_optional_parameters") or []
            parameter_names = "、".join(f"`{name}`" for name in optional_parameters)
            parameter_reference = "该参数" if len(optional_parameters) == 1 else "这些参数"
            reason = (
                f"下游重写的方法未接收上游新增的可选参数 {parameter_names}，"
                f"但目前没有证据表明运行时会把{parameter_reference}传给该下游实现。"
            )
        else:
            reason = "new exact delta masked by a preexisting incompatibility"
        reason_lines.append(f"- Review reason: {reason}")
    return [
        f"### {index}. {item['priority']} {item['relation']} / {item.get('contract_kind', '')}",
        "",
        f"- Upstream: `{upstream.get('file') or ''}:{upstream_name}`",
        f"- Downstream: `{downstream.get('file') or ''}:{downstream.get('line') or ''}`",
        *path_lines,
        *call_lines,
        *reason_lines,
        f"- Change: {item['change']}",
        f"- Suggested action: {item['suggestion']}",
        "",
    ]


def _upstream_pr_markdown(payload: dict[str, Any]) -> str:
    meta = payload["metadata"]
    summary = payload["summary"]
    findings = payload["findings"]
    review_findings = payload["review_findings"]
    result = "BREAKS FOUND" if findings else "REVIEW" if review_findings else "PASS"
    lines = [
        "# vLLM Interface Compatibility",
        "",
        f"**Result: {result}**",
        "",
        f"- vLLM range: `{meta['vllm_old_sha']}` -> `{meta['vllm_new_sha']}`",
        f"- vllm-ascend baseline: `{meta['vllm_ascend_sha']}`",
        "- Scope: downstream imports, overrides, and direct upstream-call contracts",
        "- Monkey patches, inheritance-only findings, generator reviews, "
        "and historical incompatibilities are intentionally excluded.",
        "- Exact new deltas masked by historical incompatibilities are retained as review input.",
        f"- Introduced breaks: {summary['introduced_breaks']}",
        f"- Root causes: {summary['root_causes']}",
        f"- Review findings: {summary['review_findings']}",
        f"- Review root causes: {summary['review_root_causes']}",
        "",
        "## Introduced breaks",
        "",
    ]
    if not findings:
        lines.append("No new downstream interface break was introduced by this range.")
    for index, item in enumerate(findings, start=1):
        lines.extend(_upstream_pr_finding_lines(item, index, review=False))
    lines.extend(["## Review findings", ""])
    if not review_findings:
        lines.append("No exact optional-contract or masked-delta review was found.")
    for index, item in enumerate(review_findings, start=1):
        lines.extend(_upstream_pr_finding_lines(item, index, review=True))
    return "\n".join(lines)


def _write_upstream_pr_reports(
    report: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    payload = _upstream_pr_payload(report)
    json_path = output_dir / "vllm-interface-pr-report.json"
    introduced_csv = output_dir / "vllm-interface-introduced-breaks.csv"
    markdown_path = output_dir / "vllm-interface-pr-summary.md"
    metadata_path = output_dir / "vllm-interface-analysis-metadata.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(introduced_csv, payload["findings"])
    markdown_path.write_text(_upstream_pr_markdown(payload), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "metadata": report["metadata"],
                "introduced_breaks": payload["summary"]["introduced_breaks"],
                "root_causes": payload["summary"]["root_causes"],
                "review_findings": payload["summary"]["review_findings"],
                "review_root_causes": payload["summary"]["review_root_causes"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "introduced_csv": str(introduced_csv),
        "markdown": str(markdown_path),
        "metadata_json": str(metadata_path),
    }


def write_reports(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if report.get("metadata", {}).get("scenario") == "vllm-interface":
        return _write_upstream_pr_reports(report, output_dir)
    json_path = output_dir / "main2main-range-report.json"
    all_csv = output_dir / "main2main-all-findings.csv"
    introduced_csv = output_dir / "main2main-introduced-breaks.csv"
    markdown_path = output_dir / "main2main-range-report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(all_csv, report["findings"])
    _write_csv(
        introduced_csv,
        [
            item
            for item in report["findings"]
            if item["classification"] == "introduced_break" and item["action"] == "modify"
        ],
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "all_csv": str(all_csv),
        "introduced_csv": str(introduced_csv),
        "markdown": str(markdown_path),
    }
