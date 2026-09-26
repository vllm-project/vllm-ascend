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
import posixpath
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, cast

from .analysis_plans import (
    MAIN2MAIN_SCENARIO,
    AnalysisPlan,
    resolve_analysis_plan,
)
from .annotation_names import AnnotationNamespace
from .cache import CacheResult, PersistentCache, build_identity, git_source_state, normalized_repo_path
from .call_contracts import (
    MAX_FACTORY_RETURN_DEPTH,
    CallShape,
    DirectAttributeDependency,
    DirectAttributeDetector,
    DirectCallDependency,
    DirectCallDetector,
    ReturnContract,
    ReturnShape,
    _attribute_is_read,
    _DirectDependencyResolver,
    _parents,
    _under_attribute_fallback,
    bind_call_shape,
    call_shape,
    infer_return_contract,
    replacement_return_compatible,
    return_contract_from_dict,
    return_use_compatible,
)
from .condition_contracts import conditional_state_evidence
from .construction_effects import construction_effects, has_construction_effect, has_storage_effect
from .dataclass_contracts import ClassSource, dataclass_layout, dataclass_replacement, dataclass_storage
from .factory_returns import factory_body_supported, proven_factory_return
from .generator import (
    _KNOWN_TRANSPARENT_SIGNATURE_DECORATORS,
    _KNOWN_WRAPS_SIGNATURE_DECORATORS,
    _TRITON_HEURISTICS_DECORATOR,
    _TRITON_JIT_DECORATOR,
    _TRITON_KERNEL_PROTOCOL,
    GENERATOR_VERSION,
    STDLIB_STRUCTURAL_BASES,
    HistoricalOverrideCandidate,
    InterfaceBoundaryGenerator,
    Relation,
    RelationEvidence,
    RepositoryIndex,
    SignatureContract,
    _accepts_signature_contract,
    _expression_name,
    _function_local_names,
    _function_scope_nodes,
    _import_binding_reference,
    _inspect_signature,
    _jsonable_signature,
    _scope_final_bindings,
    _ScopeBinding,
    _statements_must_terminate,
    _tag_guard_names,
)
from .method_context import FactoryCallProof, MethodContextLookup, MethodProof, method_body_hash
from .models import (
    CompatibilityState,
    RangeFinding,
    SourceEndpoint,
)
from .module_attributes import ModuleGetattrContract, module_getattr_contract, runtime_module_body
from .patch_consumers import patch_consumer_findings
from .source_facts import SourceFacts
from .type_flow import (
    TYPE_BUILTINS,
    ContainerFlow,
    FlowValue,
    HelperCallEffects,
    Lookup,
    TypeShape,
    method_path_proofs,
    resolve_flow_path,
)

RANGE_SCHEMA_VERSION = 48
RANGE_ANALYZER_VERSION = "2.34.0"
SNAPSHOT_CACHE_SCHEMA_VERSION = 16
_GIT_BATCH_CLOSE_TIMEOUT_SECONDS = 5
RELATION_CACHE_SCHEMA_VERSION = 2
DIRECT_IMPORT_CACHE_SCHEMA_VERSION = 3
DIRECT_CALL_CACHE_SCHEMA_VERSION = 38
DIRECT_ATTRIBUTE_CACHE_SCHEMA_VERSION = 37
CLASSIFICATIONS = (
    "introduced_break",
    "compatibility_warning",
    "preexisting",
    "fixed",
    "analysis_unresolved",
)


@dataclass(frozen=True)
class InheritedStateDependency:
    """A new upstream inherited read that depends on downstream instance state."""

    downstream_class: str
    downstream_file: str
    upstream_root: str
    inherited_member: str
    required_attribute: str
    read_line: int
    read_condition: str
    constructor_owner: str
    constructor_file: str
    constructor_line: int
    initialization_status: str
    initialization_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "downstream_class": self.downstream_class,
            "downstream_file": self.downstream_file,
            "upstream_root": self.upstream_root,
            "inherited_member": self.inherited_member,
            "required_attribute": self.required_attribute,
            "read_line": self.read_line,
            "read_condition": self.read_condition,
            "constructor_owner": self.constructor_owner,
            "constructor_file": self.constructor_file,
            "constructor_line": self.constructor_line,
            "initialization_status": self.initialization_status,
            "initialization_reason": self.initialization_reason,
        }


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


def _record_diagnostic_timing(
    label: str,
    elapsed: float,
    timings: dict[str, float | None],
) -> None:
    rounded = round(elapsed, 6)
    timings[label] = rounded
    if os.environ.get("VLLM_INTERFACE_TIMINGS") == "1":
        print(f"[vllm-interface] {label}: {rounded:.3f}s", file=sys.stderr, flush=True)


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
        detail = error.stderr or b""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise ValueError(
            f"Cannot resolve Git commit {revision} in {repo}: "
            f"git exited with status {error.returncode}. {detail.strip()}"
        ) from error


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
    } | _KNOWN_TRANSPARENT_SIGNATURE_DECORATORS
    builtin_descriptors = {"builtins.classmethod", "builtins.property", "builtins.staticmethod"}
    for item in node.decorator_list:
        raw = _decorator_name(item)
        resolved = resolver(raw) if resolver is not None and raw else raw
        if (
            resolved in builtin_descriptors
            or (resolver is None and raw in {"classmethod", "property", "staticmethod"})
            or resolved in known
            or (resolved in _KNOWN_WRAPS_SIGNATURE_DECORATORS and not isinstance(item, ast.Call))
        ):
            continue
        return "unknown"
    return "exact"


def _snapshot_node_signature_contract(
    node: ast.AST | None,
    resolver: Any | None,
    invocation_kind: str,
    *,
    descriptor: str | None,
    binds_receiver: bool,
    access_kind: str | None = None,
) -> SignatureContract | None:
    """Derive the callable contract used by one historical snapshot.

    Relation discovery already models Triton JIT kernels by their public
    ``kernel[grid](...)`` launch protocol. Historical range comparison must
    apply the same decorator transform instead of treating ``@triton.jit`` as
    an unknown ordinary Python wrapper.
    """

    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return None
    definition_signature = _jsonable_signature(node)
    runtime_signature = definition_signature
    reported_signature = definition_signature
    protocol = "property_access" if descriptor == "property" else "python_call"
    status = _signature_status(node, resolver)
    provenance = ["git_snapshot"]

    if invocation_kind == _TRITON_KERNEL_PROTOCOL:
        status = "exact"
        protocol = _TRITON_KERNEL_PROTOCOL
        reported_signature = None
        saw_jit = False
        for decorator in reversed(node.decorator_list):
            raw = _decorator_name(decorator)
            reference = resolver(raw) if resolver is not None and raw else raw or None
            label = reference or raw or "<dynamic-decorator>"
            if reference == _TRITON_JIT_DECORATOR:
                if saw_jit:
                    status = "unknown"
                    runtime_signature = None
                    provenance.append(f"{label}:duplicate_kernel_launch")
                    break
                saw_jit = True
                runtime_signature = definition_signature
                provenance.append(f"{label}:kernel_launch")
                continue
            if reference == _TRITON_HEURISTICS_DECORATOR:
                generated_names = InterfaceBoundaryGenerator._triton_heuristic_names(decorator)
                transformed_signature = (
                    InterfaceBoundaryGenerator._triton_heuristics_signature(
                        runtime_signature,
                        generated_names,
                    )
                    if saw_jit and generated_names is not None
                    else None
                )
                if transformed_signature is None:
                    status = "unknown"
                    runtime_signature = None
                    provenance.append(f"{label}:unresolved_kernel_heuristics")
                    break
                runtime_signature = transformed_signature
                provenance.append(f"{label}:generated={','.join(generated_names or ())}")
                continue
            status = "unknown"
            runtime_signature = None
            provenance.append(label)
            break
        if not saw_jit:
            status = "unknown"
            runtime_signature = None
            provenance.append("missing_triton_jit")

    bound_signature = (
        _bound_signature(
            runtime_signature,
            descriptor=descriptor,
            access_kind=access_kind or ("instance" if binds_receiver else "module"),
        )
        if status == "exact"
        else None
    )
    if status == "exact" and bound_signature is None:
        status = "unknown"
        provenance.append("unknown_descriptor_binding")
    return SignatureContract(
        definition_signature=definition_signature,
        runtime_entry_signature=runtime_signature,
        reported_signature=reported_signature,
        bound_call_signature=bound_signature,
        protocol=protocol,
        status=status or "unknown",
        provenance=tuple(provenance),
    )


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


def _binding_from_alternatives(alternatives: tuple[_ScopeBinding, ...]) -> _NamedBinding:
    """Return one final runtime namespace binding, or fail closed.

    The shared scope-flow interpreter handles overload stubs followed by a
    concrete implementation, conditional definitions, rebinding, and delete.
    A path-dependent final binding is ``unknown`` rather than ``missing``.
    """

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


def _body_named_binding(body: list[ast.stmt], name: str) -> _NamedBinding:
    return _binding_from_alternatives(_scope_final_bindings(body, _tag_guard_names(body)).get(name, ()))


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
    if not isinstance(positional_only, list) or not isinstance(positional_or_keyword, list):
        return None
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
    if old_parameters is None or new_parameters is None or old_signature is None or new_signature is None:
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


def _added_keyword_witness(
    old: SignatureContract | None,
    new: SignatureContract | None,
    replacement: list[object] | None,
) -> dict[str, object] | None:
    """Prove a newly rejected call independently of whole-signature debt.

    Both calls must bind to their respective upstream contracts. The old call
    must also bind to the pinned replacement; adding only new upstream keyword
    parameters must make that same replacement reject it. This is a strict
    interface witness, not an assertion that every old call was compatible.
    """
    if old is None or new is None or old.status != "exact" or new.status != "exact":
        return None
    before, after = old.bound_call_signature, new.bound_call_signature
    delta = _signature_delta(before, after)
    if delta is None or not isinstance(delta["added"], list):
        return None
    added = tuple(
        str(item["name"])
        for item in delta["added"]
        if isinstance(item, dict) and item["kind"] in {"keyword_only", "positional_or_keyword"}
    )
    parameters = _signature_parameters(before)
    if not added or parameters is None:
        return None
    positional = [p for p in parameters if p["kind"] != "keyword_only"]
    # Cover keyword-based adapters with reordered parameters as well as mixed
    # positional calls. The bound-signature oracle rejects every invalid shape.
    for count in range(len(positional) + 1):
        supplied = {str(p["name"]) for p in positional[:count]}
        keywords = tuple(str(p["name"]) for p in parameters if str(p["name"]) not in supplied)
        old_call = CallShape(count, keywords)
        new_call = CallShape(count, (*keywords, *added))
        if (
            bind_call_shape(before, old_call)[0] is True
            and bind_call_shape(replacement, old_call)[0] is True
            and bind_call_shape(after, new_call)[0] is True
            and bind_call_shape(replacement, new_call)[0] is False
        ):
            return {
                "kind": "exact_interface_call_pair",
                "old_call": old_call.as_dict(),
                "new_call": new_call.as_dict(),
                "added_keywords": list(added),
                "rejection": bind_call_shape(replacement, new_call)[1],
            }
    return None


def _local_constructor_delta_witness(
    relation: Relation,
    engine: InterfaceBoundaryGenerator,
    old: SignatureContract | None,
    new: SignatureContract | None,
    replacement: list[object] | None,
) -> dict[str, object] | None:
    """Isolate new alignment debt for adapters with extra construction context.

    For example, a factory passes a downstream-only runner into an adapted
    constructor. A concrete local call proves its baseline binding even when
    that extra required argument prevents whole-signature substitutability.
    This proof deliberately does not claim an upstream runtime dispatch.
    """
    if relation.relation != "override" or relation.downstream_name != "__init__":
        return None
    if old is None or new is None or old.status != "exact" or new.status != "exact":
        return None
    delta = _signature_delta(old.bound_call_signature, new.bound_call_signature)
    if delta is None or not isinstance(delta["added"], list):
        return None
    added = tuple(
        str(item["name"])
        for item in delta["added"]
        if isinstance(item, dict) and item["kind"] in {"keyword_only", "positional_or_keyword"}
    )
    if not added:
        return None
    module, _ = _file_module(relation.downstream_file)
    expected = f"{module}.{relation.downstream_owner}"
    for info in engine.downstream.callables.values():
        if not isinstance(info.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local_names = _function_local_names(info.node)
        for node in _function_scope_nodes(info.node):
            if not isinstance(node, ast.Call):
                continue
            expression = _expression_name(node.func)
            if expression is None or expression.split(".", 1)[0] in local_names:
                continue
            resolved = engine.downstream.canonical_name(engine.downstream.resolve_reference(info.module, expression))
            if resolved != expected:
                continue
            before = call_shape(node)
            if not before.exact or set(added) & set(before.keyword_names):
                continue
            after = replace(before, keyword_names=(*before.keyword_names, *added))
            if bind_call_shape(replacement, before)[0] is True and bind_call_shape(replacement, after)[0] is False:
                return {
                    "kind": "local_constructor_interface_alignment",
                    "file": info.file,
                    "line": node.lineno,
                    "scope": info.qualified_name,
                    "old_call": before.as_dict(),
                    "new_call": after.as_dict(),
                    "added_keywords": list(added),
                    "rejection": bind_call_shape(replacement, after)[1],
                    "runtime_dispatch_proven": False,
                }
    return None


def _relation_symbol_presence(endpoint: SourceEndpoint) -> bool | None:
    """Return proven symbol presence without conflating ambiguity with deletion."""

    if endpoint.file is None or endpoint.symbol_kind == "missing":
        return False
    if endpoint.symbol_kind in {None, "unknown"}:
        return None
    return True


def _snapshot_signature_contract(
    endpoint: SourceEndpoint,
    invocation_kind: str = "python_call",
) -> SignatureContract | None:
    """Build the provable runtime-signature view available from one Git snapshot."""

    if endpoint.symbol_kind not in {"callable", "constructor"}:
        return None
    status = endpoint.signature_status or "unknown"
    runtime_signature = endpoint.signature if status == "exact" else None
    binding_descriptor = "ordinary" if endpoint.descriptor == "property" else endpoint.descriptor
    bound_signature = (
        _bound_signature(
            runtime_signature,
            descriptor=binding_descriptor,
            access_kind="instance"
            if endpoint.owner is not None and endpoint.symbol_kind != "constructor"
            else "module",
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
        protocol=("property_access" if endpoint.descriptor == "property" else invocation_kind),
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
    old_contract: SignatureContract | None,
    new_contract: SignatureContract | None,
) -> bool:
    """Compare runtime contracts when both snapshot definitions are exact."""

    if old_contract is None or new_contract is None:
        return old_contract is not new_contract
    if old_contract.status != new_contract.status:
        return True
    if old_contract.status != "exact" or new_contract.status != "exact":
        return False
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
    _reference_prefixes: tuple[str, ...] = ("vllm.",)

    def __init__(self, root: Path, revision: str):
        self.root = root.resolve()
        self.revision = revision
        self.cache_work_seconds = 0.0
        self._lock = threading.RLock()
        self._batch_lock = threading.RLock()
        self._memo = SourceFacts()
        self._batch_process: subprocess.Popen[bytes] | None = None
        self._batch_stderr: BinaryIO | None = None
        self._batch_resources = ExitStack()
        self._closed = False
        self._files: set[str] | None = None
        self._tree_entries: dict[str, tuple[str, str]] | None = None
        self._source: dict[str, str | None] = {}
        self._trees: dict[str, ast.Module | None] = {}
        self._bindings: dict[str, dict[str, str]] = {}
        self._annotation_namespaces: dict[str, AnnotationNamespace] = {}
        self._construction_effects: dict[str, frozenset[str]] = {}
        self._replacement_protocols: dict[
            tuple[str, tuple[str, ...]], tuple[SourceEndpoint, dict[str, Any] | None]
        ] = {}
        self._attribute_endpoints: dict[tuple[str, str, str | None, str | None], SourceEndpoint] = {}
        self._module_attributes: dict[str, tuple[list[ast.stmt], ModuleGetattrContract | None]] = {}
        self._keyword_call_candidates: dict[
            tuple[tuple[str, ...], str],
            list[tuple[str, ast.Call, str | None, str]],
        ] = {}
        self._keyword_call_resolutions: dict[
            tuple[tuple[str, ...], str, int, int],
            _ResolvedCallBinding | None,
        ] = {}

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        for key in (
            "_lock",
            "_batch_lock",
            "_memo",
            "_batch_process",
            "_batch_stderr",
            "_batch_resources",
            "_closed",
        ):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.root = Path(self.root).resolve()
        if not hasattr(self, "_attribute_endpoints"):
            self._attribute_endpoints = {}
        if not hasattr(self, "_module_attributes"):
            self._module_attributes = {}
        self._lock = threading.RLock()
        self._batch_lock = threading.RLock()
        self._memo = SourceFacts()
        self._batch_process = None
        self._batch_stderr = None
        self._batch_resources = ExitStack()
        self._closed = False

    @property
    def files(self) -> set[str]:
        with self._lock:
            if self._files is not None:
                return self._files

        def build() -> tuple[dict[str, tuple[str, str]], set[str]]:
            started = time.perf_counter()
            output = _git(self.root, "ls-tree", "-r", self.revision)
            entries: dict[str, tuple[str, str]] = {}
            for line in output.splitlines():
                metadata, separator, file_name = line.partition("\t")
                parts = metadata.split()
                if separator and len(parts) == 3:
                    entries[file_name] = (parts[0], parts[2])
            with self._lock:
                self.cache_work_seconds += time.perf_counter() - started
            return entries, set(entries)

        entries, files = self._memo.get("snapshot_files", self.revision, build)
        with self._lock:
            if self._files is None:
                self._tree_entries = entries
                self._files = files
            return self._files

    def _source_at(self, normalized: str, seen: frozenset[str]) -> str | None:
        if normalized in seen or len(seen) >= 16 or normalized not in self.files:
            return None
        assert self._tree_entries is not None
        mode, _blob_sha = self._tree_entries[normalized]
        raw = self._read_blob(normalized)
        decoded = raw.decode("utf-8", errors="replace")
        if mode != "120000":
            return decoded
        target = posixpath.normpath(posixpath.join(posixpath.dirname(normalized), decoded.strip()))
        if posixpath.isabs(target) or target == ".." or target.startswith("../"):
            return None
        return self._source_at(target, seen | {normalized})

    def source(self, file_name: str) -> str | None:
        normalized = file_name.replace("\\", "/")
        with self._lock:
            if normalized in self._source:
                return self._source[normalized]

        def build() -> str | None:
            started = time.perf_counter()
            value = self._source_at(normalized, frozenset())
            with self._lock:
                self.cache_work_seconds += time.perf_counter() - started
            return value

        value = self._memo.get("snapshot_source", normalized, build)
        with self._lock:
            self._source.setdefault(normalized, value)
            return self._source[normalized]

    def __enter__(self) -> GitSnapshot:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Explicit close/ExitStack is the primary owner; retain compatibility
        # with short-lived snapshot clients that predate the lazy Git reader.
        if getattr(self, "_batch_process", None) is not None:
            with suppress(OSError, ValueError):
                self.close()

    def _read_blob(self, file_name: str) -> bytes:
        """Read on demand through one binary Git pipe per analysis snapshot."""
        with self._batch_lock:
            if self._closed:
                raise ValueError("cannot read from a closed Git snapshot")
            # The batch protocol is line-delimited. Keep the original Git path
            # handling for unusual names rather than splitting a request in two.
            if "\n" in file_name or "\r" in file_name:
                return subprocess.run(
                    ["git", "-C", str(self.root), "show", f"{self.revision}:{file_name}"],
                    check=True,
                    capture_output=True,
                ).stdout
            try:
                if self._batch_process is None:
                    # A file avoids a full stderr pipe blocking Git while Python
                    # reads stdout, including partial-clone lazy-fetch diagnostics.
                    self._batch_stderr = cast(
                        BinaryIO,
                        self._batch_resources.enter_context(
                            tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 - owned by the snapshot's ExitStack
                        ),
                    )
                    self._batch_process = subprocess.Popen(
                        ["git", "-C", str(self.root), "cat-file", "--batch"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=self._batch_stderr,
                    )
                process = self._batch_process
                assert process.stdin is not None and process.stdout is not None
                if process.poll() is not None:
                    raise ValueError(f"Git batch process exited with code {process.returncode}")
                process.stdin.write(f"{self.revision}:{file_name}\n".encode())
                process.stdin.flush()
                header = process.stdout.readline().split()
                if len(header) != 3 or header[1] != b"blob":
                    raise ValueError(f"unexpected Git batch header: {header!r}")
                size = int(header[2])
                if size < 0:
                    raise ValueError(f"invalid Git blob size: {size}")
                content = process.stdout.read(size)
                if len(content) != size or process.stdout.read(1) != b"\n":
                    raise ValueError("incomplete Git batch blob response")
                return content
            except (OSError, ValueError) as error:
                diagnostic = ""
                if self._batch_stderr is not None:
                    self._batch_stderr.seek(0)
                    diagnostic = self._batch_stderr.read().decode("utf-8", errors="replace").strip()
                self.close()
                raise ValueError(
                    f"Git batch read failed for {self.revision}:{file_name}: {error}"
                    + (f"\n{diagnostic}" if diagnostic else "")
                ) from error

    def close(self) -> None:
        """Release the lazy Git process on both successful and failed runs."""
        with self._batch_lock:
            self._closed = True
            process = self._batch_process
            try:
                if process is not None:
                    if process.stdin is not None:
                        with suppress(BrokenPipeError):
                            process.stdin.close()
                    try:
                        process.wait(timeout=_GIT_BATCH_CLOSE_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    if process.stdout is not None:
                        process.stdout.close()
            finally:
                self._batch_resources.close()
                self._batch_stderr = None
                self._batch_process = None

    def tree(self, file_name: str) -> ast.Module | None:
        normalized = file_name.replace("\\", "/")
        with self._lock:
            if normalized in self._trees:
                return self._trees[normalized]

        def build() -> ast.Module | None:
            started = time.perf_counter()
            source = self.source(normalized)
            if source is None:
                tree = None
            else:
                try:
                    tree = ast.parse(source, filename=normalized)
                except SyntaxError:
                    tree = None
            with self._lock:
                self.cache_work_seconds += time.perf_counter() - started
            return tree

        tree = self._memo.get("snapshot_tree", normalized, build)
        with self._lock:
            self._trees.setdefault(normalized, tree)
            return self._trees[normalized]

    def resolve_module(self, module: str) -> str | None:
        return next((candidate for candidate in _module_file(module) if candidate in self.files), None)

    def _scope_bindings(self, scope: ast.Module | ast.ClassDef) -> dict[str, tuple[_ScopeBinding, ...]]:
        return self._body_bindings(scope.body)

    def _body_bindings(self, body: list[ast.stmt]) -> dict[str, tuple[_ScopeBinding, ...]]:
        return self._memo.get(
            "scope_bindings", tuple(body), lambda: _scope_final_bindings(body, _tag_guard_names(body))
        )

    def _body_named_binding(self, body: list[ast.stmt], name: str) -> _NamedBinding:
        return self._memo.get(
            "named_binding",
            (tuple(body), name),
            lambda: _binding_from_alternatives(self._body_bindings(body).get(name, ())),
        )

    def named_binding(self, tree: ast.Module, owner: str | None, name: str) -> _NamedBinding:
        if owner is None:
            return self._body_named_binding(tree.body, name)
        node = self._memo.get("owner", (tree, owner), lambda: _owner_node(tree, owner))
        return self._body_named_binding(node.body, name) if node is not None else _NamedBinding(None, "missing")

    def _import_body(self, tree: ast.Module) -> list[ast.stmt]:
        return self._memo.get("runtime_module_body", tree, lambda: runtime_module_body(tree))

    def has_import_symbol(self, file_name: str, name: str) -> bool | None:
        def build() -> bool | None:
            tree = self.tree(file_name)
            if tree is None:
                return False if file_name not in self.files else None
            body = self._import_body(tree)
            final = self._body_bindings(body)
            kinds = {binding.kind for binding in final.get(name, ())}
            if kinds and kinds <= {"function", "class", "value", "alias"}:
                return True
            if not kinds or kinds == {"unbound"}:
                dynamic = self._memo.get(
                    "dynamic_exports",
                    tree,
                    lambda: (
                        any(binding.kind != "unbound" for binding in final.get("__getattr__", ()))
                        or any(
                            isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
                            for statement in body
                            for node in ast.walk(statement)
                        )
                    ),
                )
                return None if dynamic else False
            return None

        return self._memo.get("import_presence", (file_name, name), build)

    def _module_bindings(self, file_name: str) -> dict[str, str]:
        normalized = file_name.replace("\\", "/")
        if normalized in self._bindings:
            return self._bindings[normalized]
        tree = self.tree(normalized)
        module, is_package = _file_module(normalized)
        bindings: dict[str, str] = {}
        pending: dict[str, str] = {}
        if tree is not None:
            final = self._scope_bindings(tree)
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
                    elif self._body_named_binding(tree.body, root).status == "exact":
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
                if not target.startswith(self._reference_prefixes):
                    target = f"{module}.{target}"
                return self._resolve_qualified_node(target, frozenset((*seen, expression)))
            owner = ".".join(suffix[:-1]) or None
            tree = self.tree(file_name)
            if tree is None:
                return _QualifiedBinding(file_name, owner, suffix[-1], None, "unknown")
            binding = self.named_binding(tree, owner, suffix[-1])
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
        *,
        generated_storage: bool = False,
    ) -> _QualifiedBinding | None:
        key = (receiver_type, member, seen, generated_storage)
        return self._memo.get(
            "effective_member",
            key,
            lambda: self._build_effective_member(receiver_type, member, seen, generated_storage=generated_storage),
        )

    def _build_effective_member(
        self,
        receiver_type: str,
        member: str,
        seen: frozenset[str] = frozenset(),
        *,
        generated_storage: bool = False,
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
        if class_node.decorator_list and not (
            generated_storage and dataclass_storage(receiver_type, self._type_source) is not None
        ):
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                None,
                "unknown",
                _node_fingerprint(class_node),
            )
        direct = self._body_named_binding(class_node.body, member)
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
        if base is None or not base.startswith(self._reference_prefixes):
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                None,
                "unknown",
                _node_fingerprint(class_node),
            )
        return self._effective_member(
            base,
            member,
            frozenset((*seen, receiver_type)),
            generated_storage=generated_storage,
        )

    @staticmethod
    def _instance_assignment(
        statement: ast.stmt,
        receiver: str,
        member: str,
    ) -> ast.AST | None:
        targets: Iterable[ast.AST] = ()
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets = (statement.target,)
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == member
                and isinstance(target.value, ast.Name)
                and target.value.id == receiver
            ):
                return statement
        call = statement.value if isinstance(statement, ast.Expr) else None
        if (
            isinstance(call, ast.Call)
            and (_expression_name(call.func) or "").rsplit(".", 1)[-1] == "setattr"
            and len(call.args) >= 3
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == receiver
            and isinstance(call.args[1], ast.Constant)
            and call.args[1].value == member
        ):
            return statement
        return None

    @staticmethod
    def _function_may_assign_instance_member(
        function: ast.AsyncFunctionDef | ast.FunctionDef,
        receiver: str,
        member: str,
        *,
        allow_dynamic_member: bool = True,
    ) -> bool:
        for candidate in _function_scope_nodes(function):
            if (
                isinstance(candidate, ast.Attribute)
                and isinstance(candidate.ctx, (ast.Del, ast.Store))
                and candidate.attr == member
                and isinstance(candidate.value, ast.Name)
                and candidate.value.id == receiver
            ):
                return True
            if (
                allow_dynamic_member
                and isinstance(candidate, ast.Attribute)
                and candidate.attr == "__dict__"
                and isinstance(candidate.value, ast.Name)
                and candidate.value.id == receiver
            ):
                return True
            if (
                isinstance(candidate, ast.Call)
                and (_expression_name(candidate.func) or "").rsplit(".", 1)[-1] == "setattr"
                and candidate.args
                and isinstance(candidate.args[0], ast.Name)
                and candidate.args[0].id == receiver
                and (
                    (
                        len(candidate.args) >= 2
                        and isinstance(candidate.args[1], ast.Constant)
                        and candidate.args[1].value == member
                    )
                    or (
                        allow_dynamic_member
                        and (len(candidate.args) < 2 or not isinstance(candidate.args[1], ast.Constant))
                    )
                )
            ):
                return True
        return False

    @classmethod
    def _statements_definitely_assign_instance_member(
        cls,
        statements: Iterable[ast.stmt],
        receiver: str,
        member: str,
    ) -> bool:
        """Prove a field assignment on every normally completing branch."""

        for statement in statements:
            if cls._instance_assignment(statement, receiver, member) is not None:
                return True
            if not isinstance(statement, ast.If):
                if isinstance(statement, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
                    return False
                continue
            body_assigns = cls._statements_definitely_assign_instance_member(
                statement.body,
                receiver,
                member,
            )
            else_assigns = cls._statements_definitely_assign_instance_member(
                statement.orelse,
                receiver,
                member,
            )
            body_terminates = _statements_must_terminate(statement.body)
            else_terminates = _statements_must_terminate(statement.orelse)
            if (
                (body_assigns or body_terminates)
                and (else_assigns or else_terminates)
                and not (body_terminates and else_terminates)
            ):
                return True
        return False

    @classmethod
    def _statements_definitely_call_super_init(
        cls,
        statements: Iterable[ast.stmt],
    ) -> bool:
        for statement in statements:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "__init__"
                and isinstance(statement.value.func.value, ast.Call)
                and isinstance(statement.value.func.value.func, ast.Name)
                and statement.value.func.value.func.id == "super"
                and not statement.value.func.value.args
                and not statement.value.func.value.keywords
            ):
                return True
            if isinstance(statement, (ast.AsyncWith, ast.With)):
                if cls._statements_definitely_call_super_init(statement.body):
                    return True
                continue
            if isinstance(statement, ast.If):
                body_calls = cls._statements_definitely_call_super_init(statement.body)
                else_calls = cls._statements_definitely_call_super_init(statement.orelse)
                body_terminates = _statements_must_terminate(statement.body)
                else_terminates = _statements_must_terminate(statement.orelse)
                if (
                    (body_calls or body_terminates)
                    and (else_calls or else_terminates)
                    and not (body_terminates and else_terminates)
                ):
                    return True
            if isinstance(statement, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
                return False
        return False

    @classmethod
    def _calls_super_init(cls, function: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
        return cls._statements_definitely_call_super_init(function.body)

    def _slot_binding(self, class_node: ast.ClassDef, member: str) -> tuple[bool | None, ast.AST | None]:
        binding = self._body_named_binding(class_node.body, "__slots__")
        if binding.status == "missing":
            return False, None
        if binding.status != "non_callable" or not isinstance(binding.node, (ast.Assign, ast.AnnAssign)):
            return None, binding.node
        value = binding.node.value
        names: tuple[str, ...]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            names = (value.value,)
        elif isinstance(value, (ast.List, ast.Set, ast.Tuple)) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts
        ):
            names = tuple(
                item.value for item in value.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
        else:
            return None, binding.node
        return member in names, binding.node

    def _supported_dataclass(self, file_name: str, class_node: ast.ClassDef) -> bool:
        if not class_node.decorator_list:
            return False
        resolver = self._return_resolver(file_name)
        references = {resolver(raw) for decorator in class_node.decorator_list if (raw := _decorator_name(decorator))}
        return references == {"dataclasses.dataclass"}

    @staticmethod
    def _annotation_only_field(node: ast.AST | None) -> bool:
        return isinstance(node, ast.AnnAssign) and node.value is None

    def _classvar_or_initvar(self, file_name: str, node: ast.AST | None) -> bool:
        if not isinstance(node, ast.AnnAssign):
            return False
        annotation = node.annotation
        root = annotation.value if isinstance(annotation, ast.Subscript) else annotation
        reference = _expression_name(root)
        resolved = self._return_resolver(file_name)(reference) if reference else None
        return (resolved or "").rsplit(".", 1)[-1] in {"ClassVar", "InitVar"}

    def _effective_instance_field(
        self,
        receiver_type: str,
        member: str,
        seen: frozenset[str] = frozenset(),
    ) -> _QualifiedBinding | None:
        if receiver_type in seen:
            return _QualifiedBinding("", None, member, None, "unknown")
        resolved = self._resolve_qualified_node(receiver_type)
        if resolved is None:
            return None
        if resolved.status != "exact" or not isinstance(resolved.node, ast.ClassDef):
            return _QualifiedBinding(
                resolved.file,
                resolved.owner,
                member,
                None,
                "unknown",
                resolved.fingerprint,
            )
        class_node = resolved.node
        actual_owner = ".".join(item for item in (resolved.owner, class_node.name) if item)
        supported_dataclass = self._supported_dataclass(resolved.file, class_node)
        if (class_node.decorator_list and not supported_dataclass) or class_node.keywords:
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                None,
                "unknown",
                _node_fingerprint(class_node),
            )

        if supported_dataclass:
            dataclass_fields = [
                statement
                for statement in class_node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == member
                and not self._classvar_or_initvar(resolved.file, statement)
            ]
            if len(dataclass_fields) == 1:
                field = dataclass_fields[0]
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    field,
                    "non_callable",
                    _node_fingerprint(field),
                )
            if len(dataclass_fields) > 1:
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    None,
                    "unknown",
                    _node_fingerprint(class_node),
                )

        direct = self._body_named_binding(class_node.body, member)
        if direct.status == "unknown":
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                direct.node,
                "unknown",
                direct.fingerprint,
            )
        if direct.status in {"exact", "non_callable"}:
            annotation_only = self._annotation_only_field(direct.node)
            ignored_dataclass_field = self._classvar_or_initvar(resolved.file, direct.node)
            if not annotation_only or supported_dataclass and not ignored_dataclass_field:
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    direct.node,
                    direct.status,
                    direct.fingerprint,
                )

        slot_present, slot_node = self._slot_binding(class_node, member)
        if slot_present is True:
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                slot_node,
                "non_callable",
                _node_fingerprint(slot_node) if slot_node is not None else None,
            )
        if slot_present is None:
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                slot_node,
                "unknown",
                _node_fingerprint(slot_node) if slot_node is not None else None,
            )

        for dynamic_name in ("__getattr__", "__getattribute__"):
            if self._body_named_binding(class_node.body, dynamic_name).status != "missing":
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    None,
                    "unknown",
                    _node_fingerprint(class_node),
                )

        initializer = self._body_named_binding(class_node.body, "__init__")
        recurse_to_base = initializer.status == "missing"
        if initializer.status == "unknown" or initializer.status == "non_callable":
            return _QualifiedBinding(
                resolved.file,
                actual_owner,
                member,
                initializer.node,
                "unknown",
                initializer.fingerprint,
            )
        if initializer.status == "exact":
            if not isinstance(initializer.node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    initializer.node,
                    "unknown",
                    initializer.fingerprint,
                )
            positional = [*initializer.node.args.posonlyargs, *initializer.node.args.args]
            receiver = positional[0].arg if positional else None
            if receiver is None:
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    initializer.node,
                    "unknown",
                    initializer.fingerprint,
                )
            assignment = next(
                (
                    candidate
                    for statement in initializer.node.body
                    if (candidate := self._instance_assignment(statement, receiver, member)) is not None
                ),
                None,
            )
            if assignment is not None:
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    assignment,
                    "non_callable",
                    _node_fingerprint(assignment),
                )
            if self._statements_definitely_assign_instance_member(
                initializer.node.body,
                receiver,
                member,
            ):
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    initializer.node,
                    "non_callable",
                    initializer.fingerprint,
                )
            if self._function_may_assign_instance_member(initializer.node, receiver, member):
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    initializer.node,
                    "unknown",
                    initializer.fingerprint,
                )
            recurse_to_base = self._calls_super_init(initializer.node)

        for statement in class_node.body:
            if not isinstance(statement, (ast.AsyncFunctionDef, ast.FunctionDef)) or statement.name == "__init__":
                continue
            positional = [*statement.args.posonlyargs, *statement.args.args]
            receiver = positional[0].arg if positional else None
            if receiver is not None and self._function_may_assign_instance_member(
                statement,
                receiver,
                member,
                allow_dynamic_member=False,
            ):
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    statement,
                    "unknown",
                    _node_fingerprint(statement),
                )

        if not recurse_to_base or not class_node.bases:
            return _QualifiedBinding(resolved.file, actual_owner, member, None, "missing")
        if len(class_node.bases) == 1:
            base = self._base_reference(resolved.file, class_node.bases[0])
            if base == "builtins.object" or base in STDLIB_STRUCTURAL_BASES:
                return _QualifiedBinding(resolved.file, actual_owner, member, None, "missing")
            if base is None or not base.startswith(self._reference_prefixes):
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    None,
                    "unknown",
                    _node_fingerprint(class_node),
                )
            return self._effective_instance_field(
                base,
                member,
                frozenset((*seen, receiver_type)),
            )

        base_bindings: list[_QualifiedBinding] = []
        for base_node in class_node.bases:
            base = self._base_reference(resolved.file, base_node)
            if base == "builtins.object" or base in STDLIB_STRUCTURAL_BASES:
                continue
            if base is None or not base.startswith(self._reference_prefixes):
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    None,
                    "unknown",
                    _node_fingerprint(class_node),
                )
            base_binding = self._effective_instance_field(
                base,
                member,
                frozenset((*seen, receiver_type)),
            )
            if base_binding is None or base_binding.status == "unknown":
                return _QualifiedBinding(
                    resolved.file,
                    actual_owner,
                    member,
                    None,
                    "unknown",
                    _node_fingerprint(class_node),
                )
            if base_binding.status != "missing":
                base_bindings.append(base_binding)

        if not base_bindings:
            # C3 order is irrelevant when every statically resolved base proves
            # absence.  This lets a removed field stay an exact deletion even
            # when the receiver class has several mixin bases.
            return _QualifiedBinding(resolved.file, actual_owner, member, None, "missing")
        if len(base_bindings) == 1:
            return base_bindings[0]
        return _QualifiedBinding(
            resolved.file,
            actual_owner,
            member,
            None,
            "unknown",
            _node_fingerprint(class_node),
        )

    def registered_buffer_state(self, receiver_type: str, member: str) -> bool | None:
        """A closed, literal torch.nn.Module registration model, without imports.

        Absence is usable only after the old endpoint proves a buffer of this
        exact name. Unknown helpers, registries, descriptors or receiver escape
        must not turn an external nn.Module base into a blanket absence proof.
        """
        binding = self._resolve_qualified_node(receiver_type)
        if binding is None or binding.status != "exact" or not isinstance(binding.node, ast.ClassDef):
            return None
        node = binding.node
        if node.decorator_list or node.keywords or len(node.bases) != 1:
            return None
        if self._base_reference(binding.file, node.bases[0]) not in {
            "torch.nn.Module",
            "torch.nn.modules.module.Module",
        }:
            return None
        for name in ("__getattr__", "__getattribute__", "__setattr__", "register_buffer", "register_parameter"):
            if self._body_named_binding(node.body, name).status != "missing":
                return None
        initializer = self._body_named_binding(node.body, "__init__")
        if initializer.status != "exact" or not isinstance(initializer.node, ast.FunctionDef):
            return None
        function = initializer.node
        if function.decorator_list or not self._calls_super_init(function):
            return None
        positional = [*function.args.posonlyargs, *function.args.args]
        if not positional:
            return None
        receiver = positional[0].arg
        registry_names = {"_buffers", "_parameters", "_modules", "__dict__"}
        registrations: set[int] = set()
        for method in node.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for candidate in _function_scope_nodes(method):
                if (
                    isinstance(candidate, ast.Attribute)
                    and isinstance(candidate.value, ast.Name)
                    and candidate.value.id == receiver
                ):
                    if candidate.attr in registry_names:
                        return None
                if not isinstance(candidate, ast.Call):
                    continue
                if any(
                    isinstance(arg, ast.Name) and arg.id == receiver
                    for arg in [*candidate.args, *(k.value for k in candidate.keywords)]
                ):
                    return None
                if not (
                    isinstance(candidate.func, ast.Attribute)
                    and isinstance(candidate.func.value, ast.Name)
                    and candidate.func.value.id == receiver
                ):
                    continue
                if method is not function:
                    if candidate.func.attr in {"register_buffer", "register_parameter"}:
                        return None
                    continue
                if (
                    candidate.func.attr not in {"register_buffer", "register_parameter"}
                    or len(candidate.args) < 2
                    or not isinstance(candidate.args[0], ast.Constant)
                    or not isinstance(candidate.args[0].value, str)
                ):
                    return None
                if candidate.args[0].value == member:
                    registrations.add(candidate.lineno)
        if not registrations:
            if any(
                self._function_may_assign_instance_member(method, receiver, member)
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            ):
                return None
            return False

        class LiteralRegistration(ast.NodeTransformer):
            def visit_Expr(self, statement: ast.Expr) -> ast.AST:
                if isinstance(statement.value, ast.Call) and statement.value.lineno in registrations:
                    return ast.copy_location(
                        ast.Assign(
                            targets=[
                                ast.Attribute(value=ast.Name(id=receiver, ctx=ast.Load()), attr=member, ctx=ast.Store())
                            ],
                            value=ast.Constant(value=None),
                        ),
                        statement,
                    )
                return statement

        transformed = LiteralRegistration().visit(copy.deepcopy(function))
        assert isinstance(transformed, ast.FunctionDef)
        return True if self._statements_definitely_assign_instance_member(transformed.body, receiver, member) else None

    def attribute_endpoint(
        self,
        expression: str,
        access_kind: str,
        *,
        receiver_type: str | None = None,
        member: str | None = None,
    ) -> SourceEndpoint:
        cache_key = (expression, access_kind, receiver_type, member)
        with self._lock:
            cached = self._attribute_endpoints.get(cache_key)
        if cached is not None:
            return cached
        endpoint = self._attribute_endpoint_uncached(
            expression,
            access_kind,
            receiver_type=receiver_type,
            member=member,
        )
        with self._lock:
            self._attribute_endpoints[cache_key] = endpoint
        return endpoint

    def _attribute_endpoint_uncached(
        self,
        expression: str,
        access_kind: str,
        *,
        receiver_type: str | None = None,
        member: str | None = None,
    ) -> SourceEndpoint:
        endpoint = self.call_endpoint(
            expression,
            access_kind,
            receiver_type=receiver_type,
            member=member,
        )
        if access_kind == "direct":
            module_endpoint = self._module_attribute_endpoint(expression, endpoint)
            if module_endpoint is not None:
                return module_endpoint
        if (
            endpoint.symbol_kind not in {"missing", "unknown"}
            or access_kind != "instance"
            or receiver_type is None
            or member is None
        ):
            return endpoint
        binding = self._effective_instance_field(receiver_type, member)
        if binding is None:
            return SourceEndpoint(None, None, member, symbol_kind="unknown")
        if binding.status in {"exact", "non_callable"}:
            return SourceEndpoint(
                file=binding.file or None,
                owner=binding.owner,
                name=binding.name,
                line=getattr(binding.node, "lineno", None),
                descriptor=(
                    _descriptor(binding.node, self._return_resolver(binding.file))
                    if binding.file and binding.status == "exact" and binding.node is not None
                    else "instance_attribute"
                ),
                symbol_kind="attribute",
                analysis_fingerprint=binding.fingerprint,
            )
        return SourceEndpoint(
            file=binding.file or None,
            owner=binding.owner,
            name=binding.name,
            symbol_kind=binding.status,
            signature_status="unknown" if binding.status == "unknown" else None,
            analysis_fingerprint=binding.fingerprint,
        )

    def _module_attribute_endpoint(self, expression: str, fallback: SourceEndpoint) -> SourceEndpoint | None:
        module, separator, name = expression.rpartition(".")
        file_name = self.resolve_module(module) if separator else None
        if file_name is None:
            return None
        tree = self.tree(file_name)
        if tree is None:
            return SourceEndpoint(file_name, None, name, symbol_kind="unknown")

        def build() -> tuple[list[ast.stmt], ModuleGetattrContract | None]:
            body = self._import_body(tree)
            getter = self._body_named_binding(body, "__getattr__")
            contract = None
            if getter.status != "missing":
                contract = module_getattr_contract(body, getter.node)
                if self._body_named_binding(body, "AttributeError").status != "missing":
                    contract = ModuleGetattrContract(None)
            return body, contract

        # Never acquire the memo lock while holding the snapshot lock: memo
        # builders can resolve files/trees and acquire the snapshot lock.
        body, contract = self._memo.get("module_attributes", file_name, build)
        binding = self._body_named_binding(body, name)
        descriptor = "module_attribute"
        status, node = binding.status, binding.node
        fingerprint = binding.fingerprint
        if status == "missing" and contract is not None:
            status, node = contract.resolve(name)
            descriptor = "module_getattr_registry"
            fingerprint = _node_fingerprint(node) if node is not None else None
        elif status in {"exact", "non_callable"}:
            status = "attribute"
        elif status == "unknown" and fallback.symbol_kind not in {None, "missing", "unknown"}:
            # Preserve proven re-export resolution from the ordinary resolver.
            return fallback
        return SourceEndpoint(
            file=file_name,
            owner=None,
            name=name,
            line=getattr(node, "lineno", None),
            descriptor=descriptor,
            symbol_kind=status,
            signature_status="unknown" if status == "unknown" else None,
            analysis_fingerprint=fingerprint,
        )

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
        if base is None or not base.startswith(self._reference_prefixes):
            return False
        return self._constructor_class_safe(base, frozenset((*seen, class_reference)))

    def _type_source(self, reference: str) -> ClassSource | None:
        binding = self._resolve_qualified_node(reference)
        if binding is None or binding.status != "exact" or not isinstance(binding.node, ast.ClassDef):
            return None
        tree = self.tree(binding.file)
        if tree is None:
            return None
        if binding.file not in self._annotation_namespaces:
            module, is_package = _file_module(binding.file)
            self._annotation_namespaces[binding.file] = AnnotationNamespace(
                tree, module, is_package, source_facts=self._memo
            )
        namespace = self._annotation_namespaces[binding.file]

        def resolve(expression: str) -> str | None:
            if expression in TYPE_BUILTINS:
                tree = self.tree(binding.file)
                if tree is not None and self._body_named_binding(tree.body, expression).status == "missing":
                    return expression
            return namespace.resolve(expression)

        module, is_package = _file_module(binding.file)
        if binding.file not in self._construction_effects:
            self._construction_effects[binding.file] = construction_effects(tree, module, is_package)
        effects = self._construction_effects[binding.file]
        return ClassSource(
            binding.node,
            resolve,
            binding.file,
            reference,
            has_storage_effect(reference, effects) or has_storage_effect(f"{module}.{binding.node.name}", effects),
        )

    def _flow_lookup(self) -> Lookup:
        return self._type_source

    def _factory_method_has_effect(self, reference: str, file: str) -> bool:
        effects = self._construction_effects.get(file, frozenset())
        return has_construction_effect(reference, effects) or any(
            effect.startswith(reference + ".") for effect in effects
        )

    def factory_method_result(
        self,
        receiver: str,
        member: str,
        arguments: tuple[FlowValue | None, ...] = (),
        *,
        positional: int = 0,
        keywords: tuple[str, ...] = (),
        _start: str | None = None,
        _seen: frozenset[tuple[str, str, str]] = frozenset(),
    ) -> FlowValue | None:
        """Prove one endpoint's bounded class/static factory body, not Self.

        This internal evaluator uses only this snapshot's names, method owner
        and AST. A separate invocation recipe is required before discovery may
        publish its result for replay at another endpoint.
        """
        start = _start or receiver
        key = (receiver, start, member)
        if key in _seen or len(_seen) >= MAX_FACTORY_RETURN_DEPTH or not 0 <= positional <= len(arguments):
            return None
        if positional + len(keywords) != len(arguments) or len(set(keywords)) != len(keywords):
            return None
        source = self._type_source(receiver)
        if source is None or source.storage_effects:
            return None
        if dataclass_storage(receiver, self._type_source) is None and not self._constructor_class_safe(receiver):
            return None
        hook = self._effective_member(receiver, "__init_subclass__", generated_storage=True)
        if hook is None or hook.status != "missing":
            return None
        binding = self._effective_member(start, member, generated_storage=True)
        if binding is None or binding.status != "exact" or not isinstance(binding.node, ast.FunctionDef):
            return None
        function = binding.node
        namespace = self._return_resolver(binding.file)
        descriptor = _descriptor(function, namespace)
        if (
            descriptor not in {"classmethod", "staticmethod"}
            or len(function.decorator_list) != 1
            or not isinstance(function.decorator_list[0], (ast.Name, ast.Attribute))
        ):
            return None
        if not factory_body_supported(function) or function.args.vararg or function.args.kwarg:
            return None
        signature_data = _jsonable_signature(function)
        signature = _inspect_signature(signature_data) if signature_data is not None else None
        if signature is None:
            return None
        class_value = FlowValue(TypeShape("class", (TypeShape(receiver),)), (receiver,))
        try:
            bound = signature.bind(
                *((class_value,) if descriptor == "classmethod" else ()),
                *arguments[:positional],
                **dict(zip(keywords, arguments[positional:])),
            )
        except TypeError:
            return None
        if any(value is None for value in bound.arguments.values()):
            return None
        module, _ = _file_module(binding.file)
        owner = ".".join(part for part in (module, binding.owner) if part)
        owner_source = self._type_source(owner)
        if (
            owner_source is None
            or owner_source.storage_effects
            or self._factory_method_has_effect(f"{owner}.{member}", binding.file)
        ):
            return None
        local_names = _function_local_names(function)
        tree = self.tree(binding.file)
        if tree is None:
            return None

        def resolve(expression: str) -> str | None:
            if expression == "super" and self._body_named_binding(tree.body, "super").status == "missing":
                return "builtins.super"
            return owner_source.resolve(expression)

        def constructor(call: ast.Call) -> str | None:
            expression = _expression_name(call.func)
            if expression is None or expression.split(".")[0] in local_names:
                return None
            target = resolve(expression)
            return target if target is not None and self._type_source(target) is not None else None

        proxy = None
        if descriptor == "classmethod" and len(owner_source.node.bases) == 1:
            base = self._base_reference(binding.file, owner_source.node.bases[0])
            if base is not None and base.startswith(self._reference_prefixes):
                proxy = FlowValue(TypeShape("super", (TypeShape(receiver), TypeShape(base))), ("unknown",))

        def method_bind(call: ast.Call, value: FlowValue | None):
            if not isinstance(call.func, ast.Attribute):
                return None
            target, lookup_start = None, None
            if value is not None and value.shape.reference == "super" and len(value.shape.arguments) == 2:
                target, lookup_start = (item.reference for item in value.shape.arguments)
            elif value is not None and value.shape.reference == "class" and len(value.shape.arguments) == 1:
                target = value.shape.arguments[0].reference
            elif value is None:
                expression = _expression_name(call.func.value)
                if expression is not None and expression.split(".")[0] not in local_names:
                    target = resolve(expression)
            if target is None:
                return None
            bound_target = target
            method_name = call.func.attr

            def invoke(node: ast.Call, values: tuple[FlowValue | None, ...]) -> HelperCallEffects | None:
                if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
                    kw.arg is None for kw in node.keywords
                ):
                    return None
                result = self.factory_method_result(
                    bound_target,
                    method_name,
                    values,
                    positional=len(node.args),
                    keywords=tuple(kw.arg for kw in node.keywords if kw.arg is not None),
                    _start=lookup_start,
                    _seen=_seen | {key},
                )
                if result is None:
                    return None
                return HelperCallEffects(
                    frozenset(range(len(values))),
                    frozenset(),
                    None,
                    None,
                    result if result.owned is None else None,
                    result if result.owned is not None else None,
                )

            return invoke

        flow = ContainerFlow(
            function,
            resolve,
            self._type_source,
            runtime_resolve=resolve,
            constructor_resolve=constructor,
            method_bind=method_bind,
            capture_returns=True,
            parameters=dict(bound.arguments),
            super_value=proxy,
            super_receiver=next(iter(bound.arguments.items()), None) if descriptor == "classmethod" else None,
        )
        return proven_factory_return(flow, dict(bound.arguments))

    def dependency_endpoint(
        self,
        dependency: DirectCallDependency | DirectAttributeDependency,
    ) -> SourceEndpoint:
        if isinstance(dependency, DirectCallDependency) and dependency.resolution_basis == "dataclass_replace":
            return self._dataclass_replace_endpoint(dependency)
        receiver = dependency.lookup_root or dependency.receiver_type
        if dependency.receiver_path:
            resolved = resolve_flow_path(dependency.receiver_path, self._flow_lookup())
            if (
                resolved is None
                or not resolved.shape.reference.startswith(self._reference_prefixes)
                or dependency.resolution_basis == "constructed_storage"
                and not resolved.constructed
            ):
                return SourceEndpoint(None, None, dependency.member, symbol_kind="unknown")
            receiver = resolved.shape.reference
        if isinstance(dependency, DirectAttributeDependency):
            return self.attribute_endpoint(
                dependency.target, dependency.access_kind, receiver_type=receiver, member=dependency.member
            )
        return self.call_endpoint(
            dependency.target,
            dependency.access_kind,
            receiver_type=receiver,
            member=dependency.member,
            invocation_kind=dependency.invocation_kind,
            constructed_storage=dependency.resolution_basis == "constructed_storage",
        )

    def _dataclass_replace_endpoint(self, dependency: DirectCallDependency) -> SourceEndpoint:
        return self._dataclass_replace_protocol(dependency)[0]

    def _dataclass_replace_protocol(
        self, dependency: DirectCallDependency
    ) -> tuple[SourceEndpoint, dict[str, Any] | None]:
        key = (dependency.target, dependency.receiver_path)
        if key not in self._replacement_protocols:
            self._replacement_protocols[key] = self._build_dataclass_replace_protocol(dependency)
        return self._replacement_protocols[key]

    def _build_dataclass_replace_protocol(
        self, dependency: DirectCallDependency
    ) -> tuple[SourceEndpoint, dict[str, Any] | None]:
        value = resolve_flow_path(dependency.receiver_path, self._flow_lookup())
        unknown = SourceEndpoint(None, None, dependency.target, symbol_kind="unknown")
        if value is None or not value.constructed or value.shape.reference != dependency.target:
            return unknown, None
        source = self._type_source(dependency.target)
        replacement = dataclass_replacement(dependency.target, self._type_source)
        if source is None or replacement is None:
            return unknown, None
        layout, storage = replacement.layout, replacement.storage
        # replace copies each ordinary init field from the existing instance.
        # InitVars are not stored: a required InitVar must still be supplied.
        # init=False and ClassVar entries cannot become constructor keywords.
        fields = [field for field in layout.fields if field.included]
        node = ast.parse("def replacement(): pass").body[0]
        assert isinstance(node, ast.FunctionDef)
        node.args.kwonlyargs = [ast.arg(arg=field.name) for field in fields]
        node.args.kw_defaults = [
            None if field.name in replacement.required_keywords else ast.Constant(None) for field in fields
        ]
        endpoint = SourceEndpoint(
            source.file,
            None,
            dependency.target.rsplit(".", 1)[-1],
            line=source.node.lineno,
            signature=_jsonable_signature(node),
            symbol_kind="constructor",
            signature_status="exact",
            analysis_fingerprint=_node_fingerprint(source.node),
        )
        protocol = {
            "fields": [asdict(field) for field in layout.fields],
            "copied_fields": list(replacement.copied_fields),
            "excluded_fields": list(replacement.excluded_fields),
            "required_keywords": list(replacement.required_keywords),
            "stored_parameters": sorted(storage.stored_parameters),
            "initializer_owner": storage.initializer_owner,
            "initializer_signature": _bound_signature(
                _jsonable_signature(storage.layout.initializer()),
                descriptor="ordinary",
                access_kind="constructor",
            ),
        }
        return endpoint, protocol

    def call_state(self, dependency: DirectCallDependency, endpoint: SourceEndpoint) -> CompatibilityState:
        if dependency.resolution_basis != "dataclass_replace":
            return _direct_call_state(endpoint, dependency)
        _, protocol = self._dataclass_replace_protocol(dependency)
        if protocol is None:
            return _direct_call_state(endpoint, dependency)
        if not dependency.call_shape.exact:
            return CompatibilityState(True, None, "replacement keyword expansion is not statically exact")
        supplied = set(dependency.call_shape.keyword_names)
        if len(supplied) != len(dependency.call_shape.keyword_names):
            return CompatibilityState(True, False, "replacement call supplies a keyword more than once")
        forbidden = sorted(supplied.intersection(protocol["excluded_fields"]))
        if forbidden:
            return CompatibilityState(
                True, False, "replacement explicitly sets init=False fields: " + ", ".join(forbidden)
            )
        copied = set(protocol["copied_fields"])
        # The child's dataclass metadata, not the parent's initializer layout,
        # supplies implicit keywords. Bind the synthesized call separately.
        shape = CallShape(0, tuple(sorted(supplied | copied)))
        compatible, reason = bind_call_shape(protocol["initializer_signature"], shape)
        if compatible is not True:
            return CompatibilityState(True, compatible, "replacement synthesized constructor: " + reason)
        missing = sorted(set(protocol["required_keywords"]) - supplied)
        if missing:
            return CompatibilityState(
                True, False, "replacement requires unstored initialization fields: " + ", ".join(missing)
            )
        return CompatibilityState(True, True, "replacement copies stored fields and binds the generated constructor")

    def _constructor_class_source(self, name: str) -> ClassSource | None:
        binding = self._resolve_qualified_node(name)
        if binding is None or binding.status != "exact" or not isinstance(binding.node, ast.ClassDef):
            return None
        # Allocation and subclass hooks can change the generated call protocol.
        if any(
            self._body_named_binding(binding.node.body, member).status != "missing"
            for member in ("__new__", "__init_subclass__")
        ):
            return None
        return ClassSource(binding.node, self._return_resolver(binding.file), binding.file, name)

    def _dataclass_constructor(self, reference: str, seen: frozenset[str] = frozenset()) -> ast.FunctionDef | None:
        if reference in seen:
            return None
        layout = dataclass_layout(reference, self._constructor_class_source)
        if layout is None:
            return None
        initializer = layout.initializer()
        if initializer is not None or layout.generates_initializer:
            return initializer
        source = self._constructor_class_source(reference)
        if source is None or len(source.node.bases) != 1:
            return None
        if self._body_named_binding(source.node.body, "__init__").status != "missing":
            return None
        base = self._base_reference(source.file, source.node.bases[0])
        return self._dataclass_constructor(base, seen | {reference}) if base is not None else None

    def _return_resolver(self, file_name: str) -> Any:
        module, _ = _file_module(file_name)
        bindings = self._module_bindings(file_name)

        def resolve(expression: str) -> str | None:
            root, separator, remainder = expression.partition(".")
            if root in bindings:
                return f"{bindings[root]}.{remainder}" if separator else bindings[root]
            if expression.startswith(self._reference_prefixes):
                return expression
            if (
                expression in {"classmethod", "property", "staticmethod"}
                and (tree := self.tree(file_name)) is not None
                and self._body_named_binding(tree.body, expression).status == "missing"
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

    def endpoint(
        self,
        file_name: str,
        owner: str | None,
        name: str,
        *,
        invocation_kind: str = "python_call",
    ) -> SourceEndpoint:
        return self._memo.get(
            "endpoint",
            (file_name, owner, name, invocation_kind),
            lambda: self._build_endpoint(file_name, owner, name, invocation_kind=invocation_kind),
        )

    def _build_endpoint(
        self,
        file_name: str,
        owner: str | None,
        name: str,
        *,
        invocation_kind: str = "python_call",
    ) -> SourceEndpoint:
        tree = self.tree(file_name)
        binding = self.named_binding(tree, owner, name) if tree is not None else _NamedBinding(None, "unknown")
        node = binding.node if binding.status == "exact" else None
        resolver = self._return_resolver(file_name)
        descriptor = _descriptor(node, resolver) if node is not None else None
        signature_contract = _snapshot_node_signature_contract(
            node,
            resolver,
            invocation_kind,
            descriptor=descriptor,
            binds_receiver=owner is not None,
        )
        return_contract = infer_return_contract(
            node,
            resolver=resolver,
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
            descriptor=descriptor,
            symbol_kind=symbol_kind,
            signature_status=(
                "unknown"
                if binding.status == "unknown"
                else signature_contract.status
                if signature_contract is not None
                else None
            ),
            analysis_fingerprint=binding.fingerprint,
            return_contract=return_contract.as_dict() if return_contract is not None else None,
        )

    def signature_contract(
        self,
        endpoint: SourceEndpoint,
        invocation_kind: str = "python_call",
    ) -> SignatureContract | None:
        if endpoint.symbol_kind == "constructor" or endpoint.file is None or endpoint.name is None:
            return _snapshot_signature_contract(endpoint, invocation_kind)
        key = (endpoint.file, endpoint.owner, endpoint.name, endpoint.descriptor, invocation_kind)
        return self._memo.get(
            "signature_contract", key, lambda: self._build_signature_contract(endpoint, invocation_kind)
        )

    def _build_signature_contract(self, endpoint: SourceEndpoint, invocation_kind: str) -> SignatureContract | None:
        assert endpoint.file is not None and endpoint.name is not None
        tree = self.tree(endpoint.file)
        binding = self.named_binding(tree, endpoint.owner, endpoint.name) if tree is not None else None
        node = binding.node if binding is not None and binding.status == "exact" else None
        if node is None:
            return _snapshot_signature_contract(endpoint, invocation_kind)
        return _snapshot_node_signature_contract(
            node,
            self._return_resolver(endpoint.file),
            invocation_kind,
            descriptor=endpoint.descriptor,
            binds_receiver=endpoint.owner is not None,
        )

    def call_endpoint(
        self,
        expression: str,
        access_kind: str,
        *,
        receiver_type: str | None = None,
        member: str | None = None,
        invocation_kind: str = "python_call",
        constructed_storage: bool = False,
    ) -> SourceEndpoint:
        key = (expression, access_kind, receiver_type, member, invocation_kind, constructed_storage)
        return self._memo.get(
            "call_endpoint",
            key,
            lambda: self._build_call_endpoint(
                expression,
                access_kind,
                receiver_type=receiver_type,
                member=member,
                invocation_kind=invocation_kind,
                constructed_storage=constructed_storage,
            ),
        )

    def _build_call_endpoint(
        self,
        expression: str,
        access_kind: str,
        *,
        receiver_type: str | None = None,
        member: str | None = None,
        invocation_kind: str = "python_call",
        constructed_storage: bool = False,
    ) -> SourceEndpoint:
        effective_access_kind = access_kind
        resolved: _QualifiedBinding | None = None
        if access_kind == "instance" and receiver_type is not None and member is not None:
            if receiver_type.startswith(self._reference_prefixes):
                resolved = self._effective_member(receiver_type, member, generated_storage=constructed_storage)
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
            generated_initializer = self._dataclass_constructor(expression)
            generated_protocol = generated_initializer is not None
            constructor_fingerprint = hashlib.sha256(
                json.dumps(
                    [
                        resolved.fingerprint,
                        initializer_binding.fingerprint if initializer_binding is not None else None,
                        new_binding.fingerprint if new_binding is not None else None,
                        ast.dump(generated_initializer) if generated_initializer is not None else None,
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
            if generated_protocol:
                initializer = generated_initializer
                constructor_unknown = False
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
        invocation_contract = _snapshot_node_signature_contract(
            node,
            self._return_resolver(file_name),
            invocation_kind,
            descriptor=descriptor,
            binds_receiver=effective_access_kind in {"constructor", "instance"},
            access_kind=effective_access_kind,
        )
        signature = invocation_contract.bound_call_signature if invocation_contract is not None else None
        return_contract = infer_return_contract(node, resolver=self._return_resolver(file_name))
        return SourceEndpoint(
            file=file_name,
            owner=owner,
            name=name,
            line=node.lineno,
            signature=signature,
            descriptor=descriptor,
            symbol_kind="callable",
            signature_status=invocation_contract.status if invocation_contract is not None else None,
            analysis_fingerprint=resolved.fingerprint,
            return_contract=return_contract.as_dict() if return_contract is not None else None,
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
        *,
        invocation_kind: str = "python_call",
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
        resolver = self._return_resolver(file_name)
        descriptor = _descriptor(node, resolver)
        signature_contract = _snapshot_node_signature_contract(
            node,
            resolver,
            invocation_kind,
            descriptor=descriptor,
            binds_receiver=owner is not None,
        )
        return SourceEndpoint(
            file=file_name,
            owner=owner,
            name=node.name,
            line=node.lineno,
            signature=_jsonable_signature(node),
            descriptor=descriptor,
            symbol_kind="callable",
            signature_status=signature_contract.status if signature_contract is not None else None,
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
    if upstream.symbol_kind not in {"callable", "constructor"}:
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


def _direct_attribute_state(upstream: SourceEndpoint) -> CompatibilityState:
    if upstream.symbol_kind in {None, "unknown"}:
        return CompatibilityState(None, None, "upstream member binding could not be proven")
    if upstream.file is None or upstream.symbol_kind == "missing":
        return CompatibilityState(False, False, "upstream member does not exist")
    return CompatibilityState(True, True, "upstream member exists")


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
        return "Track this as a preexisting issue; do not attribute it to this upstream upgrade."
    if classification == "fixed":
        return "Upstream compatibility was restored; review whether the downstream compatibility code is still needed."
    if new.file is None:
        return (
            "Update the downstream target. If upstream removed the capability, remove the patch or inheritance edge "
            "and add an alternative implementation."
        )
    if old.name != new.name:
        return f"Update the downstream dependency from {old.name} to {new.name}, then recheck argument forwarding."
    if relation == "monkey_patch":
        return (
            "Update the replacement signature for the new upstream call contract and verify that the patch "
            "installation path is still active."
        )
    if relation == "override":
        return "Synchronize the override parameters and check super() calls and keyword forwarding."
    if relation == "inheritance":
        return (
            "Review the new base-class path and MRO; do not guess a replacement when the inheritance chain "
            "is incomplete."
        )
    if relation == "direct_import":
        return "Update the imported module or symbol path and add an import-boundary regression test."
    if relation == "direct_call":
        return (
            "Update downstream arguments or return-value consumption and add an interface regression test "
            "for this callsite."
        )
    return (
        "Update the dependency for the exact upstream/downstream contract delta and add an interface regression test."
    )


def _finding_id(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _relation_endpoints(
    relation: Relation,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    new_to_old: dict[str, str],
    invocation_kind: str,
) -> tuple[SourceEndpoint, SourceEndpoint]:
    old_file = new_to_old.get(relation.upstream_file, relation.upstream_file)
    old_endpoint = old_snapshot.endpoint(
        old_file,
        relation.upstream_owner,
        relation.upstream_name,
        invocation_kind=invocation_kind,
    )
    new_endpoint = new_snapshot.endpoint(
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
        invocation_kind=invocation_kind,
    )
    if relation.relation == "monkey_patch":
        # A function can replace a class binding (including a scoped factory).
        # Reuse the constructor resolver, which rejects custom metaclasses,
        # decorators and __new__, instead of comparing two absent signatures.
        endpoints = []
        for snapshot, endpoint in ((old_snapshot, old_endpoint), (new_snapshot, new_endpoint)):
            if endpoint.symbol_kind == "class" and endpoint.file is not None:
                module, _ = _file_module(endpoint.file)
                target = ".".join(part for part in (module, endpoint.owner, endpoint.name) if part)
                endpoint = snapshot.call_endpoint(target, "constructor")
            endpoints.append(endpoint)
        old_endpoint, new_endpoint = endpoints
    if _relation_symbol_presence(new_endpoint) is False:
        old_tree = old_snapshot.tree(old_file)
        old_node = _named_node(old_tree, relation.upstream_owner, relation.upstream_name) if old_tree else None
        renamed = new_snapshot.unique_rename(
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            _definition_fingerprint(old_node),
            invocation_kind=invocation_kind,
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
                key_target, value_target = node.target.elts
                if not isinstance(key_target, ast.Name) or not isinstance(value_target, ast.Name):
                    continue
                key_name = key_target.id
                value_name = value_target.id
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
                    name_keyword = keywords.get("name")
                    class_keyword = keywords.get("_decorated_op_cls")
                    if (
                        isinstance(name_keyword, ast.Name)
                        and name_keyword.id == key_name
                        and isinstance(class_keyword, ast.Name)
                        and class_keyword.id == value_name
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
                        "line": getattr(value, "lineno", getattr(node, "lineno", 0)),
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


def _relation_contract_changed(
    old_endpoint: SourceEndpoint,
    new_endpoint: SourceEndpoint,
    old_signature_contract: SignatureContract | None,
    new_signature_contract: SignatureContract | None,
) -> bool:
    old_exists = _relation_symbol_presence(old_endpoint)
    new_exists = _relation_symbol_presence(new_endpoint)
    return (
        old_exists != new_exists
        or old_endpoint.file != new_endpoint.file
        or old_endpoint.name != new_endpoint.name
        or old_endpoint.signature != new_endpoint.signature
        or old_endpoint.descriptor != new_endpoint.descriptor
        or old_endpoint.signature_status != new_endpoint.signature_status
        or _ambiguous_binding_changed(old_endpoint, new_endpoint)
        or _runtime_signature_changed(old_signature_contract, new_signature_contract)
    )


def _relation_findings(
    relation: Relation,
    engine: InterfaceBoundaryGenerator,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    new_to_old: dict[str, str],
    changed_upstream_files: tuple[str, ...],
    registered_overrides: dict[tuple[str, str], list[dict[str, object]]],
    *,
    strict_optional_contracts: bool,
) -> list[RangeFinding]:
    """Compare one verified replacement relation across the selected range."""
    invocation_kind = (
        _TRITON_KERNEL_PROTOCOL
        if relation.upstream_signature_contract is not None
        and relation.upstream_signature_contract.protocol == _TRITON_KERNEL_PROTOCOL
        else "python_call"
    )
    old_endpoint, new_endpoint = _relation_endpoints(
        relation,
        old_snapshot,
        new_snapshot,
        new_to_old,
        invocation_kind,
    )
    old_signature_contract = old_snapshot.signature_contract(old_endpoint, invocation_kind)
    new_signature_contract = new_snapshot.signature_contract(new_endpoint, invocation_kind)
    downstream = _relation_downstream_endpoint(relation, engine)
    old_exists = _relation_symbol_presence(old_endpoint)
    new_exists = _relation_symbol_presence(new_endpoint)
    runtime_signature_changed = _runtime_signature_changed(old_signature_contract, new_signature_contract)
    contract_changed = _relation_contract_changed(
        old_endpoint,
        new_endpoint,
        old_signature_contract,
        new_signature_contract,
    )
    evidence = [item.as_dict() for item in relation.evidence] or [
        {"file": relation.evidence_file, "line": relation.evidence_line}
    ]
    findings = patch_consumer_findings(relation, engine, old_snapshot, new_snapshot)
    if contract_changed:
        contract_kind = "base_presence" if relation.relation == "inheritance" else "call_arguments"
        old_state = _state(
            old_endpoint,
            downstream.signature,
            relation.relation,
            downstream.descriptor,
            old_signature_contract,
        )
        new_state = _state(
            new_endpoint,
            downstream.signature,
            relation.relation,
            downstream.descriptor,
            new_signature_contract,
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
            _signature_delta(
                old_signature_contract.bound_call_signature,
                new_signature_contract.bound_call_signature,
            )
            if old_signature_contract is not None
            and new_signature_contract is not None
            and old_signature_contract.status == "exact"
            and new_signature_contract.status == "exact"
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
        optional_contract_without_dispatch = bool(optional_parameters and not upstream_call_evidence)
        optional_contract_review = optional_contract_without_dispatch and not strict_optional_contracts
        masked_preexisting_delta = bool(
            relation.relation in {"monkey_patch", "override"}
            and classification == "preexisting"
            and _signature_delta_changed(parameter_delta)
        )
        removed_override_target_only = bool(
            relation.relation == "override" and old_exists is True and new_exists is False
        )
        gates = {
            "relationship_verified": True,
            "contract_changed": contract_changed,
            # A removed base declaration does not by itself prove that the
            # surviving downstream method is called. Direct/super call
            # analysis reports a hard break separately when one exists.
            "runtime_reachable": not removed_override_target_only,
            "version_lane_matches": True,
        }
        action = (
            "review"
            if optional_contract_review or masked_preexisting_delta or removed_override_target_only
            else _finding_action(classification, gates)
        )
        if optional_contract_review:
            suggestion = (
                "Review whether the new optional parameter can reach this downstream override at runtime. "
                "If it can, update the override signature and handle the new argument."
            )
        elif masked_preexisting_delta:
            suggestion = (
                "Upstream introduced another exact parameter delta while downstream was already incompatible at old. "
                "Review this delta separately instead of treating it as a confirmed upgrade regression."
            )
        elif removed_override_target_only:
            suggestion = (
                "Upstream removed the overridden declaration. Review whether the downstream method is now dead or "
                "still called through another path; remove or migrate it only with call-site evidence."
            )
        else:
            suggestion = _suggestion(
                relation.relation,
                classification,
                old_endpoint,
                new_endpoint,
            )
        review_details: dict[str, object] = {}
        if _signature_delta_changed(parameter_delta):
            review_details["parameter_delta"] = parameter_delta
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
                        else "exact_optional_contract_requires_main2main_alignment"
                        if strict_optional_contracts
                        else "strict_optional_contract_without_proven_downstream_dispatch"
                    ),
                    "strict_main2main_contract": bool(strict_optional_contracts and optional_contract_without_dispatch),
                    "parameter_delta": parameter_delta,
                }
            )
        if masked_preexisting_delta:
            review_details.update(
                {
                    "new_delta_on_preexisting_break": True,
                    "actionability_reason": "new_delta_masked_by_preexisting_incompatibility",
                    "priority_reason": "exact_new_parameter_delta_requires_separate_review",
                    "parameter_delta": parameter_delta,
                }
            )
        if removed_override_target_only:
            review_details.update(
                {
                    "removed_override_target_only": True,
                    "actionability_reason": "removed_base_declaration_without_runtime_call_evidence",
                    "priority_reason": "override_removal_requires_call_site_review",
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
                    if relation.relation == "monkey_patch" and (action == "modify" or masked_preexisting_delta)
                    else ("P1" if action == "modify" or masked_preexisting_delta else "P2")
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
                    "invocation_protocol": invocation_kind,
                    **review_details,
                    **_override_details(relation),
                },
            )
        )

        witness = (
            _added_keyword_witness(old_signature_contract, new_signature_contract, downstream.signature)
            if masked_preexisting_delta and strict_optional_contracts
            else None
        )
        if witness is None and masked_preexisting_delta and strict_optional_contracts:
            witness = _local_constructor_delta_witness(
                relation,
                engine,
                old_signature_contract,
                new_signature_contract,
                downstream.signature,
            )
        if witness is not None:
            findings.append(
                replace(
                    findings[-1],
                    finding_id=_finding_id(findings[-1].finding_id, "independent_parameter_delta"),
                    classification="introduced_break",
                    action=_finding_action("introduced_break", gates),
                    old_state=CompatibilityState(True, True, "replacement accepts the old witness call"),
                    new_state=CompatibilityState(True, False, "replacement rejects only the added witness keywords"),
                    suggestion="Adapt the newly rejected parameters; track whole-signature historical debt separately.",
                    details={
                        **findings[-1].details,
                        "new_delta_on_preexisting_break": False,
                        "historical_finding_id": findings[-1].finding_id,
                        "actionability_reason": "independently_proven_new_parameter_delta",
                        "delta_witness": witness,
                    },
                )
            )

    return_changed = new_exists is True and (
        old_exists is not True or old_endpoint.return_contract != new_endpoint.return_contract
    )
    if (
        relation.relation in {"monkey_patch", "override"}
        and return_changed
        and new_endpoint.symbol_kind != "constructor"
    ):
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
                suggestion=(
                    "Update the patch or override return protocol for the new upstream contract "
                    "and add a return-value regression test."
                ),
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
            if alias.name == "vllm" or (alias.name.startswith("vllm.") and alias.asname is None):
                self._vllm_roots.add(alias.asname or "vllm")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._version_guard_depth:
            return
        if not _attribute_is_read(node, self._parents):
            # Stores install patches or mutate state; they do not read the old
            # attribute. Still visit receivers and values through normal AST flow.
            self.generic_visit(node)
            return
        parent = self._parents.get(id(node))
        if isinstance(parent, ast.Attribute) and parent.value is node and _attribute_is_read(parent, self._parents):
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


def discover_imports(ascend_root: Path, *, trees: dict[str, ast.Module] | None = None) -> list[ImportReference]:
    references: list[ImportReference] = []
    for path in sorted((ascend_root / "vllm_ascend").rglob("*.py")):
        relative = path.relative_to(ascend_root).as_posix()
        tree = trees.get(relative) if trees is not None else None
        if tree is None:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except (OSError, SyntaxError, UnicodeError):
                continue
        visitor = _ImportVisitor(relative)
        visitor.visit(ast.Module(body=runtime_module_body(tree), type_ignores=[]))
        references.extend(visitor.references)
    unique = {(item.module, item.symbol, item.file, item.line): item for item in references}
    ordered = sorted(
        unique,
        key=lambda item: (item[0], item[1] or "", item[2], item[3]),
    )
    return [unique[key] for key in ordered]


def _top_level_symbol(snapshot: GitSnapshot, file_name: str, name: str) -> SourceEndpoint:
    if snapshot.has_import_symbol(file_name, name) is not True:
        return SourceEndpoint(file=None, owner=None, name=name)
    endpoint = snapshot.endpoint(file_name, None, name)
    if endpoint.line is not None:
        return endpoint
    tree = snapshot.tree(file_name)
    assert tree is not None
    bindings = snapshot._body_bindings(snapshot._import_body(tree))[name]
    return SourceEndpoint(file=file_name, owner=None, name=name, line=min(binding.line for binding in bindings))


def _import_findings(
    ascend_root: Path,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    old_to_new: dict[str, str],
    references: Iterable[ImportReference] | None = None,
) -> list[RangeFinding]:
    findings: list[RangeFinding] = []
    for reference in discover_imports(ascend_root) if references is None else references:
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
            if old_snapshot.has_import_symbol(old_file, symbol) is not True:
                continue
            if new_file is not None:
                if new_file.endswith("/__init__.py") and new_snapshot.resolve_module(f"{reference.module}.{symbol}"):
                    continue
                if new_snapshot.has_import_symbol(new_file, symbol) is not False:
                    continue
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


def _dispatch_state(state: CompatibilityState, binding: dict[str, Any] | None) -> CompatibilityState:
    if binding is None:
        return state
    if binding.get("kind") == "invalidated_parameter_receiver":
        return CompatibilityState(
            state.exists,
            None,
            "preceding effects invalidated receiver provenance; the parameter annotation is not an exact binding",
        )
    return CompatibilityState(
        state.exists,
        None,
        "base annotation has downstream member alternatives; concrete receiver dispatch is unresolved",
    )


class _ConstructorSnapshot(GitSnapshot):
    """Ephemeral composite view: fixed downstream ASTs and one upstream SHA.

    Constructor and concrete receiver comparisons use this view. It is never stored as an
    upstream snapshot or substituted into the running relation generator.
    """

    _reference_prefixes = ("vllm.", "vllm_ascend.")

    def __init__(
        self,
        upstream: GitSnapshot,
        downstream: RepositoryIndex,
        *,
        proofs: frozenset[MethodProof] = frozenset(),
        baseline_effects: dict[str, frozenset[str]] | None = None,
        flow_engine: InterfaceBoundaryGenerator | None = None,
    ):
        super().__init__(upstream.root, upstream.revision)
        self._upstream = upstream
        self._downstream = downstream
        self._flow_engine = flow_engine
        self._downstream_trees = downstream.source_facts.get(
            "downstream_trees", downstream, lambda: {module.file: module.tree for module in downstream.modules.values()}
        )
        self._combined_files = downstream.source_facts.get(
            "combined_files", upstream, lambda: upstream.files | self._downstream_trees.keys()
        )
        self._method_proofs = proofs
        self._baseline_effects = baseline_effects if baseline_effects is not None else {}
        self._proof_views: dict[frozenset[MethodProof], _ConstructorSnapshot] = {}
        self._proofs_valid = True
        self._context_effects: frozenset[str] = frozenset()
        if proofs:
            effects: set[str] = set()
            for module in downstream.modules.values():
                calls = [proof.call_in(module.tree) for proof in proofs if proof.file == module.file]
                if any(call is None for call in calls):
                    self._proofs_valid = False
                if calls:
                    module_effects = construction_effects(
                        module.tree,
                        module.name,
                        module.is_package,
                        bound_receiver_calls=frozenset(id(call) for call in calls if call is not None),
                    )
                else:
                    if module.file not in self._baseline_effects:
                        self._baseline_effects[module.file] = downstream.source_facts.get(
                            "construction_effects",
                            (module.tree, module.name, module.is_package),
                            partial(construction_effects, module.tree, module.name, module.is_package),
                        )
                    module_effects = self._baseline_effects[module.file]
                self._construction_effects[module.file] = module_effects
                effects.update(module_effects)
            self._context_effects = frozenset(effects)
            self._proofs_valid = self._proofs_valid and all(self._valid_method_proof(proof) for proof in proofs)

    def _valid_method_proof(self, proof: MethodProof) -> bool:
        tree = self._downstream_trees.get(proof.file)
        call = proof.call_in(tree) if tree is not None else None
        if (
            call is None
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != proof.method.rsplit(".", 1)[-1]
        ):
            return False
        creation_hook = self._effective_member(proof.receiver, "__init_subclass__", generated_storage=True)
        if creation_hook is None or creation_hook.status != "missing":
            return False
        binding = self._effective_member(proof.receiver, call.func.attr, generated_storage=True)
        if binding is None or binding.status != "exact" or not isinstance(binding.node, ast.FunctionDef):
            return False
        module, _ = _file_module(binding.file)
        reference = ".".join(part for part in (module, binding.owner, binding.name) if part)
        if isinstance(proof, FactoryCallProof):
            expression = _expression_name(call.func.value)
            # This entry point currently binds source-visible class references,
            # not annotated instances or local aliases with unproved values.
            return (
                expression is not None
                and self._return_resolver(proof.file)(expression) == proof.receiver
                and not has_construction_effect(proof.receiver, self._context_effects)
                and not has_construction_effect(reference, self._context_effects)
                and not any(effect.startswith(reference + ".") for effect in self._context_effects)
            )
        return (
            binding.file in self._downstream_trees
            and reference == proof.method
            and method_body_hash(binding.node) == proof.body_sha256
            and not has_construction_effect(proof.receiver, self._context_effects)
            and not has_construction_effect(proof.method, self._context_effects)
            and not any(effect.startswith(proof.method + ".") for effect in self._context_effects)
        )

    def _type_source(self, reference: str) -> ClassSource | None:
        source = super()._type_source(reference)
        if source is None or not self._method_proofs:
            return source
        return replace(
            source,
            storage_effects=source.storage_effects
            or has_storage_effect(reference, self._context_effects)
            or has_storage_effect(source.qualified_name, self._context_effects),
        )

    def _flow_lookup(self) -> Lookup:
        if self._method_proofs:
            return MethodContextLookup(self._type_source, self._method_proofs, self.evaluate_factory_call)
        return super()._flow_lookup()

    def _factory_method_has_effect(self, reference: str, file: str) -> bool:
        return (
            super()._factory_method_has_effect(reference, file)
            or has_construction_effect(reference, self._context_effects)
            or any(effect.startswith(reference + ".") for effect in self._context_effects)
        )

    def _factory_context(self, proof: FactoryCallProof, proofs: frozenset[MethodProof]) -> _ConstructorSnapshot | None:
        if len(proofs) > MAX_FACTORY_RETURN_DEPTH:
            return None
        frozen = proofs
        if frozen != self._method_proofs:
            candidate = self._resolve_qualified_node(proof.receiver)
            if candidate is None or candidate.status != "exact" or not isinstance(candidate.node, ast.ClassDef):
                # A cheap source-only rejection precedes the whole-downstream
                # contextual effect pass. It never substitutes for body proof.
                return None
            direct = _body_named_binding(candidate.node.body, proof.member)
            if isinstance(direct.node, ast.FunctionDef) and not factory_body_supported(direct.node):
                return None
            if frozen not in self._proof_views:
                self._proof_views[frozen] = _ConstructorSnapshot(
                    self._upstream,
                    self._downstream,
                    proofs=frozen,
                    baseline_effects=self._baseline_effects,
                    flow_engine=self._flow_engine,
                )
            view = self._proof_views[frozen]
            return view if view._proofs_valid else None
        if not self._proofs_valid:
            return None
        return self

    def evaluate_factory_call(
        self,
        proof: FactoryCallProof,
        arguments: tuple[FlowValue | None, ...],
        positional: int,
        keywords: tuple[str, ...],
    ) -> FlowValue | None:
        proofs = set(self._method_proofs) | {proof}
        for argument in arguments:
            nested = method_path_proofs(argument.path, require_complete=True) if argument is not None else None
            if nested is None:
                return None
            proofs.update(nested)
        view = self._factory_context(proof, frozenset(proofs))
        if view is None:
            return None
        return view.factory_method_result(
            proof.receiver, proof.member, arguments, positional=positional, keywords=keywords
        )

    def factory_call_inputs(self, proof: FactoryCallProof) -> tuple[FlowValue | None, ...] | None:
        # This memo belongs to a fixed endpoint and proof view. It never shares
        # mutable dataflow or bypasses reconstruction for a different call.
        return self._memo.get("factory_call_inputs", proof, lambda: self._build_factory_call_inputs(proof))

    def _build_factory_call_inputs(self, proof: FactoryCallProof) -> tuple[FlowValue | None, ...] | None:
        """Rebuild arguments in an isolated source flow, not from annotations.

        The pending callee has not executed while its arguments are evaluated.
        Exclude only that exact call's prospective effects; all other calls and
        writes remain barriers. This view cannot publish a return value itself:
        the caller must still prove every argument recipe and the method body.
        """
        engine = self._flow_engine
        if engine is None or engine.downstream is not self._downstream:
            return None
        view = self._factory_context(proof, self._method_proofs | {proof})
        if view is None:
            return None
        module_name, _ = _file_module(proof.file)
        module = engine.downstream.modules.get(module_name)
        call = proof.call_in(module.tree) if module is not None else None
        if module is None or call is None:
            return None
        parents = engine.downstream.source_facts.parents(module.tree)
        function: ast.AST | None = parents.get(id(call))
        while function is not None and not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = parents.get(id(function))
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        # No factory callback is installed on this resolver: reconstruction
        # cannot recursively invoke itself or assume another unknown return.
        resolver = _DirectDependencyResolver(engine)
        resolver._upstream_type_source = view._type_source
        resolver._reference_prefixes = self._reference_prefixes
        resolver._storage_downstream_effects = view._context_effects
        resolver._storage_module_effects = {
            (id(engine.downstream), item.name): view._construction_effects[item.file]
            for item in engine.downstream.modules.values()
        }
        flow = resolver._function_flow(function, module, capture_call_inputs=True)
        return flow.invocation_inputs.get(id(call))

    @property
    def files(self) -> set[str]:
        return self._combined_files

    def tree(self, file_name: str) -> ast.Module | None:
        normalized = file_name.replace("\\", "/")
        if normalized in self._downstream_trees:
            return self._downstream_trees[normalized]
        return self._upstream.tree(normalized)

    def upstream_fields(self, reference: str) -> list[dict[str, Any]] | None:
        layout = dataclass_layout(reference, self._constructor_class_source)
        if layout is None:
            return None
        return [asdict(field) for field in layout.fields if field.owner.startswith("vllm.")]

    def initializer_origin(self, reference: str) -> dict[str, Any] | None:
        binding = self._effective_member(reference, "__init__")
        if binding is None:
            return None
        return {
            "file": binding.file,
            "owner": binding.owner,
            "name": binding.name,
            "line": getattr(binding.node, "lineno", None),
            "status": binding.status,
        }

    def _effective_init_mutations(
        self, reference: str, mutations: tuple[dict[str, Any], ...], seen: frozenset[str] = frozenset()
    ) -> tuple[dict[str, Any], ...]:
        if reference in seen:
            return ()
        direct = tuple(item for item in mutations if item["target"] == reference and item["member"] == "__init__")
        if direct:
            return direct
        source = self._constructor_class_source(reference)
        if source is None:
            return ()
        layout = dataclass_layout(reference, self._constructor_class_source)
        if layout is not None and layout.generates_initializer:
            return ()
        if _body_named_binding(source.node.body, "__init__").status != "missing" or len(source.node.bases) != 1:
            return ()
        base = self._base_reference(source.file, source.node.bases[0])
        return self._effective_init_mutations(base, mutations, seen | {reference}) if base is not None else ()

    def dependency_endpoint(self, dependency: DirectCallDependency | DirectAttributeDependency) -> SourceEndpoint:
        if not self._proofs_valid:
            return SourceEndpoint(None, None, dependency.member, symbol_kind="unknown")
        if dependency.receiver_path and not self._method_proofs:
            proofs = method_path_proofs(dependency.receiver_path)
            if proofs is None:
                return SourceEndpoint(None, None, dependency.member, symbol_kind="unknown")
            if proofs:
                if proofs not in self._proof_views:
                    self._proof_views[proofs] = _ConstructorSnapshot(
                        self._upstream,
                        self._downstream,
                        proofs=proofs,
                        baseline_effects=self._baseline_effects,
                    )
                return self._proof_views[proofs].dependency_endpoint(dependency)
        endpoint = super().dependency_endpoint(dependency)
        if not isinstance(dependency, DirectCallDependency) or not dependency.constructor_mutations:
            return endpoint
        mutations = dependency.constructor_mutations
        allocation_changed = any(item["member"] != "__init__" for item in mutations)
        initializer_mutations = self._effective_init_mutations(dependency.target, mutations)
        if not allocation_changed and not initializer_mutations:
            return endpoint
        fingerprint = hashlib.sha256(
            json.dumps([endpoint.analysis_fingerprint, mutations], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        unknown_return = dict(endpoint.return_contract or {}, status="unknown")
        unknown = replace(
            endpoint, signature_status="unknown", analysis_fingerprint=fingerprint, return_contract=unknown_return
        )
        if allocation_changed or not initializer_mutations:
            return unknown
        if len({item["file"] for item in initializer_mutations}) != 1 or any(
            item["scope"] is not None for item in initializer_mutations
        ):
            return unknown
        mutation = max(initializer_mutations, key=lambda item: item["line"])
        if not mutation["exact_module_assignment"]:
            return unknown
        reference = mutation["replacement"]
        replacement = self._resolve_qualified_node(reference) if reference is not None else None
        # A dataclass layout proves a plain single-inheritance hierarchy and
        # excludes custom allocation/subclass hooks through the shared lookup.
        layout = dataclass_layout(dependency.target, self._constructor_class_source)
        if (
            layout is None
            or replacement is None
            or replacement.status != "exact"
            or not isinstance(replacement.node, ast.FunctionDef)
            or replacement.node.decorator_list
        ):
            return unknown
        signature = _bound_signature(
            _jsonable_signature(replacement.node), descriptor="ordinary", access_kind="constructor"
        )
        status = _signature_status(replacement.node, self._return_resolver(replacement.file))
        return_contract = ReturnContract(
            protocol="value",
            variants=(ReturnShape("object", type_ref=dependency.target),),
            status="exact" if status != "unknown" else "unknown",
            provenance=("class_constructor", "post_definition_initializer"),
        )
        return replace(
            endpoint,
            signature=signature,
            signature_status=status,
            analysis_fingerprint=fingerprint,
            return_contract=return_contract.as_dict(),
        )


def _direct_call_findings(
    dependencies: Iterable[DirectCallDependency],
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    old_to_new: dict[str, str] | None = None,
    *,
    downstream_index: RepositoryIndex | None = None,
) -> tuple[list[RangeFinding], list[DirectCallDependency]]:
    """Compare exact downstream call and return-use contracts at both SHAs."""
    findings: list[RangeFinding] = []
    exact_dependencies: list[DirectCallDependency] = []
    constructor_findings: list[RangeFinding] = []
    constructor_dependencies: list[DirectCallDependency] = []
    if downstream_index is not None:
        dependencies = list(dependencies)
        composite_bases = {"downstream_constructor", "dataclass_replace"}
        constructors = [item for item in dependencies if item.resolution_basis in composite_bases or item.receiver_path]
        dependencies = [
            item for item in dependencies if item.resolution_basis not in composite_bases and not item.receiver_path
        ]
        if constructors:
            old_view = _ConstructorSnapshot(old_snapshot, downstream_index)
            new_view = _ConstructorSnapshot(new_snapshot, downstream_index)
            constructor_findings, constructor_dependencies = _direct_call_findings(
                constructors, old_view, new_view, old_to_new
            )
            for finding in constructor_findings:
                target = finding.details["target"]
                if finding.details.get("resolution_basis") == "dataclass_replace":
                    finding.source = "dataclass_replace_detector"
                    finding.direction = "downstream_replace_to_dataclass_contract"
                    finding.change = "dataclass replacement field contract changed: " + finding.change
                    finding.suggestion = "Update replacement keywords for the exact dataclass field contract."
                    finding.details["replacement_inheritance"] = {
                        "class": target,
                        "upstream_fields_old": old_view.upstream_fields(target),
                        "upstream_fields_new": new_view.upstream_fields(target),
                    }
                    continue
                if finding.details.get("resolution_basis") != "downstream_constructor":
                    continue
                finding.direction = "upstream_inheritance_to_downstream_constructor_call"
                finding.source = "downstream_constructor_detector"
                finding.change = "upstream inheritance changed the effective downstream constructor: " + finding.change
                finding.suggestion = (
                    "Check this call against the fixed downstream class's effective constructor at each upstream SHA. "
                    "Use the recorded upstream field owners to identify the required adaptation."
                )
                finding.details["constructor_inheritance"] = {
                    "kind": "effective_downstream_constructor",
                    "class": target,
                    "upstream_fields_old": old_view.upstream_fields(target),
                    "upstream_fields_new": new_view.upstream_fields(target),
                    "initializer_old": old_view.initializer_origin(target),
                    "initializer_new": new_view.initializer_origin(target),
                }
    for dependency in dependencies:
        old_endpoint = old_snapshot.dependency_endpoint(dependency)
        new_endpoint = new_snapshot.dependency_endpoint(dependency)
        old_call_state = old_snapshot.call_state(dependency, old_endpoint)
        new_call_state = new_snapshot.call_state(dependency, new_endpoint)
        replacement_protocol = None
        if dependency.resolution_basis == "dataclass_replace":
            replacement_protocol = {
                "old": old_snapshot._dataclass_replace_protocol(dependency)[1],
                "new": new_snapshot._dataclass_replace_protocol(dependency)[1],
            }
        relocation: dict[str, Any] | None = None
        if (
            old_endpoint.file is not None
            and old_endpoint.file not in new_snapshot.files
            and new_endpoint.symbol_kind == "missing"
        ):
            # Longest-prefix resolution can land on a parent __init__.py when
            # the original module disappeared. Do not present that fallback as
            # the missing callable's actual definition or relocation destination.
            new_endpoint = SourceEndpoint(
                old_endpoint.file, old_endpoint.owner, old_endpoint.name, symbol_kind="missing"
            )
            destination = (old_to_new or {}).get(old_endpoint.file)
            if destination is not None and old_endpoint.name is not None:
                relocation = new_snapshot.endpoint(destination, old_endpoint.owner, old_endpoint.name).as_dict()
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
            or replacement_protocol is not None
            and old_call_state != new_call_state
        )
        if parameter_changed:
            contract_kind = (
                "call_target_presence"
                if old_endpoint.symbol_kind not in callable_kinds or new_endpoint.symbol_kind not in callable_kinds
                else "call_arguments"
            )
            old_state = old_call_state
            new_state = new_call_state
            declared_states = {"old": old_state.as_dict(), "new": new_state.as_dict()}
            old_state = _dispatch_state(old_state, dependency.receiver_binding)
            new_state = _dispatch_state(new_state, dependency.receiver_binding)
            classification = _classify(old_state, new_state, parameter_changed)
            gates = {
                "relationship_verified": dependency.receiver_binding is None,
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
                    change="declared upstream contract changed; concrete receiver dispatch is unresolved"
                    if dependency.receiver_binding is not None
                    else _change_text(
                        old_endpoint,
                        new_endpoint,
                        runtime_signature_changed=(old_endpoint.signature_status != new_endpoint.signature_status),
                    ),
                    evidence=[dependency.as_dict()],
                    gates=gates,
                    suggestion="Resolve the concrete caller receiver and its local member before proposing adaptation."
                    if dependency.receiver_binding is not None
                    else _suggestion("direct_call", classification, old_endpoint, new_endpoint),
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
                        **({"replacement_protocol": replacement_protocol} if replacement_protocol is not None else {}),
                        "receiver_binding": dependency.receiver_binding,
                        "receiver_path": list(dependency.receiver_path),
                        "declared_endpoint_compatibility": declared_states,
                        "call_shape": dependency.call_shape.as_dict(),
                        "parameter_delta": _signature_delta(
                            old_contract.bound_call_signature
                            if (
                                old_contract := old_snapshot.signature_contract(
                                    old_endpoint, dependency.invocation_kind
                                )
                            )
                            is not None
                            else None,
                            new_contract.bound_call_signature
                            if (
                                new_contract := new_snapshot.signature_contract(
                                    new_endpoint, dependency.invocation_kind
                                )
                            )
                            is not None
                            else None,
                        ),
                        "relocation_destination": relocation,
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
        declared_states = {"old": old_state.as_dict(), "new": new_state.as_dict()}
        old_state = _dispatch_state(old_state, dependency.receiver_binding)
        new_state = _dispatch_state(new_state, dependency.receiver_binding)
        classification = _classify(old_state, new_state, return_changed)
        gates = {
            "relationship_verified": dependency.receiver_binding is None,
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
                change="declared upstream return changed; concrete receiver dispatch is unresolved"
                if dependency.receiver_binding is not None
                else _change_text(old_endpoint, new_endpoint, "return_usage"),
                evidence=[dependency.as_dict()],
                gates=gates,
                suggestion=(
                    "Resolve the concrete receiver and local return contract before changing return consumption."
                    if dependency.receiver_binding is not None
                    else "Update downstream unpacking, indexing, or protocol use for the upstream return value "
                    "and add a callsite regression test."
                ),
                source="direct_call_detector",
                contract_kind="return_usage",
                direction="downstream_call_to_upstream",
                details={
                    "target": dependency.target,
                    "receiver_binding": dependency.receiver_binding,
                    "receiver_path": list(dependency.receiver_path),
                    "declared_endpoint_compatibility": declared_states,
                    "return_use": dependency.return_use.as_dict(),
                    "upstream_old_return": old_endpoint.return_contract,
                    "upstream_new_return": new_endpoint.return_contract,
                    "scope": dependency.scope,
                },
            )
        )
    return findings + constructor_findings, exact_dependencies + constructor_dependencies


def _direct_attribute_findings(
    dependencies: Iterable[DirectAttributeDependency],
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    *,
    downstream_index: RepositoryIndex | None = None,
) -> tuple[list[RangeFinding], list[DirectAttributeDependency]]:
    """Compare exact downstream member-read presence at both SHAs."""

    findings: list[RangeFinding] = []
    exact_dependencies: list[DirectAttributeDependency] = []
    old_receiver_snapshot, new_receiver_snapshot = old_snapshot, new_snapshot
    if downstream_index is not None:
        dependencies = list(dependencies)
        if any(item.receiver_path for item in dependencies):
            # Reuse the same fixed downstream ASTs at both upstream endpoints.
            # The actual child can shadow an upstream field or descriptor.
            old_receiver_snapshot = _ConstructorSnapshot(old_snapshot, downstream_index)
            new_receiver_snapshot = _ConstructorSnapshot(new_snapshot, downstream_index)
    for dependency in dependencies:
        endpoint_receiver = dependency.lookup_root or dependency.receiver_type
        old_endpoint_view = old_receiver_snapshot if dependency.receiver_path else old_snapshot
        new_endpoint_view = new_receiver_snapshot if dependency.receiver_path else new_snapshot
        old_endpoint = old_endpoint_view.dependency_endpoint(dependency)
        new_endpoint = new_endpoint_view.dependency_endpoint(dependency)
        exact_dependencies.append(dependency)
        old_state = _direct_attribute_state(old_endpoint)
        new_state = _direct_attribute_state(new_endpoint)
        registered_buffer_evidence = False
        if endpoint_receiver and dependency.member and old_state.exists is None and not dependency.receiver_path:
            old_buffer = old_snapshot.registered_buffer_state(endpoint_receiver, dependency.member)
            new_buffer = new_snapshot.registered_buffer_state(endpoint_receiver, dependency.member)
            if old_buffer is True:
                old_state = CompatibilityState(
                    True, True, "literal nn.Module buffer registered on every constructor path"
                )
                old_endpoint = replace(old_endpoint, symbol_kind="instance_field")
                registered_buffer_evidence = True
                if new_state.exists is None and new_buffer is not None:
                    new_state = CompatibilityState(
                        new_buffer,
                        new_buffer,
                        "literal nn.Module buffer registration present"
                        if new_buffer
                        else "previously registered buffer is absent from the closed new layout",
                    )
                    new_endpoint = replace(new_endpoint, symbol_kind="instance_field" if new_buffer else "missing")
        declared_states = {"old": old_state.as_dict(), "new": new_state.as_dict()}
        endpoint_unresolved = old_state.exists is None or new_state.exists is None
        contract_changed = old_state.exists != new_state.exists or old_state.compatible != new_state.compatible
        if not contract_changed and not endpoint_unresolved:
            continue
        old_state = _dispatch_state(old_state, dependency.receiver_binding)
        new_state = _dispatch_state(new_state, dependency.receiver_binding)
        unresolved = endpoint_unresolved or dependency.receiver_binding is not None
        classification = _classify(old_state, new_state, contract_changed)
        gates = {
            "relationship_verified": dependency.receiver_binding is None,
            "contract_changed": contract_changed and not unresolved,
            "runtime_reachable": True,
            "version_lane_matches": True,
        }
        action = _finding_action(classification, gates)
        findings.append(
            RangeFinding(
                finding_id=_finding_id(
                    "direct_attribute",
                    "attribute_presence",
                    dependency.file,
                    dependency.line,
                    dependency.column,
                    dependency.target,
                    old_snapshot.revision,
                    new_snapshot.revision,
                ),
                classification=classification,
                relation="direct_attribute",
                priority="P1" if action == "modify" else "P2",
                action=action,
                confidence="high" if classification != "analysis_unresolved" else "medium",
                upstream_old=old_endpoint,
                upstream_new=new_endpoint,
                downstream=SourceEndpoint(
                    file=dependency.file,
                    owner=dependency.owner,
                    name=dependency.expression,
                    line=dependency.line,
                    symbol_kind="attribute_read",
                ),
                old_state=old_state,
                new_state=new_state,
                change=(
                    "declared upstream attribute changed; preceding effects invalidated the receiver proof"
                    if dependency.receiver_binding is not None
                    and dependency.receiver_binding.get("kind") == "invalidated_parameter_receiver"
                    else "declared upstream attribute changed; concrete receiver may resolve a downstream member"
                    if dependency.receiver_binding is not None
                    else "upstream attribute presence could not be proven at both endpoints"
                    if unresolved
                    else _change_text(old_endpoint, new_endpoint, "attribute_presence")
                ),
                evidence=[dependency.as_dict()],
                gates=gates,
                suggestion=(
                    "Re-establish receiver provenance after preceding effects before proposing field adaptation."
                    if dependency.receiver_binding is not None
                    and dependency.receiver_binding.get("kind") == "invalidated_parameter_receiver"
                    else "Resolve the caller receiver and local property; retain separate breaks inside its getter."
                    if dependency.receiver_binding is not None
                    else "Inspect the unresolved upstream runtime attribute binding; do not assume an introduced break."
                    if unresolved
                    else "Update the downstream member read for the new upstream object layout and add an "
                    "attribute-presence regression test."
                ),
                source="direct_attribute_detector",
                contract_kind="attribute_presence",
                direction="downstream_attribute_read_to_upstream",
                details={
                    "target": dependency.target,
                    "access_kind": dependency.access_kind,
                    "receiver_type": dependency.receiver_type,
                    "member": dependency.member,
                    "lookup_root": dependency.lookup_root,
                    "resolution_basis": dependency.resolution_basis,
                    "receiver_binding": dependency.receiver_binding,
                    "declared_endpoint_compatibility": declared_states,
                    "scope": dependency.scope,
                    "registered_buffer_evidence": registered_buffer_evidence,
                    "receiver_path": list(dependency.receiver_path),
                },
            )
        )
    return findings, exact_dependencies


def _instance_attribute_reads(
    function: ast.AsyncFunctionDef | ast.FunctionDef,
) -> dict[str, tuple[int, bool]]:
    """Return exact receiver fields read by one inherited callable.

    Direct method dispatch (``self.run()``) is a callable contract, not state.
    A receiver used as the base of a longer chain (``self.client.send()``),
    however, still requires the ``client`` instance attribute.  Explicit
    ``hasattr`` or ``AttributeError`` fallbacks are not hard requirements.
    """

    positional = [*function.args.posonlyargs, *function.args.args]
    if not positional:
        return {}
    receiver = positional[0].arg
    parents = _parents(function)
    reads: dict[str, tuple[int, bool]] = {}
    for node in _function_scope_nodes(function):
        if not (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == receiver
            and _attribute_is_read(node, parents)
        ):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        if _under_attribute_fallback(node, parents):
            continue
        conditional = _under_conditional_branch(node, parents)
        previous_line, previous_conditional = reads.get(node.attr, (node.lineno, True))
        reads[node.attr] = (
            min(previous_line, node.lineno),
            previous_conditional and conditional,
        )
    return reads


def _under_conditional_branch(
    node: ast.AST,
    parents: dict[int, ast.AST],
) -> bool:
    current = node
    parent = parents.get(id(current))
    while parent is not None:
        if isinstance(parent, ast.If) and current is not parent.test:
            return True
        if isinstance(parent, ast.IfExp) and current is not parent.test:
            return True
        if isinstance(parent, (ast.AsyncFor, ast.For, ast.Match, ast.While)):
            return True
        current = parent
        parent = parents.get(id(current))
    return False


def _snapshot_inherited_reads(
    snapshot: GitSnapshot,
    upstream_root: str,
    member: str,
) -> tuple[dict[str, tuple[int, bool]] | None, SourceEndpoint]:
    endpoint = snapshot.call_endpoint(
        f"{upstream_root}.{member}",
        "instance",
        receiver_type=upstream_root,
        member=member,
    )
    binding = snapshot._effective_member(upstream_root, member)
    if binding is None or binding.status == "unknown":
        return None, endpoint
    if binding.status != "exact":
        return {}, endpoint
    if not isinstance(binding.node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return None, endpoint
    return _instance_attribute_reads(binding.node), endpoint


def _downstream_class_binding_state(
    engine: InterfaceBoundaryGenerator,
    downstream_mro: tuple[str, ...],
    upstream_index: int,
    member: str,
) -> CompatibilityState | None:
    """Resolve a class-level fallback before the first upstream MRO owner."""

    for owner in downstream_mro[:upstream_index]:
        alternatives = engine._final_bindings(f"{owner}.{member}")
        if not alternatives:
            continue
        live = [
            alternative
            for alternative in alternatives
            if alternative.kind != "unbound"
            and not (isinstance(alternative.node, ast.AnnAssign) and alternative.node.value is None)
        ]
        if not live:
            continue
        if len(alternatives) == 1:
            return CompatibilityState(
                True,
                True,
                f"downstream class {owner} provides the required member",
            )
        return CompatibilityState(
            None,
            None,
            f"downstream class {owner} provides the member only on some source paths",
        )
    return None


def _initializer_has_unresolved_base_call(
    function: ast.AsyncFunctionDef | ast.FunctionDef,
    receiver: str,
) -> bool:
    for node in _function_scope_nodes(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "__init__":
            continue
        if (
            isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
            and not node.func.value.args
            and not node.func.value.keywords
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == receiver:
            return True
    return False


def _other_downstream_method_may_initialize(
    engine: InterfaceBoundaryGenerator,
    downstream_mro: tuple[str, ...],
    upstream_index: int,
    member: str,
) -> bool:
    for owner in downstream_mro[:upstream_index]:
        class_info = engine.downstream.find_class(owner)
        if class_info is None:
            continue
        for name, node in class_info.methods.items():
            if name == "__init__" or not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            positional = [*node.args.posonlyargs, *node.args.args]
            if positional and GitSnapshot._function_may_assign_instance_member(
                node,
                positional[0].arg,
                member,
            ):
                return True
    return False


def _downstream_initialization_state(
    engine: InterfaceBoundaryGenerator,
    new_snapshot: GitSnapshot,
    downstream_mro: tuple[str, ...],
    upstream_root: str,
    member: str,
) -> tuple[CompatibilityState, SourceEndpoint]:
    """Prove whether the effective downstream constructor establishes a field."""

    upstream_index = downstream_mro.index(upstream_root)
    class_binding = _downstream_class_binding_state(
        engine,
        downstream_mro,
        upstream_index,
        member,
    )
    if class_binding is not None:
        owner = downstream_mro[0]
        class_info = engine.downstream.find_class(owner)
        return class_binding, SourceEndpoint(
            class_info.file if class_info is not None else None,
            class_info.name if class_info is not None else owner.rsplit(".", 1)[-1],
            member,
            symbol_kind="class_attribute",
        )

    for dynamic_name in ("__getattr__", "__getattribute__"):
        resolution = engine._effective_method_resolution(
            downstream_mro[:upstream_index],
            dynamic_name,
        )
        if resolution.callable_owners:
            owner = resolution.callable_owners[0]
            callable_info = engine.downstream.find_callable(f"{owner}.{dynamic_name}")
            return CompatibilityState(
                None,
                None,
                f"downstream {dynamic_name} may provide the required member dynamically",
            ), SourceEndpoint(
                callable_info.file if callable_info is not None else None,
                owner.rsplit(".", 1)[-1],
                dynamic_name,
                getattr(callable_info.node, "lineno", None) if callable_info is not None else None,
                symbol_kind="callable",
            )

    def resolve_initializer(start: int) -> tuple[CompatibilityState, SourceEndpoint]:
        resolution = engine._effective_method_resolution(
            downstream_mro[start:],
            "__init__",
        )
        if not resolution.is_total_callable or len(resolution.callable_owners) != 1:
            owner = downstream_mro[start]
            return CompatibilityState(
                None,
                None,
                "effective constructor could not be proven through the complete MRO",
            ), SourceEndpoint(None, owner.rsplit(".", 1)[-1], "__init__", symbol_kind="unknown")
        owner = resolution.callable_owners[0]
        owner_index = downstream_mro.index(owner, start)
        if owner.startswith("vllm."):
            field = new_snapshot._effective_instance_field(upstream_root, member)
            if field is not None and field.status in {"exact", "non_callable"}:
                return CompatibilityState(
                    True,
                    True,
                    "the effective upstream constructor establishes the required attribute",
                ), SourceEndpoint(
                    field.file or None,
                    field.owner,
                    field.name,
                    getattr(field.node, "lineno", None),
                    descriptor="instance_attribute",
                    symbol_kind="attribute",
                )
            return CompatibilityState(
                None,
                None,
                "the upstream field initializer could not be proven",
            ), SourceEndpoint(None, owner.rsplit(".", 1)[-1], member, symbol_kind="unknown")

        callable_info = engine.downstream.find_callable(f"{owner}.__init__")
        node = callable_info.node if callable_info is not None else None
        endpoint = SourceEndpoint(
            callable_info.file if callable_info is not None else None,
            owner.rsplit(".", 1)[-1],
            "__init__",
            getattr(node, "lineno", None),
            descriptor=callable_info.descriptor_kind if callable_info is not None else None,
            symbol_kind="callable" if callable_info is not None else "unknown",
        )
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return CompatibilityState(
                None,
                None,
                f"downstream constructor {owner} is not an exact function definition",
            ), endpoint
        positional = [*node.args.posonlyargs, *node.args.args]
        if not positional:
            return CompatibilityState(
                None,
                None,
                f"downstream constructor {owner} has no provable instance receiver",
            ), endpoint
        receiver = positional[0].arg
        assignment = next(
            (
                candidate
                for statement in node.body
                if (candidate := GitSnapshot._instance_assignment(statement, receiver, member)) is not None
            ),
            None,
        )
        if assignment is not None:
            return CompatibilityState(
                True,
                True,
                f"downstream constructor {owner} assigns {member} on its main path",
            ), endpoint
        if GitSnapshot._statements_definitely_assign_instance_member(
            node.body,
            receiver,
            member,
        ):
            return CompatibilityState(
                True,
                True,
                f"downstream constructor {owner} assigns {member} on every normal path",
            ), endpoint
        if GitSnapshot._function_may_assign_instance_member(node, receiver, member):
            return CompatibilityState(
                None,
                None,
                f"downstream constructor {owner} assigns {member} only through unproven control flow",
            ), endpoint
        if GitSnapshot._calls_super_init(node):
            return resolve_initializer(owner_index + 1)
        if _initializer_has_unresolved_base_call(node, receiver):
            return CompatibilityState(
                None,
                None,
                f"downstream constructor {owner} invokes an explicit initializer that could not be resolved",
            ), endpoint
        if _other_downstream_method_may_initialize(
            engine,
            downstream_mro,
            upstream_index,
            member,
        ):
            return CompatibilityState(
                None,
                None,
                f"another downstream method may initialize {member}",
            ), endpoint
        return CompatibilityState(
            False,
            False,
            f"downstream constructor {owner} neither assigns {member} nor calls super().__init__()",
        ), endpoint

    return resolve_initializer(0)


def _inherited_state_findings(
    engine: InterfaceBoundaryGenerator,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
) -> tuple[list[RangeFinding], list[InheritedStateDependency]]:
    """Detect newly required upstream state missing from downstream constructors."""

    findings_by_id: dict[str, RangeFinding] = {}
    dependencies: list[InheritedStateDependency] = []
    old_read_cache: dict[
        tuple[str, str],
        tuple[dict[str, tuple[int, bool]] | None, SourceEndpoint],
    ] = {}
    new_read_cache: dict[str, dict[str, tuple[int, bool]]] = {}
    new_field_cache: dict[tuple[str, str], bool] = {}
    state_cache: dict[tuple[str, str, str], tuple[CompatibilityState, SourceEndpoint]] = {}

    for class_info in sorted(engine.downstream.classes.values(), key=lambda item: item.qualified_name):
        mro_result = engine._linearized_mro(class_info.qualified_name)
        if not mro_result.complete:
            continue
        downstream_mro = mro_result.owners
        upstream_owners = tuple(owner for owner in downstream_mro if owner.startswith("vllm."))
        if not upstream_owners:
            continue
        upstream_root = upstream_owners[0]
        initializer_resolution = engine._effective_method_resolution(
            downstream_mro,
            "__init__",
        )
        if (
            not initializer_resolution.is_total_callable
            or len(initializer_resolution.callable_owners) != 1
            or not initializer_resolution.callable_owners[0].startswith("vllm_ascend.")
        ):
            continue
        method_names = {
            method_name
            for owner in upstream_owners
            if (upstream_class := engine.upstream.find_class(owner)) is not None
            for method_name in upstream_class.methods
            if method_name not in {"__init__", "__new__"}
        }
        for method_name in sorted(method_names):
            resolution = engine._effective_method_resolution(downstream_mro, method_name)
            if not resolution.is_total_callable or len(resolution.callable_owners) != 1:
                continue
            inherited_owner = resolution.callable_owners[0]
            if not inherited_owner.startswith("vllm."):
                continue
            qualified_member = f"{inherited_owner}.{method_name}"
            callable_info = engine.upstream.find_callable(qualified_member)
            if callable_info is None:
                continue
            node = callable_info.node
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            if qualified_member not in new_read_cache:
                new_read_cache[qualified_member] = _instance_attribute_reads(node)
            new_reads = new_read_cache[qualified_member]
            if not new_reads:
                continue
            old_key = (upstream_root, method_name)
            if old_key not in old_read_cache:
                old_read_cache[old_key] = _snapshot_inherited_reads(
                    old_snapshot,
                    upstream_root,
                    method_name,
                )
            old_reads, old_endpoint = old_read_cache[old_key]
            new_endpoint = new_snapshot.endpoint(
                callable_info.file,
                callable_info.owner,
                callable_info.name,
            )
            for required_attribute, (read_line, conditional_read) in sorted(new_reads.items()):
                if old_reads is not None and required_attribute in old_reads:
                    continue
                field_key = (upstream_root, required_attribute)
                if field_key not in new_field_cache:
                    field = new_snapshot._effective_instance_field(upstream_root, required_attribute)
                    new_field_cache[field_key] = field is not None and field.status in {"exact", "non_callable"}
                if not new_field_cache[field_key]:
                    continue
                state_key = (class_info.qualified_name, upstream_root, required_attribute)
                if state_key not in state_cache:
                    state_cache[state_key] = _downstream_initialization_state(
                        engine,
                        new_snapshot,
                        downstream_mro,
                        upstream_root,
                        required_attribute,
                    )
                initialization_state, constructor = state_cache[state_key]
                if constructor.file is None or not constructor.owner:
                    continue
                condition_evidence: list[dict[str, object]] = []
                if conditional_read and initialization_state.compatible is False:
                    constructor_module, _ = _file_module(constructor.file)
                    constructor_info = engine.downstream.find_callable(
                        f"{constructor_module}.{constructor.owner}.__init__"
                    )
                    if constructor_info is not None and isinstance(
                        constructor_info.node, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        helpers: dict[str, ast.FunctionDef] = {}
                        for statement in constructor_info.node.body:
                            call = statement.value if isinstance(statement, ast.Expr) else None
                            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
                                continue
                            helper_resolution = engine._effective_method_resolution(downstream_mro, call.func.attr)
                            if not helper_resolution.is_total_callable or len(helper_resolution.callable_owners) != 1:
                                continue
                            helper_info = engine.downstream.find_callable(
                                f"{helper_resolution.callable_owners[0]}.{call.func.attr}"
                            )
                            if (
                                helper_info is not None
                                and helper_info.file == constructor_info.file
                                and isinstance(helper_info.node, ast.FunctionDef)
                            ):
                                helpers[call.func.attr] = helper_info.node
                        condition_evidence = conditional_state_evidence(
                            node,
                            constructor_info.node,
                            required_attribute,
                            read_line,
                            _tag_guard_names(engine.downstream.modules[constructor_info.module].tree.body),
                            helpers,
                        )
                conditional_unresolved = conditional_read and not condition_evidence
                new_state = (
                    CompatibilityState(
                        initialization_state.exists,
                        None,
                        f"{initialization_state.reason}; the inherited read is conditional",
                    )
                    if conditional_unresolved and initialization_state.compatible is False
                    else initialization_state
                )
                old_state = (
                    CompatibilityState(
                        None,
                        None,
                        "the old inherited callable binding could not be proven",
                    )
                    if old_reads is None
                    else CompatibilityState(
                        True,
                        True,
                        f"the old inherited callable did not read {required_attribute}",
                    )
                )
                classification = _classify(old_state, new_state, True)
                gates = {
                    "relationship_verified": True,
                    "contract_changed": True,
                    "runtime_reachable": not conditional_unresolved,
                    "version_lane_matches": True,
                }
                action = _finding_action(classification, gates)
                dependency = InheritedStateDependency(
                    downstream_class=class_info.qualified_name,
                    downstream_file=class_info.file,
                    upstream_root=upstream_root,
                    inherited_member=qualified_member,
                    required_attribute=required_attribute,
                    read_line=read_line,
                    read_condition="supported_conditional"
                    if condition_evidence
                    else "conditional"
                    if conditional_read
                    else "unconditional",
                    constructor_owner=constructor.owner,
                    constructor_file=constructor.file,
                    constructor_line=constructor.line or 0,
                    initialization_status=(
                        "established"
                        if initialization_state.compatible is True
                        else "missing"
                        if initialization_state.compatible is False
                        else "unknown"
                    ),
                    initialization_reason=initialization_state.reason,
                )
                dependencies.append(dependency)
                finding_id = _finding_id(
                    "inherited_state",
                    constructor.file,
                    constructor.owner,
                    required_attribute,
                    upstream_root,
                    old_snapshot.revision,
                    new_snapshot.revision,
                )
                candidate_finding = RangeFinding(
                    finding_id=finding_id,
                    classification=classification,
                    relation="inherited_state",
                    priority="P1" if action == "modify" else "P2",
                    action=action,
                    confidence="high" if classification != "analysis_unresolved" else "medium",
                    upstream_old=old_endpoint,
                    upstream_new=new_endpoint,
                    downstream=constructor,
                    old_state=old_state,
                    new_state=new_state,
                    change=(f"inherited upstream member newly reads required instance attribute {required_attribute}"),
                    evidence=[dependency.as_dict()],
                    gates=gates,
                    suggestion=(
                        f"Initialize {required_attribute} in the downstream constructor or call "
                        "super().__init__(), then add an inherited-state regression test."
                    ),
                    source="inherited_state_detector",
                    contract_kind="required_instance_attribute",
                    direction="upstream_inherited_read_to_downstream_state",
                    details={
                        "upstream_root": upstream_root,
                        "inherited_member": qualified_member,
                        "required_attribute": required_attribute,
                        "read_line": read_line,
                        "read_condition": dependency.read_condition,
                        "condition_evidence": condition_evidence,
                        "downstream_class": class_info.qualified_name,
                        "initialization_status": dependency.initialization_status,
                        "initialization_reason": dependency.initialization_reason,
                        "inherited_members": [qualified_member],
                        "impacted_downstream_classes": [class_info.qualified_name],
                    },
                )
                previous = findings_by_id.get(finding_id)
                if previous is None:
                    findings_by_id[finding_id] = candidate_finding
                    continue
                inherited_members = set(previous.details["inherited_members"])
                inherited_members.add(qualified_member)
                impacted_classes = set(previous.details["impacted_downstream_classes"])
                impacted_classes.add(class_info.qualified_name)
                retained = (
                    candidate_finding
                    if CLASSIFICATIONS.index(candidate_finding.classification)
                    < CLASSIFICATIONS.index(previous.classification)
                    else previous
                )
                retained.evidence = [*previous.evidence, dependency.as_dict()]
                retained.details["inherited_members"] = sorted(inherited_members)
                retained.details["impacted_downstream_classes"] = sorted(impacted_classes)
                findings_by_id[finding_id] = retained
    return list(findings_by_id.values()), dependencies


def _dataclass_definition_findings(
    engine: InterfaceBoundaryGenerator,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
) -> list[RangeFinding]:
    """Compare class-definition legality against the pinned downstream source."""

    def lookup(snapshot: GitSnapshot, reference: str) -> ClassSource | None:
        if reference.startswith("vllm."):
            binding = snapshot._resolve_qualified_node(reference)
            if binding is None or binding.status != "exact" or not isinstance(binding.node, ast.ClassDef):
                return None
            return ClassSource(binding.node, snapshot._return_resolver(binding.file), binding.file, reference)
        callable_info = engine.downstream.find_callable(reference)
        if callable_info is None or not isinstance(callable_info.node, ast.ClassDef):
            return None
        module = callable_info.module
        return ClassSource(
            callable_info.node,
            lambda expression: engine.downstream.canonical_name(
                engine.downstream.resolve_reference(module, expression)
            ),
            callable_info.file,
            reference,
        )

    findings: list[RangeFinding] = []
    for reference in sorted(engine.downstream.classes):
        old_layout = dataclass_layout(reference, lambda name: lookup(old_snapshot, name))
        new_layout = dataclass_layout(reference, lambda name: lookup(new_snapshot, name))
        if old_layout is None or new_layout is None or old_layout.fields == new_layout.fields:
            continue
        old_error = old_layout.ordering_error()
        new_error = new_layout.ordering_error()
        if new_error is None:
            continue
        default, required = new_error
        if not default.owner.startswith("vllm.") or not required.owner.startswith("vllm_ascend."):
            continue
        source = lookup(new_snapshot, reference)
        if source is None:
            continue
        old_state = CompatibilityState(True, old_error is None, "old dataclass initializer field order")
        new_state = CompatibilityState(
            True, False, f"non-default argument {required.name!r} follows default argument {default.name!r}"
        )
        classification = _classify(old_state, new_state, True)
        gates = {
            "relationship_verified": True,
            "contract_changed": True,
            "runtime_reachable": True,
            "version_lane_matches": True,
        }
        action = _finding_action(classification, gates)
        old_base = old_snapshot._resolve_qualified_node(default.owner)
        new_base = new_snapshot._resolve_qualified_node(default.owner)
        if old_base is None or new_base is None:
            continue
        findings.append(
            RangeFinding(
                finding_id=_finding_id(
                    "dataclass_definition", reference, required.name, old_snapshot.revision, new_snapshot.revision
                ),
                classification=classification,
                relation="inheritance",
                priority="P1" if action == "modify" else "P2",
                action=action,
                confidence="high",
                upstream_old=old_snapshot.endpoint(old_base.file, old_base.owner, old_base.name),
                upstream_new=new_snapshot.endpoint(new_base.file, new_base.owner, new_base.name),
                downstream=SourceEndpoint(source.file, None, source.node.name, required.line, symbol_kind="class"),
                old_state=old_state,
                new_state=new_state,
                change="upstream dataclass defaults invalidate the downstream generated initializer",
                evidence=[
                    {
                        "upstream_default": default.name,
                        "upstream_line": default.line,
                        "downstream_required": required.name,
                        "downstream_line": required.line,
                    }
                ],
                gates=gates,
                suggestion=(
                    "Align the downstream dataclass field defaults or keyword-only contract and test class creation."
                ),
                source="dataclass_definition_detector",
                contract_kind="dataclass_field_order",
                direction="upstream_dataclass_layout_to_downstream_definition",
                details={
                    "upstream_default": default.name,
                    "downstream_required": required.name,
                    "old_fields": [field.__dict__ for field in old_layout.fields],
                    "new_fields": [field.__dict__ for field in new_layout.fields],
                },
            )
        )
    return findings


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


def _verified_historical_direct_attributes(
    candidates: Iterable[DirectAttributeDependency],
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
) -> list[DirectAttributeDependency]:
    """Promote old-proven self/super reads when the new binding is absent or unresolved."""

    verified: list[DirectAttributeDependency] = []
    for candidate in candidates:
        if candidate.lookup_root is None or candidate.member is None:
            continue
        old_endpoint = old_snapshot.attribute_endpoint(
            candidate.target,
            candidate.access_kind,
            receiver_type=candidate.lookup_root,
            member=candidate.member,
        )
        new_endpoint = new_snapshot.attribute_endpoint(
            candidate.target,
            candidate.access_kind,
            receiver_type=candidate.lookup_root,
            member=candidate.member,
        )
        if (
            _direct_attribute_state(old_endpoint).compatible is True
            and _direct_attribute_state(new_endpoint).compatible is not True
        ):
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
) -> tuple[
    list[DirectCallDependency],
    list[DirectAttributeDependency],
    list[dict[str, Any]],
]:
    """Validate exact call, member-presence, and return contracts for one source pair."""

    plan = plan or resolve_analysis_plan()
    factory_snapshot = _ConstructorSnapshot(snapshot, engine.downstream, flow_engine=engine)
    receiver_flow_cache_key = object()
    discovered_dependencies = (
        DirectCallDetector(
            engine,
            factory_evaluator=factory_snapshot.evaluate_factory_call,
            factory_input_resolver=factory_snapshot.factory_call_inputs,
            receiver_flow_cache_key=receiver_flow_cache_key,
        ).discover()
        if plan.analyze_direct_calls
        else []
    )
    discovered_attributes = (
        DirectAttributeDetector(
            engine,
            factory_evaluator=factory_snapshot.evaluate_factory_call,
            factory_input_resolver=factory_snapshot.factory_call_inputs,
            receiver_flow_cache_key=receiver_flow_cache_key,
        ).discover()
        if plan.analyze_direct_attributes
        else []
    )
    dependencies: list[DirectCallDependency] = []
    attribute_dependencies: list[DirectAttributeDependency] = []
    findings: list[dict[str, Any]] = []
    constructor_snapshot = (
        _ConstructorSnapshot(snapshot, engine.downstream)
        if any(
            item.resolution_basis in {"downstream_constructor", "dataclass_replace"} or item.receiver_path
            for item in discovered_dependencies
        )
        or any(item.receiver_path for item in discovered_attributes)
        else snapshot
    )
    for dependency in discovered_dependencies:
        endpoint_snapshot = (
            constructor_snapshot
            if dependency.resolution_basis in {"downstream_constructor", "dataclass_replace"}
            or dependency.receiver_path
            else snapshot
        )
        upstream = endpoint_snapshot.dependency_endpoint(dependency)
        dependencies.append(dependency)
        argument_state = endpoint_snapshot.call_state(dependency, upstream)
        if argument_state.compatible is not True:
            argument_state = _dispatch_state(argument_state, dependency.receiver_binding)
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
                return_state = _dispatch_state(return_state, dependency.receiver_binding)
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

    for attribute_dependency in discovered_attributes:
        endpoint_snapshot = constructor_snapshot if attribute_dependency.receiver_path else snapshot
        upstream = endpoint_snapshot.dependency_endpoint(attribute_dependency)
        attribute_dependencies.append(attribute_dependency)
        state = _direct_attribute_state(upstream)
        if state.compatible is True:
            continue
        state = _dispatch_state(state, attribute_dependency.receiver_binding)
        findings.append(
            {
                "relation": "direct_attribute",
                "contract_kind": "attribute_presence",
                "status": "risk" if state.compatible is False else "review",
                "upstream": upstream.as_dict(),
                "downstream": {
                    "file": attribute_dependency.file,
                    "owner": attribute_dependency.owner,
                    "name": attribute_dependency.expression,
                    "line": attribute_dependency.line,
                },
                "reason": state.reason,
                "evidence": attribute_dependency.as_dict(),
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
    return dependencies, attribute_dependencies, ordered


def _cache_bypass_event(component: str, commit_sha: str, reason: str) -> CacheResult:
    return CacheResult(
        component=component,
        enabled=True,
        status="bypassed",
        commit_sha=commit_sha,
        reason=reason,
    )


def _snapshot_cache_identity(
    root: Path,
    revision: str,
) -> dict[str, object]:
    return build_identity(
        component="upstream_snapshot",
        repo_root=root,
        commit_sha=revision,
        analyzer_version=RANGE_ANALYZER_VERSION,
        component_schema=SNAPSHOT_CACHE_SCHEMA_VERSION,
        config={
            "known_transparent_decorators": sorted(_KNOWN_TRANSPARENT_SIGNATURE_DECORATORS),
            "known_wrapped_decorators": sorted(_KNOWN_WRAPS_SIGNATURE_DECORATORS),
        },
    )


def _load_snapshot(
    cache: PersistentCache,
    root: Path,
    revision: str,
    *,
    cache_safe: bool,
    unsafe_reason: str,
) -> tuple[GitSnapshot, CacheResult]:
    identity = _snapshot_cache_identity(root, revision)
    if not cache_safe and cache.enabled:
        result = _cache_bypass_event("upstream_snapshot", revision, unsafe_reason)
        cache.events.append(result)
        return GitSnapshot(root, revision), result
    payload, result = cache.load(
        "upstream_snapshot",
        identity,
        validator=lambda item: (
            isinstance(item, GitSnapshot) and item.revision == revision and item.root == root.resolve()
        ),
    )
    if isinstance(payload, GitSnapshot):
        payload.cache_work_seconds = 0.0
        return payload, result
    return GitSnapshot(root, revision), result


def _store_snapshot(
    cache: PersistentCache,
    snapshot: GitSnapshot,
    result: CacheResult,
    *,
    cache_safe: bool,
) -> None:
    if not cache_safe or result.status == "hit":
        return
    cache.store(
        "upstream_snapshot",
        _snapshot_cache_identity(snapshot.root, snapshot.revision),
        snapshot,
        build_seconds=snapshot.cache_work_seconds,
        result=result,
    )


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
    analysis_workers: int = 3,
    downstream_index_cache_dir: Path | None = None,
    upstream_file_index_cache_dir: Path | None = None,
    index_workers: int = 1,
    cache_dir: Path | None = None,
    cache_enabled: bool = True,
) -> dict[str, Any]:
    """Run analysis and release all lazy Git readers even on failure."""
    with ExitStack() as resources:
        return _analyze_range(
            vllm_root=vllm_root,
            ascend_root=ascend_root,
            old=old,
            new=new,
            expect_ascend_sha=expect_ascend_sha,
            external_roots=external_roots,
            external_shas=external_shas,
            profile=profile,
            scenario=scenario,
            analysis_workers=analysis_workers,
            downstream_index_cache_dir=downstream_index_cache_dir,
            upstream_file_index_cache_dir=upstream_file_index_cache_dir,
            index_workers=index_workers,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            resources=resources,
        )


def _analyze_range(
    *,
    resources: ExitStack,
    vllm_root: Path,
    ascend_root: Path,
    old: str,
    new: str,
    expect_ascend_sha: str,
    external_roots: dict[str, Path] | None = None,
    external_shas: dict[str, str] | None = None,
    profile: str = "exact-contracts",
    scenario: str = MAIN2MAIN_SCENARIO,
    analysis_workers: int = 3,
    downstream_index_cache_dir: Path | None = None,
    upstream_file_index_cache_dir: Path | None = None,
    index_workers: int = 1,
    cache_dir: Path | None = None,
    cache_enabled: bool = True,
) -> dict[str, Any]:
    """Run the selected source-analysis plan for an exact vLLM range."""
    analysis_started = time.perf_counter()
    phase_started = time.perf_counter()
    timings: dict[str, float | None] = {}
    plan = resolve_analysis_plan(scenario)
    if analysis_workers < 1:
        raise ValueError("analysis_workers must be at least 1")
    if index_workers < 1:
        raise ValueError("index_workers must be at least 1")
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

    persistent_cache = PersistentCache(cache_dir, enabled=cache_enabled)
    upstream_cache_safe, upstream_cache_reason = git_source_state(vllm_root, "vllm")
    downstream_cache_safe, downstream_cache_reason = git_source_state(
        ascend_root,
        "vllm_ascend",
    )
    if not cache_enabled:
        downstream_index_cache_dir = None
        upstream_file_index_cache_dir = None
    elif persistent_cache.enabled and persistent_cache.root is not None:
        downstream_index_cache_dir = downstream_index_cache_dir or (
            persistent_cache.root / "repository-index" / "downstream"
        )
        upstream_file_index_cache_dir = upstream_file_index_cache_dir or (
            persistent_cache.root / "repository-index" / "upstream-fragments"
        )

    generator = InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        external_roots,
        source_versions={"vllm": new_sha, "vllm_ascend": ascend_sha, **external_shas},
        downstream_index_cache_dir=downstream_index_cache_dir,
        upstream_file_index_cache_dir=upstream_file_index_cache_dir,
        index_workers=index_workers,
    )
    phase_started = _diagnostic_timing("repository_indexing", phase_started, timings)
    timings.update(
        {f"repository_indexing.{name}": duration for name, duration in generator.repository_index_timings.items()}
    )
    relation_identity = build_identity(
        component="downstream_relations",
        repo_root=ascend_root,
        commit_sha=ascend_sha,
        analyzer_version=GENERATOR_VERSION,
        component_schema=RELATION_CACHE_SCHEMA_VERSION,
        config={
            "analysis_plan": plan.as_dict(),
            "vllm_repo_root": normalized_repo_path(vllm_root),
            "vllm_new_sha": new_sha,
            "external_roots": {package: normalized_repo_path(root) for package, root in sorted(external_roots.items())},
            "external_shas": dict(sorted(external_shas.items())),
        },
    )
    relation_cache_safe = upstream_cache_safe and downstream_cache_safe
    if relation_cache_safe or not persistent_cache.enabled:
        relation_payload, relation_cache_result = persistent_cache.load(
            "downstream_relations",
            relation_identity,
            validator=lambda item: (
                isinstance(item, dict)
                and isinstance(item.get("relations"), list)
                and isinstance(item.get("findings"), list)
                and isinstance(item.get("historical_override_candidates"), list)
            ),
        )
    else:
        relation_payload = None
        relation_cache_result = _cache_bypass_event(
            "downstream_relations",
            ascend_sha,
            f"upstream: {upstream_cache_reason}; downstream: {downstream_cache_reason}",
        )
        persistent_cache.events.append(relation_cache_result)
    if isinstance(relation_payload, dict):
        relations = relation_payload["relations"]
        generator.relations = relations
        generator_findings = relation_payload["findings"]
        generator.historical_override_candidates = relation_payload["historical_override_candidates"]
        cached_phase_timings = relation_payload.get("phase_timings")
        if isinstance(cached_phase_timings, dict):
            generator.phase_timings = cached_phase_timings
    else:
        relation_started = time.perf_counter()
        relations, generator_findings = generator.generate(plan)
        relation_build_seconds = time.perf_counter() - relation_started
        if relation_cache_safe:
            persistent_cache.store(
                "downstream_relations",
                relation_identity,
                {
                    "relations": relations,
                    "findings": generator_findings,
                    "historical_override_candidates": generator.historical_override_candidates,
                    "phase_timings": generator.phase_timings,
                },
                build_seconds=relation_build_seconds,
                result=relation_cache_result,
            )
    timings.update({f"relation_generation.{name}": duration for name, duration in generator.phase_timings.items()})
    phase_started = time.perf_counter()
    old_snapshot, old_snapshot_cache = _load_snapshot(
        persistent_cache,
        vllm_root,
        old_sha,
        cache_safe=upstream_cache_safe,
        unsafe_reason=upstream_cache_reason,
    )
    resources.callback(old_snapshot.close)
    new_snapshot, new_snapshot_cache = _load_snapshot(
        persistent_cache,
        vllm_root,
        new_sha,
        cache_safe=upstream_cache_safe,
        unsafe_reason=upstream_cache_reason,
    )
    resources.callback(new_snapshot.close)
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

    import_identity = build_identity(
        component="downstream_direct_imports",
        repo_root=ascend_root,
        commit_sha=ascend_sha,
        analyzer_version=RANGE_ANALYZER_VERSION,
        component_schema=DIRECT_IMPORT_CACHE_SCHEMA_VERSION,
        config={"analysis_plan": plan.as_dict()},
    )
    direct_call_identity = build_identity(
        component="downstream_direct_calls",
        repo_root=ascend_root,
        commit_sha=ascend_sha,
        analyzer_version=f"{GENERATOR_VERSION}/{RANGE_ANALYZER_VERSION}",
        component_schema=DIRECT_CALL_CACHE_SCHEMA_VERSION,
        config={
            "analysis_plan": plan.as_dict(),
            "vllm_repo_root": normalized_repo_path(vllm_root),
            "vllm_old_sha": old_sha,
            "vllm_new_sha": new_sha,
        },
    )
    direct_attribute_identity = build_identity(
        component="downstream_direct_attributes",
        repo_root=ascend_root,
        commit_sha=ascend_sha,
        analyzer_version=f"{GENERATOR_VERSION}/{RANGE_ANALYZER_VERSION}",
        component_schema=DIRECT_ATTRIBUTE_CACHE_SCHEMA_VERSION,
        config={
            "analysis_plan": plan.as_dict(),
            "vllm_repo_root": normalized_repo_path(vllm_root),
            "vllm_old_sha": old_sha,
            "vllm_new_sha": new_sha,
        },
    )
    new_receiver_flow_cache_key = object()
    old_receiver_flow_cache_key = object()

    discovery_metrics: dict[str, dict[str, Any]] = {}

    def analyze_relations() -> tuple[list[RangeFinding], float]:
        started = time.perf_counter()
        branch_findings = [
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
                strict_optional_contracts=plan.scenario == MAIN2MAIN_SCENARIO,
            )
        ]
        if plan.analyze_inheritance:
            branch_findings.extend(_dataclass_definition_findings(generator, old_snapshot, new_snapshot))
        return branch_findings, time.perf_counter() - started

    def analyze_imports() -> tuple[list[RangeFinding], float]:
        started = time.perf_counter()
        if downstream_cache_safe or not persistent_cache.enabled:
            cached_imports, import_cache_result = persistent_cache.load(
                "downstream_direct_imports",
                import_identity,
                validator=lambda item: (
                    isinstance(item, list) and all(isinstance(reference, ImportReference) for reference in item)
                ),
            )
        else:
            cached_imports = None
            import_cache_result = _cache_bypass_event(
                "downstream_direct_imports",
                ascend_sha,
                downstream_cache_reason,
            )
            persistent_cache.events.append(import_cache_result)
        if isinstance(cached_imports, list):
            import_references = cached_imports
        else:
            discovery_started = time.perf_counter()
            import_references = discover_imports(
                ascend_root, trees={m.file: m.tree for m in generator.downstream.modules.values()}
            )
            discovery_seconds = time.perf_counter() - discovery_started
            if downstream_cache_safe:
                persistent_cache.store(
                    "downstream_direct_imports",
                    import_identity,
                    import_references,
                    build_seconds=discovery_seconds,
                    result=import_cache_result,
                )
        branch_findings = _import_findings(
            ascend_root,
            old_snapshot,
            new_snapshot,
            old_to_new,
            import_references,
        )
        return branch_findings, time.perf_counter() - started

    def analyze_direct_calls() -> tuple[
        list[RangeFinding],
        list[DirectCallDependency],
        float,
        float,
    ]:
        discovery_started = time.perf_counter()
        if relation_cache_safe or not persistent_cache.enabled:
            cached_calls, direct_call_cache_result = persistent_cache.load(
                "downstream_direct_calls",
                direct_call_identity,
                validator=lambda item: (
                    isinstance(item, dict)
                    and isinstance(item.get("dependencies"), list)
                    and all(isinstance(dependency, DirectCallDependency) for dependency in item.get("dependencies", []))
                    and isinstance(item.get("historical_candidates"), list)
                ),
            )
        else:
            cached_calls = None
            direct_call_cache_result = _cache_bypass_event(
                "downstream_direct_calls",
                ascend_sha,
                f"upstream: {upstream_cache_reason}; downstream: {downstream_cache_reason}",
            )
            persistent_cache.events.append(direct_call_cache_result)
        if isinstance(cached_calls, dict):
            discovered_direct_calls = cached_calls["dependencies"]
            historical_candidates = cached_calls["historical_candidates"]
        else:
            call_old_factory = _ConstructorSnapshot(old_snapshot, generator.downstream, flow_engine=generator)
            call_new_factory = _ConstructorSnapshot(new_snapshot, generator.downstream, flow_engine=generator)
            direct_call_detector = DirectCallDetector(
                generator,
                historical_type_source=old_snapshot._type_source,
                factory_evaluator=call_new_factory.evaluate_factory_call,
                historical_factory_evaluator=call_old_factory.evaluate_factory_call,
                factory_input_resolver=call_new_factory.factory_call_inputs,
                historical_factory_input_resolver=call_old_factory.factory_call_inputs,
                receiver_flow_cache_key=new_receiver_flow_cache_key,
                historical_receiver_flow_cache_key=old_receiver_flow_cache_key,
            )
            discovered_direct_calls = direct_call_detector.discover()
            discovery_metrics["direct_calls"] = direct_call_detector.metrics()
            historical_candidates = direct_call_detector.historical_candidates
            if relation_cache_safe:
                persistent_cache.store(
                    "downstream_direct_calls",
                    direct_call_identity,
                    {
                        "dependencies": discovered_direct_calls,
                        "historical_candidates": historical_candidates,
                    },
                    build_seconds=time.perf_counter() - discovery_started,
                    result=direct_call_cache_result,
                )
        discovered_direct_calls.extend(
            _verified_historical_direct_calls(
                historical_candidates,
                old_snapshot,
                new_snapshot,
            )
        )
        discovery_elapsed = time.perf_counter() - discovery_started
        comparison_started = time.perf_counter()
        branch_findings, dependencies = _direct_call_findings(
            discovered_direct_calls,
            old_snapshot,
            new_snapshot,
            old_to_new,
            downstream_index=generator.downstream,
        )
        return (
            branch_findings,
            dependencies,
            discovery_elapsed,
            time.perf_counter() - comparison_started,
        )

    def analyze_direct_attributes() -> tuple[
        list[RangeFinding],
        list[DirectAttributeDependency],
        float,
        float,
    ]:
        discovery_started = time.perf_counter()
        if relation_cache_safe or not persistent_cache.enabled:
            cached_attributes, direct_attribute_cache_result = persistent_cache.load(
                "downstream_direct_attributes",
                direct_attribute_identity,
                validator=lambda item: (
                    isinstance(item, dict)
                    and isinstance(item.get("dependencies"), list)
                    and all(
                        isinstance(dependency, DirectAttributeDependency) for dependency in item.get("dependencies", [])
                    )
                    and isinstance(item.get("historical_candidates"), list)
                    and all(
                        isinstance(dependency, DirectAttributeDependency)
                        for dependency in item.get("historical_candidates", [])
                    )
                ),
            )
        else:
            cached_attributes = None
            direct_attribute_cache_result = _cache_bypass_event(
                "downstream_direct_attributes",
                ascend_sha,
                f"upstream: {upstream_cache_reason}; downstream: {downstream_cache_reason}",
            )
            persistent_cache.events.append(direct_attribute_cache_result)
        if isinstance(cached_attributes, dict):
            discovered_attributes = cached_attributes["dependencies"]
            historical_candidates = cached_attributes["historical_candidates"]
        else:
            attribute_old_factory = _ConstructorSnapshot(old_snapshot, generator.downstream, flow_engine=generator)
            attribute_new_factory = _ConstructorSnapshot(new_snapshot, generator.downstream, flow_engine=generator)
            direct_attribute_detector = DirectAttributeDetector(
                generator,
                historical_type_source=old_snapshot._type_source,
                factory_evaluator=attribute_new_factory.evaluate_factory_call,
                historical_factory_evaluator=attribute_old_factory.evaluate_factory_call,
                factory_input_resolver=attribute_new_factory.factory_call_inputs,
                historical_factory_input_resolver=attribute_old_factory.factory_call_inputs,
                receiver_flow_cache_key=new_receiver_flow_cache_key,
                historical_receiver_flow_cache_key=old_receiver_flow_cache_key,
            )
            discovered_attributes = direct_attribute_detector.discover()
            discovery_metrics["direct_attributes"] = direct_attribute_detector.metrics()
            historical_candidates = direct_attribute_detector.historical_attribute_candidates
            if relation_cache_safe:
                persistent_cache.store(
                    "downstream_direct_attributes",
                    direct_attribute_identity,
                    {
                        "dependencies": discovered_attributes,
                        "historical_candidates": historical_candidates,
                    },
                    build_seconds=time.perf_counter() - discovery_started,
                    result=direct_attribute_cache_result,
                )
        discovered_attributes.extend(
            _verified_historical_direct_attributes(
                historical_candidates,
                old_snapshot,
                new_snapshot,
            )
        )
        discovery_elapsed = time.perf_counter() - discovery_started
        comparison_started = time.perf_counter()
        branch_findings, dependencies = _direct_attribute_findings(
            discovered_attributes,
            old_snapshot,
            new_snapshot,
            downstream_index=generator.downstream,
        )
        return (
            branch_findings,
            dependencies,
            discovery_elapsed,
            time.perf_counter() - comparison_started,
        )

    def analyze_inherited_state() -> tuple[
        list[RangeFinding],
        list[InheritedStateDependency],
        float,
    ]:
        started = time.perf_counter()
        branch_findings, dependencies = _inherited_state_findings(
            generator,
            old_snapshot,
            new_snapshot,
        )
        return branch_findings, dependencies, time.perf_counter() - started

    branch_count = (
        1
        + int(plan.analyze_direct_imports)
        + int(plan.analyze_direct_calls)
        + int(plan.analyze_direct_attributes)
        + int(plan.analyze_inherited_state)
    )
    effective_workers = min(analysis_workers, branch_count)
    branches_started = time.perf_counter()
    if effective_workers > 1:
        with ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="vllm-interface",
        ) as executor:
            relation_future = executor.submit(analyze_relations)
            import_future = executor.submit(analyze_imports) if plan.analyze_direct_imports else None
            direct_call_future = executor.submit(analyze_direct_calls) if plan.analyze_direct_calls else None
            direct_attribute_future = (
                executor.submit(analyze_direct_attributes) if plan.analyze_direct_attributes else None
            )
            inherited_state_future = executor.submit(analyze_inherited_state) if plan.analyze_inherited_state else None
            relation_result = relation_future.result()
            import_result = import_future.result() if import_future is not None else None
            direct_call_result = direct_call_future.result() if direct_call_future is not None else None
            direct_attribute_result = direct_attribute_future.result() if direct_attribute_future is not None else None
            inherited_state_result = inherited_state_future.result() if inherited_state_future is not None else None
    else:
        relation_result = analyze_relations()
        import_result = analyze_imports() if plan.analyze_direct_imports else None
        direct_call_result = analyze_direct_calls() if plan.analyze_direct_calls else None
        direct_attribute_result = analyze_direct_attributes() if plan.analyze_direct_attributes else None
        inherited_state_result = analyze_inherited_state() if plan.analyze_inherited_state else None

    _record_diagnostic_timing("analysis_branches_wall", time.perf_counter() - branches_started, timings)
    findings, relation_elapsed = relation_result
    _record_diagnostic_timing("relation_comparison", relation_elapsed, timings)
    if import_result is not None:
        import_findings, import_elapsed = import_result
        findings.extend(import_findings)
        _record_diagnostic_timing("direct_import_analysis", import_elapsed, timings)
    else:
        timings["direct_import_analysis"] = None

    direct_call_dependencies: list[DirectCallDependency] = []
    if direct_call_result is not None:
        direct_call_findings, direct_call_dependencies, discovery_elapsed, comparison_elapsed = direct_call_result
        findings.extend(direct_call_findings)
        _record_diagnostic_timing("direct_call_discovery", discovery_elapsed, timings)
        _record_diagnostic_timing("direct_call_comparison", comparison_elapsed, timings)
    else:
        timings["direct_call_discovery"] = None
        timings["direct_call_comparison"] = None

    direct_attribute_dependencies: list[DirectAttributeDependency] = []
    if direct_attribute_result is not None:
        (
            direct_attribute_findings,
            direct_attribute_dependencies,
            discovery_elapsed,
            comparison_elapsed,
        ) = direct_attribute_result
        findings.extend(direct_attribute_findings)
        _record_diagnostic_timing("direct_attribute_discovery", discovery_elapsed, timings)
        _record_diagnostic_timing("direct_attribute_comparison", comparison_elapsed, timings)
    else:
        timings["direct_attribute_discovery"] = None
        timings["direct_attribute_comparison"] = None

    inherited_state_dependencies: list[InheritedStateDependency] = []
    if inherited_state_result is not None:
        inherited_state_findings, inherited_state_dependencies, inherited_state_elapsed = inherited_state_result
        findings.extend(inherited_state_findings)
        _record_diagnostic_timing("inherited_state_comparison", inherited_state_elapsed, timings)
    else:
        timings["inherited_state_comparison"] = None

    generator_finding_started = time.perf_counter()
    exact_relations_by_signature_key: dict[
        tuple[str, str, str | None, str],
        list[Relation],
    ] = {}
    for relation in relations:
        if relation.upstream_package != "vllm" or relation.relation not in plan.relation_types:
            continue
        exact_relations_by_signature_key.setdefault(
            (
                relation.relation,
                relation.downstream_file,
                relation.downstream_owner,
                relation.downstream_name,
            ),
            [],
        ).append(relation)
    exact_relation_signature_keys = {
        (
            finding.relation,
            finding.downstream.file,
            finding.downstream.owner,
            finding.downstream.name,
        )
        for finding in findings
        if finding.source == "dynamic_relation_graph"
        and finding.contract_kind == "call_arguments"
        and finding.old_state.compatible is not None
        and finding.new_state.compatible is not None
    }
    for candidate in generator_findings if plan.include_generator_findings else []:
        if candidate.status not in {"review", "risk"}:
            continue
        candidate_signature_key = (
            candidate.relation,
            candidate.downstream_file,
            candidate.downstream_owner,
            candidate.downstream_name,
        )
        if (
            candidate.supplemental
            and candidate.reason_code == "signature_incompatible"
            and candidate_signature_key in exact_relation_signature_keys
        ):
            # The range finding already owns the exact old/new compatibility
            # conclusion. Keeping the current-snapshot supplemental risk would
            # duplicate the same root cause as an unresolved P2 review.
            continue
        if candidate.supplemental and candidate.reason_code == "signature_incompatible":
            exact_relations = exact_relations_by_signature_key.get(candidate_signature_key, [])
            evidence_matches = [
                relation
                for relation in exact_relations
                if any(evidence.target_expression == candidate.target_expression for evidence in relation.evidence)
            ]
            if evidence_matches:
                exact_relations = evidence_matches
            elif len(exact_relations) != 1:
                exact_relations = []
            converted_preexisting = False
            for relation in exact_relations:
                invocation_kind = (
                    _TRITON_KERNEL_PROTOCOL
                    if relation.upstream_signature_contract is not None
                    and relation.upstream_signature_contract.protocol == _TRITON_KERNEL_PROTOCOL
                    else "python_call"
                )
                old_relation_endpoint, new_relation_endpoint = _relation_endpoints(
                    relation,
                    old_snapshot,
                    new_snapshot,
                    new_to_old,
                    invocation_kind,
                )
                old_contract = old_snapshot.signature_contract(old_relation_endpoint, invocation_kind)
                new_contract = new_snapshot.signature_contract(new_relation_endpoint, invocation_kind)
                if _relation_contract_changed(
                    old_relation_endpoint,
                    new_relation_endpoint,
                    old_contract,
                    new_contract,
                ):
                    continue
                downstream = _relation_downstream_endpoint(relation, generator)
                old_state = _state(
                    old_relation_endpoint,
                    downstream.signature,
                    relation.relation,
                    downstream.descriptor,
                    old_contract,
                )
                new_state = _state(
                    new_relation_endpoint,
                    downstream.signature,
                    relation.relation,
                    downstream.descriptor,
                    new_contract,
                )
                if old_state.compatible is not False or new_state.compatible is not False:
                    continue
                evidence = [item.as_dict() for item in relation.evidence] or [
                    {"file": relation.evidence_file, "line": relation.evidence_line}
                ]
                findings.append(
                    RangeFinding(
                        finding_id=_finding_id(
                            relation.exact_key(),
                            "call_arguments",
                            old_snapshot.revision,
                            new_snapshot.revision,
                        ),
                        classification="preexisting",
                        relation=relation.relation,
                        priority="P2",
                        action="review",
                        confidence="high",
                        upstream_old=old_relation_endpoint,
                        upstream_new=new_relation_endpoint,
                        downstream=downstream,
                        old_state=old_state,
                        new_state=new_state,
                        change=(
                            "upstream contract is unchanged; the installed downstream callable remains incompatible"
                        ),
                        evidence=evidence,
                        gates={
                            "relationship_verified": True,
                            "contract_changed": False,
                            "runtime_reachable": True,
                            "version_lane_matches": True,
                        },
                        suggestion=(
                            "This incompatibility predates the selected range. Track it as baseline debt rather than "
                            "attributing it to the current upgrade."
                        ),
                        contract_kind="call_arguments",
                        direction="upstream_contract_to_downstream_implementation",
                        details={
                            "installed_signature": downstream.signature,
                            "installed_descriptor": downstream.descriptor,
                            "invocation_protocol": invocation_kind,
                            "unchanged_preexisting_contract": True,
                        },
                    )
                )
                converted_preexisting = True
                break
            if converted_preexisting:
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
                    else (
                        "Static evidence is insufficient. Confirm the target binding or extend the analysis rule "
                        "before changing downstream code."
                    )
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
    finding_payloads = [item.as_dict() for item in ordered]
    for payload in finding_payloads:
        payload["root_cause_id"] = _finding_id(
            "root_cause",
            _root_cause_key(payload),
            old_sha,
            new_sha,
        )
    introduced_payloads = [item for item in finding_payloads if item["classification"] == "introduced_break"]
    actionable_payloads = [item for item in introduced_payloads if item["action"] == "modify"]
    review_payloads = [item for item in finding_payloads if item["action"] == "review"]
    counts = Counter(item.classification for item in ordered)
    action_counts = Counter(item.action for item in ordered)
    relation_counts = Counter(item.relation for item in ordered)
    contract_counts = Counter(item.contract_kind for item in ordered)
    analyzed_relation_count = sum(
        relation.upstream_package == "vllm" and relation.relation in plan.relation_types for relation in relations
    )
    _store_snapshot(
        persistent_cache,
        old_snapshot,
        old_snapshot_cache,
        cache_safe=upstream_cache_safe,
    )
    _store_snapshot(
        persistent_cache,
        new_snapshot,
        new_snapshot_cache,
        cache_safe=upstream_cache_safe,
    )
    ordered_cache_events = sorted(
        persistent_cache.events,
        key=lambda event: (event.component, event.commit_sha or "", event.key or ""),
    )
    cache_events = [event.as_dict() for event in ordered_cache_events]
    cache_saved_seconds = round(sum(event.saved_seconds for event in ordered_cache_events), 6)
    timings["cache_estimated_saved"] = cache_saved_seconds
    timings["total"] = round(time.perf_counter() - analysis_started, 6)

    def timing_value(name: str) -> float:
        value = timings.get(name)
        return float(value) if isinstance(value, (float, int)) else 0.0

    relation_cache_hit = relation_cache_result.status == "hit"
    stage_timings = {
        "analysis_branches_wall": timing_value("analysis_branches_wall"),
        "direct_call_discovery": timings.get("direct_call_discovery"),
        "direct_attribute_discovery": timings.get("direct_attribute_discovery"),
        "downstream_scanning_parsing": timing_value("repository_indexing.downstream"),
        "downstream_relation_generation": round(
            relation_cache_result.load_seconds
            if relation_cache_hit
            else timing_value("relation_generation.inheritance_mro") + timing_value("relation_generation.override"),
            6,
        ),
        "monkey_patch_generation": round(
            0.0 if relation_cache_hit else timing_value("relation_generation.monkey_patch"),
            6,
        ),
        "upstream_old_new_snapshot_index": round(
            old_snapshot_cache.load_seconds
            + new_snapshot_cache.load_seconds
            + old_snapshot.cache_work_seconds
            + new_snapshot.cache_work_seconds,
            6,
        ),
        "contract_comparison": round(
            timing_value("relation_comparison")
            + timing_value("direct_import_analysis")
            + timing_value("direct_call_comparison")
            + timing_value("direct_attribute_comparison")
            + timing_value("inherited_state_comparison")
            + timing_value("generator_finding_conversion"),
            6,
        ),
        "report_generation": None,
        "total": timings["total"],
    }
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
            "execution": {
                "analysis_workers_requested": analysis_workers,
                "analysis_workers_used": effective_workers,
                "parallel_branches": effective_workers > 1,
                "branches": [
                    "relation_comparison",
                    *(["direct_import_analysis"] if plan.analyze_direct_imports else []),
                    *(["direct_call_analysis"] if plan.analyze_direct_calls else []),
                    *(["direct_attribute_analysis"] if plan.analyze_direct_attributes else []),
                    *(["inherited_state_analysis"] if plan.analyze_inherited_state else []),
                ],
            },
            "repository_index_cache": generator.repository_index_cache,
            "persistent_cache": {
                "enabled": persistent_cache.enabled,
                "root": str(persistent_cache.root) if persistent_cache.root is not None else None,
                "upstream_source_state": upstream_cache_reason,
                "downstream_source_state": downstream_cache_reason,
                "estimated_saved_seconds": cache_saved_seconds,
                "events": cache_events,
            },
            "stage_timings_seconds": stage_timings,
            "reuse_metrics": {
                "discovery": discovery_metrics,
                "downstream_source_facts": generator.downstream.source_facts.metrics(),
                "old_snapshot": old_snapshot._memo.metrics(),
                "new_snapshot": new_snapshot._memo.metrics(),
            },
            "timing_semantics": {
                "version": 1,
                "analysis_branches_wall": "Elapsed time covering all scheduled analysis branches, including discovery.",
                "contract_comparison": "Sum of branch durations, not wall time; parallel branches overlap.",
                "stage_totals_additive": False,
                "snapshot_work": "Lazy snapshot work may also be included in branch durations.",
                "cache_estimated_saved": "Historical build times minus loads, not measured wall-time savings.",
            },
            "timings_seconds": timings,
        },
        "summary": {
            "relations": analyzed_relation_count,
            "relations_collected": len(relations),
            "direct_call_dependencies": len(direct_call_dependencies),
            "direct_attribute_dependencies": len(direct_attribute_dependencies),
            "inherited_state_dependencies": len(inherited_state_dependencies),
            "generator_findings": (len(generator_findings) if plan.include_generator_findings else 0),
            "total": len(ordered),
            "root_causes": len({item["root_cause_id"] for item in finding_payloads}),
            "by_relation": dict(sorted(relation_counts.items())),
            "by_contract": dict(sorted(contract_counts.items())),
            "actionable_introduced_break": len({item["root_cause_id"] for item in actionable_payloads}),
            "actionable_introduced_findings": len(actionable_payloads),
            "introduced_break_root_causes": len({item["root_cause_id"] for item in introduced_payloads}),
            "review_root_causes": len({item["root_cause_id"] for item in review_payloads}),
            "by_action": dict(sorted(action_counts.items())),
            **{name: counts[name] for name in CLASSIFICATIONS},
        },
        "findings": finding_payloads,
    }


def _csv_rows(findings: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for item in findings:
        old = item["upstream"]["old"]
        new = item["upstream"]["new"]
        downstream = item["downstream"]
        yield {
            "root_cause_id": item.get("root_cause_id", ""),
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
    "root_cause_id",
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
        "# vLLM main2main Interface Compatibility Report",
        "",
        f"- vLLM range: `{meta['vllm_old_sha']}` → `{meta['vllm_new_sha']}`",
        f"- vllm-ascend baseline: `{meta['vllm_ascend_sha']}`",
        (
            "- Required downstream changes: "
            f"{summary['actionable_introduced_break']} root causes "
            f"({summary['actionable_introduced_findings']} relation findings)"
        ),
        (
            "- Strict contract incompatibilities, including review items: "
            f"{summary['introduced_break_root_causes']} root causes "
            f"({summary['introduced_break']} relation findings)"
        ),
        f"- Compatibility warnings: {summary['compatibility_warning']}",
        f"- Preexisting issues: {summary['preexisting']}",
        f"- Statically unresolved: {summary['analysis_unresolved']}",
        "",
        "## Required Upgrade Work",
        "",
    ]
    introduced = [
        item
        for item in report["findings"]
        if item["classification"] == "introduced_break" and item["action"] == "modify"
    ]
    if not introduced:
        lines.append("No interface break was proven to be introduced by this range.")
    for item in introduced:
        downstream = item["downstream"]
        contract_labels = {
            "call_arguments": "call arguments",
            "call_target_presence": "call target",
            "return_usage": "return-value use",
            "replacement_return": "replacement return protocol",
            "symbol_presence": "imported symbol",
            "target_presence": "upstream target",
            "base_presence": "base class",
            "attribute_presence": "member attribute",
            "required_instance_attribute": "required inherited instance state",
        }
        contract_label = contract_labels.get(item.get("contract_kind"), item.get("contract_kind", "interface contract"))
        lines.extend(
            [
                f"### {item['relation']} / {contract_label}: {downstream['file']}:{downstream['line'] or ''}",
                "",
                f"- Change: {item['change']}",
                f"- Downstream interface: `{downstream['owner'] or ''}.{downstream['name'] or ''}`",
                f"- Suggested action: {item['suggestion']}",
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
    lines.extend(["## Manual Review", ""])
    if not reviews:
        lines.append("No optional-contract delta or masked preexisting incompatibility was proven.")
    for item in reviews:
        details = item.get("details", {})
        downstream = item["downstream"]
        if details.get("optional_contract_only"):
            optional_parameters = details.get("new_optional_parameters") or []
            parameter_names = ", ".join(f"`{name}`" for name in optional_parameters)
            parameter_label = "parameter" if len(optional_parameters) == 1 else "parameters"
            parameter_reference = "that parameter" if len(optional_parameters) == 1 else "those parameters"
            reason = (
                f"The downstream override does not accept the new optional {parameter_label} {parameter_names}, "
                f"and no evidence proves that runtime dispatch passes {parameter_reference} to this implementation."
            )
        else:
            reason = "Downstream was already incompatible at old, but this range adds another exact parameter delta."
        lines.extend(
            [
                f"### {item['relation']} / review: {downstream['file']}:{downstream['line'] or ''}",
                "",
                f"- Reason: {reason}",
                f"- Suggested action: {item['suggestion']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "`preexisting` means old and new are both incompatible and is not attributed to this upgrade. "
            "`analysis_unresolved` means the available static evidence was insufficient and the analyzer did not "
            "guess.",
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
    """Group affected relations by the upstream change that caused them."""

    details = item.get("details", {})
    if item.get("contract_kind") == "patch_consumer_route":
        return (
            "patch_consumer_route",
            details.get("patch_target"),
            item["upstream"]["new"].get("owner"),
            details.get("factory_attribute"),
        )
    old = item["upstream"]["old"]
    new = item["upstream"]["new"]
    root = details.get("root_upstream") or old

    def present(endpoint: dict[str, Any]) -> bool:
        return endpoint.get("file") is not None and endpoint.get("symbol_kind") != "missing"

    def identity(endpoint: dict[str, Any]) -> tuple[object, ...]:
        return (
            endpoint.get("file"),
            endpoint.get("owner"),
            endpoint.get("name"),
        )

    old_present = present(old)
    new_present = present(new)
    if item.get("contract_kind") == "symbol_presence":
        # A relocation destination is diagnostic evidence, not proof that the
        # old import path still exists. Use the import resolver's presence fact.
        old_present = item.get("compatibility", {}).get("old", {}).get("exists") is True
        new_present = item.get("compatibility", {}).get("new", {}).get("exists") is True
    if old_present and not new_present:
        return ("upstream_presence_removed", *identity(root))
    if not old_present and new_present:
        return ("upstream_presence_added", *identity(new))

    endpoint = new if new.get("file") is not None else root
    parameter_delta = details.get("parameter_delta")
    if _signature_delta_changed(parameter_delta):
        parameter_identity = identity(endpoint)
        if endpoint.get("symbol_kind") == "constructor":
            parameter_identity = (
                endpoint.get("file"),
                ".".join(part for part in (endpoint.get("owner"), endpoint.get("name")) if part),
                "__init__",
            )
        return (
            "upstream_parameter_delta",
            *parameter_identity,
            json.dumps(parameter_delta, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    if item.get("contract_kind") == "required_instance_attribute":
        return (
            "upstream_inherited_state_requirement",
            *identity(endpoint),
            details.get("required_attribute"),
        )
    return (
        "upstream_contract",
        *identity(endpoint),
        item.get("contract_kind"),
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
            parameter_names = ", ".join(f"`{name}`" for name in optional_parameters)
            parameter_label = "parameter" if len(optional_parameters) == 1 else "parameters"
            parameter_reference = "it" if len(optional_parameters) == 1 else "them"
            reason = (
                f"The downstream override does not accept the new optional {parameter_label} {parameter_names}, "
                f"and the analyzer found no upstream call in this range that passes {parameter_reference} "
                "to that override."
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
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if report.get("metadata", {}).get("scenario") == "vllm-interface":
        outputs = _write_upstream_pr_reports(report, output_dir)
    else:
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
        outputs = {
            "json": str(json_path),
            "all_csv": str(all_csv),
            "introduced_csv": str(introduced_csv),
            "markdown": str(markdown_path),
        }

    report_seconds = round(time.perf_counter() - started, 6)
    stage_timings = report.get("metadata", {}).get("stage_timings_seconds")
    if isinstance(stage_timings, dict):
        stage_timings["report_generation"] = report_seconds
        analysis_total = stage_timings.get("total")
        if isinstance(analysis_total, (float, int)):
            stage_timings["total_with_report"] = round(float(analysis_total) + report_seconds, 6)

    if report.get("metadata", {}).get("scenario") == "vllm-interface":
        payload = _upstream_pr_payload(report)
        Path(outputs["json"]).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        Path(outputs["metadata_json"]).write_text(
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
    else:
        Path(outputs["json"]).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return outputs
