#
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
#

from __future__ import annotations

import ast
import importlib.util
import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import pytest

_BOUNDARIES_PATH = Path(__file__).parents[1] / "interface_boundaries.jsonl"
_VLLM_ASCEND_ROOT = Path(__file__).parents[4]
_CALLABLE_RELATIONS = {
    "monkey_patch",
    "override",
    "override_candidate",
    "patch_call_closure",
}


def _load_boundaries() -> list[dict[str, Any]]:
    records = []
    for line in _BOUNDARIES_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "_meta" not in record:
            records.append(record)
    return records


def _vllm_root() -> Path:
    configured_root = os.getenv("VLLM_SOURCE_ROOT")
    if configured_root:
        candidate = Path(configured_root).resolve()
        if (candidate / "vllm" / "__init__.py").is_file():
            return candidate
        if candidate.name == "vllm" and (candidate / "__init__.py").is_file():
            return candidate.parent
        raise AssertionError(f"VLLM_SOURCE_ROOT does not contain the vllm package: {candidate}")

    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.submodule_search_locations, (
        "Cannot locate the vllm source package. Install vLLM or set VLLM_SOURCE_ROOT."
    )
    return Path(next(iter(spec.submodule_search_locations))).resolve().parent


@cache
def _parse(path: Path) -> ast.Module:
    assert path.is_file(), f"Interface source file was removed or moved: {path}"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_named_node(nodes: list[ast.stmt], name: str) -> ast.AST | None:
    for node in reversed(nodes):
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return node
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.rsplit(".", 1)[-1]) == name for alias in node.names):
                return node
    return None


def _find_node(tree: ast.Module, owner: str | None, name: str) -> ast.AST | None:
    if owner is None:
        top_level = _find_named_node(tree.body, name)
        if top_level is not None:
            return top_level
        return next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name
            ),
            None,
        )
    owner_node = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == owner),
        None,
    )
    return _find_named_node(owner_node.body, name) if owner_node else None


def _parameter(name: str, required: bool) -> list[object]:
    return [name, required]


def _signature(node: ast.AST) -> list[object] | None:
    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return None

    arguments = node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    required_count = len(positional) - len(arguments.defaults)
    return [
        "async" if isinstance(node, ast.AsyncFunctionDef) else "sync",
        [_parameter(argument.arg, index < required_count) for index, argument in enumerate(arguments.posonlyargs)],
        [
            _parameter(argument.arg, index + len(arguments.posonlyargs) < required_count)
            for index, argument in enumerate(arguments.args)
        ],
        arguments.vararg.arg if arguments.vararg else None,
        [
            _parameter(argument.arg, default is None)
            for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
        ],
        arguments.kwarg.arg if arguments.kwarg else None,
    ]


def _boundary_id(boundary: dict[str, Any]) -> str:
    upstream_file, owner, name, _ = boundary["u"]
    qualified_name = f"{owner}.{name}" if owner else name
    return f"{upstream_file}::{qualified_name}"


def _callable_consumers() -> list[tuple[dict[str, Any], list[Any]]]:
    return [
        (boundary, consumer)
        for boundary in _load_boundaries()
        for consumer in boundary["c"]
        if consumer[0] in _CALLABLE_RELATIONS
    ]


def _direct_consumers() -> list[tuple[dict[str, Any], list[Any]]]:
    return [
        (boundary, consumer)
        for boundary in _load_boundaries()
        for consumer in boundary["c"]
        if consumer[0] == "direct_callable"
    ]


def _inheritance_consumers() -> list[tuple[dict[str, Any], list[Any]]]:
    return [
        (boundary, consumer)
        for boundary in _load_boundaries()
        for consumer in boundary["c"]
        if consumer[0] == "inheritance"
    ]


def _inherited_callable_consumers() -> list[tuple[dict[str, Any], list[Any]]]:
    return [
        (boundary, consumer)
        for boundary in _load_boundaries()
        for consumer in boundary["c"]
        if consumer[0] == "inherited_callable"
    ]


def _consumer_id(item: tuple[dict[str, Any], list[Any]]) -> str:
    boundary, consumer = item
    return f"{_boundary_id(boundary)}->{consumer[1]}::{consumer[2] or ''}.{consumer[3]}"


@pytest.mark.parametrize("boundary", _load_boundaries(), ids=_boundary_id)
def test_upstream_callable_boundary(boundary: dict[str, Any]) -> None:
    upstream_file, owner, name, expected_signature = boundary["u"]
    node = _find_node(_parse(_vllm_root() / upstream_file), owner, name)
    assert node is not None, f"Upstream callable was removed or moved: {_boundary_id(boundary)}"
    assert _signature(node) == expected_signature, f"Upstream callable boundary changed: {_boundary_id(boundary)}"


@pytest.mark.parametrize("item", _callable_consumers(), ids=_consumer_id)
def test_downstream_callable_boundary(item: tuple[dict[str, Any], list[Any]]) -> None:
    boundary, consumer = item
    relation, source_file, owner, name, expected_signature = consumer
    node = _find_node(_parse(_VLLM_ASCEND_ROOT / source_file), owner, name)
    assert node is not None, (
        f"Downstream {relation} endpoint was removed or moved: "
        f"{_boundary_id(boundary)} -> {source_file}::{owner or ''}.{name}"
    )
    assert _signature(node) == expected_signature, (
        f"Downstream {relation} boundary changed: {_boundary_id(boundary)} -> {source_file}::{owner or ''}.{name}"
    )


def _call_errors(call: ast.Call, signature: list[object]) -> list[str]:
    _, positional_only, positional_or_keyword, variadic_positional, keyword_only, variadic_keyword = signature
    positional_parameters = [*positional_only, *positional_or_keyword]
    explicit_keywords = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
    has_starred = any(isinstance(argument, ast.Starred) for argument in call.args)
    has_double_starred = any(keyword.arg is None for keyword in call.keywords)
    errors = []

    if variadic_positional is None and not has_starred and len(call.args) > len(positional_parameters):
        errors.append(f"too many positional arguments: {len(call.args)} > {len(positional_parameters)}")

    supported_keywords = {parameter[0] for parameter in [*positional_or_keyword, *keyword_only]}
    if variadic_keyword is None:
        unsupported = explicit_keywords - supported_keywords
        if unsupported:
            errors.append(f"unsupported keywords: {sorted(unsupported)}")

    if not has_starred and not has_double_starred:
        supplied = {parameter[0] for parameter in positional_parameters[: len(call.args)]} | explicit_keywords
        required = {parameter[0] for parameter in [*positional_parameters, *keyword_only] if parameter[1]}
        missing = required - supplied
        if missing:
            errors.append(f"missing required arguments: {sorted(missing)}")
    return errors


@pytest.mark.parametrize("item", _direct_consumers(), ids=_consumer_id)
def test_direct_call_boundary(item: tuple[dict[str, Any], list[Any]]) -> None:
    boundary, consumer = item
    _, source_file, _, local_name, _ = consumer
    calls = [
        node
        for node in ast.walk(_parse(_VLLM_ASCEND_ROOT / source_file))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == local_name
    ]
    assert calls, f"Direct call was removed or renamed: {source_file}::{local_name}"

    patched_signatures = [
        candidate[4] for candidate in boundary["c"] if candidate[0] == "monkey_patch" and candidate[4] is not None
    ]
    candidate_signatures = patched_signatures or [boundary["u"][3]]
    candidate_signatures = [signature for signature in candidate_signatures if signature is not None]
    if not candidate_signatures:
        return

    for call in calls:
        errors = [_call_errors(call, signature) for signature in candidate_signatures]
        assert any(not candidate_errors for candidate_errors in errors), (
            f"Direct call boundary is incompatible: {source_file}::{local_name} -> "
            f"{_boundary_id(boundary)}; candidates={errors}"
        )


@pytest.mark.parametrize("item", _inheritance_consumers(), ids=_consumer_id)
def test_inheritance_boundary(item: tuple[dict[str, Any], list[Any]]) -> None:
    boundary, consumer = item
    _, source_file, class_name, base_name, _ = consumer
    tree = _parse(_VLLM_ASCEND_ROOT / source_file)
    class_node = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    assert class_node is not None, f"Downstream class was removed: {source_file}::{class_name}"
    base_names = {
        base.id if isinstance(base, ast.Name) else base.attr
        for base in class_node.bases
        if isinstance(base, (ast.Attribute, ast.Name))
    }
    assert base_name in base_names, (
        f"Inheritance boundary changed: {_boundary_id(boundary)} -> {source_file}::{class_name}({base_name})"
    )


@pytest.mark.parametrize("item", _inherited_callable_consumers(), ids=_consumer_id)
def test_inherited_callable_boundary(item: tuple[dict[str, Any], list[Any]]) -> None:
    boundary, consumer = item
    _, source_file, class_name, method_name, _ = consumer
    tree = _parse(_VLLM_ASCEND_ROOT / source_file)
    class_node = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    assert class_node is not None, f"Downstream class was removed: {source_file}::{class_name}"
    upstream_owner = boundary["u"][1]
    base_names = {
        base.id if isinstance(base, ast.Name) else base.attr
        for base in class_node.bases
        if isinstance(base, (ast.Attribute, ast.Name))
    }
    assert upstream_owner in base_names, (
        f"Inherited callable boundary changed: {_boundary_id(boundary)} -> {source_file}::{class_name}.{method_name}"
    )
