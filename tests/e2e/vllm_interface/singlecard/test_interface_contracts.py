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

_MANIFEST_PATH = Path(__file__).parents[1] / "interface_contracts.json"
_VLLM_ASCEND_SOURCE_ROOT = Path(__file__).parents[4]


def _load_contracts() -> list[dict[str, Any]]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    return manifest["contracts"]


def _vllm_source_root() -> Path:
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
    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    return package_root.parent


@cache
def _parse_source(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_named_node(nodes: list[ast.stmt], name: str) -> ast.AST | None:
    # Python binds the last definition, which matters for overloaded functions.
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


def _find_contract_node(tree: ast.Module, owner: str | None, callable_name: str) -> ast.AST | None:
    if owner is None:
        return _find_named_node(tree.body, callable_name)

    owner_node = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == owner),
        None,
    )
    if owner_node is None:
        return None
    return _find_named_node(owner_node.body, callable_name)


def _parameter(name: str, required: bool) -> dict[str, object]:
    return {"name": name, "required": required}


def _callable_signature(node: ast.AST) -> dict[str, object] | None:
    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return None

    arguments = node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    required_count = len(positional) - len(arguments.defaults)

    return {
        "kind": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
        "positional_only": [
            _parameter(argument.arg, index < required_count) for index, argument in enumerate(arguments.posonlyargs)
        ],
        "positional_or_keyword": [
            _parameter(argument.arg, index + len(arguments.posonlyargs) < required_count)
            for index, argument in enumerate(arguments.args)
        ],
        "variadic_positional": arguments.vararg.arg if arguments.vararg else None,
        "keyword_only": [
            _parameter(argument.arg, default is None)
            for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
        ],
        "variadic_keyword": arguments.kwarg.arg if arguments.kwarg else None,
    }


def _contract_id(contract: dict[str, Any]) -> str:
    owner = f"{contract['owner']}." if contract["owner"] else ""
    return f"{contract['upstream_file']}::{owner}{contract['callable']}"


def _direct_callable_contracts() -> list[dict[str, Any]]:
    return [
        contract
        for contract in _load_contracts()
        if contract["signature"] is not None
        and any(consumer["relation"] == "direct_callable" for consumer in contract["consumers"])
        and not any(consumer["relation"] == "monkey_patch" for consumer in contract["consumers"])
    ]


@pytest.mark.parametrize("contract", _load_contracts(), ids=_contract_id)
def test_watched_vllm_callable_contract(contract: dict[str, Any]) -> None:
    """Detect changes to vLLM callables coupled to vllm-ascend without importing NPU code."""
    source_path = _vllm_source_root() / contract["upstream_file"]
    consumers = "\n".join(
        f"  - {consumer['relation']}: {consumer['file']}::{consumer['qualified_name']}"
        for consumer in contract["consumers"]
    )
    interface_name = _contract_id(contract)

    assert source_path.is_file(), (
        f"Watched vLLM source file was removed or moved: {contract['upstream_file']}\n"
        f"vllm-ascend consumers that need review:\n{consumers}"
    )

    tree = _parse_source(source_path)
    node = _find_contract_node(tree, contract["owner"], contract["callable"])
    assert node is not None, (
        f"Watched vLLM interface was removed or moved: {interface_name}\n"
        f"vllm-ascend consumers that need review:\n{consumers}"
    )

    expected_signature = contract["signature"]
    if expected_signature is None:
        return

    actual_signature = _callable_signature(node)
    assert actual_signature == expected_signature, (
        f"Watched vLLM interface changed: {interface_name}\n"
        f"Expected: {json.dumps(expected_signature, sort_keys=True)}\n"
        f"Actual:   {json.dumps(actual_signature, sort_keys=True)}\n"
        f"vllm-ascend consumers that need review:\n{consumers}\n"
        "Review the downstream implementation before accepting a new baseline."
    )


@pytest.mark.parametrize("contract", _direct_callable_contracts(), ids=_contract_id)
def test_direct_callable_keywords_are_supported(contract: dict[str, Any]) -> None:
    """Ensure direct vLLM calls do not use keywords removed by an upstream change."""
    upstream_path = _vllm_source_root() / contract["upstream_file"]
    upstream_node = _find_contract_node(
        _parse_source(upstream_path),
        contract["owner"],
        contract["callable"],
    )
    assert upstream_node is not None
    upstream_signature = _callable_signature(upstream_node)
    assert upstream_signature is not None

    supported_keywords = {
        parameter["name"]
        for parameter in [
            *upstream_signature["positional_or_keyword"],
            *upstream_signature["keyword_only"],
        ]
    }
    if upstream_signature["variadic_keyword"] is not None:
        return

    for consumer in contract["consumers"]:
        if consumer["relation"] != "direct_callable":
            continue

        local_name = consumer["qualified_name"].removesuffix("(...)")
        consumer_path = _VLLM_ASCEND_SOURCE_ROOT / consumer["file"]
        consumer_tree = _parse_source(consumer_path)
        calls = [
            node
            for node in ast.walk(consumer_tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == local_name
        ]
        assert calls, f"Direct callable consumer was removed or renamed: {consumer['file']}::{local_name}"

        for call in calls:
            unsupported_keywords = {
                keyword.arg
                for keyword in call.keywords
                if keyword.arg is not None and keyword.arg not in supported_keywords
            }
            assert not unsupported_keywords, (
                f"{consumer['file']}::{local_name} uses keyword arguments no longer accepted by "
                f"{_contract_id(contract)}: {sorted(unsupported_keywords)}"
            )
