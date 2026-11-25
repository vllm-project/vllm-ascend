# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

RFORK_ROOT = Path(__file__).resolve().parents[4] / "vllm_ascend/model_loader/rfork"
MODELS_ROOT = RFORK_ROOT.parents[1] / "models"


def _load_module(monkeypatch, module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, RFORK_ROOT / file_name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _stub(monkeypatch, module_name: str, **attributes):
    module = ModuleType(module_name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _defined_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.FunctionDef):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _exec_source_definitions(source: Path, names: set[str], namespace: dict, class_name: str | None = None) -> None:
    # Runs real model-source definitions without importing the model module and its NPU dependencies.
    body = ast.parse(source.read_text(encoding="utf-8")).body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    nodes = [node for node in body if _defined_name(node) in names]
    assert {_defined_name(node) for node in nodes} == names, f"{source} no longer defines {names}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
