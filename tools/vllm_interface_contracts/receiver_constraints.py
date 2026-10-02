# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source evidence that a base annotation does not prove concrete dispatch.

Local MRO members are alternatives, not guessed runtime receivers. Caller guard
contexts are retained for adjudication; they do not prove closed-world dispatch.
"""

from __future__ import annotations

import ast
from typing import Any

from .annotation_names import AnnotationNamespace
from .generator import (
    InterfaceBoundaryGenerator,
    ModuleInfo,
    _function_local_names,
    _inspect_signature,
)


class ReceiverConstraints:
    """Index alternate member owners and bounded incoming helper-call evidence."""

    def __init__(self, engine: InterfaceBoundaryGenerator):
        self.engine = engine
        self._members: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
        self._callers: dict[tuple[str, str, str], tuple[dict[str, Any], ...]] = {}

    def local_members(self, declared_type: str, member: str) -> tuple[dict[str, Any], ...]:
        key = (declared_type, member)
        if key in self._members:
            return self._members[key]
        result: dict[str, dict[str, Any]] = {}
        for candidate in self.engine.downstream.classes.values():
            mro = self.engine._linearized_mro(candidate.qualified_name)
            if not mro.complete or declared_type not in mro.owners[1:]:
                continue
            for owner in mro.owners:
                target = f"{owner}.{member}"
                bindings = self.engine._final_bindings(target)
                bound = [
                    item
                    for item in bindings
                    if item.kind != "unbound" and not (isinstance(item.node, ast.AnnAssign) and item.node.value is None)
                ]
                if not bound:
                    continue
                if owner.startswith("vllm_ascend."):
                    info = self.engine.downstream.find_class(owner)
                    if info is not None:
                        result.setdefault(
                            target,
                            {
                                "target": target,
                                "receiver": candidate.qualified_name,
                                "file": info.file,
                                "line": min(item.line for item in bound),
                                "binding_kinds": sorted({item.kind for item in bound}),
                                "conditional": any(item.kind == "unbound" for item in bindings),
                            },
                        )
                # A nearer binding blocks inherited lookup even when it is not
                # callable. Do not invent an upstream owner behind that value.
                break
        self._members[key] = tuple(result[name] for name in sorted(result))
        return self._members[key]

    def caller_contexts(
        self, module: ModuleInfo, function: ast.FunctionDef | ast.AsyncFunctionDef, parameter: str
    ) -> tuple[dict[str, Any], ...]:
        key = (module.name, function.name, parameter)
        if key in self._callers:
            return self._callers[key]
        self._callers[key] = ()
        info = self.engine.downstream.find_callable(f"{module.name}.{function.name}")
        if info is None or info.node is not function or info.owner is not None or function.decorator_list:
            return ()
        signature = _inspect_signature(info.signature) if info.signature is not None else None
        if signature is None:
            return ()
        namespace = AnnotationNamespace(module.tree, module.name, module.is_package)
        parents = {id(child): parent for parent in ast.walk(module.tree) for child in ast.iter_child_nodes(parent)}
        contexts = []
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != function.name:
                continue
            caller: ast.AST | None = parents.get(id(node))
            while caller is not None and not isinstance(caller, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller = parents.get(id(caller))
            if caller is not None and function.name in _function_local_names(caller):
                continue
            if namespace.runtime(function.name) != f"{module.name}.{function.name}":
                continue
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
                continue
            try:
                bound = signature.bind(*node.args, **{kw.arg: kw.value for kw in node.keywords if kw.arg is not None})
            except TypeError:
                continue
            argument = bound.arguments.get(parameter)
            if not isinstance(argument, ast.AST):
                continue
            guards = []
            current: ast.AST = node
            while (parent := parents.get(id(current))) is not None and parent is not caller:
                if isinstance(parent, ast.If) and current in parent.body:
                    guards.append({"line": parent.lineno, "expression": ast.unparse(parent.test)})
                current = parent
            contexts.append(
                {
                    "file": module.file,
                    "line": node.lineno,
                    "parameter": parameter,
                    "argument": ast.unparse(argument),
                    "enclosing_positive_guards": guards,
                    "status": "syntactic_context_not_concrete_dispatch_proof",
                }
            )
        self._callers[key] = tuple(contexts)
        return self._callers[key]

    def evidence(
        self,
        declared_type: str,
        member: str,
        module: ModuleInfo,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        parameter: str | None,
    ) -> dict[str, Any] | None:
        alternatives = self.local_members(declared_type, member)
        if not alternatives:
            return None
        return {
            "kind": "base_annotation_with_local_member_alternatives",
            "declared_type": declared_type,
            "local_members": alternatives,
            "caller_contexts": self.caller_contexts(module, function, parameter) if parameter is not None else (),
            "status": "concrete_receiver_unresolved",
        }
