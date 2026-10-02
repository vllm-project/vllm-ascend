# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, source-only contracts for registry-backed module attributes.

No module or getter is executed. Unknown keys, escaping/mutated registries,
and unsupported getters cannot prove a member present or absent.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass


def _name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def runtime_module_body(tree: ast.Module) -> list[ast.stmt]:
    """Exclude annotation-only bindings and proven typing-only branches.

    A typing alias with any other binding is deliberately not constant-folded.
    Nested function/class bodies retain their own annotation semantics.
    """

    return _typing_module_body(tree, annotation_mode=False)


def annotation_module_body(tree: ast.Module) -> list[ast.stmt]:
    """Select proven TYPE_CHECKING branches for annotation lookup only."""
    return _typing_module_body(tree, annotation_mode=True)


def _typing_module_body(tree: ast.Module, *, annotation_mode: bool) -> list[ast.stmt]:
    candidates: dict[str, str] = {}
    counts: dict[str, int] = {}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "typing" and not statement.level:
            for alias in statement.names:
                if alias.name == "TYPE_CHECKING":
                    local = alias.asname or alias.name
                    candidates[local] = "flag"
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "typing":
                    candidates[alias.asname or alias.name] = "module"
    # Function/class-local names cannot rebind a module typing alias. A nested
    # global declaration can, so retain a conservative barrier for that case.
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            for name in node.names:
                candidates.pop(name, None)
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pending.extend(node.decorator_list)
            pending.extend(node.args.defaults)
            pending.extend(default for default in node.args.kw_defaults if default is not None)
        elif isinstance(node, ast.ClassDef):
            pending.extend([*node.decorator_list, *node.bases, *(keyword.value for keyword in node.keywords)])
        elif not isinstance(node, ast.Lambda):
            pending.extend(ast.iter_child_nodes(node))
        names: list[str] = []
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            candidates.clear()
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.extend(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            names.append(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.append(node.rest)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if isinstance(node.value, ast.Name) and node.attr == "TYPE_CHECKING":
                names.append(node.value.id)
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    aliases = {name: kind for name, kind in candidates.items() if counts.get(name) == 1}

    class RuntimeBody(ast.NodeTransformer):
        def generic_visit(self, node: ast.AST) -> ast.AST:
            # Copy only the visited module-level structure. Function/class
            # bodies are immutable inputs and need not be duplicated. Copy
            # lists as well: NodeTransformer edits list fields in place.
            result = copy.copy(node)
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    setattr(result, field, list(value))
            return super().generic_visit(result)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
            return node

        def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
            return node if node.value is not None else None

        def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
            test = node.test
            is_typing = isinstance(test, ast.Name) and aliases.get(test.id) == "flag"
            if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name):
                is_typing = test.attr == "TYPE_CHECKING" and aliases.get(test.value.id) == "module"
            if is_typing:
                replacement = ast.Module(body=node.body if annotation_mode else node.orelse, type_ignores=[])
                self.generic_visit(replacement)
                return replacement.body
            return self.generic_visit(node)

    transformed = RuntimeBody().visit(tree)
    assert isinstance(transformed, ast.Module)
    return transformed.body


@dataclass(frozen=True)
class ModuleGetattrContract:
    entries: dict[str, ast.expr] | None
    invokes_values: bool = False

    def resolve(self, name: str) -> tuple[str, ast.AST | None]:
        if self.entries is None:
            return "unknown", None
        value = self.entries.get(name)
        if value is None:
            return "missing", None
        if self.invokes_values and not isinstance(value, ast.Lambda):
            return "unknown", value
        if self.invokes_values and isinstance(value, ast.Lambda):
            args = value.args
            if len(args.posonlyargs) + len(args.args) > len(args.defaults) or any(
                default is None for default in args.kw_defaults
            ):
                return "unknown", value
        return "attribute", value


def _registry_is_closed(body: list[ast.stmt], declaration: ast.stmt, registry: str) -> bool:
    """Allow only read-only registry uses; reject aliases, writes and escapes."""

    tree = ast.Module(body=body, type_ignores=[])
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"exec", "eval", "globals", "locals", "vars"}:
                return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == registry:
            return False
        if isinstance(node, ast.arg) and node.arg == registry:
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(alias.name == "*" or (alias.asname or alias.name.split(".")[0]) == registry for alias in node.names):
                return False
        if isinstance(node, ast.ExceptHandler) and node.name == registry:
            return False
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == registry:
            return False
        if isinstance(node, ast.MatchMapping) and node.rest == registry:
            return False
        if not _name(node, registry):
            continue
        parent = parents.get(node)
        if parent is declaration and isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            continue
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            return False
        if isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Load):
            continue
        if (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))  # codespell:ignore notin
            and parent.comparators[0] is node
        ):
            continue
        if isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is node:
            continue
        if isinstance(parent, ast.Attribute) and parent.attr in {"keys", "items", "values"}:
            call = parents.get(parent)
            if isinstance(call, ast.Call) and call.func is parent and not call.args and not call.keywords:
                continue
        return False
    return True


def module_getattr_contract(body: list[ast.stmt], getter: ast.AST | None) -> ModuleGetattrContract:
    """Recognize `if name in registry: return registry[name]()` + AttributeError.

    Direct value registries are supported too. This models attribute presence,
    not arbitrary failures inside callbacks (just as for property getters).
    """

    unknown = ModuleGetattrContract(None)
    if not isinstance(getter, ast.FunctionDef) or getter.decorator_list:
        return unknown
    args = getter.args
    positional = [*args.posonlyargs, *args.args]
    if len(positional) != 1 or args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
        return unknown
    parameter = positional[0].arg
    statements = getter.body
    if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
        if isinstance(statements[0].value.value, str):
            statements = statements[1:]
    if len(statements) != 2 or not isinstance(statements[0], ast.If):
        return unknown
    branch, failure = statements
    assert isinstance(branch, ast.If)
    test = branch.test
    if not (
        isinstance(test, ast.Compare)
        and _name(test.left, parameter)
        and len(test.ops) == len(test.comparators) == 1
        and isinstance(test.ops[0], ast.In)
        and isinstance(test.comparators[0], ast.Name)
        and not branch.orelse
        and len(branch.body) == 1
        and isinstance(branch.body[0], ast.Return)
        and isinstance(failure, ast.Raise)
        and isinstance(failure.exc, ast.Call)
        and _name(failure.exc.func, "AttributeError")
    ):
        return unknown
    registry = test.comparators[0].id
    if registry == parameter:
        return unknown
    result = branch.body[0].value
    invokes_values = isinstance(result, ast.Call)
    if isinstance(result, ast.Call):
        if result.args or result.keywords:
            return unknown
        result = result.func
    if not (isinstance(result, ast.Subscript) and _name(result.value, registry) and _name(result.slice, parameter)):
        return unknown
    declarations = [
        node
        for node in body
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and _name(node.targets[0], registry))
        or (isinstance(node, ast.AnnAssign) and _name(node.target, registry))
    ]
    if len(declarations) != 1:
        return unknown
    declaration = declarations[0]
    assert isinstance(declaration, (ast.Assign, ast.AnnAssign))
    if not isinstance(declaration.value, ast.Dict) or not _registry_is_closed(body, declaration, registry):
        return unknown
    entries: dict[str, ast.expr] = {}
    for key, value in zip(declaration.value.keys, declaration.value.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            return unknown
        entries[key.value] = value
    return ModuleGetattrContract(entries, invokes_values)
