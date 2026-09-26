# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit class/function writes and escapes resolved by shared scope flow."""

from __future__ import annotations

import ast

from .generator import (
    _expression_name,
    _function_local_names,
    _scope_reference_variants,
    _ScopeBinding,
    _tag_guard_names,
)


def has_construction_effect(reference: str, effects: frozenset[str]) -> bool:
    return reference in effects or any(
        effect.endswith(".*") and (reference == effect[:-2] or reference.startswith(effect[:-1])) for effect in effects
    )


def has_storage_effect(reference: str, effects: frozenset[str]) -> bool:
    """Include mutations to allocation/storage hook objects and metadata."""
    hooks = (
        "__init__",
        "__new__",
        "__post_init__",
        "__setattr__",
        "__delattr__",
        "__getattribute__",
        "__getattr__",
        "__init_subclass__",
        "__dataclass_fields__",
        "__dataclass_params__",
    )
    return has_construction_effect(reference, effects) or any(
        has_construction_effect(f"{reference}.{hook}", effects) for hook in hooks
    )


def construction_effects(
    tree: ast.Module, module: str, is_package: bool, *, bound_receiver_calls: frozenset[int] = frozenset()
) -> frozenset[str]:
    """Invalidate allocation assumptions for source-visible callable effects.

    This is a conservative effect collector, not a replacement symbol resolver.
    Candidate roots only avoid resolving every ordinary local argument. Imports,
    classes, functions and simple aliases require exact shared scope resolution.
    Constructor *results* are not class aliases.
    """
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    roots: set[str] = set()
    assignments: list[ast.Assign | ast.AnnAssign] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            roots.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments.append(node)
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            if not isinstance(assignment.value, (ast.Name, ast.Attribute)):
                continue
            value = _expression_name(assignment.value)
            if value is None or value.split(".")[0] not in roots:
                continue
            targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
            additions = {target.id for target in targets if isinstance(target, ast.Name)} - roots
            roots.update(additions)
            changed = changed or bool(additions)
    module_guards = _tag_guard_names(tree.body)
    final_line = max((getattr(node, "end_lineno", 0) or 0 for node in tree.body), default=0) + 1
    local_names: dict[int, set[str]] = {}
    guards: dict[int, set[str]] = {}
    effects: set[str] = set()
    # All module fallbacks read the same immutable statement/guard view.
    # The shared resolver bounds this cache; keep it invocation-local so
    # different source snapshots and bound-receiver proofs remain isolated.
    module_states: dict[int, dict[str, tuple[_ScopeBinding, ...]]] = {}

    def record(expression: ast.AST, at: ast.AST, *, subtree: bool = False) -> None:
        name = _expression_name(expression)
        if name is None or name.split(".")[0] not in roots:
            return
        scope = parents.get(id(at))
        while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = parents.get(id(scope))
        statements = scope.body if scope is not None else tree.body
        if scope is not None:
            key = id(scope)
            if key not in local_names:
                local_names[key] = set(_function_local_names(scope))
                guards[key] = _tag_guard_names(scope.body)
            blocked = local_names[key]
        else:
            blocked = set()

        def fallback(node: ast.AST) -> set[str | None]:
            root = (_expression_name(node) or "").split(".")[0]
            if root in blocked:
                return {None}
            return _scope_reference_variants(
                node,
                statements=tree.body,
                line=final_line,
                tag_guard_names=module_guards,
                module=module,
                is_package=is_package,
                state_cache=module_states,
            )

        references = _scope_reference_variants(
            expression,
            statements=statements,
            line=getattr(at, "lineno", 0),
            tag_guard_names=guards[id(scope)] if scope is not None else module_guards,
            module=module,
            is_package=is_package,
            fallback=fallback if scope is not None else None,
            state_cache=module_states if scope is None else None,
        )
        effects.update(reference + (".*" if subtree else "") for reference in references if reference is not None)

    def record_value(expression: ast.AST, at: ast.AST) -> None:
        if isinstance(expression, (ast.Name, ast.Attribute)):
            record(expression, at, subtree=True)
        elif not isinstance(expression, ast.Call):
            # Passing a container containing a class also exposes that class.
            # A nested call contributes its result instead; its own arguments
            # are checked separately by the outer AST walk.
            for child in ast.iter_child_nodes(expression):
                record_value(child, at)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            record(node.value, node, subtree=True)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if isinstance(node.value, ast.Attribute):
                record(node.value.value, node, subtree=True)
        elif isinstance(node, ast.Call):
            # A class passed to an unknown callable, including setattr/delattr,
            # can acquire a new initializer or descriptor. Calls to class
            # methods can also modify the class. Nested instance constructions
            # are not mistaken for escaping the class itself.
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                record_value(argument, node)
            # A caller may separately prove the bound method body. Excluding
            # that receiver does not exclude argument escapes or nested calls.
            if isinstance(node.func, ast.Attribute) and id(node) not in bound_receiver_calls:
                record(node.func.value, node)
        elif isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) and node.value is not None:
            record_value(node.value, node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            if not isinstance(node.value, (ast.Name, ast.Attribute)):
                record_value(node.value, node)
    return frozenset(effects)
