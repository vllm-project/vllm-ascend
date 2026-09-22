# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restricted evidence for configuration-conditional inherited state contracts."""

from __future__ import annotations

import ast

from .call_contracts import _parents
from .generator import _expression_name, _function_scope_nodes, _main_condition_value


def conditional_state_evidence(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    initializer: ast.FunctionDef | ast.AsyncFunctionDef,
    member: str,
    read_line: int,
    tag_guard_names: set[str] | None = None,
    initialization_helpers: dict[str, ast.FunctionDef] | None = None,
) -> list[dict[str, object]]:
    """Prove a supported element-flag path, not execution for every workload.

    The constructor must explicitly enable the exact flag on an element of
    the collection being iterated. Additional guards may only be a positive
    numeric argument (or its pure arithmetic derivation). Unknown branch
    predicates, fallback paths and contradictory writes are not promoted.
    """
    parents = _parents(function)
    nodes = list(_function_scope_nodes(function))
    reads = [
        node for node in nodes if isinstance(node, ast.Attribute) and node.attr == member and node.lineno == read_line
    ]
    if len(reads) != 1 or not function.args.args or not initializer.args.args:
        return []
    receiver = function.args.args[0].arg
    init_receiver = initializer.args.args[0].arg
    helper_calls: dict[int, tuple[str, int]] = {}
    helper_scopes: dict[int, ast.FunctionDef] = {}
    active_helpers: list[ast.FunctionDef] = []
    for statement in initializer.body:
        if isinstance(statement, (ast.Return, ast.Raise)):
            break
        call = statement.value if isinstance(statement, ast.Expr) else None
        if not (
            isinstance(call, ast.Call)
            and not call.args
            and not call.keywords
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == init_receiver
        ):
            continue
        helper = (initialization_helpers or {}).get(call.func.attr)
        if helper is None or helper.decorator_list:
            continue
        arguments = helper.args
        positional = [*arguments.posonlyargs, *arguments.args]
        if (
            len(positional) != 1
            or positional[0].arg != init_receiver
            or arguments.kwonlyargs
            or arguments.vararg is not None
            or arguments.kwarg is not None
        ):
            continue
        active_helpers.append(helper)
        for candidate in _function_scope_nodes(helper):
            helper_calls[id(candidate)] = (call.func.attr, call.lineno)
            helper_scopes[id(candidate)] = helper
    if any(isinstance(node, (ast.Return, ast.Raise, ast.Assert)) and node.lineno < read_line for node in nodes):
        return []
    predicates: list[ast.expr] = []
    loops: dict[str, str] = {}
    child: ast.AST = reads[0]
    while (parent := parents.get(id(child))) is not None:
        if isinstance(parent, ast.If):
            if child not in parent.body:
                return []
            predicates.append(parent.test)
        elif isinstance(parent, ast.For):
            if not isinstance(parent.target, ast.Name) or child not in parent.body:
                return []
            if not (
                isinstance(parent.iter, ast.Attribute)
                and isinstance(parent.iter.value, ast.Name)
                and parent.iter.value.id == receiver
            ):
                return []
            loops[parent.target.id] = parent.iter.attr
        elif isinstance(parent, (ast.While, ast.AsyncFor, ast.Match, ast.IfExp, ast.Try)):
            return []
        child = parent
    if not loops or not predicates:
        return []

    numeric_parameters = {
        argument.arg
        for argument in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        if argument.annotation is not None and ast.unparse(argument.annotation) in {"int", "float"}
    }

    def arithmetic_input(node: ast.AST, seen: frozenset[str] = frozenset()) -> bool:
        if isinstance(node, ast.Name):
            if node.id in numeric_parameters:
                return not any(
                    isinstance(candidate, ast.Name)
                    and isinstance(candidate.ctx, ast.Store)
                    and candidate.id == node.id
                    and candidate.lineno < read_line
                    for candidate in nodes
                )
            if node.id in seen:
                return False
            values = [
                statement.value
                for statement in nodes
                if isinstance(statement, ast.Assign)
                and statement.lineno < read_line
                and any(isinstance(target, ast.Name) and target.id == node.id for target in statement.targets)
            ]
            return bool(values) and all(arithmetic_input(value, seen | {node.id}) for value in values)
        # Only model the common alignment protocol n // block * block. Merely
        # depending on a numeric parameter is not a satisfiability proof: n-n
        # and n%n are constant zero. Other algebra stays unknown.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            quotient, block = node.left, node.right
            if not (
                isinstance(quotient, ast.BinOp)
                and isinstance(quotient.op, ast.FloorDiv)
                and ast.dump(quotient.right) == ast.dump(block)
            ):
                return False
            if isinstance(block, ast.Constant):
                block_safe = isinstance(block.value, int) and block.value > 0
            elif isinstance(block, ast.Attribute) and isinstance(block.value, ast.Name) and block.value.id == receiver:
                values = [
                    statement.value
                    for statement in _function_scope_nodes(initializer)
                    if isinstance(statement, ast.Assign)
                    and any(_expression_name(target) == f"{init_receiver}.{block.attr}" for target in statement.targets)
                ]
                block_safe = bool(values) and not any(
                    isinstance(value, ast.Constant) and (not isinstance(value.value, int) or value.value <= 0)
                    for value in values
                )
            else:
                block_safe = False
            return block_safe and arithmetic_input(quotient.left, seen)
        return False

    evidence: list[dict[str, object]] = []
    initializer_nodes = list(_function_scope_nodes(initializer))
    init_parents = _parents(initializer)
    for helper in active_helpers:
        initializer_nodes.extend(_function_scope_nodes(helper))
        init_parents.update(_parents(helper))

    def prove(predicate: ast.expr) -> bool:
        if isinstance(predicate, ast.BoolOp) and isinstance(predicate.op, ast.And):
            return all(prove(value) for value in predicate.values)
        if isinstance(predicate, ast.Constant):
            return predicate.value is True
        if (
            isinstance(predicate, ast.Compare)
            and len(predicate.ops) == 1
            and isinstance(predicate.ops[0], ast.Gt)
            and isinstance(predicate.comparators[0], ast.Constant)
            and predicate.comparators[0].value == 0
        ):
            return arithmetic_input(predicate.left)
        if not (
            isinstance(predicate, ast.Attribute)
            and isinstance(predicate.value, ast.Name)
            and predicate.value.id in loops
        ):
            return False
        collection = loops[predicate.value.id]
        matches: list[ast.Assign] = []
        for statement in initializer_nodes:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if not (
                    isinstance(target, ast.Attribute)
                    and target.attr == predicate.attr
                    and isinstance(target.value, ast.Subscript)
                ):
                    continue
                if _expression_name(target.value.value) != f"{init_receiver}.{collection}":
                    continue
                write_scope = helper_scopes.get(id(statement), initializer)
                if any(
                    isinstance(candidate, (ast.Return, ast.Raise)) and candidate.lineno < statement.lineno
                    for candidate in _function_scope_nodes(write_scope)
                ):
                    return False
                if not isinstance(statement.value, ast.Constant) or statement.value.value is not True:
                    return False
                current: ast.AST = statement
                while (ancestor := init_parents.get(id(current))) is not None:
                    if isinstance(ancestor, ast.If):
                        condition = _main_condition_value(ancestor.test, tag_guard_names or set())
                        if condition is not None and condition != (current in ancestor.body):
                            return False
                    current = ancestor
                matches.append(statement)
        if not matches:
            return False
        evidence.append(
            {
                "kind": "constructor_enables_collection_element_flag",
                "collection": collection,
                "flag": predicate.attr,
                "assignment_lines": [item.lineno for item in matches],
                "read_condition": ast.unparse(predicate),
                "initialization_helper_calls": [
                    {"helper": helper_calls[id(item)][0], "constructor_call_line": helper_calls[id(item)][1]}
                    for item in matches
                    if id(item) in helper_calls
                ],
            }
        )
        return True

    if not all(prove(predicate) for predicate in predicates) or not evidence:
        return []
    for item in evidence:
        item["path_conditions"] = [ast.unparse(predicate) for predicate in predicates]
        item["scope"] = "conditional_interface_contract_not_all_workloads"
    return evidence
