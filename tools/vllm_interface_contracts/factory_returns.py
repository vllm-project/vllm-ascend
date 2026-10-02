# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared bounded body and result proof for source-derived factory evaluation."""

from __future__ import annotations

import ast
from dataclasses import replace

from .type_flow import ContainerFlow, FlowValue, evaluated_factory_value, guarded_factory_value, literal_index


def factory_body_supported(function: ast.FunctionDef) -> bool:
    """Reject runtime effects outside the existing pure factory flow grammar."""
    for node in ast.walk(function):
        if (
            isinstance(node, ast.stmt)
            and node is not function
            and not isinstance(
                node, (ast.Assign, ast.AnnAssign, ast.If, ast.Return, ast.Expr, ast.Pass, ast.Assert, ast.For)
            )
        ):
            return False
        if isinstance(node, ast.Assign) and not all(isinstance(target, ast.Name) for target in node.targets):
            return False
        if isinstance(node, ast.AnnAssign) and not isinstance(node.target, ast.Name):
            return False
        if isinstance(node, ast.expr) and not isinstance(
            node,
            (
                ast.Name,
                ast.Constant,
                ast.Call,
                ast.Subscript,
                ast.Tuple,
                ast.List,
                ast.Compare,
                ast.Attribute,
                ast.ListComp,
            ),
        ):
            if isinstance(node, ast.UnaryOp) and literal_index(node) is not None:
                continue
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                continue
            return False
    return True


def proven_factory_return(flow: ContainerFlow, parameters: dict[str, FlowValue | None]) -> FlowValue | None:
    """Keep one proven result, its borrowed identity and all prior obligations."""
    if (
        any(value is None for value in parameters.values())
        or not flow.factory_safe
        or not flow.return_values
        or any(value is None or not value.constructed for value in flow.return_values)
    ):
        return None
    input_roots = frozenset(root for value in parameters.values() if value is not None for root in value.roots)
    variants = {
        value if value.owned in input_roots else replace(value, owned=None)
        for value in flow.return_values
        if value is not None
    }
    if len(variants) != 1:
        return None
    result = guarded_factory_value(next(iter(variants)), tuple(flow.factory_requirements))
    return (
        evaluated_factory_value(result, tuple(value for value in parameters.values() if value is not None))
        if result is not None
        else None
    )
