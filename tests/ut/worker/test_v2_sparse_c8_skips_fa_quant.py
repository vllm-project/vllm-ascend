# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""V2 sparse C8 reshape must not call enable_fa_quant."""

import ast
from pathlib import Path

ATTN_UTILS = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/attn_utils.py"


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(ATTN_UTILS.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _parents(root: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(root):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _is_sparse_sfa_c8_test(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "sparse_sfa_c8"


def test_reshape_calls_enable_fa_quant_only_after_sparse_c8() -> None:
    function = _function("_reshape_kv_cache_v2")
    parents = _parents(function)
    calls = [node for node in ast.walk(function) if _call_name(node) == "enable_fa_quant"]
    assert len(calls) == 1

    if_node = parents[calls[0]]
    assert isinstance(if_node, ast.If)
    assert if_node.test is calls[0]

    cursor: ast.AST = if_node
    guarded = False
    while cursor in parents:
        parent = parents[cursor]
        if isinstance(parent, ast.If) and _is_sparse_sfa_c8_test(parent.test) and cursor in parent.orelse:
            guarded = True
            break
        cursor = parent
    assert guarded

    sparse_ifs = [node for node in ast.walk(function) if isinstance(node, ast.If) and _is_sparse_sfa_c8_test(node.test)]
    assert any(_call_name(child) == "kv_cache_dtype_str_to_dtype" for node in sparse_ifs for child in ast.walk(node))
