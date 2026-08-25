# SPDX-License-Identifier: Apache-2.0
"""Source-level memory-safety regressions for rejection-sampler kernels.

The invalid predecessor read is allocator-dependent at runtime: a numerically
correct result does not prove that ``ptr[-1]`` was not evaluated.  Keep an
import-free source check alongside the NPU test so CI deterministically guards
the load masks without initializing a device runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TypeGuard

ROOT = Path(__file__).resolve().parents[4]
KERNEL = ROOT / "vllm_ascend" / "ops" / "triton" / "reject_sample.py"


def _function(name: str) -> ast.FunctionDef:
    for node in ast.parse(KERNEL.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {KERNEL}")


def _is_tl_call(node: ast.AST, name: str) -> TypeGuard[ast.Call]:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == name
    )


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _assignment(function: ast.FunctionDef, name: str) -> ast.AST:
    values = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(values) == 1, f"expected one assignment to {name}, found {len(values)}"
    return values[0]


def _resolve_name(function: ast.FunctionDef, node: ast.AST) -> ast.AST:
    return _assignment(function, node.id) if isinstance(node, ast.Name) else node


def _names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _has_positive_index_guard(node: ast.AST, index_name: str) -> bool:
    return any(
        isinstance(child, ast.Compare)
        and isinstance(child.left, ast.Name)
        and child.left.id == index_name
        and len(child.ops) == 1
        and isinstance(child.ops[0], ast.Gt)
        and len(child.comparators) == 1
        and isinstance(child.comparators[0], ast.Constant)
        and child.comparators[0].value == 0
        for child in ast.walk(node)
    )


def _predecessor_load(function: ast.FunctionDef, pointer_name: str) -> ast.Call:
    loads = [
        node
        for node in ast.walk(function)
        if _is_tl_call(node, "load")
        and node.args
        and pointer_name in _names(node.args[0])
        and "- 1" in ast.unparse(node.args[0])
    ]
    assert len(loads) == 1, f"expected one predecessor load in {function.name}, found {len(loads)}"
    return loads[0]


def test_predecessor_loads_mask_request_zero_and_padded_lanes() -> None:
    cases = {
        "rejection_greedy_sample_triton": (
            "cu_num_draft_tokens_ptr",
            "offset",
            "is_greedy_mask",
        ),
        "rejection_random_sample_kernel": (
            "cu_num_draft_tokens_ptr",
            "offsets",
            "not_greedy_mask",
        ),
        "expand_kernel": ("cu_num_tokens_ptr", "offset", "len_mask"),
        "sample_recovered_tokens_kernel": (
            "cu_num_draft_tokens_ptr",
            "req_idx",
            None,
        ),
    }

    for function_name, (pointer_name, index_name, lane_mask_name) in cases.items():
        function = _function(function_name)
        load = _predecessor_load(function, pointer_name)
        mask = _keyword(load, "mask")
        other = _keyword(load, "other")

        assert mask is not None, f"{function_name} predecessor load must be masked"
        resolved_mask = _resolve_name(function, mask)
        assert _has_positive_index_guard(resolved_mask, index_name), (
            f"{function_name} predecessor mask must reject index zero"
        )
        if lane_mask_name is not None:
            assert lane_mask_name in _names(resolved_mask), (
                f"{function_name} predecessor mask must reject padded/non-target lanes"
            )
        assert isinstance(other, ast.Constant) and other.value == 0, (
            f"{function_name} predecessor load must use neutral other=0"
        )


def test_expand_kernel_masks_padded_lane_reads_and_writes() -> None:
    function = _function("expand_kernel")

    input_loads = [
        node
        for node in ast.walk(function)
        if _is_tl_call(node, "load") and node.args and "input_ptr" in _names(node.args[0])
    ]
    assert len(input_loads) == 1
    input_mask = _keyword(input_loads[0], "mask")
    assert input_mask is not None
    assert ast.unparse(input_mask) == "len_mask"
    input_other = _keyword(input_loads[0], "other")
    assert isinstance(input_other, ast.Constant) and input_other.value == 0

    stores = [node for node in ast.walk(function) if _is_tl_call(node, "store")]
    assert len(stores) == 1
    store_mask = _keyword(stores[0], "mask")
    assert store_mask is not None
    assert "valid_req" in _names(store_mask)

    valid_req = _assignment(function, "valid_req")
    assert isinstance(valid_req, ast.Call)
    assert "len_mask" in _names(valid_req)
