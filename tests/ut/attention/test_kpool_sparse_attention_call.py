# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kpool sparse attention call must match the operator's real schema.

``npu_sparse_flash_attention`` names the key-side length and layout ``*_kv``,
while ``npu_lightning_indexer`` -- called a few lines earlier in the same chain
-- names its own ``layout_key``. Passing the indexer's spelling to the
attention operator raises ``Unknown keyword argument`` only once a request is
actually decoded, i.e. after a full multi-card model load, so pin the call
against the schema this repo registers for the operator instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[3]
KPOOL_MLA = ROOT / "vllm_ascend" / "attention" / "kpool_mla_v1.py"
TORCH_BINDING = ROOT / "csrc" / "torch_binding.cpp"
OPERATOR = "npu_sparse_flash_attention"
# (attention_out, softmax_max, softmax_sum), returned whether or not
# return_softmax_lse was asked for.
OPERATOR_RETURN_ARITY = 3


def _schema_parameter_names() -> set[str]:
    """Parameter names of the operator as ``ops.def`` declares them."""
    source = TORCH_BINDING.read_text(encoding="utf-8")
    start = source.index(f'"{OPERATOR}(')
    declaration = source[start : source.index(");", start)]
    # The schema is written as a run of adjacent C++ string literals.
    schema = "".join(re.findall(r'"([^"]*)"', declaration))
    arguments = schema[schema.index("(") + 1 : schema.rindex("->")].rstrip(") ")
    names = set()
    for argument in arguments.split(","):
        argument = argument.strip().removeprefix("*").strip()
        if argument:
            names.add(argument.split("=")[0].split()[-1])
    return names


def _module() -> ast.Module:
    return ast.parse(KPOOL_MLA.read_text(encoding="utf-8"))


def _operator_call(module: ast.Module) -> ast.Call:
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == OPERATOR
    ]
    assert len(calls) == 1, f"expected one {OPERATOR} call in {KPOOL_MLA.name}, found {len(calls)}"
    return calls[0]


def _module_constants(module: ast.Module) -> dict[str, int]:
    return {
        node.targets[0].id: node.value.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int)
    }


def test_schema_is_parsed_at_all() -> None:
    """Guard the guard: a parse that silently returns nothing proves nothing."""
    names = _schema_parameter_names()

    assert {"query", "key", "value", "sparse_indices", "scale_value"} <= names
    assert {"actual_seq_lengths_kv", "layout_kv"} <= names
    # The spellings npu_lightning_indexer uses and this operator does not.
    assert not {"actual_seq_lengths_key", "layout_key"} & names


def test_every_keyword_passed_exists_in_the_schema() -> None:
    call = _operator_call(_module())
    passed = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
    unknown = passed - _schema_parameter_names()

    assert not unknown, (
        f"{sorted(unknown)} are not parameters of {OPERATOR}; the operator "
        "raises 'Unknown keyword argument' for these only at decode time."
    )


def test_the_three_result_tensors_are_unpacked() -> None:
    """Returning the raw tuple would hand a 3-tuple to the up-projection."""
    module = _module()
    assignment = next(
        node for node in ast.walk(module) if isinstance(node, ast.Assign) and node.value is _operator_call(module)
    )

    target = assignment.targets[0]
    assert isinstance(target, ast.Tuple) and len(target.elts) == OPERATOR_RETURN_ARITY, (
        f"{OPERATOR} returns {OPERATOR_RETURN_ARITY} tensors; unpack them so the "
        "attention output rather than the tuple reaches the up-projection."
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [("sparse_mode", 3), ("attention_mode", 2), ("sparse_block_size", 1)],
)
def test_mode_constants_match_the_probed_values(name: str, expected: int) -> None:
    """``sparse_mode`` and ``attention_mode`` are independent axes.

    Sharing one constant between them silently swapped the causal crop (3,
    RightDownCausal) for the MLA kernel selector (2).
    """
    module = _module()
    keyword = next(k for k in _operator_call(module).keywords if k.arg == name)

    assert isinstance(keyword.value, ast.Name), f"{name} should be passed as a named constant"
    assert _module_constants(module)[keyword.value.id] == expected
