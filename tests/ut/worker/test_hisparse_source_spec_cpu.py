# SPDX-License-Identifier: Apache-2.0
"""Native C8 Host geometry must be declared before Core storage allocation."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("leader", [False, True])
def test_native_host_source_spec_bytes_and_dtype_preserve_original(c8, leader):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/attn_utils.py"
    tree = ast.parse(path.read_text())
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "isinstance(attn_module, MLAAttention) and vllm_config.attention_config.hisparse_config is not None"
    )

    @dataclass(frozen=True)
    class Spec:
        head_size: int
        dtype: torch.dtype
        is_index_group_leader: bool = False
        block_size: int = 128

    class Layer:
        pass

    layer = Layer()
    layer.impl = SimpleNamespace(enable_sparse_sfa_c8=c8, c8_cache_dtype=torch.float8_e4m3fn)
    layer.hisparse_cache = SimpleNamespace(runtime=SimpleNamespace(is_group_leader=leader))
    original = Spec(head_size=576, dtype=torch.uint8 if c8 else torch.bfloat16)
    namespace = dict(
        attn_module=layer,
        MLAAttention=Layer,
        vllm_config=SimpleNamespace(attention_config=SimpleNamespace(hisparse_config=object())),
        spec=original,
        replace=replace,
        kv_cache_spec={},
        layer_name="attention",
    )
    # Keep the production branch inside its real loop so its continue is valid.
    code = ast.Module(
        body=[
            ast.For(
                target=ast.Name(id="once", ctx=ast.Store()),
                iter=ast.List(elts=[ast.Constant(0)], ctx=ast.Load()),
                body=[branch],
                orelse=[],
            )
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    actual = namespace["kv_cache_spec"]["attention"]
    assert actual.is_index_group_leader is leader
    assert actual.head_size == (656 if c8 else 576)
    assert actual.dtype == (torch.float8_e4m3fn if c8 else torch.bfloat16)
    assert actual.block_size * actual.head_size * actual.dtype.itemsize == 128 * (656 if c8 else 1152)
    assert original.head_size == 576 and original.dtype == (torch.uint8 if c8 else torch.bfloat16)
