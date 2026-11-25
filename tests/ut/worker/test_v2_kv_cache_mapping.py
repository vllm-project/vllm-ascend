# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the existing V2 cache consumers without importing NPU dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock


def initialize_cache_consumers(mapping, device):
    """Execute the production initialization tail with allocation already done."""
    path = Path(__file__).parents[3] / "vllm_ascend/patch/worker/patch_model_runner.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "initialize_kv_cache")
    start = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "register_sparse_kv_caches" for target in node.targets)
    )
    runner = NS(
        device=device,
        vllm_config=NS(aux_output_config=NS(enabled=False)),
        _register_sparse_kv_caches=Mock(),
    )
    connector = Mock()
    namespace = dict(
        self=runner,
        kv_caches_dict=mapping,
        upstream=NS(get_kv_connector=connector),
        is_profiling=False,
        vllm_version_is=lambda version: version == "0.33.0",
    )
    exec(compile(ast.Module(body=function.body[start:], type_ignores=[]), str(path), "exec"), namespace)
    return runner, connector


def test_tuple_list_and_tensor_caches_preserve_identity_and_layer_mapping():
    key, value, scale, single = [NS(device="npu:0") for _ in range(4)]
    host = NS(device="cpu")
    mapping = {"kv": (key, value), "scale": [scale, None], "single": single, "host": host}
    runner, connector = initialize_cache_consumers(mapping, "npu:0")
    assert [id(tensor) for tensor in runner.kv_caches] == [id(tensor) for tensor in (key, value, scale, single)]
    assert connector.call_args.args[1] is mapping
    assert runner._register_sparse_kv_caches.call_args.args[0] is mapping
