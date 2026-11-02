# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Preserve the native allocation path and reject unsupported custom ownership."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def allocator():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/attn_utils.py"
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "allocate_kv_cache_main"
    )
    for argument in function.args.args + function.args.kwonlyargs:
        argument.annotation = None
    function.returns = None
    calls = []
    result = {"native_cache": object()}
    namespace = {
        "get_current_vllm_config": lambda: SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8")),
        "get_layers_from_vllm_config": lambda *args: {},
        "AttentionLayerBase": object,
        "_allocate_kv_cache": lambda *args, **kwargs: calls.append("allocate") or {},
        "_reshape_kv_cache_v2": lambda **kwargs: calls.append("reshape") or result,
    }
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[function.name], calls, result


@pytest.mark.parametrize("kwargs", [{}, {"allocate": None}])
def test_default_callback_retains_native_allocate_and_reshape(allocator, kwargs):
    function, calls, result = allocator
    assert function(SimpleNamespace(kv_cache_groups=[]), "npu", None, [], **kwargs) is result
    assert calls == ["allocate", "reshape"]


def test_custom_callback_is_not_silently_ignored(allocator):
    function, calls, _ = allocator
    with pytest.raises(NotImplementedError, match="external allocation callback"):
        function(SimpleNamespace(kv_cache_groups=[]), "npu", None, [], allocate=lambda size: None)
    assert not calls
