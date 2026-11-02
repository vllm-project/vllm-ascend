# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Preserve native views while forwarding external allocation ownership."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


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
        "_allocate_kv_cache": lambda *args, **kwargs: calls.append(("allocate", kwargs)) or {},
        "_reshape_kv_cache_v2": lambda **kwargs: calls.append(("reshape", kwargs)) or result,
    }
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[function.name], calls, result


@pytest.mark.parametrize("kwargs", [{}, {"allocate": None}])
def test_default_callback_retains_native_allocate_and_reshape(allocator, kwargs):
    function, calls, result = allocator
    assert function(SimpleNamespace(kv_cache_groups=[]), "npu", None, [], **kwargs) is result
    assert [name for name, _ in calls] == ["allocate", "reshape"]
    assert calls[0][1]["allocate"] is None


def test_custom_callback_reaches_raw_allocator(allocator):
    function, calls, result = allocator
    callback = lambda size: torch.empty(size, dtype=torch.uint8)
    assert function(SimpleNamespace(kv_cache_groups=[]), "npu", None, [], allocate=callback) is result
    assert calls[0][1]["allocate"] is callback
    assert [name for name, _ in calls] == ["allocate", "reshape"]


@pytest.mark.parametrize("cache_dtype,expected", [("fp8", torch.float8_e4m3fn), ("turboquant_4bit_nc", torch.int8)])
def test_packed_sfa_cache_retains_native_dtype_and_exact_storage(cache_dtype, expected):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/attn_utils.py"
    branch = next(
        n
        for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "packed_sfa_main_cache"
        and isinstance(n.body[0], ast.Assign)
        and isinstance(n.body[0].targets[0], ast.Name)
        and n.body[0].targets[0].id == "raw_k_tensor"
    )
    raw = torch.arange(256, dtype=torch.uint8)
    namespace = {
        "torch": torch,
        "raw_cache": raw,
        "k_shape": (2, 128, 1, 1),
        "vllm_config": SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype), model_config=object()),
        "TURBOQUANT_CACHE_DTYPE": "turboquant_4bit_nc",
        "native_c8_cache_dtype": lambda *args: torch.float8_e4m3fn,
        "kv_cache_dtype_str_to_dtype": lambda *args: torch.uint8,
    }
    module = ast.fix_missing_locations(ast.Module(body=branch.body[:3], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    view = namespace["k_cache"]
    assert view.dtype == expected
    assert view.data_ptr() == raw.data_ptr()
    assert torch.equal(view.view(torch.uint8).flatten(), raw)
