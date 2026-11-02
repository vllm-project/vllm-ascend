# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Forward the current Core hash-proof window through all coordinator routes."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_kv_cache_coordinator.py"


@pytest.mark.parametrize("route", ["hisparse", "ordinary", "ascend_hybrid"])
@pytest.mark.parametrize("lookahead", [None, 0, 11])
def test_factory_forwards_hash_lookahead_without_breaking_old_callers(route, lookahead):
    function = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "get_kv_cache_coordinator"
    )
    for arg in function.args.args:
        arg.annotation = None
    function.returns = None
    calls = []

    def original(**kwargs):
        calls.append(("original", kwargs))
        return kwargs

    def hybrid(**kwargs):
        calls.append(("hybrid", kwargs))
        return kwargs

    namespace = dict(
        _select_kv_token_budget=lambda length, budget: length if budget is None else budget,
        KVCacheGroupRole=SimpleNamespace(HISPARSE_SOURCE="source"),
        _orig_get_kv_cache_coordinator=original,
        AscendHybridKVCacheCoordinator=hybrid,
        is_prefix_cacheable=lambda spec: True,
        _is_deepseek_v4_kv_cache_config=lambda config: route == "ascend_hybrid",
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(SOURCE), "exec"), namespace
    )
    groups = [SimpleNamespace(role="source" if route == "hisparse" else "ordinary", kv_cache_spec=object())]
    options = {} if lookahead is None else {"num_hash_lookahead_tokens": lookahead}
    result = namespace[function.name](
        SimpleNamespace(kv_cache_groups=groups), 4096, max_in_flight_tokens=128, **options
    )
    assert calls[0][0] == ("hybrid" if route == "ascend_hybrid" else "original")
    assert result["max_in_flight_tokens"] == 128
    if lookahead is None:
        assert "num_hash_lookahead_tokens" not in result
    else:
        assert result["num_hash_lookahead_tokens"] == lookahead


@pytest.mark.parametrize("lookahead", [None, 0, 11])
def test_hybrid_block_pool_receives_the_actual_hash_proof_window(lookahead):
    owner = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendHybridKVCacheCoordinator"
    )
    init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    selected = [
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(ast.unparse(target) in ("hash_lookahead_kwargs", "self.block_pool") for target in n.targets)
    ]
    assert len(selected) == 2
    instance = SimpleNamespace()
    namespace = dict(
        self=instance,
        num_hash_lookahead_tokens=lookahead,
        kv_cache_config=SimpleNamespace(num_blocks=100),
        enable_caching=True,
        hash_block_size=128,
        enable_kv_cache_events=False,
        metrics_collector=None,
        BlockPool=lambda **kwargs: kwargs,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(SOURCE), "exec"), namespace)
    expected = {} if lookahead is None else {"num_hash_lookahead_tokens": lookahead}
    actual = instance.block_pool
    assert actual.pop("num_gpu_blocks") == 100
    assert {key: value for key, value in actual.items() if key == "num_hash_lookahead_tokens"} == expected
