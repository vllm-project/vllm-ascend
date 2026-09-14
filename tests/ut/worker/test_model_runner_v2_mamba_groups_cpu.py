# SPDX-License-Identifier: Apache-2.0
"""Run the MRv2 Mamba group lookup on CPU without importing the NPU stack."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@dataclass(frozen=True)
class MambaSpec:
    block_size: int = 16
    shapes: tuple = ((2, 3), (2, 2))


@dataclass
class UniformTypeKVCacheSpecs:
    kv_cache_specs: dict


@pytest.fixture
def lookup():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_states/mamba_hybrid.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_mamba_group_info"
    )
    namespace = {
        "MambaSpec": MambaSpec,
        "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs,
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(method), str(source), "exec"), namespace)
    return namespace[method.name]


def group(spec):
    return SimpleNamespace(kv_cache_spec=spec)


@pytest.mark.parametrize("wrapped_ids", [(), (1, 3), (1,)], ids=["plain", "wrapped", "mixed"])
def test_preserves_original_group_indices_and_wrappers(lookup, wrapped_ids):
    mamba = MambaSpec()
    attention = object()
    specs = [attention, mamba, attention, mamba]
    for index in (0, *wrapped_ids):
        specs[index] = UniformTypeKVCacheSpecs({f"layer.{index}.{j}": specs[index] for j in range(2)})
    groups = [group(spec) for spec in specs]
    config = SimpleNamespace(kv_cache_groups=groups)
    state = SimpleNamespace(_mamba_spec=None, _mamba_group_ids=[])

    ids, spec = lookup(state, config)

    assert ids == [1, 3]
    assert spec is mamba
    assert state._mamba_group_ids is ids and state._mamba_spec is spec
    assert config.kv_cache_groups is groups
    assert all(entry.kv_cache_spec is original for entry, original in zip(groups, specs))
    # Subsequent state preprocessing must reuse the cached result, without
    # revisiting or normalizing the scheduler's cache configuration.
    cached_ids, cached_spec = lookup(state, None)
    assert cached_ids is ids and cached_spec is spec


@pytest.mark.parametrize("case", ["missing", "plain", "wrapped", "within_wrapper"])
def test_rejects_missing_or_inconsistent_specs_without_caching(lookup, case):
    spec = MambaSpec()
    other = replace(spec, shapes=((2, 4), (2, 2)))
    if case == "missing":
        groups = [group(UniformTypeKVCacheSpecs({"attention": object()}))]
    elif case == "plain":
        groups = [group(spec), group(replace(spec, block_size=32))]
    elif case == "wrapped":
        groups = [group(spec), group(UniformTypeKVCacheSpecs({"other.0": other, "other.1": other}))]
    else:
        groups = [group(UniformTypeKVCacheSpecs({"mamba.0": spec, "mamba.1": other}))]
    state = SimpleNamespace(_mamba_spec=None, _mamba_group_ids=[])
    expected = "no mamba layers in the model" if case == "missing" else "Mamba specs are inconsistent"

    with pytest.raises(AssertionError, match=expected):
        lookup(state, SimpleNamespace(kv_cache_groups=groups))

    assert state._mamba_spec is None and state._mamba_group_ids == []
