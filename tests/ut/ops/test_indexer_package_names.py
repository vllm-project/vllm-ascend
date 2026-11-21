# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend.ops import packaged_attention as ops


@pytest.mark.parametrize("name", ["quant_lightning_indexer", "quant_sparse_lightning_indexer"])
@pytest.mark.parametrize("legacy", [False, True])
def test_indexer_package_names(monkeypatch, name, legacy):
    compute, metadata = object(), object()
    suffix = "_dsl" if legacy else ""
    modules = {
        f"ops.{name}{suffix}": SimpleNamespace(**{name: compute}),
        f"ops.{name}_metadata{suffix}": SimpleNamespace(**{f"{name}_metadata": metadata}),
    }

    def load(module):
        if module not in modules:
            raise ModuleNotFoundError(name=module)
        return modules[module]

    monkeypatch.setattr(ops, "import_module", load)
    ops._get_dsl_ops.cache_clear()
    try:
        assert ops._get_dsl_ops(name) == (compute, metadata)
    finally:
        ops._get_dsl_ops.cache_clear()


def test_indexer_package_dependency_error_is_preserved(monkeypatch):
    def load(module):
        raise ModuleNotFoundError(name="cannbotdsl")

    monkeypatch.setattr(ops, "import_module", load)
    ops._get_dsl_ops.cache_clear()
    with pytest.raises(ModuleNotFoundError) as error:
        ops._get_dsl_ops("quant_lightning_indexer")
    assert error.value.name == "cannbotdsl"


@pytest.mark.parametrize("missing", ["metadata", "dependency"])
def test_available_legacy_compute_never_uses_native_metadata(monkeypatch, missing):
    name = "quant_lightning_indexer"
    calls = []
    absent = f"ops.{name}_metadata_dsl" if missing == "metadata" else "cannbot_dsl"

    def load(module):
        calls.append(module)
        if module == f"ops.{name}":
            raise ModuleNotFoundError(name=module)
        if module == f"ops.{name}_dsl":
            return SimpleNamespace(**{name: object()})
        raise ModuleNotFoundError(name=absent)

    monkeypatch.setattr(ops, "import_module", load)
    ops._get_dsl_ops.cache_clear()
    with pytest.raises(ModuleNotFoundError) as error:
        ops._get_dsl_ops(name)
    assert error.value.name == absent
    assert calls == [f"ops.{name}", f"ops.{name}_dsl", f"ops.{name}_metadata_dsl"]


@pytest.mark.parametrize("name", ["quant_lightning_indexer", "quant_sparse_lightning_indexer"])
def test_native_provider_wins_over_stale_legacy_distribution(monkeypatch, name):
    compute, metadata = object(), object()
    calls = []

    def load(module):
        calls.append(module)
        if module == f"ops.{name}":
            return SimpleNamespace(**{name: compute})
        if module == f"ops.{name}_metadata":
            return SimpleNamespace(**{f"{name}_metadata": metadata})
        raise ImportError("Stale legacy module imports an incompatible system DSL")

    monkeypatch.setattr(ops, "import_module", load)
    ops._get_dsl_ops.cache_clear()
    try:
        assert ops._get_dsl_ops(name) == (compute, metadata)
        assert calls == [f"ops.{name}", f"ops.{name}_metadata"]
    finally:
        ops._get_dsl_ops.cache_clear()


def test_available_native_compute_never_uses_legacy_metadata(monkeypatch):
    name = "quant_lightning_indexer"
    calls = []

    def load(module):
        calls.append(module)
        if module == f"ops.{name}":
            return SimpleNamespace(**{name: object()})
        raise ModuleNotFoundError(name=module)

    monkeypatch.setattr(ops, "import_module", load)
    ops._get_dsl_ops.cache_clear()
    try:
        with pytest.raises(ModuleNotFoundError) as error:
            ops._get_dsl_ops(name)
        assert error.value.name == f"ops.{name}_metadata"
        assert calls == [f"ops.{name}", f"ops.{name}_metadata"]
    finally:
        ops._get_dsl_ops.cache_clear()
