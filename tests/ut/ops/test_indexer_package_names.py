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
