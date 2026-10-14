# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the canonical capability detector without importing device plugins."""

import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def detector():
    path = ROOT / "vllm_ascend/attention/utils.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_has_flashmla_module", "get_flashmla_ops")
    ]
    scope = {"lru_cache": lru_cache, "find_spec": Mock(return_value=object()), "import_module": Mock()}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"), scope
    )
    return scope


def test_absent_package_never_imports_ops(detector):
    detector["find_spec"].return_value = None
    assert detector["get_flashmla_ops"]() is None
    detector["import_module"].assert_not_called()


@pytest.mark.parametrize(
    "apis",
    [{}, {"flash_mla": Mock()}, {"flash_mla": 1, "flash_mla_metadata": Mock()}],
)
def test_incomplete_apis_fall_back(detector, apis):
    detector["import_module"].return_value = SimpleNamespace(**apis)
    assert detector["get_flashmla_ops"]() is None


def test_complete_apis_are_selected(detector):
    attention, metadata = Mock(), Mock()
    detector["import_module"].return_value = SimpleNamespace(flash_mla=attention, flash_mla_metadata=metadata)
    assert detector["get_flashmla_ops"]() == (attention, metadata)


@pytest.mark.parametrize("missing", ["cann_ops_transformer", "cann_ops_transformer.ops", "internal_dependency"])
def test_import_failure_boundary(detector, missing):
    detector["import_module"].side_effect = ModuleNotFoundError(missing, name=missing)
    if missing == "internal_dependency":
        with pytest.raises(ModuleNotFoundError, match=missing):
            detector["get_flashmla_ops"]()
    else:
        assert detector["get_flashmla_ops"]() is None
