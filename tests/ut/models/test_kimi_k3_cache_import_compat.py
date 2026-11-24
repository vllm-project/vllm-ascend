# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Import compatibility for vLLM cache planning helpers."""

import builtins
import importlib
import importlib.util
import sys
from types import ModuleType

import pytest

CASES = [
    ("vllm_ascend.models.kimi_k3.cache_config", "create_kv_cache_group_specs"),
    ("vllm_ascend.models.kimi_k3.paired_sfa_cache", "may_override_num_blocks"),
]
PLANNING_MODULE = "vllm.v1.core.kv_cache_planning"
LEGACY_MODULE = "vllm.v1.core.kv_cache_utils"


def _module_with_helper(name, helper, value):
    module = ModuleType(name)
    setattr(module, helper, value)
    return module


@pytest.mark.parametrize(("module_name", "helper"), CASES)
@pytest.mark.parametrize("layout", ["planning", "legacy", "planning_dependency_error"])
def test_cache_helper_import_compatibility(monkeypatch, module_name, helper, layout):
    # Execute the actual consumer source in isolation, simulating only upstream
    # imports so the real fallback and exception logic runs unchanged.
    source_module = importlib.import_module(module_name)
    isolated_name = module_name.rpartition(".")[0] + "._cache_import_compat_test"
    spec = importlib.util.spec_from_file_location(isolated_name, source_module.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[isolated_name] = module
    marker = object()
    imports = []
    planning = _module_with_helper(PLANNING_MODULE, helper, marker)
    legacy = _module_with_helper(LEGACY_MODULE, helper, marker)
    real_import = builtins.__import__

    def simulated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == PLANNING_MODULE:
            imports.append(name)
            if layout == "legacy":
                raise ModuleNotFoundError("planning module is absent", name=PLANNING_MODULE)
            if layout == "planning_dependency_error":
                raise ModuleNotFoundError("dependency is absent", name="planning_dependency")
            return planning
        if name == LEGACY_MODULE:
            imports.append(name)
            return legacy
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", simulated_import)
    try:
        if layout == "planning_dependency_error":
            with pytest.raises(ModuleNotFoundError, match="dependency is absent") as exc_info:
                spec.loader.exec_module(module)
            assert exc_info.value.name == "planning_dependency"
            assert LEGACY_MODULE not in imports
        else:
            spec.loader.exec_module(module)
            assert getattr(module, helper) is marker
            expected_imports = [PLANNING_MODULE]
            if layout == "legacy":
                expected_imports.append(LEGACY_MODULE)
            assert imports == expected_imports
    finally:
        sys.modules.pop(isolated_name, None)
