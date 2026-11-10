# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("plan_name", ["KDAStateCopyPlan", "StridedKDAFallbackPlan"])
def test_metadata_import_and_unsupported_prepare_never_load_triton(monkeypatch, plan_name):
    """A 310P worker can import metadata and reject KDA before device access."""
    modules: dict[str, dict[str, object]] = {
        "vllm.forward_context": {"get_forward_context": Mock()},
        "vllm.utils.torch_utils": {"direct_register_custom_op": Mock()},
        "vllm_ascend.utils": {"is_950": lambda: False, "supports_triton": lambda: False},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("vllm_ascend.ops.triton") or "jit_warmup_triton_helper" in name:
            raise AssertionError(f"Unsupported worker imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    namespace = runpy.run_path(str(ROOT / "vllm_ascend/ops/kda_state_copy_plan.py"))
    with pytest.raises(ValueError, match="Triton-capable hardware"):
        namespace[plan_name].prepare(object(), 4)
