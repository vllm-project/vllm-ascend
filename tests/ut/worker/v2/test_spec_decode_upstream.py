# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import builtins
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

UPSTREAM = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/upstream.py"


@pytest.mark.parametrize("renamed", [False, True])
def test_resolves_same_class_and_patch_module(monkeypatch, renamed):
    original_import = builtins.__import__
    ar_class, manager = object(), object()
    ar_module = SimpleNamespace(TargetDependentARSpeculator=ar_class, AutoRegressiveSpeculator=ar_class)

    def import_module(name, *args, **kwargs):
        if name.startswith("vllm.v1.worker.gpu.spec_decode."):
            if "target_dependent_ar" in name and not renamed:
                raise ModuleNotFoundError(name=name)
            if name.endswith(".speculator"):
                return ar_module
            if name.endswith(".cudagraph_utils"):
                return SimpleNamespace(SpeculatorCudaGraphManager=manager)
            return SimpleNamespace(speculator=ar_module)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    resolved = runpy.run_path(str(UPSTREAM))
    assert resolved["AutoRegressiveSpeculator"] is ar_class
    assert resolved["SpeculatorCudaGraphManager"] is manager
    assert resolved["ar_speculator_module"] is ar_module


def test_does_not_mask_missing_upstream_dependencies(monkeypatch):
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name.startswith("vllm.v1.worker.gpu.spec_decode."):
            raise ModuleNotFoundError(name="torch_npu")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(ModuleNotFoundError) as error:
        runpy.run_path(str(UPSTREAM))
    assert error.value.name == "torch_npu"
