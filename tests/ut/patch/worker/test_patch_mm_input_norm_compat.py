# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_mm_input_norm.py"
NORM_MODULE = "vllm.model_executor.layers.fusion.mm_input_norm"


@pytest.mark.parametrize("missing", ["vllm.model_executor.layers.fusion", NORM_MODULE])
def test_patch_skips_unavailable_normalizer(monkeypatch, missing):
    original_import = builtins.__import__

    def import_without_normalizer(name, *args, **kwargs):
        if name == NORM_MODULE:
            raise ModuleNotFoundError(name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_normalizer)
    # Module-level installation and repeated installation must both be safe.
    namespace = runpy.run_path(str(PATCH_PATH))
    namespace["patch_mm_input_norm"]()


def test_patch_does_not_hide_broken_dependencies(monkeypatch):
    original_import = builtins.__import__

    def import_with_broken_dependency(name, *args, **kwargs):
        if name == NORM_MODULE:
            raise ModuleNotFoundError(name="broken_dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_broken_dependency)
    with pytest.raises(ModuleNotFoundError) as exc:
        runpy.run_path(str(PATCH_PATH))
    assert exc.value.name == "broken_dependency"


@pytest.mark.parametrize("persistent", [True, False])
def test_patch_preserves_buffers_and_is_idempotent(monkeypatch, persistent):
    # Exercise the persistence contract even on v0.30, which lacks the module.
    class InputNorm(torch.nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.register_buffer("weight", weight, persistent=persistent)
            self.register_buffer("bias", bias, persistent=persistent)

    module = ModuleType(NORM_MODULE)
    module.FusedMMInputNorm = InputNorm
    monkeypatch.setitem(sys.modules, NORM_MODULE, module)
    namespace = runpy.run_path(str(PATCH_PATH))
    patched_init = InputNorm.__init__
    namespace["patch_mm_input_norm"]()
    assert InputNorm.__init__ is patched_init

    weight, bias = torch.randn(3), torch.randn(3)
    norm = InputNorm(weight, bias)
    assert norm.weight is weight
    assert norm.bias is bias
    assert set(dict(norm.named_buffers())) == {"weight", "bias"}
    assert not norm.state_dict()
    norm.to(dtype=torch.float64)
    torch.testing.assert_close(norm.weight, weight.double())
    torch.testing.assert_close(norm.bias, bias.double())
