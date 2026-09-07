# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
import vllm.model_executor.layers.fused_moe as fused_moe_pkg
import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

from vllm_ascend.patch.platform import patch_fused_moe


@pytest.mark.parametrize("model_name", ["qwen3_moe", "deepseek_v2"])
def test_patch_rebinds_early_model_import_and_preserves_custom_factory(monkeypatch, model_name):
    """Repair cached upstream factories without overwriting custom factories."""
    original = fused_moe_layer.FusedMoE
    stale = ModuleType(f"vllm.model_executor.models.{model_name}")
    stale.__dict__["FusedMoE"] = original
    custom = ModuleType("vllm.model_executor.models.custom_test")
    custom_factory = object()
    custom.__dict__["FusedMoE"] = custom_factory
    unrelated = ModuleType("unrelated_test")
    unrelated.__dict__["FusedMoE"] = original
    monkeypatch.setitem(sys.modules, stale.__name__, stale)
    monkeypatch.setitem(sys.modules, custom.__name__, custom)
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.missing_test", None)

    # Execute the production patch without reloading its global state.
    # Register every binding it may update for restoration after this test.
    monkeypatch.setattr(fused_moe_layer, "FusedMoE", original)
    monkeypatch.setattr(fused_moe_pkg, "FusedMoE", fused_moe_pkg.FusedMoE)
    for name, module in list(sys.modules.items()):
        if name.startswith("vllm.model_executor.models") and module is not None:
            if module.__dict__.get("FusedMoE") is original:
                monkeypatch.setattr(module, "FusedMoE", original)

    patched = runpy.run_path(str(Path(patch_fused_moe.__file__)))

    assert stale.FusedMoE is patched["_ascend_FusedMoE"]
    assert fused_moe_pkg.FusedMoE is stale.FusedMoE
    assert fused_moe_layer.FusedMoE is stale.FusedMoE
    assert custom.FusedMoE is custom_factory
    assert unrelated.FusedMoE is original
