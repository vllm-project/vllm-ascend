# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib.util
import runpy
from pathlib import Path


def test_engram_patch_is_noop_without_upstream_config(monkeypatch):
    original_import = builtins.__import__

    def without_engram(name, *args, **kwargs):
        if name == "vllm.config.engram":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_engram)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(str(root / "vllm_ascend/patch/platform/patch_engram_config.py"))
    assert "AscendEngramConfig" not in namespace
