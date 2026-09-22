# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib
import sys

import pytest


def test_engram_patch_is_noop_without_upstream_module(monkeypatch):
    """vLLM 0.29 nightly images have no ``vllm.config.engram``."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "vllm.config.engram":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    sys.modules.pop("vllm_ascend.patch.platform.patch_engram_config", None)
    try:
        module = importlib.import_module("vllm_ascend.patch.platform.patch_engram_config")
        assert not hasattr(module, "AscendEngramConfig")
    finally:
        monkeypatch.undo()
        sys.modules.pop("vllm_ascend.patch.platform.patch_engram_config", None)
        importlib.import_module("vllm_ascend.patch.platform.patch_engram_config")


def test_engram_patch_defines_ascend_config_when_upstream_present():
    spec = importlib.util.find_spec("vllm.config.engram")
    if spec is None:
        pytest.skip("this vLLM build does not expose vllm.config.engram")
    from vllm_ascend.patch.platform.patch_engram_config import AscendEngramConfig

    assert AscendEngramConfig().dp_shared_memory is False
