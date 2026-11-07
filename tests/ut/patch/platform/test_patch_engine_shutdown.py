# SPDX-License-Identifier: Apache-2.0
"""Preserve request-drain deadlines while allowing NPU process cleanup."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("device", ["npu", "cuda", "rocm"])
@pytest.mark.parametrize("request_timeout,remaining", [(0, 0), (0, None), (0, 3), (30, 0), (30, 2), (None, 0)])
def test_only_npu_immediate_abort_gets_new_cleanup_grace(monkeypatch, device, request_timeout, remaining):
    upstream = Mock(return_value=remaining)
    module = types.ModuleType("vllm.v1.engine.utils")
    module.get_engine_process_shutdown_timeout = upstream
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.utils", module)
    platform = types.ModuleType("vllm.platforms")
    platform.current_platform = types.SimpleNamespace(device_type=device)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform)
    for name in ("vllm", "vllm.v1", "vllm.v1.engine"):
        parent = types.ModuleType(name)
        parent.__path__ = []
        monkeypatch.setitem(sys.modules, name, parent)
    sys.modules["vllm.v1.engine"].utils = module
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_engine_shutdown.py"
    spec = importlib.util.spec_from_file_location("_engine_shutdown_ut", path)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    actual = module.get_engine_process_shutdown_timeout(request_timeout, remaining)
    if device == "npu" and request_timeout == 0 and remaining in (None, 0):
        assert actual == 15
        upstream.assert_not_called()
    else:
        assert actual == remaining
        upstream.assert_called_once_with(request_timeout, remaining)
