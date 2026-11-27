# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep raw probe handle safety covered independently of production sleep."""

import sys
from ctypes import c_void_p
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def probe(monkeypatch):
    logger_module = ModuleType("vllm.logger")
    logger_module.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/device_allocator/hccl_suspend.py"
    spec = spec_from_file_location("_hccl_probe_under_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ctypes_and_nested_handles(probe):
    assert probe.comm_to_int(c_void_p(17)) == 17
    assert probe.comm_to_int(SimpleNamespace(value=c_void_p(23))) == 23
    assert probe.comm_to_int(SimpleNamespace(handle=SimpleNamespace(ptr=29))) == 29
    assert probe.comm_to_int(object()) == 0
    assert probe.comm_to_int(c_void_p()) == 0


def test_resume_failure_retains_pending_handle(probe, monkeypatch):
    monkeypatch.setattr(probe, "hccl_suspend_available", lambda: True)
    probe._suspended_handles.add(17)
    probe._resume_fn = MagicMock(return_value=1)
    assert probe.resume_hccl_comm(17) is False
    assert probe._suspended_handles == {17}


def test_raw_fallback_cannot_resume_a_replaced_pg_handle(probe, monkeypatch):
    torch = ModuleType("torch")
    torch.device = str
    torch.npu = SimpleNamespace(current_device=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(probe, "get_backend_hccl_comm", lambda _: 29)
    resume = MagicMock()
    monkeypatch.setattr(probe, "resume_hccl_comm", resume)
    group = SimpleNamespace(_get_backend=lambda _: object())
    assert probe.resume_process_group_hccl(group, expected_handle=17) is False
    resume.assert_not_called()
