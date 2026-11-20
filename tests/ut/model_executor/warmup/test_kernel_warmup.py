# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``kernel_warmup``."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.model_executor.warmup.helpers import make_mock_vllm_config, make_mock_worker, make_registry

kw = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")


@patch.object(kw, "logger")
@patch.object(kw, "triton_rms_warmup")
@patch.object(kw, "penalties_triton_warmup")
@patch.object(kw, "rejection_sampler_triton_warmup")
@patch.object(kw, "indexer_triton_warmup")
@patch.object(kw, "HAS_TRITON", True)
def test_kernel_warmup(mock_indexer, mock_rej, mock_pen, mock_rms, mock_logger):
    worker = make_mock_worker()
    kw.kernel_warmup(worker)

    mock_rej.assert_called_once_with(worker)
    mock_pen.assert_called_once_with(worker)
    mock_rms.assert_called_once_with(worker)
    mock_indexer.assert_called_once_with(worker)


def test_kda_copy_layouts_compile_after_release_warmups_without_touching_cache(monkeypatch):
    from vllm_ascend.ops import kda_state_copy_plan as state_copy

    config = make_mock_vllm_config()
    registry = make_registry(config)
    state = torch.full((4, 2, 4, 4), 123.0)
    monkeypatch.setattr(state_copy, "_validate_cache_layout", lambda _: (32, 4, 32))
    monkeypatch.setattr(state_copy, "_configuration", lambda: ("test",))
    monkeypatch.setattr(state_copy._kda_state_copy_kernel, "pre_run_hooks", [], raising=False)
    compiled = []
    monkeypatch.setattr(state_copy.StateCopyJitKernel, "compile", lambda self, key: compiled.append(key))
    with registry.activate():
        plan = state_copy.KDAStateCopyPlan.prepare(state, max_selected=8)
        plan.seal()
    assert compiled == []
    assert len(plan._compile_keys) == 4
    assert all(not isinstance(value, torch.Tensor) for key in plan._compile_keys.values() for _, value in key.arguments)
    worker = SimpleNamespace(vllm_config=config, model_runner=SimpleNamespace(jit_warmup_registry=registry))
    legacy = []
    monkeypatch.setattr(kw, "HAS_TRITON", True)
    for name in (
        "rejection_sampler_triton_warmup",
        "penalties_triton_warmup",
        "triton_rms_warmup",
        "indexer_triton_warmup",
    ):
        monkeypatch.setattr(kw, name, lambda _: legacy.append(len(compiled)))
    kw.kernel_warmup(worker)
    assert legacy == [0, 0, 0, 0]
    assert set(compiled) == set(plan._compile_keys.values())
    torch.testing.assert_close(state, torch.full_like(state, 123))
