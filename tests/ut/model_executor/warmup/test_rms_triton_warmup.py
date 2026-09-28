# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for kernel-owned RMS warmup."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

rw = importlib.import_module("vllm_ascend.model_executor.warmup.rms_triton_warmup")


def _make_vllm_config():
    model_config = SimpleNamespace(dtype=torch.float16, get_head_size=lambda: 128)
    return SimpleNamespace(
        model_config=model_config,
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
    )


def test_model_uses_triton_q_rms_accepts_backend_subclasses():
    class DummyDSABackend:
        pass

    class DummyDSAChildBackend(DummyDSABackend):
        pass

    fake_module = SimpleNamespace(AscendDSABackend=DummyDSABackend)
    model_runner = SimpleNamespace(attn_groups=[[SimpleNamespace(backend=DummyDSAChildBackend)]])

    with patch.dict(sys.modules, {"vllm_ascend.attention.dsa_v1": fake_module}):
        assert rw._model_uses_triton_q_rms(model_runner)


def test_triton_rms_warmup_direct_uses_compile_only_path():
    vllm_config = _make_vllm_config()
    worker = SimpleNamespace(vllm_config=vllm_config, model_runner=SimpleNamespace())

    with (
        patch.object(rw, "HAS_TRITON", True),
        patch.object(rw, "_model_uses_triton_q_rms", return_value=True),
        patch.object(rw._TRITON_Q_RMS_KERNEL, "compile") as mock_compile,
    ):
        rw.triton_rms_warmup(worker)

    assert mock_compile.call_count == 5


def test_triton_rms_warmup_registers_with_registry():
    vllm_config = _make_vllm_config()
    registry = JitWarmupRegistry(vllm_config)
    worker = SimpleNamespace(
        vllm_config=vllm_config,
        model_runner=SimpleNamespace(jit_warmup_registry=registry),
    )

    with (
        patch.object(rw, "HAS_TRITON", True),
        patch.object(rw, "_model_uses_triton_q_rms", return_value=True),
        patch.object(rw._TRITON_Q_RMS_KERNEL, "compile") as mock_compile,
    ):
        with registry.activate():
            assert rw.register_triton_rms_warmup(worker)
        registry.warmup()

    assert mock_compile.call_count == 5
