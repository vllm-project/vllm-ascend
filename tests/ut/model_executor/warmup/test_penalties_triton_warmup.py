# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for kernel-owned penalties/bincount warmup."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

pw = importlib.import_module("vllm_ascend.model_executor.warmup.penalties_triton_warmup")
penalty_module = importlib.import_module("vllm_ascend.ops.triton.penalty")


def _make_vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float16, get_vocab_size=lambda: 1024),
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
    )


def test_penalties_triton_warmup_uses_compile_only_path():
    worker = SimpleNamespace(vllm_config=_make_vllm_config(), model_runner=SimpleNamespace())

    with (
        patch.object(pw, "HAS_TRITON", True),
        patch.object(penalty_module, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(pw._TOKEN_BIN_COUNTS_AND_MASK_KERNEL, "compile") as mock_bincount,
        patch.object(pw._APPLY_ALL_PENALTIES_KERNEL, "compile") as mock_penalties,
    ):
        pw.penalties_triton_warmup(worker)

    mock_bincount.assert_called_once()
    mock_penalties.assert_called_once()


def test_penalties_triton_warmup_registers_with_registry():
    vllm_config = _make_vllm_config()
    registry = JitWarmupRegistry(vllm_config)
    worker = SimpleNamespace(
        vllm_config=vllm_config,
        model_runner=SimpleNamespace(jit_warmup_registry=registry),
    )

    with (
        patch.object(pw, "HAS_TRITON", True),
        patch.object(penalty_module, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(pw._TOKEN_BIN_COUNTS_AND_MASK_KERNEL, "compile") as mock_bincount,
        patch.object(pw._APPLY_ALL_PENALTIES_KERNEL, "compile") as mock_penalties,
    ):
        with registry.activate():
            assert pw.register_penalties_triton_warmup(worker)
        registry.warmup()

    mock_bincount.assert_called_once()
    mock_penalties.assert_called_once()
