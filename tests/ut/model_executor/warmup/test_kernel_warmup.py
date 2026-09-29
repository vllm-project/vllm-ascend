# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for shared-registry kernel warmup orchestration."""

import importlib
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

kw = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")


class _Registry:
    def __init__(self):
        self.activations = 0
        self.warmups = 0

    @contextmanager
    def activate(self):
        self.activations += 1
        yield

    def warmup(self):
        self.warmups += 1


def test_kernel_warmup_uses_shared_registry_once():
    registry = _Registry()
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(kernel_config=SimpleNamespace(enable_jit_warmup=True)),
        model_runner=SimpleNamespace(jit_warmup_registry=registry),
    )

    with (
        patch.object(kw, "HAS_TRITON", True),
        patch.object(kw, "register_triton_rms_warmup", return_value=True) as mock_rms,
        patch.object(kw, "register_penalties_triton_warmup", return_value=True) as mock_pen,
        patch.object(kw, "register_rejection_sampler_triton_warmup", return_value=True) as mock_rej,
        patch.object(kw, "register_indexer_triton_warmup", return_value=True) as mock_indexer,
    ):
        kw.kernel_warmup(worker)

    mock_rms.assert_called_once_with(worker)
    mock_pen.assert_called_once_with(worker)
    mock_rej.assert_called_once_with(worker)
    mock_indexer.assert_called_once_with(worker)
    assert registry.activations == 1
    assert registry.warmups == 1
