# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the fused Q-RMS ``TritonQRmsKernel`` warmup wrapper."""

from unittest.mock import patch

import torch

from tests.ut.model_executor.warmup.helpers import (
    make_mock_vllm_config,
    make_registry,
)
from vllm_ascend.ops.triton import rms_norm


def test_block_m_enumeration_covers_every_power_of_two():
    """BLOCK_M = pow2floor(min(ceiling, cdiv(batch, cores))) with the batch
    spanning every runtime size, so every power of two up to the UB-derived
    ceiling is a distinct JIT key."""
    config = make_mock_vllm_config(head_size=128, dtype=torch.float16)
    with patch.object(rms_norm, "_rms_row_block_size", return_value=16):
        keys = rms_norm._TRITON_Q_RMS_KERNEL.get_warmup_keys(config)

    assert [key.block_m for key in keys] == [1, 2, 4, 8, 16]
    assert all(key.dim == 128 and key.dtype == torch.float16 for key in keys)


def test_registration_only_collects_inside_window():
    from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

    config = make_mock_vllm_config()
    registry = make_registry(config)

    # Outside a window the registration is a no-op.
    rms_norm._TRITON_Q_RMS_KERNEL.register_warmup(config)
    assert len(registry) == 0

    with registry.activate():
        rms_norm._TRITON_Q_RMS_KERNEL.register_warmup(config)
    assert len(registry) == 1
    assert isinstance(rms_norm._TRITON_Q_RMS_KERNEL, AscendTritonJitKernel)
