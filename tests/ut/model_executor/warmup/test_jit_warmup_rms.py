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


def test_row_block_size_ub_formula():
    """(UB - 6 KiB reserve) / (dim * elem * multiplier)."""
    with patch.object(rms_norm, "get_ub_size_bytes", return_value=27648):
        # fp16: 7 live buffers of dim * 2 bytes.
        assert rms_norm._rms_row_block_size(128, torch.float16) == 12
        # fp32: 5 live buffers of dim * 4 bytes.
        assert rms_norm._rms_row_block_size(128, torch.float32) == 8


def test_enumeration_clamps_ceiling_to_single_row():
    """A zero UB budget must still compile the single-row key."""
    config = make_mock_vllm_config(head_size=128)
    with patch.object(rms_norm, "get_ub_size_bytes", return_value=6144):
        keys = rms_norm._TRITON_Q_RMS_KERNEL.get_warmup_keys(config)
    assert [key.block_m for key in keys] == [1]


def test_head_size_over_2048_skips_triton_path():
    config = make_mock_vllm_config(head_size=4096)
    assert rms_norm._TRITON_Q_RMS_KERNEL.get_warmup_keys(config) == []


def test_warmup_inputs_shape_and_strides():
    config = make_mock_vllm_config(head_size=128, dtype=torch.bfloat16)
    with patch.object(rms_norm, "_rms_row_block_size", return_value=16):
        key = rms_norm._TRITON_Q_RMS_KERNEL.get_warmup_keys(config)[1]

    inputs = rms_norm._TRITON_Q_RMS_KERNEL.warmup_inputs(key)
    hidden = inputs["hidden_state"]
    assert hidden.dtype == torch.bfloat16
    assert hidden.shape == (1, 128)
    assert hidden.strides == (128, 1)
    assert inputs["DIM"] == 128
    assert inputs["BLOCK_M"] == key.block_m


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
