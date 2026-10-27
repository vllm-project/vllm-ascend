# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the v1 rejection-sampler warmup wrappers."""

from unittest.mock import patch

import torch

from tests.ut.model_executor.warmup.helpers import (
    make_mock_ascend_config,
    make_mock_spec_config,
    make_mock_vllm_config,
    make_registry,
)
from vllm_ascend.ops.triton import reject_sample as rs


def _context(config):
    with (
        patch.object(rs, "get_vectorcore_num", return_value=4),
        patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=make_mock_ascend_config(),
        ),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1),
    ):
        return rs.build_rejection_warmup_context(config)


def test_context_synthetic_and_block_verify_gates():
    spec = make_mock_spec_config(num_speculative_tokens=3, rejection_sample_method="synthetic")
    config = make_mock_vllm_config(speculative_config=spec)
    with (
        patch.object(rs, "get_vectorcore_num", return_value=4),
        patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=make_mock_ascend_config(enable_block_verify=True),
        ),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1),
    ):
        context = rs.build_rejection_warmup_context(config)
    assert context.synthetic_mode is True
    assert context.block_verify is True

    # block verify additionally requires max_spec_len >= 3.
    spec = make_mock_spec_config(num_speculative_tokens=2)
    context = _context(make_mock_vllm_config(speculative_config=spec))
    assert context.block_verify is False


def test_register_registers_six_kernels_inside_window():
    spec = make_mock_spec_config(num_speculative_tokens=3)
    config = make_mock_vllm_config(speculative_config=spec)
    registry = make_registry(config)

    with (
        patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=make_mock_ascend_config(),
        ),
        patch.object(rs, "get_vectorcore_num", return_value=4),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1),
    ):
        rs.register_rejection_warmup(config)  # outside window: registers nothing
        assert len(registry) == 0
        with registry.activate():
            assert rs.register_rejection_warmup(config) is True
    assert len(registry) == 6


def test_expand_dtype_axis_covers_runtime_dtypes():
    spec = make_mock_spec_config(num_speculative_tokens=3)
    context = _context(make_mock_vllm_config(speculative_config=spec))
    keys = rs._EXPAND_KERNEL.get_warmup_keys(context)
    assert {key.dtype for key in keys} == {
        torch.int32,
        torch.int64,
        torch.float32,
    }
