# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the v1 rejection-sampler warmup wrappers."""

from types import SimpleNamespace
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


def test_collect_block_sizes_returns_one_representative_per_block_size():
    """cal_grid_and_block_size maps batches 1..max to BLOCK_SIZE buckets;
    only one representative per bucket is a JIT key."""
    with patch.object(rs, "get_vectorcore_num", return_value=4):
        # batches 1-4 -> block 1; 5-8 -> block 2; bucket max gets batch 8.
        assert rs.collect_rejection_warmup_block_sizes(8) == [1, 8]
        assert rs.collect_rejection_warmup_block_sizes(0) == []


def test_context_is_none_without_speculative_decoding():
    assert _context(make_mock_vllm_config(speculative_config=None)) is None
    spec = make_mock_spec_config(num_speculative_tokens=0)
    assert _context(make_mock_vllm_config(speculative_config=spec)) is None


def test_context_ngram_pins_no_draft_probs():
    spec = make_mock_spec_config(num_speculative_tokens=3, method="ngram")
    context = _context(make_mock_vllm_config(speculative_config=spec))
    assert context.no_draft_probs_values == (True,)


def test_context_multi_pipeline_ranks_widen_no_draft_axis():
    spec = make_mock_spec_config(num_speculative_tokens=3, method="eagle")
    config = make_mock_vllm_config(
        speculative_config=spec, pipeline_parallel_size=2
    )
    context = _context(config)
    assert context.no_draft_probs_values == (False, True)


def test_context_synthetic_and_block_verify_gates():
    spec = make_mock_spec_config(
        num_speculative_tokens=3, rejection_sample_method="synthetic"
    )
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


def test_greedy_spec_len_1_kernel_gated_on_max_spec_len():
    spec1 = make_mock_spec_config(num_speculative_tokens=1)
    context = _context(make_mock_vllm_config(speculative_config=spec1))
    assert rs._REJECTION_GREEDY_SPEC_LEN_1_KERNEL.get_warmup_keys(context)

    spec2 = make_mock_spec_config(num_speculative_tokens=2)
    context = _context(make_mock_vllm_config(speculative_config=spec2))
    assert rs._REJECTION_GREEDY_SPEC_LEN_1_KERNEL.get_warmup_keys(context) == []


def test_greedy_has_is_greedy_axis_widens_with_spec_len():
    spec1 = make_mock_spec_config(num_speculative_tokens=1)
    context = _context(make_mock_vllm_config(speculative_config=spec1))
    keys = rs._REJECTION_GREEDY_KERNEL.get_warmup_keys(context)
    assert {key.has_is_greedy for key in keys} == {True}

    spec3 = make_mock_spec_config(num_speculative_tokens=3)
    context = _context(make_mock_vllm_config(speculative_config=spec3))
    keys = rs._REJECTION_GREEDY_KERNEL.get_warmup_keys(context)
    assert {key.has_is_greedy for key in keys} == {False, True}
    assert {key.block_size for key in keys} == set(context.block_sizes)


def test_expand_dtype_axis_covers_runtime_dtypes():
    spec = make_mock_spec_config(num_speculative_tokens=3)
    context = _context(make_mock_vllm_config(speculative_config=spec))
    keys = rs._EXPAND_KERNEL.get_warmup_keys(context)
    assert {key.dtype for key in keys} == {
        torch.int32,
        torch.int64,
        torch.float32,
    }


def test_random_kernel_no_ori_tied_to_entropy_verify():
    spec = make_mock_spec_config(num_speculative_tokens=3)
    config = make_mock_vllm_config(vocab_size=1024, speculative_config=spec)
    context = _context(config)
    keys = rs._REJECTION_RANDOM_SAMPLE_KERNEL.get_warmup_keys(context)
    # ori_target_probs exists exactly when entropy verify is on; here it is
    # off, so every key has NO_ORI_TARGET_PROBS=True.
    assert keys and all(key.no_ori_target_probs for key in keys)
    assert {key.vocab_size for key in keys} == {16}


def test_block_verify_kernel_gated_on_context():
    spec = make_mock_spec_config(num_speculative_tokens=3)
    config = make_mock_vllm_config(speculative_config=spec)
    context = _context(config)
    assert rs._REJECTION_RANDOM_SAMPLE_BLOCK_VERIFY_KERNEL.get_warmup_keys(context) == []
