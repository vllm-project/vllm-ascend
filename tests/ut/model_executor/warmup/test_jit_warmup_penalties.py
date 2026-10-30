# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the penalty and bincount warmup wrappers."""

from unittest.mock import patch

import torch

from tests.ut.model_executor.warmup.helpers import make_mock_vllm_config
from vllm_ascend.ops.triton import penalty


def _keys(module, kernel, config):
    with patch.object(module, "get_tensor_model_parallel_world_size", return_value=1):
        return kernel.get_warmup_keys(config)


def test_penalty_block_size_matches_runtime_launch():
    """The live sampler launches with BLOCK_SIZE=2048 (max_num_seqs capped by
    num_vector_core cores); the warmup key must hit the same JIT cache."""
    assert penalty._APPLY_ALL_PENALTIES_KERNEL.BLOCK_SIZE == 2048


def test_penalty_key_uses_vocab_specialization_class():
    config = make_mock_vllm_config(vocab_size=1024)
    (key,) = _keys(penalty, penalty._APPLY_ALL_PENALTIES_KERNEL, config)
    assert key.block_size == 2048
    # 1024 is divisible by 16 -> representative 16 of the div-16 class.
    assert key.vocab_size == 16
    assert key.logits_dtype == torch.float32

    config = make_mock_vllm_config(vocab_size=1025)
    (key,) = _keys(penalty, penalty._APPLY_ALL_PENALTIES_KERNEL, config)
    assert key.vocab_size == 2  # generic class
