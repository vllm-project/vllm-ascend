# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``penalties_triton_warmup``."""

from unittest.mock import patch

import torch

from tests.ut.model_executor.warmup.helpers import make_mock_worker
from vllm_ascend.model_executor.warmup import penalties_triton_warmup as pw


def test_make_history_tokens():
    vocab_size = 64
    tokens = pw._make_history_tokens(3, 8, vocab_size, torch.device("cpu"))
    assert tokens.shape == (3, 8)
    assert torch.all(tokens[:, -1] == vocab_size)


def test_local_vocab_size():
    model_config = type("Cfg", (), {"get_vocab_size": lambda self: 1024})()
    with patch.object(pw, "get_tensor_model_parallel_world_size", return_value=4):
        assert pw._local_vocab_size(model_config) == 256


@patch.object(pw, "apply_penalties_triton")
@patch.object(pw, "get_tensor_model_parallel_world_size", return_value=1)
@patch.object(pw, "HAS_TRITON", True)
def test_penalties_triton_warmup(mock_tp, mock_apply):
    worker = make_mock_worker(max_num_seqs=4, max_num_batched_tokens=300, vocab_size=512)
    pw.penalties_triton_warmup(worker)

    mock_apply.assert_called_once()
    logits, prompt_tokens, output_tokens, _, _, _ = mock_apply.call_args[0]
    expected_seq_len = min(pw._BINCOUNT_SEQ_BLOCK + 1, 300)
    assert logits.shape == (4, 512)
    assert prompt_tokens.shape == (4, expected_seq_len)
    assert output_tokens.shape == (4, expected_seq_len)
