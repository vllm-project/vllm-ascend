# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``rms_triton_warmup``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.model_executor.warmup.helpers import make_mock_worker
from vllm_ascend.model_executor.warmup import rms_triton_warmup as rw


def test_triton_rms_warmup():
    worker = make_mock_worker(head_size=128, dtype=torch.float16)
    mock_triton_q_rms = MagicMock()
    fake_module = SimpleNamespace(triton_q_rms=mock_triton_q_rms)
    num_vectorcore = 4

    with (
        patch.object(rw, "HAS_TRITON", True),
        patch.object(rw, "_model_uses_triton_q_rms", return_value=True),
        patch.object(rw, "get_vectorcore_num", return_value=num_vectorcore),
        patch.dict("sys.modules", {"vllm_ascend.ops.triton.rms_norm": fake_module}),
    ):
        rw.triton_rms_warmup(worker)

    assert mock_triton_q_rms.call_count == len(rw.collect_triton_rms_warmup_block_m_values())
    q, eps = mock_triton_q_rms.call_args_list[0][0]
    assert q.shape == (1 * num_vectorcore, 1, 128)
    assert eps == 1e-5


def test_collect_block_m_values_are_powers_of_two():
    """The kernel floors BLOCK_M to a power of two, so only those are JIT keys."""
    values = rw.collect_triton_rms_warmup_block_m_values()

    assert values == [1, 2, 4, 8, 16]
    assert values[-1] == rw._ROW_BLOCK_SIZE


def test_triton_rms_warmup_assume_used_skips_model_probe():
    """Construct-time warmup runs before ``model_runner.attn_groups`` exists."""
    worker = make_mock_worker(head_size=128, dtype=torch.float16)
    fake_module = SimpleNamespace(triton_q_rms=MagicMock())

    with (
        patch.object(rw, "HAS_TRITON", True),
        patch.object(rw, "_model_uses_triton_q_rms") as mock_probe,
        patch.object(rw, "get_vectorcore_num", return_value=4),
        patch.dict("sys.modules", {"vllm_ascend.ops.triton.rms_norm": fake_module}),
    ):
        rw.triton_rms_warmup(worker, assume_used=True)

    mock_probe.assert_not_called()
    assert fake_module.triton_q_rms.call_count == len(rw.collect_triton_rms_warmup_block_m_values())


def test_triton_rms_warmup_joins_early_thread():
    worker = make_mock_worker(head_size=128, dtype=torch.float16)
    fake_module = SimpleNamespace(triton_q_rms=MagicMock())

    with (
        patch.object(rw, "HAS_TRITON", True),
        patch.object(rw, "_model_uses_triton_q_rms", return_value=False),
        patch.object(rw, "get_vectorcore_num", return_value=4),
        patch.dict("sys.modules", {"vllm_ascend.ops.triton.rms_norm": fake_module}),
        patch("vllm_ascend.model_executor.warmup.early_kernel_warmup.join_early_kernel_warmup") as mock_join,
    ):
        rw.triton_rms_warmup(worker)

    mock_join.assert_called_once_with("rms")
    fake_module.triton_q_rms.assert_not_called()
