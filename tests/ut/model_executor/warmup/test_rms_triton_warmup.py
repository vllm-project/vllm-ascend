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

    assert mock_triton_q_rms.call_count == rw._ROW_BLOCK_SIZE
    q, eps = mock_triton_q_rms.call_args_list[0][0]
    assert q.shape == (1 * num_vectorcore, 1, 128)
    assert eps == 1e-5
