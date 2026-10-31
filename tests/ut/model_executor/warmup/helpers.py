# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared helpers for Triton warmup unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


def make_mock_worker(
    *,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 512,
    vocab_size: int = 1024,
    head_size: int = 128,
    dtype: torch.dtype = torch.float16,
    pipeline_parallel_size: int = 1,
    speculative_config=None,
    attn_groups=None,
    device: str = "cpu",
) -> MagicMock:
    worker = MagicMock()
    worker.device = torch.device(device)
    worker.scheduler_config.max_num_seqs = max_num_seqs
    worker.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    worker.model_config.get_vocab_size.return_value = vocab_size
    worker.model_config.dtype = dtype
    worker.vllm_config.model_config.get_head_size.return_value = head_size
    worker.vllm_config.model_config.get_vocab_size.return_value = vocab_size
    worker.vllm_config.model_config.hf_text_config = SimpleNamespace(rms_norm_eps=1e-5)
    worker.vllm_config.speculative_config = speculative_config
    worker.vllm_config.parallel_config.pipeline_parallel_size = pipeline_parallel_size
    worker.model_runner.attn_groups = attn_groups
    return worker
