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
    enable_jit_warmup: bool = True,
) -> MagicMock:
    worker = MagicMock()
    worker.device = torch.device(device)
    worker.vllm_config.kernel_config.enable_jit_warmup = enable_jit_warmup
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


def make_mock_spec_config(
    *,
    num_speculative_tokens: int = 3,
    method: str = "eagle",
    rejection_sample_method: str | None = None,
    use_ngram_gpu=None,
):
    return SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        method=method,
        rejection_sample_method=rejection_sample_method,
        use_ngram_gpu=use_ngram_gpu,
    )


def make_mock_vllm_config(
    *,
    vocab_size: int = 1024,
    head_size: int = 128,
    dtype: torch.dtype = torch.float16,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 512,
    pipeline_parallel_size: int = 1,
    speculative_config=None,
    index_head_dim: int = 128,
    enable_jit_warmup: bool = True,
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_head_size=lambda: head_size,
            get_vocab_size=lambda: vocab_size,
            dtype=dtype,
            hf_text_config=SimpleNamespace(index_head_dim=index_head_dim),
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size),
        speculative_config=speculative_config,
        kernel_config=SimpleNamespace(enable_jit_warmup=enable_jit_warmup),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    return config


def make_registry(vllm_config=None):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    return JitWarmupRegistry(vllm_config)
