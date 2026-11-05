# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared helpers for JIT kernel warmup unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


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
    )
    return config


def make_mock_worker(**kwargs) -> MagicMock:
    """Worker mock for ``kernel_warmup`` orchestration tests."""
    config = make_mock_vllm_config(
        speculative_config=kwargs.pop("speculative_config", None),
        **kwargs,
    )
    worker = MagicMock()
    worker.vllm_config = config
    model_runner = SimpleNamespace(
        vllm_config=config,
        jit_warmup_registry=make_registry(config),
        drafter=None,
        input_batch=None,
    )
    worker.model_runner = model_runner
    return worker


def make_registry(vllm_config=None):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    return JitWarmupRegistry(vllm_config)


def make_mock_ascend_config(
    *,
    enable_entropy_verify: bool = False,
    enable_block_verify: bool = False,
    posterior_threshold: float = 0.95,
    posterior_alpha: float = 0.4,
):
    """Patching target for ``vllm_ascend.ascend_config.get_ascend_config``."""
    return SimpleNamespace(
        rejection_sampler_config=SimpleNamespace(
            enable_entropy_verify=enable_entropy_verify,
            enable_block_verify=enable_block_verify,
            posterior_threshold=posterior_threshold,
            posterior_alpha=posterior_alpha,
        )
    )
