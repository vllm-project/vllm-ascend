# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up DSA-related Triton kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.model_executor.warmup.spec_decode_triton_warmup import (
    collect_warmup_token_sizes,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


def _model_uses_dsa_attention(model_runner) -> bool:
    from vllm_ascend.attention.dsa_v1 import AscendDSABackend

    attn_groups = getattr(model_runner, "attn_groups", None)
    if not attn_groups:
        return False

    for groups in attn_groups:
        for group in groups:
            if group.backend is AscendDSABackend:
                return True
    return False


def _estimate_local_num_heads(model_config) -> int:
    hf_config = getattr(model_config, "hf_text_config", None) or getattr(
        model_config,
        "hf_config",
        None,
    )
    num_heads = getattr(hf_config, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(hf_config, "n_heads", 8)
    tp_size = get_tensor_model_parallel_world_size()
    return max(1, num_heads // tp_size)


@torch.inference_mode()
def dsa_triton_warmup(worker: NPUWorker) -> None:
    """Warm `triton_q_rms` for DSA models on non-A5 devices."""
    if not HAS_TRITON:
        return
    if not _model_uses_dsa_attention(worker.model_runner):
        return

    try:
        from vllm_ascend.ops.triton.rms_norm import triton_q_rms
    except ImportError:
        return

    head_dim = worker.vllm_config.model_config.get_head_size()
    if head_dim > 2048:
        return

    device = worker.device
    num_heads = _estimate_local_num_heads(worker.vllm_config.model_config)
    capture_sizes = worker.vllm_config.compilation_config.cudagraph_capture_sizes
    max_num_tokens = worker.scheduler_config.max_num_batched_tokens
    token_sizes = collect_warmup_token_sizes(max_num_tokens, capture_sizes)

    logger.info(
        "Warming up DSA Triton Q-RMS kernel: head_dim=%d, num_heads=%d, "
        "token_sizes=%s (max_num_batched_tokens=%d)",
        head_dim,
        num_heads,
        token_sizes,
        max_num_tokens,
    )

    variance_epsilon = 1e-6
    hf_config = getattr(worker.vllm_config.model_config, "hf_text_config", None)
    if hf_config is not None:
        variance_epsilon = getattr(hf_config, "rms_norm_eps", variance_epsilon)

    for num_tokens in token_sizes:
        q = torch.randn(
            num_tokens,
            num_heads,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
        triton_q_rms(q, variance_epsilon)

    if device.type == "npu":
        torch.npu.synchronize()
