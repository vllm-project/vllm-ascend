# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up ``triton_q_rms`` / ``triton_rms_kernel`` (see ``ops/triton/rms_norm.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker

# Must match ``triton_q_rms`` limits and launch in ``rms_norm.py``.
_MAX_TRITON_RMS_HEAD_DIM = 2048


def _model_uses_triton_q_rms(model_runner) -> bool:
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


def _variance_epsilon(model_config) -> float:
    variance_epsilon = 1e-6
    hf_config = getattr(model_config, "hf_text_config", None)
    if hf_config is not None:
        variance_epsilon = getattr(hf_config, "rms_norm_eps", variance_epsilon)
    return variance_epsilon


def collect_triton_rms_warmup_token_sizes(max_num_tokens: int) -> list[int]:
    """Token counts for ``triton_q_rms`` warmup.

    ``total_batch`` and strides are dynamic in the kernel; two sizes are enough
    to JIT before the first DSA forward.
    """
    max_num_tokens = max(max_num_tokens, 1)
    return [1, max_num_tokens] if max_num_tokens > 1 else [1]


def _warm_triton_q_rms(
    device: torch.device,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    variance_epsilon: float,
) -> None:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

    q = torch.randn(
        num_tokens,
        num_heads,
        head_dim,
        dtype=q_dtype,
        device=device,
    )
    triton_q_rms(q, variance_epsilon)


@torch.inference_mode()
def triton_rms_warmup(worker: NPUWorker) -> None:
    """JIT ``triton_rms_kernel`` before the first ``triton_q_rms`` call."""
    if not HAS_TRITON:
        return
    if not _model_uses_triton_q_rms(worker.model_runner):
        return

    try:
        from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F401
    except ImportError:
        return

    head_dim = worker.vllm_config.model_config.get_head_size()
    if head_dim > _MAX_TRITON_RMS_HEAD_DIM:
        return

    device = worker.device
    num_heads = _estimate_local_num_heads(worker.vllm_config.model_config)
    max_num_tokens = worker.scheduler_config.max_num_batched_tokens
    token_sizes = collect_triton_rms_warmup_token_sizes(max_num_tokens)
    q_dtype = worker.model_config.dtype
    variance_epsilon = _variance_epsilon(worker.vllm_config.model_config)
    launch_grid = max(get_vectorcore_num(), 1)

    logger.info(
        "Warming up Triton Q-RMS kernel: head_dim=%d, num_heads=%d, "
        "token_sizes=%s, launch_grid=%d, dtype=%s, eps=%g, "
        "max_num_batched_tokens=%d",
        head_dim,
        num_heads,
        token_sizes,
        launch_grid,
        q_dtype,
        variance_epsilon,
        max_num_tokens,
    )

    for num_tokens in token_sizes:
        _warm_triton_q_rms(
            device,
            num_tokens,
            num_heads,
            head_dim,
            q_dtype,
            variance_epsilon,
        )

    if device.type == "npu":
        torch.npu.synchronize()
