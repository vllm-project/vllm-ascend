# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up ``triton_q_rms`` (see ``ops/triton/rms_norm.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker

# Must match ``triton_q_rms`` limits and launch in ``rms_norm.py``.
_MAX_TRITON_RMS_HEAD_DIM = 2048
_ROW_BLOCK_SIZE = 16


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


def _variance_epsilon(model_config) -> float:
    variance_epsilon = 1e-6
    hf_config = getattr(model_config, "hf_text_config", None)
    if hf_config is not None:
        variance_epsilon = getattr(hf_config, "rms_norm_eps", variance_epsilon)
    return variance_epsilon


def collect_triton_rms_warmup_block_m_values() -> list[int]:
    """``BLOCK_M`` constexpr values selected by ``triton_q_rms``.

    ``BLOCK_M = min(ROW_BLOCK_SIZE, cdiv(total_batch, num_vectorcore))``, so every
    integer in ``[1, ROW_BLOCK_SIZE]`` must be JIT-compiled once.
    """
    return list(range(1, _ROW_BLOCK_SIZE + 1))


@torch.inference_mode()
def triton_rms_warmup(worker: NPUWorker) -> None:
    """JIT ``triton_q_rms`` kernels before the first real call."""
    if not HAS_TRITON:
        return
    if not _model_uses_triton_q_rms(worker.model_runner):
        return

    try:
        from vllm_ascend.ops.triton.rms_norm import triton_q_rms
    except ImportError:
        return

    head_dim = worker.vllm_config.model_config.get_head_size()
    if head_dim > _MAX_TRITON_RMS_HEAD_DIM:
        return

    device = worker.device
    block_m_values = collect_triton_rms_warmup_block_m_values()
    q_dtype = worker.model_config.dtype
    variance_epsilon = _variance_epsilon(worker.vllm_config.model_config)
    num_vectorcore = max(get_vectorcore_num(), 1)

    # Choose shapes so ``triton_q_rms`` selects each ``BLOCK_M`` value:
    # ``BLOCK_M = min(16, cdiv(total_batch, num_vectorcore))``.
    # Use ``head_num=1`` so ``bs * head_num == total_batch``.
    for block_m in block_m_values:
        total_batch = block_m * num_vectorcore
        q = torch.randn(
            total_batch,
            1,
            head_dim,
            dtype=q_dtype,
            device=device,
        )
        triton_q_rms(q, variance_epsilon)
