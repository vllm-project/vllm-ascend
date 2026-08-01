# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up ``triton_rms_kernel`` (see ``ops/triton/rms_norm.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.logger import logger
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


def _warm_triton_rms_kernel(
    device: torch.device,
    total_batch: int,
    dim: int,
    block_m: int,
    q_dtype: torch.dtype,
    variance_epsilon: float,
    num_vectorcore: int,
) -> None:
    from vllm_ascend.ops.triton.rms_norm import triton_rms_kernel

    hidden_state = torch.randn(
        total_batch,
        dim,
        dtype=q_dtype,
        device=device,
    )
    norm_output = torch.empty_like(hidden_state)
    grid = (num_vectorcore,)

    triton_rms_kernel[grid](
        hidden_state,
        hidden_state.stride(0),
        norm_output,
        variance_epsilon,
        total_batch,
        dim,
        block_m,
    )


@torch.inference_mode()
def triton_rms_warmup(worker: NPUWorker) -> None:
    """JIT ``triton_rms_kernel`` before the first ``triton_q_rms`` call."""
    if not HAS_TRITON:
        return
    if not _model_uses_triton_q_rms(worker.model_runner):
        return

    try:
        from vllm_ascend.ops.triton.rms_norm import triton_rms_kernel  # noqa: F401
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

    logger.info(
        "Warming up Triton RMS kernel: head_dim=%d, block_m_values=%s, "
        "num_vectorcore=%d, dtype=%s, eps=%g",
        head_dim,
        block_m_values,
        num_vectorcore,
        q_dtype,
        variance_epsilon,
    )

    for block_m in block_m_values:
        total_batch = block_m * num_vectorcore
        _warm_triton_rms_kernel(
            device,
            total_batch,
            head_dim,
            block_m,
            q_dtype,
            variance_epsilon,
            num_vectorcore,
        )

    if device.type == "npu":
        torch.npu.synchronize()
