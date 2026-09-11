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
            if issubclass(group.backend, AscendDSABackend):
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

    The live kernel floors to a power of two, so only these five values are
    JIT keys. Warming ``range(1, 17)`` would re-compile 1/2/4/8 and waste
    launches on 3/5/6/7/9-15.
    """
    values = []
    block_m = 1
    while block_m <= _ROW_BLOCK_SIZE:
        values.append(block_m)
        block_m *= 2
    return values


@torch.inference_mode()
def triton_rms_warmup(worker: NPUWorker, assume_used: bool = False) -> None:
    """JIT ``triton_q_rms`` kernels before the first real call.

    ``assume_used`` is for construct-time early warmup: ``model_runner.attn_groups``
    does not exist yet. The regular ``kernel_warmup`` call leaves it False.
    """
    if not HAS_TRITON:
        return
    try:
        from vllm_ascend.model_executor.warmup.early_kernel_warmup import (
            join_early_kernel_warmup,
        )

        join_early_kernel_warmup("rms")
    except ImportError:
        pass
    if not assume_used and not _model_uses_triton_q_rms(worker.model_runner):
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

    # Choose shapes so ``triton_q_rms`` selects each power-of-two ``BLOCK_M``.
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
