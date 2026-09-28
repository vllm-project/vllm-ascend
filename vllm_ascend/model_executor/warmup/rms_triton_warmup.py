# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Register and materialize Triton RMS specializations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.rms_norm import _TRITON_Q_RMS_KERNEL

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker

_MAX_TRITON_RMS_HEAD_DIM = 2048


def _model_uses_triton_q_rms(model_runner) -> bool:
    from vllm_ascend.attention.dsa_v1 import AscendDSABackend

    attn_groups = getattr(model_runner, "attn_groups", None)
    if not attn_groups:
        return False

    for groups in attn_groups:
        for group in groups:
            backend = getattr(group, "backend", None)
            if isinstance(backend, type) and issubclass(backend, AscendDSABackend):
                return True
    return False


def _kernel_enabled(worker: NPUWorker, *, assume_used: bool) -> bool:
    if not HAS_TRITON:
        return False
    kernel_config = getattr(worker.vllm_config, "kernel_config", None)
    if kernel_config is not None and not kernel_config.enable_jit_warmup:
        return False
    if not assume_used and not _model_uses_triton_q_rms(worker.model_runner):
        return False
    return worker.vllm_config.model_config.get_head_size() <= _MAX_TRITON_RMS_HEAD_DIM


def register_triton_rms_warmup(worker: NPUWorker, *, assume_used: bool = False) -> bool:
    """Register the RMS wrapper with the active warmup registry."""
    if not _kernel_enabled(worker, assume_used=assume_used):
        return False
    _TRITON_Q_RMS_KERNEL.register_warmup(worker.vllm_config)
    return True


def triton_rms_warmup(worker: NPUWorker, assume_used: bool = False) -> None:
    """Materialize RMS directly for early-warmup/fallback paths."""
    if not _kernel_enabled(worker, assume_used=assume_used):
        return
    _TRITON_Q_RMS_KERNEL.warmup(worker.vllm_config)
