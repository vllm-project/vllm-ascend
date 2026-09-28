# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Register and materialize Triton penalties/bincount specializations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.bincount import _TOKEN_BIN_COUNTS_AND_MASK_KERNEL
from vllm_ascend.ops.triton.penalty import _APPLY_ALL_PENALTIES_KERNEL

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


def _enabled(worker: NPUWorker) -> bool:
    if not HAS_TRITON:
        return False
    kernel_config = getattr(worker.vllm_config, "kernel_config", None)
    return kernel_config is None or bool(kernel_config.enable_jit_warmup)


def register_penalties_triton_warmup(worker: NPUWorker) -> bool:
    """Register bincount and penalties wrappers with the active registry."""
    if not _enabled(worker):
        return False
    _TOKEN_BIN_COUNTS_AND_MASK_KERNEL.register_warmup()
    _APPLY_ALL_PENALTIES_KERNEL.register_warmup(worker.vllm_config)
    return True


def penalties_triton_warmup(worker: NPUWorker) -> None:
    """Materialize penalties directly for early-warmup/fallback paths."""
    if not _enabled(worker):
        return
    _TOKEN_BIN_COUNTS_AND_MASK_KERNEL.warmup()
    _APPLY_ALL_PENALTIES_KERNEL.warmup(worker.vllm_config)
