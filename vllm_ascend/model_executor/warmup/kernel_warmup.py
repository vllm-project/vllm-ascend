# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up Triton kernels used during model execution on Ascend NPU."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.model_executor.warmup.dsa_triton_warmup import dsa_triton_warmup
from vllm_ascend.model_executor.warmup.penalties_triton_warmup import (
    penalties_triton_warmup,
)
from vllm_ascend.model_executor.warmup.rms_triton_warmup import triton_rms_warmup
from vllm_ascend.model_executor.warmup.rejection_sampler_triton_warmup import (
    rejection_sampler_triton_warmup,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


def _run_warmup(name: str, warmup_fn: Callable[["NPUWorker"], None], worker: "NPUWorker") -> None:
    warmup_fn(worker)
    logger.info("%s Triton warmup complete.", name, exc_info=True)


def kernel_warmup(worker: "NPUWorker") -> None:
    """Run Triton kernel warmups before ACL graph capture."""
    if not HAS_TRITON:
        return

    _run_warmup("rejection_sampler", rejection_sampler_triton_warmup, worker)
    _run_warmup("penalties", penalties_triton_warmup, worker)
    _run_warmup("rms", triton_rms_warmup, worker)
