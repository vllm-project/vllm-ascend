# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up Triton kernels used during model execution on Ascend NPU."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.model_executor.warmup.dsa_triton_warmup import dsa_triton_warmup
from vllm_ascend.model_executor.warmup.spec_decode_triton_warmup import (
    spec_decode_triton_warmup,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


def _run_warmup(name: str, warmup_fn: Callable[["NPUWorker"], None], worker: "NPUWorker") -> None:
    try:
        warmup_fn(worker)
    except Exception:
        logger.warning("Skipping %s Triton warmup.", name, exc_info=True)


def kernel_warmup(worker: "NPUWorker") -> None:
    """Run Triton kernel warmups before ACL graph capture."""
    if not HAS_TRITON:
        return

    _run_warmup("spec_decode", spec_decode_triton_warmup, worker)
    _run_warmup("dsa", dsa_triton_warmup, worker)
