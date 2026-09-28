# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up Triton kernels used during model execution on Ascend NPU."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.model_executor.warmup.indexer_triton_warmup import (
    register_indexer_triton_warmup,
)
from vllm_ascend.model_executor.warmup.penalties_triton_warmup import (
    register_penalties_triton_warmup,
)
from vllm_ascend.model_executor.warmup.rejection_sampler_triton_warmup import (
    register_rejection_sampler_triton_warmup,
)
from vllm_ascend.model_executor.warmup.rms_triton_warmup import (
    register_triton_rms_warmup,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


def kernel_warmup(worker: NPUWorker) -> None:
    """Materialize all selected Triton kernels before ACL graph capture."""
    if not HAS_TRITON:
        return

    kernel_config = getattr(worker.vllm_config, "kernel_config", None)
    if kernel_config is not None and not kernel_config.enable_jit_warmup:
        return

    warmup_start_time = time.perf_counter()
    registry = worker.model_runner.jit_warmup_registry
    logger.info("Starting Triton kernel warmup through shared registry.")
    with registry.activate():
        registered = any(
            (
                register_triton_rms_warmup(worker),
                register_penalties_triton_warmup(worker),
                register_rejection_sampler_triton_warmup(worker),
                register_indexer_triton_warmup(worker),
            )
        )

    if registered:
        registry.warmup()

    warmup_elapsed = time.perf_counter() - warmup_start_time
    logger.info(
        "Triton kernel warmup total time: %.3f seconds",
        warmup_elapsed,
    )
