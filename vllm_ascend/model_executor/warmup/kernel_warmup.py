# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up Triton kernels used during model execution on Ascend NPU."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.model_executor.warmup.indexer_triton_warmup import (
    indexer_triton_warmup,
)
from vllm_ascend.model_executor.warmup.penalties_triton_warmup import (
    penalties_triton_warmup,
)
from vllm_ascend.model_executor.warmup.rejection_sampler_triton_warmup import (
    rejection_sampler_triton_warmup,
)
from vllm_ascend.model_executor.warmup.rms_triton_warmup import triton_rms_warmup

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker


def _run_warmup(
    name: str,
    warmup_fn: Callable[[NPUWorker], None],
    worker: NPUWorker,
) -> None:
    start = time.perf_counter()
    warmup_fn(worker)
    elapsed = time.perf_counter() - start
    logger.info("%s Triton warmup complete in %.3fs.", name, elapsed)


def kernel_warmup(worker: NPUWorker) -> None:
    """Run Triton kernel warmups before ACL graph capture."""
    if not HAS_TRITON:
        return

    logger.info("Starting Triton kernel warmup.")
    start = time.perf_counter()

    if not worker.use_v2_model_runner:
        # V1-only triton kernels: MRv2 samples through worker/v2 kernels that
        # are already exercised by warmup_kernels' spec-decode steps and
        # for_sampler_warmup() sampling params.
        _run_warmup("rejection_sampler", rejection_sampler_triton_warmup, worker)
        _run_warmup("penalties", penalties_triton_warmup, worker)
    # rms / indexer enumerate constexpr specializations (BLOCK_M 1..16 and the
    # BLOCK_ROWS ladder) that warmup_kernels' limited batch shapes cannot
    # cover before ACL graph capture, so they stay unconditional.
    _run_warmup("rms", triton_rms_warmup, worker)
    _run_warmup("indexer", indexer_triton_warmup, worker)

    elapsed = time.perf_counter() - start
    logger.info("Triton kernel warmup finished in %.3fs.", elapsed)
