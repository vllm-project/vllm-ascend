# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner


@contextmanager
def expert_offload_model_loading(runner: NPUModelRunner, load_dummy_weights: bool) -> Iterator[None]:
    """Own offload finalization and failure cleanup around the native loader."""
    manager = runner.expert_offload_manager
    if manager is None:
        yield
        return
    if load_dummy_weights:
        raise ValueError("expert offload requires checkpoint shard tracking; dummy checkpoint loading is unsupported")
    try:
        yield
        allocated_before = torch.npu.memory_allocated(runner.device)
        manager.finalize_loading(runner.model)
        # The shared pool is allocated after the parent's loading profiler.
        runner.model_memory_usage += torch.npu.memory_allocated(runner.device) - allocated_before
    except Exception:
        try:
            manager.close()
        except Exception:
            logger.exception("Failed to clean up expert offload after model loading failure")
        raise
