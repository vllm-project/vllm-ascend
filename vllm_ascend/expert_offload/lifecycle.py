# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

_F = TypeVar("_F", bound=Callable[..., None])


def guard_expert_offload_loading(load_model: _F) -> _F:
    """Close offload resources on failure without changing the native loader body."""

    @wraps(load_model)
    def wrapped(runner: NPUModelRunner, load_dummy_weights: bool = False, *args: Any, **kwargs: Any) -> None:
        manager = runner.expert_offload_manager
        if manager is None:
            return load_model(runner, load_dummy_weights, *args, **kwargs)
        if load_dummy_weights:
            raise ValueError(
                "expert offload requires checkpoint shard tracking; dummy checkpoint loading is unsupported"
            )
        try:
            return load_model(runner, load_dummy_weights, *args, **kwargs)
        except Exception:
            try:
                manager.close()
            except Exception:
                logger.exception("Failed to clean up expert offload after model loading failure")
            raise

    return cast(_F, wrapped)


def finalize_expert_offload_loading(runner: NPUModelRunner) -> None:
    """Finalize after native warmup joins, before the runner initializes its dumper."""
    manager = runner.expert_offload_manager
    if manager is None:
        return
    allocated_before = torch.npu.memory_allocated(runner.device)
    manager.finalize_loading(runner.model)
    # The shared pool is allocated after the parent's loading profiler.
    runner.model_memory_usage += torch.npu.memory_allocated(runner.device) - allocated_before


def close_expert_offload_on_shutdown(shutdown: _F) -> _F:
    """Drain offload before native shutdown, which must run even when draining fails."""

    @wraps(shutdown)
    def wrapped(runner: NPUModelRunner) -> None:
        try:
            manager = runner.expert_offload_manager
            if manager is not None:
                manager.close()
        finally:
            shutdown(runner)

    return cast(_F, wrapped)
