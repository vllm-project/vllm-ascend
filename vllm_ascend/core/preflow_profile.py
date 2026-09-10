# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side timing helpers for PREFLOW startup calibration."""

import time
from typing import Any

import torch

from vllm_ascend.core.preflow_cost_model import (
    PREFLOW_PROFILE_ELAPSED_MS_ATTR,
    PREFLOW_PROFILE_METADATA_ATTR,
)


def start_preflow_profile_timing(scheduler_output: Any) -> float | None:
    """Synchronize and start timing only for a calibration batch."""
    if getattr(scheduler_output, PREFLOW_PROFILE_METADATA_ATTR, None) is None:
        return None
    torch.npu.synchronize()
    return time.perf_counter()


def finish_preflow_profile_timing(start_time: float | None) -> float | None:
    """Finish a calibration timing interval in milliseconds."""
    if start_time is None:
        return None
    torch.npu.synchronize()
    return (time.perf_counter() - start_time) * 1000.0


def attach_preflow_profile_timing(model_runner: Any, output: Any) -> None:
    """Attach a completed timing to the scheduler-visible model output."""
    start_time = getattr(model_runner, "_preflow_profile_start_time", None)
    model_runner._preflow_profile_start_time = None
    elapsed_ms = finish_preflow_profile_timing(start_time)
    if elapsed_ms is None:
        return

    # Async workers wrap the scheduler-visible result in an outer output.
    model_runner_output = getattr(output, "model_runner_output", output)
    if model_runner_output is not None:
        setattr(model_runner_output, PREFLOW_PROFILE_ELAPSED_MS_ATTR, elapsed_ms)


__all__ = [
    "attach_preflow_profile_timing",
    "finish_preflow_profile_timing",
    "start_preflow_profile_timing",
]
