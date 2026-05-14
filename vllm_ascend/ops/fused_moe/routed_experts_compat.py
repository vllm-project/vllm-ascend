#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Compatibility shim around vLLM's RoutedExpertsCapturer.
- 0.20.2 exposed `RoutedExpertsCapturer.get_instance()` plus
  `clear_buffer()` / `save_captured_experts(indices=...)` methods.
- main moved to module-level helpers (`get_global_experts_capturer`,
  `issue_routing_d2h_copy`, `extract_routed_experts_for_current_batch`,
  `free_routing_buffers`, `init_routed_experts_capturer_with_shared_cache`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.model_executor.layers.fused_moe import routed_experts_capturer as _rec

from vllm_ascend.utils import vllm_version_is

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

USE_LEGACY_API = vllm_version_is("0.20.2")


def get_capturer():
    """Return the global capturer instance, or None if not initialized."""
    if USE_LEGACY_API:
        return _rec.RoutedExpertsCapturer.get_instance()
    return _rec.get_global_experts_capturer()


def clear_step_buffers(scheduler_output: SchedulerOutput) -> None:
    """Free per-request routing buffers for finished/preempted reqs.

    main: `free_routing_buffers(finished, preempted)`.
    0.20.2: `capturer.clear_buffer()` (full-buffer reset).
    """
    if USE_LEGACY_API:
        capturer = get_capturer()
        if capturer is not None:
            capturer.clear_buffer()
        return

    _rec.free_routing_buffers(
        scheduler_output.finished_req_ids,
        getattr(scheduler_output, "preempted_req_ids", None),
    )


def issue_d2h_copy(
    *,
    input_batch_req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
    positions: torch.Tensor,
    positions_cpu: torch.Tensor | None,
    legacy_indices: torch.Tensor | None = None,
) -> None:
    """Trigger the per-step D2H copy of routed experts.

    main: `issue_routing_d2h_copy(...)` (async copy).
    0.20.2: `capturer.save_captured_experts(indices=legacy_indices)`.
    """
    if USE_LEGACY_API:
        capturer = get_capturer()
        if capturer is not None:
            capturer.save_captured_experts(indices=legacy_indices)
        return

    _rec.issue_routing_d2h_copy(
        input_batch_req_ids=input_batch_req_ids,
        num_scheduled_tokens=num_scheduled_tokens,
        positions=positions,
        positions_cpu=positions_cpu,
    )


def extract_for_current_batch(
    *,
    req_ids: list[str],
    requests: dict,
    req_id_to_index: dict[str, int],
    num_tokens_no_spec: np.ndarray,
    max_model_len: int,
) -> dict[str, np.ndarray] | None:
    """Pull routing data for requests finishing this step.

    main: `extract_routed_experts_for_current_batch(...)`.
    0.20.2: routing data flows through a different channel inside
    `save_captured_experts`, so this returns None.
    """
    if USE_LEGACY_API:
        return None
    return _rec.extract_routed_experts_for_current_batch(
        req_ids=req_ids,
        requests=requests,
        req_id_to_index=req_id_to_index,
        num_tokens_no_spec=num_tokens_no_spec,
        max_model_len=max_model_len,
    )


def call_capture(capturer, *, layer_id: int, topk_ids: torch.Tensor) -> None:
    """Invoke `.capture(...)` on a capturer instance.

    Both 0.20.2 and main expose `capture(layer_id, topk_ids)`, so this
    is a thin pass-through kept for symmetry with the other helpers.
    """
    if capturer is None:
        return
    capturer.capture(layer_id=layer_id, topk_ids=topk_ids)
