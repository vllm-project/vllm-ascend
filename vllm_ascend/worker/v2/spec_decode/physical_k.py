# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime physical-K support for the V2 DSpark/DFlash workers.

The V2 worker keeps maximum-size buffers for graph capture.  A smaller K can
still use those buffers safely when the active batch is uniform: the runtime
width is passed to the input-preparation kernel and the graph manager selects
the matching width-specific descriptor.  Mixed per-request K is intentionally
left at the maximum width until a ragged query kernel is available.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch


def _method_params(vllm_config: Any) -> dict[str, Any]:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    dynamic_config = additional_config.get("dynamic_spec_config", {})
    if not isinstance(dynamic_config, dict):
        return {}
    params = dynamic_config.get("method_params", {})
    return params if isinstance(params, dict) else {}


def v2_varlen_physical_k_enabled(vllm_config: Any) -> bool:
    """Whether the explicit V2 variable-width graph path is enabled."""

    params = _method_params(vllm_config)
    return bool(
        params.get(
            "v2_varlen_physical_k",
            params.get("adaptive_draft_k_v2", False),
        )
    )


def configured_capture_k(vllm_config: Any, max_k: int) -> tuple[int, ...]:
    """Return the physical K values for which V2 FULL graphs are captured."""

    params = _method_params(vllm_config)
    configured = params.get("v2_varlen_capture_k")
    if configured is None:
        values = range(1, max_k + 1)
    elif isinstance(configured, (list, tuple)):
        values = configured
    else:
        values = (configured,)

    result = sorted({max(1, min(int(value), max_k)) for value in values})
    return tuple(result)


def query_width(sample_from_anchor: bool, draft_k: int) -> int:
    """Convert draft token K to the query rows emitted by DSpark/DFlash."""

    return draft_k if sample_from_anchor else draft_k + 1


def initialize_physical_k_buffers(speculator: Any) -> None:
    """Preallocate all width-dependent index buffers before graph capture.

    ``physical_k_scope`` is also entered from the FULL graph capture path.  A
    ``torch.arange(..., device=npu)`` created inside that scope is then created
    while an ACL graph is being captured.  The resulting host/device copy can
    outlive the capture allocator and has triggered an Ascend MTE DDR address
    error.  Build the small width-specific buffers once during speculator
    initialization and only swap stable tensor references at runtime.
    """

    if getattr(speculator, "_vllm_ascend_physical_k_buffers_initialized", False):
        return

    max_k = int(
        getattr(
            speculator,
            "_vllm_ascend_max_speculative_steps",
            getattr(speculator, "num_speculative_steps"),
        )
    )
    max_num_reqs = int(getattr(speculator, "max_num_reqs", 0))
    sample_col = getattr(speculator, "sample_col", None)
    anchor_idx = getattr(speculator, "_anchor_idx", None)
    sample_from_anchor = bool(getattr(speculator, "sample_from_anchor", False))

    sample_cols: dict[int, Any] = {}
    anchor_indices: dict[int, Any] = {}
    for active_k in range(1, max_k):
        if sample_col is not None:
            sample_cols[active_k] = torch.arange(
                active_k,
                dtype=sample_col.dtype,
                device=sample_col.device,
            ).repeat(max_num_reqs)
        if anchor_idx is not None:
            anchor_indices[active_k] = (
                torch.arange(
                    max_num_reqs,
                    dtype=anchor_idx.dtype,
                    device=anchor_idx.device,
                )
                * query_width(sample_from_anchor, active_k)
            )

    speculator._vllm_ascend_physical_k_sample_cols = sample_cols
    speculator._vllm_ascend_physical_k_anchor_indices = anchor_indices
    speculator._vllm_ascend_physical_k_buffers_initialized = True


def _uniform_runtime_k(input_batch: Any, max_k: int) -> int | None:
    counts = getattr(input_batch, "num_draft_tokens_per_req", None)
    if counts is None:
        return None
    if hasattr(counts, "detach"):
        counts = counts.detach().cpu().tolist()
    elif hasattr(counts, "tolist"):
        counts = counts.tolist()
    counts = [int(value) for value in counts]
    if not counts or max(counts, default=0) <= 0:
        return None
    if len(set(counts)) != 1:
        # The current NPU input kernel has one query width per request block.
        # A mixed batch is therefore kept on the fixed-width safe path.
        return None
    return max(1, min(counts[0], max_k))


@contextmanager
def physical_k_scope(
    speculator: Any,
    input_batch: Any | None = None,
    *,
    draft_k: int | None = None,
) -> Iterator[int]:
    """Temporarily expose a smaller K to the upstream DFlash/DSpark code.

    The upstream implementation reads ``num_speculative_steps`` and
    ``num_query_per_req`` as runtime attributes in its input kernel, sampler,
    and graph dispatch path.  Updating those attributes together keeps all
    three paths consistent without changing the vLLM checkout.
    """

    max_k = int(
        getattr(
            speculator,
            "_vllm_ascend_max_speculative_steps",
            getattr(speculator, "num_speculative_steps"),
        )
    )
    if not v2_varlen_physical_k_enabled(
        getattr(speculator, "vllm_config", None)
    ):
        speculator._vllm_ascend_last_runtime_k = max_k
        yield max_k
        return
    sample_from_anchor = bool(getattr(speculator, "sample_from_anchor", False))
    active_k = draft_k
    if active_k is None and input_batch is not None:
        active_k = _uniform_runtime_k(input_batch, max_k)
    if active_k is None or active_k >= max_k:
        speculator._vllm_ascend_last_runtime_k = max_k
        yield max_k
        return

    active_k = max(1, min(int(active_k), max_k))
    initialize_physical_k_buffers(speculator)
    old_steps = speculator.num_speculative_steps
    old_query_width = speculator.num_query_per_req
    old_sample_col = getattr(speculator, "sample_col", None)
    old_anchor_idx = getattr(speculator, "_anchor_idx", None)
    old_confidence_probs = getattr(speculator, "draft_token_confidence_probs", None)
    try:
        speculator.num_speculative_steps = active_k
        speculator.num_query_per_req = query_width(sample_from_anchor, active_k)
        if old_sample_col is not None:
            speculator.sample_col = speculator._vllm_ascend_physical_k_sample_cols[
                active_k
            ]
        if old_confidence_probs is not None and old_confidence_probs.ndim >= 2:
            # DSpark's upstream confidence-head path assigns the freshly
            # computed [num_reqs, K] result to the request buffer.  Keep the
            # backing allocation at max-K for record_confidences(), but expose
            # an active-width view while the physical-K scope is running.
            speculator.draft_token_confidence_probs = old_confidence_probs[
                :, :active_k
            ]
        if old_anchor_idx is not None:
            speculator._anchor_idx = (
                speculator._vllm_ascend_physical_k_anchor_indices[active_k]
            )
        speculator._vllm_ascend_active_speculative_steps = active_k
        speculator._vllm_ascend_last_runtime_k = active_k
        yield active_k
    finally:
        speculator.num_speculative_steps = old_steps
        speculator.num_query_per_req = old_query_width
        if old_sample_col is not None:
            speculator.sample_col = old_sample_col
        if old_confidence_probs is not None:
            speculator.draft_token_confidence_probs = old_confidence_probs
        if old_anchor_idx is not None:
            speculator._anchor_idx = old_anchor_idx
        speculator._vllm_ascend_active_speculative_steps = max_k
