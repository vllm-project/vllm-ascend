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

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.dynamic_spec_config import resolve_method_params, v2_physical_k_enabled


def _dynamic_config(vllm_config: Any) -> dict[str, Any]:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    dynamic_config = additional_config.get("dynamic_spec_config", {})
    if not isinstance(dynamic_config, dict):
        return {}
    return dynamic_config


def v2_varlen_physical_k_enabled(vllm_config: Any) -> bool:
    """Whether the explicit V2 variable-width graph path is enabled."""

    return v2_physical_k_enabled(_dynamic_config(vllm_config))


def configured_capture_k(vllm_config: Any, max_k: int) -> tuple[int, ...]:
    """Return the physical K values for which V2 FULL graphs are captured."""

    params = resolve_method_params(_dynamic_config(vllm_config))
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
            speculator.num_speculative_steps,
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
            anchor_indices[active_k] = torch.arange(
                max_num_reqs,
                dtype=anchor_idx.dtype,
                device=anchor_idx.device,
            ) * query_width(sample_from_anchor, active_k)

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
            speculator.num_speculative_steps,
        )
    )
    if not v2_varlen_physical_k_enabled(getattr(speculator, "vllm_config", None)):
        speculator._vllm_ascend_last_runtime_k = max_k
        yield max_k
        return
    sample_from_anchor = bool(getattr(speculator, "sample_from_anchor", False))
    active_k = draft_k
    if active_k is None and input_batch is not None:
        # ``num_draft_tokens_per_req`` describes drafts being verified in the
        # current target pass.  The scheduler output's physical K describes
        # how many new drafts the drafter must create for the next pass; V2
        # publishes that value on the input batch explicitly.
        active_k = getattr(input_batch, "_vllm_ascend_physical_draft_k", None)
        if active_k is None:
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
            speculator.sample_col = speculator._vllm_ascend_physical_k_sample_cols[active_k]
        if old_confidence_probs is not None and old_confidence_probs.ndim >= 2:
            # DSpark's upstream confidence-head path assigns the freshly
            # computed [num_reqs, K] result to the request buffer.  Keep the
            # backing allocation at max-K for record_confidences(), but expose
            # an active-width view while the physical-K scope is running.
            speculator.draft_token_confidence_probs = old_confidence_probs[:, :active_k]
        if old_anchor_idx is not None:
            speculator._anchor_idx = speculator._vllm_ascend_physical_k_anchor_indices[active_k]
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


def _request_count(key: slice, maximum: int) -> int:
    if not isinstance(key, slice) or key.start not in (None, 0) or key.step not in (None, 1):
        raise TypeError("unsupported request buffer index")
    count = maximum if key.stop is None else key.stop
    if not isinstance(count, int) or not 0 <= count <= maximum:
        raise IndexError("request count outside buffer")
    return count


class IndexedDraftTokenBuffer:
    """One device write per column, independent of the request count.

    Construct before graph capture against the full contiguous backing tensor.
    Indices and flat storage views are persistent across capture and replay.
    """

    def __init__(self, buffer: torch.Tensor):
        if buffer.ndim != 2 or not buffer.is_contiguous():
            raise ValueError("draft token backing buffer must be contiguous and 2-D")
        self._buffer = buffer
        self._flat = buffer.view(-1)
        rows = torch.arange(buffer.shape[0], device=buffer.device, dtype=torch.int64) * buffer.shape[1]
        self._columns = tuple(rows + col for col in range(buffer.shape[1]))

    def __setitem__(self, key, value) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("unsupported draft token buffer index")
        reqs, col = key
        count = _request_count(reqs, self._buffer.shape[0])
        if not isinstance(col, int) or not 0 <= col < self._buffer.shape[1]:
            raise IndexError("draft column outside buffer")
        self._flat.index_copy_(0, self._columns[col][:count], value.reshape(count).to(self._buffer.dtype))


class IndexedConfidenceBuffer:
    """Write an active prefix without flattening a noncontiguous narrow view."""

    def __init__(self, buffer: torch.Tensor, active_k: int):
        if buffer.ndim != 2 or not buffer.is_contiguous():
            raise ValueError("confidence backing buffer must be contiguous and 2-D")
        if not 1 <= active_k <= buffer.shape[1]:
            raise ValueError("active K outside confidence buffer")
        self._buffer = buffer
        self._active_k = active_k
        self._flat = buffer.view(-1)
        rows = torch.arange(buffer.shape[0], device=buffer.device, dtype=torch.int64) * buffer.shape[1]
        cols = torch.arange(active_k, device=buffer.device, dtype=torch.int64)
        self._indices = (rows[:, None] + cols).flatten()

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self):
        return (self._buffer.shape[0], self._active_k)

    def __setitem__(self, key, value) -> None:
        count = _request_count(key, self._buffer.shape[0]) * self._active_k
        self._flat.index_copy_(0, self._indices[:count], value.reshape(count).to(self._buffer.dtype))


def initialize_dspark_physical_k(self) -> None:
    """Initialize DSpark only after upstream has finalized sample_from_anchor."""
    self._vllm_ascend_max_speculative_steps = self.num_speculative_steps
    self._physical_k_log_count = 0
    # DSpark changes ``sample_from_anchor`` after DFlash initialization,
    # so initialize the width-dependent anchor indices only now.
    initialize_physical_k_buffers(self)
    self._physical_token_buffer = IndexedDraftTokenBuffer(self.draft_tokens)
    # Retain the full backing allocation, not physical_k_scope's narrow
    # (noncontiguous) view. Allocate indices before graph capture.
    confidence_probs = getattr(self, "draft_token_confidence_probs", None)
    self._physical_confidence_buffers = (
        {k: IndexedConfidenceBuffer(confidence_probs, k) for k in range(1, self.num_speculative_steps)}
        if confidence_probs is not None
        else {}
    )


class PhysicalKDSparkMixin:
    """Adapt storage only; both confidence and sampling execute upstream."""

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        """Keep DSpark confidence writes compatible with active physical K.

        Upstream DSpark assigns the confidence result to the whole fixed-width
        request buffer.  During V2 graph capture the physical-K scope exposes a
        smaller ``num_speculative_steps``, so the result has shape ``[B, K]``
        while the buffer is still ``[B, max_K]``.  A narrow view preserves the
        fixed backing allocation and makes both the dense and top-k sampling
        paths shape-safe; the full view is restored before the caller records
        confidences for the next scheduler step.
        """
        active_k = int(self.num_speculative_steps)
        max_k = int(getattr(self, "_vllm_ascend_max_speculative_steps", active_k))
        if active_k >= max_k:
            # Keep the fixed-K path unchanged; the indexed adapter is only
            # needed by a smaller physical-K graph.
            super()._sample_sequential(num_reqs, head_hidden)
            return

        confidence_probs = getattr(self, "draft_token_confidence_probs", None)
        old_draft_tokens = self.draft_tokens
        self.draft_tokens = self._physical_token_buffer
        if confidence_probs is not None and confidence_probs.ndim >= 2:
            self.draft_token_confidence_probs = self._physical_confidence_buffers[active_k]
        try:
            super()._sample_sequential(num_reqs, head_hidden)
        finally:
            self.draft_tokens = old_draft_tokens
            if confidence_probs is not None and confidence_probs.ndim >= 2:
                self.draft_token_confidence_probs = confidence_probs


class PhysicalKDFlashMixin:
    """Keep parallel DFlash writes inside the active fixed-storage prefix."""

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Run DFlash with a variable physical K without resizing buffers.

        The upstream implementation assigns the sampled ``[num_reqs, K]``
        result to ``draft_tokens[:num_reqs]``.  That is valid for the fixed
        configured K, but V2 physical-K replay temporarily exposes a smaller
        ``num_speculative_steps`` while ``draft_tokens`` remains allocated at
        the maximum width.  Assign only the active prefix so the fixed buffer
        contract used by the scheduler and rejection sampler is preserved.
        """
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        num_steps = self.num_speculative_steps
        num_sample = num_reqs * num_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        draft_tokens = self.sample_draft(
            sample_hidden_states,
            self.sample_pos[:num_sample] - 2,
            self.sample_idx_mapping[:num_sample],
            self.temperature,
            self.seeds,
            self.sample_col[:num_sample],
            self.draft_logits,
        )
        draft_tokens = draft_tokens.view(num_reqs, num_steps)
        # ``draft_tokens[:, :num_steps]`` is a strided view when physical K
        # is smaller than the configured maximum.  Use contiguous per-request
        # rows so ACL graph capture never records an invalid strided copy.
        for req_idx in range(num_reqs):
            self.draft_tokens[req_idx, :num_steps].copy_(draft_tokens[req_idx])
