# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""V2 hardware-aware adapters layered over upstream speculative decoding.

Sections own runtime widths/buffers, width-matched draft graphs, and the
PIECEWISE verification fallback respectively. Target FULL graphs and their
profiling remain owned by upstream vLLM/vLLM-Ascend; uncaptured draft widths
retain eager fallback. Confidence and prefix allocation stay in vLLM. Runner
call sites keep their explicit input/attention ordering.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import MethodType
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.dynamic_spec import resolve_physical_k, v2_physical_k_enabled

# Runtime physical widths and persistent buffers.


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

    params = resolve_physical_k(_dynamic_config(vllm_config)) or {}
    configured = params.get("capture_k")
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


# Width-specific draft graph descriptors and capture scopes.


def _sample_from_anchor(self) -> bool:
    speculative_config = self.vllm_config.speculative_config
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    if getattr(speculative_config, "use_dspark", lambda: False)():
        return bool(getattr(hf_config, "sample_from_anchor", True))
    return False


def extend_capture_descriptors(self) -> None:
    """Capture one FULL descriptor for each configured physical K.

    The upstream manager only expands capture widths for its native
    dynamic-spec configuration.  Ascend's hardware-aware policy is an
    independent scheduler path, so add the same descriptor matrix here.
    If a width is not captured, normal dispatch falls back to eager mode;
    it never reuses a graph with a different query width.
    """

    decode_mode = self.cudagraph_mode.decode_mode()
    if decode_mode == CUDAGraphMode.NONE:
        return
    capture_sizes = sorted(self.compilation_config.cudagraph_capture_sizes or [])
    if not capture_sizes:
        return

    speculative_config = self.vllm_config.speculative_config
    max_k = int(getattr(speculative_config, "num_speculative_tokens", 0))
    if max_k <= 0:
        return
    sample_from_anchor = _sample_from_anchor(self)
    max_capture_size = self.compilation_config.max_cudagraph_capture_size or (1 << 60)
    max_decode_tokens = self.max_num_reqs * self.decode_query_len

    capture_descs = self._capture_descs.setdefault(decode_mode, [])
    for raw_tokens in capture_sizes:
        for draft_k in configured_capture_k(self.vllm_config, max_k):
            width = query_width(sample_from_anchor, draft_k)
            rounded_tokens = ((raw_tokens + width - 1) // width) * width
            num_reqs = rounded_tokens // width
            if rounded_tokens > max_decode_tokens or rounded_tokens > max_capture_size or num_reqs > self.max_num_reqs:
                continue
            for num_active_loras in self.lora_capture_cases:
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=rounded_tokens,
                    num_reqs=num_reqs,
                    uniform_token_count=width,
                    num_active_loras=num_active_loras,
                )
                if desc not in capture_descs:
                    capture_descs.append(desc)
                self._candidates.setdefault((rounded_tokens, num_active_loras), []).append(desc)
                for token_count in range(0, rounded_tokens + 1):
                    self._candidates.setdefault((token_count, num_active_loras), []).append(desc)

    capture_descs.sort(key=lambda item: item.num_tokens, reverse=True)

    # Ascend's FIA graph-parameter table is keyed by ``num_tokens`` only.
    # Keeping two FULL graphs with the same token count but different
    # physical K would make them share one workspace/event/parameter bucket
    # (for example, N=16 with K=1, 2 and 4), which can feed a workspace
    # captured for one query width to another kernel.  That combination
    # fails asynchronously in the FIA MTE with an invalid DDR address.
    # Keep the widest graph for each token count; narrower K values remain
    # valid dynamic decisions and dispatch eagerly when no width-matching
    # graph exists.
    widest_by_tokens: dict[int, BatchExecutionDescriptor] = {}
    for desc in capture_descs:
        current = widest_by_tokens.get(desc.num_tokens)
        if current is None or (desc.uniform_token_count or 0) > (current.uniform_token_count or 0):
            widest_by_tokens[desc.num_tokens] = desc
    kept_descs = set(widest_by_tokens.values())
    capture_descs[:] = sorted(
        kept_descs,
        key=lambda item: item.num_tokens,
        reverse=True,
    )
    for candidates in self._candidates.values():
        candidates[:] = list(
            dict.fromkeys(desc for desc in candidates if desc.cg_mode != decode_mode or desc in kept_descs)
        )
    for key, candidates in self._candidates.items():
        unique = list(dict.fromkeys(candidates))
        candidates[:] = unique


@contextmanager
def physical_k_capture_scope(self, forward_fn: Callable):
    """Keep metadata preparation and forward on the same K, restoring on error."""
    # The upstream DFlash graph manager builds ``attn_state`` before it
    # invokes the supplied forward function.  For a variable-width graph,
    # that preparation must observe the same physical K as the forward
    # itself: the Ascend metadata builder derives TND
    # ``actual_seq_lengths_q`` and query padding from the prepared input
    # batch.  Temporarily wrap the upstream helper so capture-time
    # metadata and the model forward use one width.  This also avoids
    # changing the upstream vLLM checkout or the shared attention backend.
    dflash_cudagraph_module = None
    original_prepare_inputs = None
    if self._v2_varlen_physical_k:
        import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as dflash_cudagraph_module

        original_prepare_inputs = dflash_cudagraph_module._prepare_dflash_inputs_to_capture

        def prepare_inputs_with_runtime_width(
            num_reqs: int,
            num_tokens: int,
            input_buffers: InputBuffers,
            block_tables: BlockTables,
            attn_groups: list[list[AttentionGroup]],
            kv_cache_config: KVCacheConfig,
            max_model_len: int,
            skip_attn: bool,
            causal: bool | Mapping[int, bool],
        ):
            width = num_tokens // num_reqs
            draft_k = width if _sample_from_anchor(self) else width - 1
            with physical_k_scope(self.speculator, draft_k=draft_k):
                attn_state = original_prepare_inputs(
                    num_reqs,
                    num_tokens,
                    input_buffers,
                    block_tables,
                    attn_groups,
                    kv_cache_config,
                    max_model_len,
                    skip_attn,
                    causal,
                )

            # ``make_dummy`` normally produces this sequence, but an
            # Ascend graph capture may add padding requests.  Make the
            # exact descriptor contract explicit for FIA's TND layout:
            # the final cumulative query length must equal num_tokens.
            if attn_state.attn_metadata is not None:
                query_lens = [(idx + 1) * width for idx in range(num_reqs)]
                for metadata in attn_state.attn_metadata.values():
                    metadata.actual_seq_lengths_q = query_lens
            return attn_state

        dflash_cudagraph_module._prepare_dflash_inputs_to_capture = prepare_inputs_with_runtime_width

    def forward_with_runtime_width(
        num_reqs: int,
        num_tokens: int,
        attn_metadata: Any,
        slot_mappings: Any,
        num_tokens_across_dp: Any,
        cg_mode: CUDAGraphMode,
    ):
        if not self._v2_varlen_physical_k or num_reqs <= 0:
            return forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )
        width = num_tokens // num_reqs
        draft_k = width if _sample_from_anchor(self) else width - 1
        with physical_k_scope(self.speculator, draft_k=draft_k):
            return forward_fn(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cg_mode,
            )

    try:
        yield forward_with_runtime_width
    finally:
        if dflash_cudagraph_module is not None:
            dflash_cudagraph_module._prepare_dflash_inputs_to_capture = original_prepare_inputs


# PIECEWISE profiling and upstream verification-manager adaptation.


def configure_piecewise_manager(
    manager,
):
    """Make one upstream manager consume Ascend PIECEWISE timings.

    Upstream seeds its step-cost tables from FULL-decode-graph dummy runs
    (``full_cudagraph=True`` samples) and profile sizes derived from the
    captured full-graph token counts. Under Plan A the target decode runs
    through PIECEWISE graphs, so there are no full graphs to price: profile
    a representative grid of piecewise batch sizes instead and price the
    drafter curve from every sample (not only ``full_cudagraph`` ones).

    Keep the object returned by the upstream factory. It owns the
    confidence buffers, copy stream, events, and validation state used by
    the rest of #47808; constructing and discarding it would allocate all
    of those resources twice during startup.
    """

    def batches_to_profile(self, capture_sizes):
        del capture_sizes
        # No FULL graphs: leave ``_cudagraph_limit`` at 0 so the cost
        # tables stay smooth (nothing pads to a captured size).
        self._cudagraph_limit = 0
        max_num_tokens = self.req_states.max_num_batched_tokens
        base_size = max(1, self.num_speculative_steps + 1)
        grid = [base_size]
        while grid[-1] < max_num_tokens:
            grid.append(min(grid[-1] * 2, max_num_tokens))
        from vllm import envs

        context_len = envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN
        for num_tokens in grid:
            for _ in range(3):
                yield {
                    "num_tokens": num_tokens,
                    "context_len": context_len,
                }

    def set_initial_cost_curves(self, samples):
        from collections import defaultdict

        def median_curve(points):
            grouped: dict[int, list[float]] = defaultdict(list)
            for key, value in points:
                grouped[key].append(value)
            return [(k, float(np.median(v))) for k, v in sorted(grouped.items())]

        draft_curve = median_curve((s.num_reqs, s.drafter_ms) for s in samples)
        verify_curve = median_curve((s.num_target_tokens, s.forward_ms) for s in samples)
        self.set_cost_curves(draft_curve, verify_curve)
        logger.debug("ASCEND_AV_COST_CURVES draft=%s verify=%s", draft_curve, verify_curve)

    manager.batches_to_profile = MethodType(  # type: ignore[method-assign]
        batches_to_profile, manager
    )
    manager.set_initial_cost_curves = MethodType(  # type: ignore[method-assign]
        set_initial_cost_curves, manager
    )
    return manager


@contextmanager
def adaptive_verification_gate_wrapper(
    runner_module,
    cudagraph_mode: CUDAGraphMode,
):
    """Adapt upstream verification only for the Ascend PIECEWISE fallback.

    PR #15098 owns the native FULL path, including graph validation, varlen
    capture and startup cost profiling.  Do not replace any of that behavior.
    PIECEWISE has no target FULL graphs to profile, so only that mode relaxes
    the upstream ``AttentionCGSupport.ALWAYS`` gate and installs the Ascend
    PIECEWISE cost-curve sampler.
    """
    if cudagraph_mode != CUDAGraphMode.PIECEWISE:
        yield
        return

    original_factory = getattr(runner_module, "maybe_create_adaptive_verification_manager", None)
    if original_factory is None:
        yield
        return

    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
    )

    def relaxed_factory(
        *,
        enable_adaptive_verification: bool,
        attn_groups,
        attn_cg_support,
        req_states,
        query_start_loc,
        num_bonus_tokens,
        max_total_logits,
        **factory_kwargs,
    ):
        # Keep upstream validation inputs (config, target layer names and
        # additional attention support) intact; only relax the graph gate.
        if not enable_adaptive_verification:
            return original_factory(
                enable_adaptive_verification=enable_adaptive_verification,
                attn_groups=attn_groups,
                attn_cg_support=attn_cg_support,
                req_states=req_states,
                query_start_loc=query_start_loc,
                num_bonus_tokens=num_bonus_tokens,
                max_total_logits=max_total_logits,
                **factory_kwargs,
            )
        try:
            manager = original_factory(
                enable_adaptive_verification=enable_adaptive_verification,
                attn_groups=attn_groups,
                attn_cg_support=attn_cg_support,
                req_states=req_states,
                query_start_loc=query_start_loc,
                num_bonus_tokens=num_bonus_tokens,
                max_total_logits=max_total_logits,
                **factory_kwargs,
            )
        except ValueError as exc:
            # Only the ALWAYS requirement is relaxed on Ascend; any other
            # validation failure must keep failing loudly.
            if "AttentionCGSupport.ALWAYS" not in str(exc):
                raise
            logger.warning(
                "Relaxing the adaptive-verification AttentionCGSupport.ALWAYS "
                "gate for Ascend; decode runs through PIECEWISE graphs: %s",
                exc,
            )
            manager = None

        # Preserve the upstream manager whenever validation succeeds. If only
        # the ALWAYS gate rejected Ascend, instantiate the same upstream class
        # once, then configure either instance for PIECEWISE profiling.
        if manager is None:
            manager = AdaptiveVerificationManager(
                req_states,
                query_start_loc,
                num_bonus_tokens,
                max_total_logits=max_total_logits,
            )
        logger.info("Using the upstream adaptive-verification manager with Ascend PIECEWISE cost profiling.")
        return configure_piecewise_manager(manager)

    try:
        runner_module.maybe_create_adaptive_verification_manager = relaxed_factory
        yield
    finally:
        runner_module.maybe_create_adaptive_verification_manager = original_factory
