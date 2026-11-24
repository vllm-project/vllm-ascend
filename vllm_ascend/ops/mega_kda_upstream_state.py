# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Map vLLM decode pages to the public MegaRecurrentKda state/replay ABI."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_ascend.ops.triton.kda.candidate_pages import store_candidate_pages
from vllm_ascend.ops.triton.mamba.state_ops import gather_initial_states


@dataclass
class UpstreamTokenLayout:
    lengths: torch.Tensor
    token_indices: torch.Tensor
    real_tokens: torch.Tensor
    requests: torch.Tensor | None
    cu_seqlens: torch.Tensor | None
    metadata: torch.Tensor | None


def prepare_token_layout(prefix, num_tokens, window, build_metadata=None) -> UpstreamTokenLayout:
    batch = prefix.numel() - 1
    if prefix.ndim != 1 or batch < 1 or window < 1 or not 0 < num_tokens <= batch * window:
        raise ValueError("Invalid MegaRecurrentKda token layout")
    device = prefix.device
    prefix = prefix.to(torch.int32)
    lengths = prefix[1:] - prefix[:-1]
    cu_seqlens = metadata = None
    if window == 1:
        token_indices = prefix[:-1].long()
        requests = None
        real_tokens = lengths > 0
    else:
        # TH requires strictly increasing offsets, while vLLM graph padding
        # allows empty requests. Append dummy rows AFTER real tokens so that
        # recurrence and replay records for the real prefix remain unchanged.
        # At most B-ceil(A/Q) rows can be empty for A real tokens. The
        # largest A allowed by this graph shape is T. This yields a static
        # bound without adding any rows to an already full B*Q batch.
        total = num_tokens + batch - (num_tokens + window - 1) // window
        strict_lengths = lengths.clamp_min(1)
        slack = window - strict_lengths
        preceding_slack = torch.cumsum(slack, 0, dtype=torch.int32) - slack
        extra = torch.minimum((total - strict_lengths.sum(dtype=torch.int32) - preceding_slack).clamp_min(0), slack)
        strict_lengths = strict_lengths + extra
        cu_seqlens = torch.cat((prefix.new_zeros(1), strict_lengths.cumsum(0, dtype=torch.int32)))
        # Mark the strictly separated boundaries, then scan to get the request
        # row for each physical token. No host readback or dynamic-size select.
        boundaries = torch.zeros(total, dtype=torch.int32, device=device)
        boundaries.index_fill_(0, cu_seqlens[1:-1].long(), 1)
        requests = boundaries.cumsum(0, dtype=torch.int64)
        local = torch.arange(total, device=device) - cu_seqlens.index_select(0, requests)
        token_indices = prefix.index_select(0, requests) + local
        real_tokens = local < lengths.index_select(0, requests)
        if build_metadata is not None:
            metadata = build_metadata(cu_seqlens, tokens=total)

    return UpstreamTokenLayout(
        lengths, token_indices.clamp(0, num_tokens - 1).long(), real_tokens, requests, cu_seqlens, metadata
    )


@dataclass
class UpstreamStatePlan:
    """Device tensors shared by all KDA layers for one metadata group."""

    window: int
    lengths: torch.Tensor
    active: torch.Tensor
    initial: torch.Tensor
    source: torch.Tensor
    destinations: torch.Tensor
    conv_slots: torch.Tensor
    history_indices: torch.Tensor
    conv_write_rows: torch.Tensor
    output_write_rows: torch.Tensor
    write_starts: torch.Tensor
    conv_prefix: torch.Tensor
    scratch_slots: torch.Tensor
    scratch_accepted: torch.Tensor
    token_indices: torch.Tensor
    real_tokens: torch.Tensor
    cu_seqlens: torch.Tensor | None
    metadata: torch.Tensor | None


def prepare_upstream_state_plan(
    layout: UpstreamTokenLayout,
    state_slots: torch.Tensor,
    conv_slots: torch.Tensor,
    accepted: torch.Tensor | None,
    has_initial: torch.Tensor | None,
    *,
    group_indices: torch.Tensor | None = None,
    max_query_len: int,
    num_state_slots: int,
    num_conv_slots: int,
) -> UpstreamStatePlan:
    """Prepare one fixed-shape plan from replay-time device inputs.

    max_query_len is the scheduler's speculative window (one for ordinary
    decode). Multi-token non-speculative prefill uses the normal prefill path.
    State columns and conv slots are already physical page IDs from the GDN
    metadata builder, including its align-mode block-table lookup.
    """
    slots = state_slots.unsqueeze(1) if state_slots.ndim == 1 else state_slots
    conv_slots = conv_slots[:, 0] if conv_slots.ndim == 2 else conv_slots
    lengths = layout.lengths
    batch = lengths.numel()
    window = max_query_len
    if (
        slots.ndim != 2
        or slots.shape[0] != batch
        or slots.shape[1] < window
        or conv_slots.shape != (batch,)
        or (accepted is not None and accepted.shape != (batch,))
        or (has_initial is not None and has_initial.shape != (batch,))
        or min(num_state_slots, num_conv_slots) < 1
    ):
        raise ValueError("Invalid decode window, page table or request metadata for MegaRecurrentKda")

    device = lengths.device
    scratch_slots = torch.arange(batch, device=device, dtype=torch.int32)
    scratch_accepted = torch.ones_like(scratch_slots)
    if accepted is None:
        accepted = scratch_accepted
    accepted_offset = (accepted.to(torch.int64) - 1).clamp(0, window - 1)
    source = slots.gather(1, accepted_offset.unsqueeze(1)).squeeze(1)
    active = (
        (lengths > 0)
        & (accepted >= 1)
        & (accepted <= window)
        & (source > 0)
        & (source < num_state_slots)
        & (conv_slots > 0)
        & (conv_slots < num_conv_slots)
    )
    history_indices = accepted_offset[:, None] + torch.arange(3, device=device)
    token_indices = layout.token_indices
    real_tokens = layout.real_tokens & (active if layout.requests is None else active.index_select(0, layout.requests))
    if group_indices is not None:
        token_indices = group_indices.index_select(0, token_indices).long()

    return UpstreamStatePlan(
        window,
        lengths,
        active,
        active if has_initial is None else active & has_initial.bool(),
        source.clamp(0, num_state_slots - 1).long(),
        slots,
        conv_slots.clamp(0, num_conv_slots - 1).long(),
        history_indices,
        torch.where(active, conv_slots, -1).to(torch.int32).contiguous(),
        torch.where(real_tokens, token_indices, -1).to(torch.int32).contiguous(),
        torch.arange(max(batch, token_indices.numel()) + 1, dtype=torch.int32, device=device),
        torch.arange(window + 2, device=device)[None, :] < lengths[:, None] + 2,
        scratch_slots,
        scratch_accepted,
        token_indices,
        real_tokens,
        layout.cu_seqlens,
        layout.metadata,
    )


def _scatter_rows(cache: torch.Tensor, values: torch.Tensor, indices: torch.Tensor, starts: torch.Tensor) -> None:
    """Copy active, non-overlapping rows without redirecting invalid writes to page zero."""
    if cache.device.type == "cpu":
        valid = indices >= 0
        cache.index_copy_(0, indices[valid].long(), values[valid])
        return
    from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state

    # The packed-to-cache copy only reads packed rows [0, B); the dummy
    # packed row and packed_indices argument are used by the reverse direction.
    if cache.ndim == 2:
        cache = cache.unsqueeze(1)
    elif cache.ndim == 4:
        cache = cache.view(cache.size(0), cache.size(1), -1)
    copy_conv_state(
        cache,
        values.contiguous().view(indices.numel(), *cache.shape[1:]),
        indices,
        starts,
        indices,
        write_back=True,
    )


def run_upstream_state_batch(
    hidden_states: torch.Tensor,
    conv_cache: torch.Tensor,
    state_cache: torch.Tensor,
    plan: UpstreamStatePlan,
    launch: Callable,
    output: torch.Tensor,
) -> None:
    """Run one ordinary or speculative group. Long prefill dispatches before this call.

    The native checkpoint is read-only. Ordinary decode and multi-page verify
    reconstruct the appropriate pages from the returned u/k/decay factors.
    """
    hidden = hidden_states.shape[1]
    window = plan.window
    batch = plan.lengths.numel()
    if (
        state_cache.ndim != 4
        or state_cache.size(2) != state_cache.size(3)
        or state_cache.dtype != torch.float32
        or conv_cache.ndim != 3
        or conv_cache.size(1) < window + 2
        or conv_cache.dtype != torch.bfloat16
    ):
        raise ValueError("Incompatible MegaRecurrentKda state, convolution or metadata layout")
    heads, dim = state_cache.shape[1:3]
    # Physical cache pages have a padded row stride. Gather the selected
    # pages directly instead of making the whole cache contiguous.
    state = gather_initial_states(state_cache, plan.source, plan.initial)
    cache_window = conv_cache[:, : window + 2]
    old_conv = gather_initial_states(cache_window, plan.conv_slots, plan.active)
    capacity = window
    conv = conv_cache.new_zeros((batch, capacity + 2, conv_cache.size(2)))
    history = old_conv.gather(1, plan.history_indices[:, :, None].expand(-1, -1, old_conv.size(2)))
    conv[:, :3].copy_(torch.where(plan.initial[:, None, None], history, 0))
    u = state.new_zeros((batch, heads, capacity, dim))
    k = torch.zeros_like(u, dtype=torch.float16)
    decay = torch.zeros_like(u)
    if hidden_states.stride(1) == 1:
        packed = gather_initial_states(hidden_states, plan.token_indices, plan.real_tokens)
    else:
        packed = torch.where(plan.real_tokens[:, None], hidden_states.index_select(0, plan.token_indices), 0)
    kwargs = {}
    if plan.cu_seqlens is None:
        packed = packed.unsqueeze(1)
    else:
        kwargs = {"cu_seqlens": plan.cu_seqlens, "metadata": plan.metadata}
    out, conv, u, k, decay = launch(
        packed, conv, state, u, k, decay, plan.scratch_slots, plan.scratch_accepted, **kwargs
    )

    store_candidate_pages(state_cache, state, u, k, decay, plan.destinations, plan.lengths, plan.active)

    # conv stores [h1,h2,q0,...]. Commit with c accepted tokens reads the
    # three entries starting at c-1. Retain the old suffix beyond each real
    # query to keep synthetic TH rows out of the scheduler's state pages.
    new_conv = torch.where(plan.conv_prefix[:, :, None], conv[:, : window + 2], old_conv)
    _scatter_rows(cache_window, new_conv, plan.conv_write_rows, plan.write_starts)
    _scatter_rows(output, out.reshape(-1, hidden), plan.output_write_rows, plan.write_starts)
