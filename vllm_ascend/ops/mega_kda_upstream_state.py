# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Map vLLM decode pages to the public MegaRecurrentKda state/replay ABI."""

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class UpstreamStatePlan:
    """Device tensors shared by all KDA layers for one metadata group."""

    num_tokens: int
    window: int
    recover: bool
    lengths: torch.Tensor
    active: torch.Tensor
    initial: torch.Tensor
    source: torch.Tensor
    destinations: torch.Tensor
    conv_slots: torch.Tensor
    history_indices: torch.Tensor
    scratch_slots: torch.Tensor
    scratch_accepted: torch.Tensor
    token_indices: torch.Tensor
    real_tokens: torch.Tensor
    cu_seqlens: torch.Tensor | None
    metadata: torch.Tensor | None


def prepare_upstream_state_plan(
    prefix: torch.Tensor,
    state_slots: torch.Tensor,
    conv_slots: torch.Tensor,
    accepted: torch.Tensor,
    has_initial: torch.Tensor,
    *,
    num_tokens: int,
    max_query_len: int,
    num_state_slots: int,
    num_conv_slots: int,
    recover: bool = False,
    build_metadata: Callable | None = None,
) -> UpstreamStatePlan:
    """Prepare one fixed-shape plan from replay-time device inputs.

    max_query_len is the scheduler's speculative window (one for ordinary
    decode). Multi-token non-speculative prefill uses the normal prefill path.
    State columns and conv slots are already physical page IDs from the GDN
    metadata builder, including its align-mode block-table lookup.
    """
    slots = state_slots.unsqueeze(1) if state_slots.ndim == 1 else state_slots
    conv_slots = conv_slots[:, 0] if conv_slots.ndim == 2 else conv_slots
    batch = prefix.numel() - 1
    window = max_query_len
    if (
        prefix.ndim != 1
        or slots.ndim != 2
        or slots.shape[0] != batch
        or slots.shape[1] < (1 if recover else window)
        or conv_slots.shape != (batch,)
        or accepted.shape != (batch,)
        or has_initial.shape != (batch,)
        or not (window >= 1 and 0 <= num_tokens <= batch * window)
        or min(num_state_slots, num_conv_slots) < 1
    ):
        raise ValueError("Invalid decode window, page table or request metadata for MegaRecurrentKda")

    device = prefix.device
    scratch_slots = torch.arange(batch, device=device, dtype=torch.int32)
    accepted_offset = (accepted.to(torch.int64) - 1).clamp(0, window - 1)
    source = slots[:, 0] if recover else slots.gather(1, accepted_offset.unsqueeze(1)).squeeze(1)
    prefix = prefix.to(torch.int32)
    lengths = prefix[1:] - prefix[:-1]
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
    cu_seqlens = metadata = None
    if window == 1:
        token_indices = prefix[:-1].long()
        real_tokens = active
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
        real_tokens = (local < lengths.index_select(0, requests)) & active.index_select(0, requests)
        if build_metadata is not None:
            metadata = build_metadata(cu_seqlens, tokens=total)

    return UpstreamStatePlan(
        num_tokens,
        window,
        recover,
        lengths,
        active,
        active & has_initial.bool(),
        source.clamp(0, num_state_slots - 1).long(),
        slots,
        conv_slots.clamp(0, num_conv_slots - 1).long(),
        history_indices,
        scratch_slots,
        torch.ones_like(scratch_slots),
        token_indices.clamp(0, max(num_tokens - 1, 0)).long(),
        real_tokens,
        cu_seqlens,
        metadata,
    )


def _scatter_rows(cache: torch.Tensor, values: torch.Tensor, slots: torch.Tensor, valid: torch.Tensor) -> None:
    """Copy active, non-overlapping rows without redirecting invalid writes to page zero."""
    if cache.device.type == "cpu":
        cache.index_copy_(0, slots[valid].long(), values[valid])
        return
    from vllm_ascend.ops.triton.kda.conv_state import copy_conv_state

    indices = torch.where(valid, slots, -1).to(torch.int32).contiguous()
    starts = torch.arange(indices.numel() + 1, device=indices.device, dtype=torch.int32)
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
    prefix: torch.Tensor,
    state_slots: torch.Tensor,
    conv_slots: torch.Tensor,
    accepted: torch.Tensor,
    has_initial: torch.Tensor,
    launch: Callable,
    *,
    max_query_len: int,
    records=None,
    plan: UpstreamStatePlan | None = None,
    build_metadata: Callable | None = None,
) -> torch.Tensor:
    """Run one ordinary or speculative group. Long prefill dispatches before this call.

    The native checkpoint is read-only. Ordinary decode and multi-page verify
    reconstruct the appropriate pages from the returned u/k/decay factors.
    Single-page RecoverSSM saves factors by request row for post-sampling commit.
    """
    tokens, hidden = hidden_states.shape
    if not tokens:
        return hidden_states.new_zeros((0, hidden))
    if plan is None:
        plan = prepare_upstream_state_plan(
            prefix,
            state_slots,
            conv_slots,
            accepted,
            has_initial,
            num_tokens=tokens,
            max_query_len=max_query_len,
            num_state_slots=state_cache.size(0),
            num_conv_slots=conv_cache.size(0),
            recover=records is not None,
            build_metadata=build_metadata,
        )
    window = plan.window
    batch = plan.lengths.numel()
    if (
        tokens != plan.num_tokens
        or window != max_query_len
        or plan.recover != (records is not None)
        or state_cache.ndim != 4
        or state_cache.size(2) != state_cache.size(3)
        or state_cache.dtype != torch.float32
        or conv_cache.ndim != 3
        or conv_cache.size(1) < window + 2
        or conv_cache.dtype != torch.bfloat16
    ):
        raise ValueError("Incompatible MegaRecurrentKda state, convolution or metadata layout")
    heads, dim = state_cache.shape[1:3]
    if records is not None:
        correction, key_gate = records if isinstance(records, tuple) else (records.correction, records.key_gate)
        if (
            correction.shape[0] < batch
            or correction.shape[1:] != (heads, window, dim)
            or key_gate.shape != (*correction.shape[:-1], 2 * dim)
            or correction.dtype != torch.float32
            or key_gate.dtype != torch.float32
        ):
            raise ValueError("RecoverSSM needs FP32 request-indexed correction and normalized-key/decay storage")

    state = state_cache.index_select(0, plan.source)
    state = torch.where(plan.initial[:, None, None, None], state, 0)
    capacity = window
    conv = conv_cache.new_zeros((batch, capacity + 2, conv_cache.size(2)))
    history = conv_cache[plan.conv_slots[:, None], plan.history_indices]
    conv[:, :3].copy_(torch.where(plan.initial[:, None, None], history, 0))
    u = state.new_zeros((batch, heads, capacity, dim))
    k = torch.zeros_like(u, dtype=torch.float16)
    decay = torch.zeros_like(u)
    packed = torch.where(plan.real_tokens[:, None], hidden_states.index_select(0, plan.token_indices), 0)
    kwargs = {}
    if plan.cu_seqlens is None:
        packed = packed.unsqueeze(1)
    else:
        kwargs = {"cu_seqlens": plan.cu_seqlens, "metadata": plan.metadata}
    out, conv, u, k, decay = launch(
        packed, conv, state, u, k, decay, plan.scratch_slots, plan.scratch_accepted, **kwargs
    )

    steps = torch.arange(window, device=state.device)
    live = plan.active[:, None] & (steps[None, :] < plan.lengths[:, None])
    if records is not None:
        valid = live[:, None, :, None]
        correction[:batch].copy_(torch.where(valid, u[:, :, :window], correction[:batch]))
        kg = torch.cat((k[:, :, :window].float(), decay[:, :, :window]), dim=-1)
        key_gate[:batch].copy_(torch.where(valid, kg, key_gate[:batch]))
    else:
        for step in range(window):
            # Upstream stores normalized k in FP16. Reconstruct from the public
            # replay values in FP32, using the checkpoint's [value,key] order.
            state = state * decay[:, :, step].unsqueeze(-2) + (
                u[:, :, step].unsqueeze(-1) * k[:, :, step].float().unsqueeze(-2)
            )
            destinations = plan.destinations[:, step]
            valid = live[:, step] & (destinations > 0) & (destinations < state_cache.size(0))
            _scatter_rows(state_cache, state, destinations, valid)

    # conv stores [h1,h2,q0,...]. Commit with c accepted tokens reads the
    # three entries starting at c-1. Retain the old suffix beyond each real
    # query to keep synthetic TH rows out of the scheduler's state pages.
    cache_window = conv_cache[:, : window + 2]
    old = cache_window.index_select(0, plan.conv_slots)
    rows = torch.arange(window + 2, device=state.device)
    new_conv = torch.where(rows[None, :, None] < plan.lengths[:, None, None] + 2, conv[:, : window + 2], old)
    _scatter_rows(cache_window, new_conv, plan.conv_slots, plan.active)
    output = hidden_states.new_zeros((tokens, hidden))
    _scatter_rows(output, out.reshape(-1, hidden), plan.token_indices, plan.real_tokens)
    return output
