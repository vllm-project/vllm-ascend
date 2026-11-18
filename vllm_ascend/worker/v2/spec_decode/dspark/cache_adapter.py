# SPDX-License-Identifier: Apache-2.0
"""Compact request metadata at the NPU rejection-kernel boundary.

The existing NPU kernels share one index for cache, seed and temperature.
Gather the local requests' q rows and seed/temperature once, then give those
kernels compact request indices. External scheduler/request state is untouched.
"""
import torch


def compact_rejection_inputs(cache, draft_sampled, idx_mapping,
                             expanded_idx_mapping, expanded_local_pos,
                             temperature, seed):
    slots = idx_mapping.to(torch.int64)
    expanded_slots = expanded_idx_mapping.to(torch.int64)
    mapped = cache.row_mapping.to(torch.int64)
    owners = cache.inverse_row_mapping.to(torch.int64)

    # Ownership changes when requests are skipped, reordered or slots reused.
    # A cached row is usable only if its last writer is this request state.
    rows = mapped.index_select(0, slots)
    row_exists = rows >= 0
    holders = owners.index_select(0, rows.clamp(min=0))
    valid = row_exists & (holders == slots)

    expanded_rows = mapped.index_select(0, expanded_slots)
    expanded_holders = owners.index_select(0, expanded_rows.clamp(min=0))
    stale = ((expanded_local_pos > 0)
             & ((expanded_rows < 0) | (expanded_holders != expanded_slots)))
    draft_sampled = draft_sampled.masked_fill(stale, -1)

    # Shape is [num_local_requests, K, V], never [all_requests, K, V].
    compact_q = cache.values.index_select(0, rows.clamp(min=0))
    compact_q.masked_fill_(~valid[:, None, None], 0)
    compact_temperature = temperature.index_select(0, slots)
    compact_seed = seed.index_select(0, slots)

    local_indices = torch.arange(slots.numel(), dtype=idx_mapping.dtype, device=slots.device)
    state_to_local = torch.full_like(cache.row_mapping, -1)
    state_to_local.scatter_(0, slots, local_indices.to(state_to_local.dtype))
    compact_expanded = state_to_local.index_select(0, expanded_slots).to(expanded_idx_mapping.dtype)
    return (compact_q, draft_sampled, local_indices, compact_expanded,
            compact_temperature, compact_seed)
