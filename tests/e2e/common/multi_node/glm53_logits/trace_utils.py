# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project


def mask_trace_logits(logits, slots, positions, prompt_lengths, traces, lengths):
    import torch

    safe_slots = slots.long().clamp(min=0)
    steps = positions.long() + 1 - prompt_lengths[safe_slots].long()
    active = (slots >= 0) & (steps >= 0) & (steps < lengths[safe_slots])
    safe_steps = steps.clamp(min=0, max=traces.shape[1] - 1)
    forced_ids = traces[safe_slots, safe_steps]
    vocabulary = torch.arange(logits.shape[1], device=logits.device)
    mask = active[:, None] & (vocabulary[None, :] != forced_ids[:, None])
    logits.masked_fill_(mask, -float("inf"))
    return logits
