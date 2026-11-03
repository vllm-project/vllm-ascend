# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Test-only fixed-history processor using vLLM's supported MRV2 extension.

The sampler clones raw logits before calling active processors. Masking only
that copy preserves full model scores while MTP verification follows the same
reference history. This is a numerical regression tool, not a benchmark.
"""

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor


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


class FixedTraceProcessor(LogitsProcessor):
    MAX_TRACE = 128

    def __init__(self, vllm_config, req_states):
        import numpy as np
        import torch
        from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

        self.states = req_states
        self.lengths = UvaBackedTensor(req_states.max_num_reqs, dtype=torch.int32)
        self.traces = StagedWriteTensor(
            (req_states.max_num_reqs, self.MAX_TRACE), dtype=torch.int64, device=req_states.device
        )
        self.enabled = np.zeros(req_states.max_num_reqs, dtype=bool)

    @classmethod
    def validate_params(cls, sampling_params):
        trace = (sampling_params.extra_args or {}).get("regression_trace")
        if trace is None:
            return
        if not isinstance(trace, list) or not 1 <= len(trace) <= cls.MAX_TRACE:
            raise ValueError("regression_trace must contain 1..128 token IDs")
        if not all(isinstance(token, int) and token >= 0 for token in trace):
            raise ValueError("Invalid regression_trace token ID")
        if sampling_params.temperature != 0 or sampling_params.max_tokens != len(trace):
            raise ValueError("Trace regression requires greedy sampling and max_tokens=len(trace)")

    def add_request(self, req_idx, sampling_params):
        trace = (sampling_params.extra_args or {}).get("regression_trace")
        self.lengths.np[req_idx] = len(trace) if trace else 0
        self.enabled[req_idx] = bool(trace)
        if trace:
            if max(trace) >= self.states.vocab_size:
                raise ValueError("Regression trace token exceeds vocabulary")
            self.traces.stage_write(req_idx, 0, trace)
        return bool(trace)

    def apply_staged_writes(self):
        self.lengths.copy_to_uva()
        self.traces.apply_write()

    def apply(self, logits, ctx):
        if not self.enabled[ctx.idx_mapping_np].any():
            return logits
        return mask_trace_logits(
            logits, ctx.expanded_idx_mapping, ctx.pos, self.states.prompt_len.gpu, self.traces.gpu, self.lengths.gpu
        )
