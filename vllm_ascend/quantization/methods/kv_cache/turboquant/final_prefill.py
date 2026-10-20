# SPDX-License-Identifier: Apache-2.0
"""Colocated BF16 active/prefix pages with an incremental TQ4 decode mirror.

Both pools are charged to the scheduler's page budget. BF16 pages remain the
source of truth for prefix sharing and mixed batches; this compatibility mode
does not claim the capacity of a TQ4-only cache. No page format is changed in
place and both views retain stable addresses for FULL decode graphs.
"""

from dataclasses import dataclass

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.distributed.kv_transfer.kv_p2p.final_prefill import (
    LATENT_DIM,
    ROPE_DIM,
    TQ_SLOT_BYTES,
)

FINAL_PREFILL_CHUNK_TOKENS = 4096
BF16_SLOT_BYTES = (LATENT_DIM + ROPE_DIM) * 2


def validate_colocated_final_prefill(config):
    parallel = config.parallel_config
    if config.cache_config.cache_dtype != "auto" or config.model_config.dtype != torch.bfloat16:
        raise ValueError("Final-prefill active cache requires --kv-cache-dtype auto and BF16 model dtype")
    if config.kv_transfer_config is not None:
        raise ValueError("Use final_prefill_export on the P/D connector; the colocated mirror needs no connector")
    if getattr(config, "use_v2_model_runner", False):
        raise ValueError("Final-prefill active cache requires the V1 model runner")
    if (not config.model_config.enforce_eager
            and config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY):
        raise ValueError("Final-prefill active cache supports eager or FULL_DECODE_ONLY")
    if (parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1
            or parallel.pipeline_parallel_size != 1 or config.speculative_config is not None):
        raise ValueError("Colocated final-prefill cache requires PCP=DCP=PP=1 and no speculative decoding")
    hf = config.model_config.hf_text_config
    if (getattr(hf, "kv_lora_rank", None), getattr(hf, "qk_rope_head_dim", None)) != (LATENT_DIM, ROPE_DIM):
        raise ValueError("Final-prefill active cache requires GLM SFA latent512/RoPE64")


@dataclass(frozen=True)
class FinalPrefillPlan:
    has_prefill: bool = False
    block_ids: tuple[int, ...] = ()


def plan_final_prefill(requests, scheduled, group_id, block_size, pending_req_ids=None):
    """Use scheduler state, including prefix hits/resume, never query length."""
    has_prefill = False
    blocks = []
    for req_id, count in scheduled.items():
        request = requests[req_id]
        computed = request.num_computed_tokens
        # Recompute may include previously generated tokens after preemption.
        target = request.num_tokens
        pending = (req_id in pending_req_ids
                   if pending_req_ids is not None else computed < request.num_prompt_tokens or target - computed > 1)
        if computed < target and pending:
            has_prefill = True
            if computed + count >= target:
                pages = (target + block_size - 1) // block_size
                blocks.extend(request.block_ids[group_id][:pages])
    return FinalPrefillPlan(has_prefill, tuple(dict.fromkeys(blocks)))


def colocated_cache_views(raw, num_blocks, block_size):
    rows = num_blocks * block_size
    nope_end = rows * LATENT_DIM * 2
    rope_end = rows * BF16_SLOT_BYTES
    expected = rows * (BF16_SLOT_BYTES + TQ_SLOT_BYTES)
    if raw.numel() != expected:
        raise ValueError("Final-prefill cache allocation does not match its charged page size")
    shape = (num_blocks, block_size, 1)
    return (
        raw[:nope_end].view(torch.bfloat16).view(*shape, LATENT_DIM),
        raw[nope_end:rope_end].view(torch.bfloat16).view(*shape, ROPE_DIM),
        raw[rope_end:].view(torch.int8).view(*shape, TQ_SLOT_BYTES),
    )


def plan_incremental_slots(requests, scheduled, group_id, block_size, pending, initial):
    """Convert new rows only; on admission/resume also cover a BF16 prefix hit."""
    slots = []
    for req_id, count in scheduled.items():
        if req_id not in pending:
            continue
        request = requests[req_id]
        start = 0 if req_id in initial else request.num_computed_tokens
        end = min(request.num_tokens, request.num_computed_tokens + count)
        blocks = request.block_ids[group_id]
        for token in range(start, end):
            slots.append(blocks[token // block_size] * block_size + token % block_size)
    return tuple(dict.fromkeys(slots))


class PrefillQuantizationPipeline:
    """One bounded stream per runner; join before retiring each model step.

    Sources/destinations are persistent cache views. Step-local indices are
    retained until the join and recorded on the side stream. No cache ownership
    crosses a scheduler boundary, including abort and preemption boundaries.
    """

    def __init__(self, device):
        self.stream = torch.npu.Stream(device=device)
        self.pending = False

    def submit(self, state):
        if state.prefill_slots is None or state.prefill_slots.numel() == 0:
            return
        self.stream.wait_stream(torch.npu.current_stream())
        self.pending = True
        with torch.npu.stream(self.stream):
            state.prefill_slots.record_stream(self.stream)
            state.update_decode_tokens(state.prefill_slots)

    def join(self):
        if self.pending:
            torch.npu.current_stream().wait_stream(self.stream)
            self.pending = False


class ColocatedFinalPrefill:

    def __init__(self, quantizer):
        self.quantizer = quantizer
        self.plan = FinalPrefillPlan()
        self.cache = None
        self.pipeline = None
        self.prefill_slots = None

    def bind(self, cache):
        self.cache = cache

    def start_prefill(self):
        if self.plan.has_prefill and self.pipeline is not None:
            self.pipeline.submit(self)

    def update_decode_tokens(self, slots):
        """BF16 prolog already wrote these tokens. Invalid graph slots stay -1."""
        nope, rope, packed_cache = self.cache
        slots = slots.reshape(-1).long()
        for start in range(0, slots.numel(), FINAL_PREFILL_CHUNK_TOKENS):
            target = slots[start:start + FINAL_PREFILL_CHUNK_TOKENS]
            indices = target.clamp_min(0)
            self.quantizer.compress_kv_into(
                nope.view(-1, LATENT_DIM).index_select(0, indices),
                rope.view(-1, ROPE_DIM).index_select(0, indices),
                target,
                packed_cache.view(-1, TQ_SLOT_BYTES),
            )
