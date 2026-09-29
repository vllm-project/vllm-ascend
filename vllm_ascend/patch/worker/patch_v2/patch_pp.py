# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""V2 PP token transport with no-consumer broadcast skipping.

Reuses the 0.30 protocol for both speculative and non-speculative PP.
Remove when upstream skips broadcasts for requests with no subsequent step.
"""

import torch
from vllm.v1.worker.gpu.pp_utils import PendingRecv

_INSTALLED = "_vllm_ascend_pp_token_transport_installed"


def compute_need_sampled_mask(input_batch, req_states=None):
    """Use shared batch/request limits, never rank-local progress bounds."""
    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    if not produces_sample.any():
        return None
    if req_states is not None:
        # Final prefill produces one token. Skip only if every sampling row
        # reaches its request limit; a continuing decode keeps the whole batch.
        finished_with_sample = input_batch.is_prefilling_np & (
            prefill_len + 1 >= req_states.max_seq_len[input_batch.idx_mapping_np]
        )
        if finished_with_sample[produces_sample].all():
            return None
    return produces_sample


def install_pp_token_transport(pp_handler, req_states) -> None:
    """Override PP send/receive while retaining upstream deferred writeback."""
    if getattr(pp_handler, _INSTALLED, False):
        return

    device = pp_handler.device
    num_speculative_steps = pp_handler.num_speculative_steps

    def receive(input_batch):
        assert not pp_handler.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch, req_states)
        if need_sampled_mask is None:
            return False

        gen_at_receive_np = pp_handler.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(pp_handler.broadcast_stream):
            pp_handler.broadcast_stream.wait_stream(pp_handler.main_stream)
            sampled_tokens = torch.empty(num_reqs, pp_handler.max_sample_len, dtype=torch.int64, device=device)
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=device)
            torch.distributed.broadcast(sampled_tokens, src=pp_handler.last_rank, group=pp_handler.broadcast_group)
            torch.distributed.broadcast(combined, src=pp_handler.last_rank, group=pp_handler.broadcast_group)
            draft_tokens = None
            if num_speculative_steps > 0:
                draft_tokens = torch.empty(num_reqs, num_speculative_steps, dtype=torch.int64, device=device)
                torch.distributed.broadcast(draft_tokens, src=pp_handler.last_rank, group=pp_handler.broadcast_group)
            event = pp_handler.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            sampled_tokens.record_stream(pp_handler.main_stream)
            combined.record_stream(pp_handler.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(pp_handler.main_stream)
        pp_handler.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())

    def broadcast(sampled_token_ids, num_sampled, num_rejected, input_batch):
        assert pp_handler.is_last_rank
        if compute_need_sampled_mask(input_batch, req_states) is None:
            return

        assert sampled_token_ids.dtype == torch.int64
        with torch.cuda.stream(pp_handler.broadcast_stream):
            pp_handler.broadcast_stream.wait_stream(pp_handler.main_stream)
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids,
                (0, pp_handler.max_sample_len - sampled_token_ids.shape[-1]),
            )
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=pp_handler.last_rank,
                group=pp_handler.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(combined, src=pp_handler.last_rank, group=pp_handler.broadcast_group)
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(pp_handler.broadcast_stream)

    def broadcast_drafts(draft_tokens, input_batch):
        assert pp_handler.is_last_rank
        if compute_need_sampled_mask(input_batch, req_states) is None:
            return
        with torch.cuda.stream(pp_handler.broadcast_stream):
            pp_handler.broadcast_stream.wait_stream(pp_handler.main_stream)
            # Native callers pass the full request-slot buffer, not batch rows.
            send = draft_tokens[input_batch.idx_mapping].contiguous()
            input_batch.idx_mapping.record_stream(pp_handler.broadcast_stream)
            torch.distributed.broadcast(send, src=pp_handler.last_rank, group=pp_handler.broadcast_group)

    pp_handler.receive = receive
    pp_handler.broadcast = broadcast
    pp_handler.broadcast_drafts = broadcast_drafts
    setattr(pp_handler, _INSTALLED, True)
