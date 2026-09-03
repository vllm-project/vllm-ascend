# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Speculative decoding support for Model Runner V2 PP."""

import numpy as np
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

_BROADCAST_PATCHED = "_vllm_ascend_spec_pp_broadcast_patched"


def sync_spec_pp_num_computed_tokens_cpu(runner, scheduler_output) -> None:
    """Refresh ``num_computed_tokens_cpu`` for MTP with PP.

    MTP reverts ``num_computed_tokens`` on rejection, and only the device value
    (and therefore the D2H copy issued by ``postprocess_sampled``) sees that
    revert, so requests that were sampled must read the copy. Requests inside a
    non-final prefill chunk were not sampled this step: their scheduler-side
    value is the authoritative one, and a fully prefilling batch does not have
    to wait on the copy event at all.
    """
    req_ids = scheduler_output.scheduled_cached_reqs.req_ids
    if not req_ids:
        return

    req_indices = [runner.req_states.req_id_to_index[req_id] for req_id in req_ids]
    is_prefilling = (
        runner.req_states.num_computed_tokens_np[req_indices] < runner.req_states.prefill_len.np[req_indices]
    )
    if not is_prefilling.all():
        runner.num_computed_tokens_event.synchronize()

    for req_index, req_is_prefilling in zip(req_indices, is_prefilling, strict=True):
        source = runner.req_states.num_computed_tokens_np if req_is_prefilling else runner.num_computed_tokens_cpu
        runner.req_states.num_computed_tokens_cpu[req_index] = source[req_index]


def install_spec_pp_token_broadcast(pp_handler, req_states) -> None:
    """Send accepted and next-draft tokens through the same V2 PP slot."""
    if getattr(pp_handler, _BROADCAST_PATCHED, False):
        return

    max_sample_len = pp_handler.max_sample_len
    draft_width = max_sample_len - 1
    token_payload_width = max_sample_len + draft_width
    original_get_prev_sampled_outputs = pp_handler.get_prev_sampled_outputs
    original_broadcast = pp_handler.broadcast
    pending_send = None

    def get_prev_sampled_outputs():
        slot = pp_handler.queue[0] if pp_handler.queue else None
        outputs = original_get_prev_sampled_outputs()
        if outputs is None:
            return None
        assert slot is not None
        token_payload = outputs["sampled_tokens"]
        outputs["sampled_tokens"] = token_payload[:, :max_sample_len]
        draft_tokens = token_payload[:, max_sample_len:]

        # Preserve valid rows on CPU; NPU bool indexing lowers to NonzeroV2.
        freed = pp_handler.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        if exclude_mask.any():
            valid_rows = np.flatnonzero(~exclude_mask)
            update_indices = np.stack(
                (valid_rows, slot.idx_mapping_np[valid_rows]),
            )
            draft_rows, draft_req_indices = async_copy_to_gpu(
                update_indices,
                device=pp_handler.device,
            ).unbind(dim=0)
            req_states.draft_tokens.index_copy_(
                0,
                draft_req_indices,
                draft_tokens.index_select(0, draft_rows),
            )
        else:
            req_states.draft_tokens.index_copy_(
                0,
                outputs["idx_mapping"],
                draft_tokens,
            )
        return outputs

    def broadcast(sampled_token_ids, num_sampled, num_rejected, input_batch):
        nonlocal pending_send
        assert pp_handler.is_last_rank
        if pending_send is not None:
            raise RuntimeError("Speculative PP already has a pending sampled-token broadcast.")
        pending_send = (
            sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch,
        )

    def broadcast_draft_tokens():
        nonlocal pending_send
        if pending_send is None:
            return

        sampled_token_ids, num_sampled, num_rejected, input_batch = pending_send
        pending_send = None
        num_reqs = input_batch.num_reqs
        draft_tokens = req_states.draft_tokens[input_batch.idx_mapping]
        token_payload = sampled_token_ids.new_zeros((num_reqs, token_payload_width))
        token_payload[:, : sampled_token_ids.shape[1]].copy_(sampled_token_ids)
        token_payload[:, max_sample_len:].copy_(draft_tokens)
        original_broadcast(
            token_payload,
            num_sampled,
            num_rejected,
            input_batch,
        )

    pp_handler.max_sample_len = token_payload_width
    pp_handler.get_prev_sampled_outputs = get_prev_sampled_outputs
    pp_handler.broadcast = broadcast
    pp_handler.broadcast_draft_tokens = broadcast_draft_tokens
    setattr(pp_handler, _BROADCAST_PATCHED, True)
