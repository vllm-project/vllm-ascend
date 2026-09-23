# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""0.30 PP sampled-token protocol for release trains (vLLM 0.28/0.29).

Replaces the deferred-broadcast transport that deadlocked under KV
saturation with async EPLB.  Delete once the paired vLLM version ships
the protocol natively.
"""

from dataclasses import dataclass

import numpy as np
import torch

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.29.0"):
    from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
else:
    # vLLM main (#56888) replaced buffer_utils.async_copy_to_gpu with
    # torch_utils.async_tensor_h2d (gaining out=/device=None support).
    from vllm.utils.torch_utils import async_tensor_h2d as async_copy_to_gpu

_INSTALLED = "_vllm_ascend_upstream_spec_pp_installed"


def compute_need_sampled_mask(input_batch):
    """Participation gate: pure function of the shared batch (no
    rank-local max_seq_len bound).  Finished requests are filtered on
    the receive side via generation counters."""
    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


@dataclass
class SpecPPPendingRecv:
    """Per-step slot: release fields plus the received draft rows."""

    event: torch.cuda.Event
    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    need_sampled_mask: np.ndarray  # [num_reqs]
    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]


def install_upstream_spec_pp_protocol(pp_handler, req_states, num_speculative_steps) -> None:
    """Bind the 0.30 broadcast/receive/consume methods onto the release
    PPHandler, adapted to fetch draft rows from ``req_states``."""
    if getattr(pp_handler, _INSTALLED, False):
        return

    device = pp_handler.device
    captured_batch = [None]

    def receive(input_batch):
        assert not pp_handler.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
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
        pp_handler.queue[-1] = SpecPPPendingRecv(
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
        if compute_need_sampled_mask(input_batch) is None:
            captured_batch[0] = None
            return
        captured_batch[0] = input_batch

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

    def broadcast_drafts(draft_tokens=None, input_batch=None):
        assert pp_handler.is_last_rank
        if input_batch is None:
            input_batch = captured_batch[0]
        if input_batch is None or compute_need_sampled_mask(input_batch) is None:
            return
        if draft_tokens is None:
            draft_tokens = req_states.draft_tokens[input_batch.idx_mapping]
        with torch.cuda.stream(pp_handler.broadcast_stream):
            pp_handler.broadcast_stream.wait_stream(pp_handler.main_stream)
            send = draft_tokens.contiguous()
            input_batch.idx_mapping.record_stream(pp_handler.broadcast_stream)
            torch.distributed.broadcast(send, src=pp_handler.last_rank, group=pp_handler.broadcast_group)

    def get_prev_sampled_outputs(draft_tokens_to_update=None):
        if not pp_handler.queue:
            return None
        slot = pp_handler.queue.popleft()
        pp_handler.queue.append(None)
        if slot is None:
            return None

        freed = pp_handler.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                return None
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=device)

        pp_handler.main_stream.wait_event(slot.event)
        if draft_tokens_to_update is None:
            draft_tokens_to_update = req_states.draft_tokens
        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            draft_tokens = slot.draft_tokens
            draft_idx_mapping = slot.idx_mapping
            if exclude_mask.any():
                keep = ~exclude_mask
                keep_t = torch.as_tensor(keep, device=device)
                draft_tokens = draft_tokens[keep_t]
                draft_idx_mapping = async_copy_to_gpu(slot.idx_mapping_np[keep], device=device)
            draft_tokens_to_update[draft_idx_mapping] = draft_tokens

        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    pp_handler.receive = receive
    pp_handler.broadcast = broadcast
    pp_handler.broadcast_drafts = broadcast_drafts
    pp_handler.broadcast_draft_tokens = broadcast_drafts
    pp_handler.get_prev_sampled_outputs = get_prev_sampled_outputs
    setattr(pp_handler, _INSTALLED, True)
