# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU fallbacks for MRv2 spec-decode helpers (310P has no Triton)."""

from __future__ import annotations

import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers


def expand_idx_mapping_cpu(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits_np: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = idx_mapping.device
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = torch.empty(total_num_logits, dtype=torch.int32, device=device)
    for req_idx in range(cu_num_logits_np.shape[0] - 1):
        start = int(cu_num_logits_np[req_idx])
        end = int(cu_num_logits_np[req_idx + 1])
        num_tokens = end - start
        if num_tokens <= 0:
            continue
        expanded_idx_mapping[start:end] = idx_mapping[req_idx]
        expanded_local_pos[start:end] = torch.arange(num_tokens, dtype=torch.int32, device=device)
    return expanded_idx_mapping, expanded_local_pos


def combine_sampled_and_draft_tokens_cpu(
    input_ids: torch.Tensor,
    idx_mapping_np: np.ndarray,
    last_sampled_tokens: torch.Tensor,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    prefill_len_np: np.ndarray,
    draft_tokens: torch.Tensor,
    cu_num_logits_np: np.ndarray,
    num_logits: int,
    num_new_sampled_tokens: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    del device
    assert num_new_sampled_tokens in (0, 1)
    num_reqs = idx_mapping_np.shape[0]
    logits_indices = torch.empty(num_logits, dtype=torch.int64, device=input_ids.device)
    input_ids_cpu = input_ids.detach().cpu()
    last_sampled_cpu = last_sampled_tokens.detach().cpu()
    draft_cpu = draft_tokens.detach().cpu()

    for batch_idx in range(num_reqs):
        req_state_idx = int(idx_mapping_np[batch_idx])
        cu_start = int(cu_num_logits_np[batch_idx])
        cu_end = int(cu_num_logits_np[batch_idx + 1])
        num_req_logits = cu_end - cu_start
        num_draft_tokens = num_req_logits - num_new_sampled_tokens

        query_end = int(query_start_loc_np[batch_idx + 1])
        logits_start = query_end - num_req_logits
        for offset in range(num_req_logits):
            logits_indices[cu_start + offset] = logits_start + offset

        seq_len = int(seq_lens_np[batch_idx])
        prefill_len = int(prefill_len_np[batch_idx])
        if seq_len <= prefill_len:
            continue

        first_logit_seq_pos = seq_len - num_req_logits
        if num_new_sampled_tokens > 0 and first_logit_seq_pos >= prefill_len:
            last_token_id = int(last_sampled_cpu[req_state_idx].item())
            input_ids_cpu[logits_start] = last_token_id

        if num_draft_tokens > 0:
            draft_row = draft_cpu[req_state_idx, :num_draft_tokens].tolist()
            draft_start = query_end - num_draft_tokens
            input_ids_cpu[draft_start:query_end] = torch.tensor(draft_row, dtype=input_ids_cpu.dtype)

    input_ids.copy_(input_ids_cpu.to(device=input_ids.device, non_blocking=True))
    return logits_indices


def get_num_sampled_and_rejected_cpu(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping_np: np.ndarray,
    prefill_len_np: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = idx_mapping_np.shape[0]
    num_rejected = torch.empty_like(num_sampled)
    cu_np = cu_num_logits.detach().cpu().numpy()
    seq_np = seq_lens.detach().cpu().numpy()
    sampled_np = num_sampled.detach().cpu().numpy().copy()

    for batch_idx in range(num_reqs):
        seq_len = int(seq_np[batch_idx])
        prefill_len_i = int(prefill_len_np[batch_idx])
        is_chunked_prefilling = seq_len < prefill_len_i
        if is_chunked_prefilling:
            sampled_np[batch_idx] = 0
            num_rejected[batch_idx] = 0
            continue
        num_logits = int(cu_np[batch_idx + 1] - cu_np[batch_idx])
        num_rejected[batch_idx] = num_logits - int(sampled_np[batch_idx])

    num_sampled_out = torch.from_numpy(sampled_np).to(device=num_sampled.device, dtype=num_sampled.dtype)
    return num_sampled_out, num_rejected.to(device=num_sampled.device)


def greedy_rejection_sample_cpu(
    target_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy (temperature=0) rejection sampling for MTP verify."""
    cu_np = cu_num_logits.detach().cpu().numpy()
    num_reqs = cu_np.shape[0] - 1
    max_tokens = num_speculative_steps + 1
    sampled = torch.full(
        (num_reqs, max_tokens),
        -1,
        dtype=torch.int32,
        device=target_logits.device,
    )
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=target_logits.device)
    draft_cpu = draft_sampled.detach().cpu()
    logits_cpu = target_logits.detach().cpu()

    for req_idx in range(num_reqs):
        start = int(cu_np[req_idx])
        end = int(cu_np[req_idx + 1])
        num_logits = end - start
        if num_logits <= 0:
            continue
        accepted = 0
        # draft_sampled = input_ids[logits_indices] = [last_sampled, draft_0, ...]
        # logits[i] predicts token i+1, which is draft_sampled[i+1] (upstream
        # rejection_sampler_utils loads draft_sampled_ptr + logit_idx + 1).
        for logit_idx in range(start, end):
            target_token = int(logits_cpu[logit_idx].argmax().item())
            is_bonus = logit_idx >= end - 1
            if accepted < num_speculative_steps and not is_bonus:
                draft_token = int(draft_cpu[logit_idx + 1].item())
                if draft_token == target_token:
                    sampled[req_idx, accepted] = target_token
                    accepted += 1
                    continue
            sampled[req_idx, accepted] = target_token
            accepted += 1
            break
        if accepted == 0:
            sampled[req_idx, 0] = int(logits_cpu[start].argmax().item())
            accepted = 1
        num_sampled[req_idx] = accepted

    return sampled, num_sampled


def prepare_prefill_inputs_cpu(
    last_token_indices: torch.Tensor,
    current_draft_step: torch.Tensor,
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    max_num_reqs: int,
) -> torch.Tensor:
    del max_num_reqs
    num_reqs = input_batch.num_reqs
    target_input_ids = input_batch.input_ids.detach().cpu()
    target_positions = input_batch.positions.detach().cpu()
    query_start_loc_np = input_batch.query_start_loc_np
    seq_lens_np = input_batch.seq_lens.detach().cpu().numpy()
    idx_mapping_np = input_batch.idx_mapping_np
    num_sampled_np = num_sampled.detach().cpu().numpy()
    num_rejected_np = num_rejected.detach().cpu().numpy()
    last_sampled_cpu = last_sampled.detach().cpu()
    next_prefill_cpu = next_prefill_tokens.detach().cpu()

    draft_input_ids = input_buffers.input_ids
    draft_positions = input_buffers.positions
    draft_input_ids_cpu = draft_input_ids.detach().cpu()
    draft_positions_cpu = draft_positions.detach().cpu()
    draft_query_start_loc_cpu = input_buffers.query_start_loc.detach().cpu()
    draft_seq_lens_cpu = input_buffers.seq_lens.detach().cpu()
    last_token_indices_cpu = last_token_indices.detach().cpu()

    for req_idx in range(num_reqs):
        req_state_idx = int(idx_mapping_np[req_idx])
        query_start = int(query_start_loc_np[req_idx])
        query_end = int(query_start_loc_np[req_idx + 1])
        query_len = query_end - query_start
        seq_len = int(seq_lens_np[req_idx])
        query_len -= int(num_rejected_np[req_idx])

        if int(num_sampled_np[req_idx]) > 0:
            next_token = int(last_sampled_cpu[req_state_idx].item())
        else:
            next_token = int(next_prefill_cpu[req_state_idx].item())

        if query_len > 1:
            draft_input_ids_cpu[query_start : query_end - 1] = target_input_ids[query_start + 1 : query_end]
        last_token_index = query_start + query_len - 1
        last_token_indices_cpu[req_idx] = last_token_index
        draft_input_ids_cpu[last_token_index] = next_token
        draft_positions_cpu[query_start:query_end] = target_positions[query_start:query_end]
        draft_query_start_loc_cpu[req_idx] = query_start
        draft_seq_lens_cpu[req_idx] = seq_len

    current_draft_step.fill_(0)
    if num_reqs > 0:
        query_end = int(query_start_loc_np[num_reqs])
        draft_query_start_loc_cpu[num_reqs:] = query_end
        draft_seq_lens_cpu[num_reqs:] = 0
        last_token_indices_cpu[num_reqs:] = 0

    draft_input_ids.copy_(draft_input_ids_cpu.to(device=draft_input_ids.device), non_blocking=True)
    draft_positions.copy_(draft_positions_cpu.to(device=draft_positions.device), non_blocking=True)
    input_buffers.query_start_loc.copy_(
        draft_query_start_loc_cpu.to(device=input_buffers.query_start_loc.device),
        non_blocking=True,
    )
    input_buffers.seq_lens.copy_(draft_seq_lens_cpu.to(device=input_buffers.seq_lens.device), non_blocking=True)
    last_token_indices.copy_(last_token_indices_cpu.to(device=last_token_indices.device), non_blocking=True)
    return last_token_indices


def prepare_decode_inputs_cpu(
    draft_tokens: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    max_model_len: int,
    max_num_reqs: int,
    advance_draft_positions: bool = True,
) -> None:
    del max_num_reqs
    num_reqs = draft_tokens.shape[0]
    draft_cpu = draft_tokens.detach().cpu()
    target_seq_cpu = target_seq_lens.detach().cpu()
    rejected_cpu = num_rejected.detach().cpu()
    input_ids_cpu = input_buffers.input_ids.detach().cpu()
    positions_cpu = input_buffers.positions.detach().cpu()
    query_start_loc_cpu = input_buffers.query_start_loc.detach().cpu()
    seq_lens_cpu = input_buffers.seq_lens.detach().cpu()

    for req_idx in range(num_reqs):
        input_ids_cpu[req_idx] = int(draft_cpu[req_idx].item())
        seq_len = int(target_seq_cpu[req_idx].item()) - int(rejected_cpu[req_idx].item())
        if advance_draft_positions:
            position = int(positions_cpu[req_idx].item())
            position = min(position + 1, max_model_len - 1)
            positions_cpu[req_idx] = position
            seq_len = min(seq_len + 1, max_model_len)
        seq_lens_cpu[req_idx] = seq_len

    for req_idx in range(num_reqs + 1):
        query_start_loc_cpu[req_idx] = min(req_idx, num_reqs)
    seq_lens_cpu[num_reqs:] = 0

    input_buffers.input_ids.copy_(input_ids_cpu.to(device=input_buffers.input_ids.device), non_blocking=True)
    input_buffers.positions.copy_(positions_cpu.to(device=input_buffers.positions.device), non_blocking=True)
    input_buffers.query_start_loc.copy_(
        query_start_loc_cpu.to(device=input_buffers.query_start_loc.device), non_blocking=True
    )
    input_buffers.seq_lens.copy_(seq_lens_cpu.to(device=input_buffers.seq_lens.device), non_blocking=True)


def update_draft_inputs_cpu(
    draft_tokens: torch.Tensor,
    current_draft_step: torch.Tensor,
    hidden_states: torch.Tensor,
    output_draft_tokens: torch.Tensor,
    next_input_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    num_reqs: int,
    max_model_len: int,
    num_speculative_steps: int,
    advance_draft_positions: bool = True,
) -> None:
    step = int(current_draft_step.item())
    draft_cpu = draft_tokens.detach().cpu()
    output_draft_cpu = output_draft_tokens.detach().cpu()
    input_ids_cpu = input_buffers.input_ids.detach().cpu()
    positions_cpu = input_buffers.positions.detach().cpu()
    seq_lens_cpu = input_buffers.seq_lens.detach().cpu()
    hidden_cpu = hidden_states.detach().cpu()
    next_hidden_cpu = next_input_hidden_states.detach().cpu()

    for req_idx in range(num_reqs):
        token = int(draft_cpu[req_idx].item())
        output_draft_cpu[req_idx, step] = token
        if step >= num_speculative_steps - 1:
            continue
        input_ids_cpu[req_idx] = token
        next_hidden_cpu[req_idx] = hidden_cpu[req_idx]
        if advance_draft_positions:
            position = min(int(positions_cpu[req_idx].item()) + 1, max_model_len - 1)
            positions_cpu[req_idx] = position
            seq_len = min(int(seq_lens_cpu[req_idx].item()) + 1, max_model_len)
            seq_lens_cpu[req_idx] = seq_len

    output_draft_tokens.copy_(output_draft_cpu.to(device=output_draft_tokens.device), non_blocking=True)
    if step < num_speculative_steps - 1:
        input_buffers.input_ids.copy_(input_ids_cpu.to(device=input_buffers.input_ids.device), non_blocking=True)
        next_input_hidden_states.copy_(next_hidden_cpu.to(device=next_input_hidden_states.device), non_blocking=True)
        if advance_draft_positions:
            input_buffers.positions.copy_(positions_cpu.to(device=input_buffers.positions.device), non_blocking=True)
            input_buffers.seq_lens.copy_(seq_lens_cpu.to(device=input_buffers.seq_lens.device), non_blocking=True)
