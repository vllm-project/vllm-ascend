# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

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
    # One small D2H for sampled/draft token tables (not the full input_ids buffer).
    last_sampled_cpu = last_sampled_tokens.detach().cpu()
    draft_cpu = draft_tokens.detach().cpu()
    # Host staging only for rows we write; copy back via indexed slices.
    writes: list[tuple[int, int]] = []
    host_vals: list[int] = []

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
            writes.append((logits_start, logits_start + 1))
            host_vals.append(last_token_id)

        if num_draft_tokens > 0:
            draft_row = draft_cpu[req_state_idx, :num_draft_tokens].tolist()
            draft_start = query_end - num_draft_tokens
            for i, tok in enumerate(draft_row):
                writes.append((draft_start + i, draft_start + i + 1))
                host_vals.append(int(tok))

    if host_vals:
        # Avoid NPU index_copy_/IndexPutV2 (unreliable on some 310P layouts).
        # Direct indexing matches gdn_310._merge_spec_and_non_spec_outputs_310.
        idx = torch.tensor([s for s, _ in writes], dtype=torch.long, device=input_ids.device)
        vals = torch.tensor(host_vals, dtype=input_ids.dtype, device=input_ids.device)
        input_ids[idx] = vals
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
    """Greedy (temperature=0) rejection sampling for MTP verify.

    Argmax runs on-device so we only D2H token ids (not the full vocab logits).
    Acceptance bookkeeping stays on CPU to avoid per-element NPU syncs.
    """
    cu_np = cu_num_logits.detach().cpu().numpy()
    num_reqs = cu_np.shape[0] - 1
    max_tokens = num_speculative_steps + 1
    sampled_cpu = torch.full((num_reqs, max_tokens), -1, dtype=torch.int32)
    num_sampled_cpu = torch.zeros(num_reqs, dtype=torch.int32)
    # Avoid full-vocab logits D2H (dominant cost on 310P MTP verify).
    target_argmax_cpu = target_logits.argmax(dim=-1).to(dtype=torch.int32).detach().cpu().numpy()
    draft_cpu = draft_sampled.detach().cpu().to(dtype=torch.int32).numpy()

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
            target_token = int(target_argmax_cpu[logit_idx])
            is_bonus = logit_idx >= end - 1
            if accepted < num_speculative_steps and not is_bonus:
                draft_token = int(draft_cpu[logit_idx + 1])
                if draft_token == target_token:
                    sampled_cpu[req_idx, accepted] = target_token
                    accepted += 1
                    continue
            sampled_cpu[req_idx, accepted] = target_token
            accepted += 1
            break
        if accepted == 0:
            sampled_cpu[req_idx, 0] = int(target_argmax_cpu[start])
            accepted = 1
        num_sampled_cpu[req_idx] = accepted

    return (
        sampled_cpu.to(device=target_logits.device, non_blocking=True),
        num_sampled_cpu.to(device=target_logits.device, non_blocking=True),
    )


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
    """Build draft-prefill inputs with minimal host/device sync.

    Prefer on-device token/position copies and host np mirrors for metadata.
    Full-buffer ``.cpu()`` of target/draft tensors was a major 310P MTP cost.
    """
    del max_num_reqs
    num_reqs = input_batch.num_reqs
    query_start_loc_np = input_batch.query_start_loc_np
    idx_mapping_np = input_batch.idx_mapping_np
    seq_lens_np = getattr(input_batch, "seq_lens_np", None)
    if seq_lens_np is None:
        seq_lens_np = input_batch.seq_lens.detach().cpu().numpy()

    # Small metadata only (one sync each); avoid D2H of full token buffers.
    num_sampled_np = num_sampled[:num_reqs].detach().cpu().numpy()
    num_rejected_np = num_rejected[:num_reqs].detach().cpu().numpy()
    last_sampled_np = last_sampled.detach().cpu().numpy()
    next_prefill_np = next_prefill_tokens.detach().cpu().numpy()

    target_input_ids = input_batch.input_ids
    target_positions = input_batch.positions
    draft_input_ids = input_buffers.input_ids
    draft_positions = input_buffers.positions
    device = draft_input_ids.device

    last_token_indices_host = np.zeros(last_token_indices.shape[0], dtype=np.int64)
    draft_qsl_host = np.zeros(input_buffers.query_start_loc.shape[0], dtype=np.int32)
    draft_seq_host = np.zeros(input_buffers.seq_lens.shape[0], dtype=np.int32)
    next_tokens_host = np.empty(num_reqs, dtype=np.int32)

    for req_idx in range(num_reqs):
        req_state_idx = int(idx_mapping_np[req_idx])
        query_start = int(query_start_loc_np[req_idx])
        query_end = int(query_start_loc_np[req_idx + 1])
        query_len = query_end - query_start
        seq_len = int(seq_lens_np[req_idx])
        query_len -= int(num_rejected_np[req_idx])

        if int(num_sampled_np[req_idx]) > 0:
            next_tokens_host[req_idx] = int(np.asarray(last_sampled_np[req_state_idx]).reshape(-1)[0])
        else:
            next_tokens_host[req_idx] = int(np.asarray(next_prefill_np[req_state_idx]).reshape(-1)[0])

        # After subtracting rejected tokens, only the kept prefix is valid
        # (match upstream Triton prepare_prefill_inputs).
        kept_end = query_start + query_len
        if query_len > 1:
            # 310P: NPU slice-assign can corrupt the destination tail element
            # (see llm_base_proposer_310.set_inputs_first_pass). Save/restore
            # draft_input_ids[kept_end-1] around the shift copy.
            tail_idx = kept_end - 1
            tail_save = draft_input_ids[tail_idx].clone()
            draft_input_ids[query_start : kept_end - 1].copy_(
                target_input_ids[query_start + 1 : kept_end],
                non_blocking=True,
            )
            draft_input_ids[tail_idx] = tail_save
        last_token_index = kept_end - 1
        last_token_indices_host[req_idx] = last_token_index
        draft_positions[query_start:kept_end].copy_(
            target_positions[query_start:kept_end],
            non_blocking=True,
        )
        draft_qsl_host[req_idx] = query_start
        draft_seq_host[req_idx] = seq_len

    current_draft_step.fill_(0)
    if num_reqs > 0:
        query_end = int(query_start_loc_np[num_reqs])
        draft_qsl_host[num_reqs:] = query_end
        draft_seq_host[num_reqs:] = 0
        last_token_indices_host[num_reqs:] = 0
        # Scatter next tokens without index_copy_ (310P IndexPutV2 issues).
        next_tokens = torch.from_numpy(next_tokens_host).to(device=device, non_blocking=True)
        last_idx = torch.from_numpy(last_token_indices_host[:num_reqs]).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        draft_input_ids[last_idx] = next_tokens.to(dtype=draft_input_ids.dtype)

    input_buffers.query_start_loc.copy_(
        torch.from_numpy(draft_qsl_host).to(device=device, non_blocking=True),
        non_blocking=True,
    )
    input_buffers.seq_lens.copy_(
        torch.from_numpy(draft_seq_host).to(device=device, dtype=input_buffers.seq_lens.dtype, non_blocking=True),
        non_blocking=True,
    )
    last_token_indices.copy_(
        torch.from_numpy(last_token_indices_host).to(device=device, dtype=last_token_indices.dtype, non_blocking=True),
        non_blocking=True,
    )
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
    """Prepare draft decode inputs with small host syncs (K>1 path)."""
    del max_num_reqs
    num_reqs = draft_tokens.shape[0]
    device = input_buffers.input_ids.device
    draft_np = draft_tokens[:num_reqs].detach().cpu().numpy()
    target_seq_np = target_seq_lens[:num_reqs].detach().cpu().numpy()
    rejected_np = num_rejected[:num_reqs].detach().cpu().numpy()

    input_ids_host = draft_np.astype(np.int64, copy=False)
    seq_host = np.zeros(input_buffers.seq_lens.shape[0], dtype=np.int64)
    qsl_host = np.arange(num_reqs + 1, dtype=np.int64)
    if qsl_host.shape[0] < input_buffers.query_start_loc.shape[0]:
        qsl_full = np.zeros(input_buffers.query_start_loc.shape[0], dtype=np.int64)
        qsl_full[: qsl_host.shape[0]] = qsl_host
        qsl_full[qsl_host.shape[0] :] = num_reqs
        qsl_host = qsl_full

    for req_idx in range(num_reqs):
        seq_len = int(target_seq_np[req_idx]) - int(rejected_np[req_idx])
        if advance_draft_positions:
            seq_len = min(seq_len + 1, max_model_len)
        seq_host[req_idx] = seq_len

    input_buffers.input_ids[:num_reqs].copy_(
        torch.from_numpy(input_ids_host).to(device=device, dtype=input_buffers.input_ids.dtype, non_blocking=True),
        non_blocking=True,
    )
    if advance_draft_positions:
        # positions += 1 on-device for the active rows (avoid full-buffer D2H).
        pos = input_buffers.positions[:num_reqs]
        pos.add_(1)
        pos.clamp_max_(max_model_len - 1)
    input_buffers.query_start_loc.copy_(
        torch.from_numpy(qsl_host).to(device=device, dtype=input_buffers.query_start_loc.dtype, non_blocking=True),
        non_blocking=True,
    )
    input_buffers.seq_lens.copy_(
        torch.from_numpy(seq_host).to(device=device, dtype=input_buffers.seq_lens.dtype, non_blocking=True),
        non_blocking=True,
    )


# Host mirror for draft step. ``current_draft_step.item()`` is a sync D2H and
# is illegal under NPU GLOBAL ACLGraph capture; callers set this before fill_.
_DRAFT_STEP_HOST: int = 0


def set_draft_step_host(step: int) -> None:
    global _DRAFT_STEP_HOST
    _DRAFT_STEP_HOST = int(step)


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
    """Update draft buffers for the next step using on-device ops where possible."""
    # ``.item()`` is a sync D2H and is illegal under NPU GLOBAL ACLGraph capture.
    if torch.npu.is_current_stream_capturing():
        step = _DRAFT_STEP_HOST
    else:
        step = int(current_draft_step.item())
        set_draft_step_host(step)
    tokens = draft_tokens[:num_reqs]
    output_draft_tokens[:num_reqs, step].copy_(tokens)
    if step >= num_speculative_steps - 1:
        return
    input_buffers.input_ids[:num_reqs].copy_(tokens)
    next_input_hidden_states[:num_reqs].copy_(hidden_states[:num_reqs])
    if advance_draft_positions:
        pos = input_buffers.positions[:num_reqs]
        pos.add_(1)
        pos.clamp_max_(max_model_len - 1)
        seq = input_buffers.seq_lens[:num_reqs]
        seq.add_(1)
        seq.clamp_max_(max_model_len)
