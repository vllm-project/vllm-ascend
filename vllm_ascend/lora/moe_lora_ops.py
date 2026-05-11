import torch

from vllm_ascend.lora.lora_ops import bgmv_expand, bgmv_shrink, sgmv_expand, sgmv_shrink


def _compute_moe_sgmv_metadata(lora_expert_indices: torch.Tensor):
    lora_indices_tensor, seq_len_tensor = torch.unique_consecutive(
        lora_expert_indices, return_counts=True
    )
    cum_result = torch.cumsum(seq_len_tensor, dim=0)
    b_seq_start_loc = torch.zeros_like(seq_len_tensor)
    b_seq_start_loc[1:].copy_(cum_result[:-1])

    batches = lora_indices_tensor.size(0)
    max_seq_length = seq_len_tensor.max().item()
    token_nums = seq_len_tensor.sum().item()

    return b_seq_start_loc, seq_len_tensor, lora_indices_tensor, batches, max_seq_length, token_nums


def _build_lora_expert_indices_allgather(
    lora_indices: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    top_k = topk_ids.shape[1]
    expanded_lora_indices = lora_indices.repeat_interleave(top_k)

    num_dispatched_tokens = expanded_row_idx.shape[0]
    perm = torch.abs(expanded_row_idx)[:num_dispatched_tokens]
    dispatched_lora_indices = expanded_lora_indices[perm]

    flat_topk_ids = topk_ids.reshape(-1)
    dispatched_expert_ids = flat_topk_ids[perm]

    return dispatched_lora_indices * num_experts + dispatched_expert_ids


def _build_lora_expert_indices_alltoall(
    lora_indices: torch.Tensor,
    reversed_global_input_permutation_mapping: torch.Tensor,
    topk_ids: torch.Tensor,
    group_list: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    top_k = topk_ids.shape[1]
    expanded_lora_indices = lora_indices.repeat_interleave(top_k)

    num_dispatched_tokens = reversed_global_input_permutation_mapping.shape[0]
    perm = torch.abs(reversed_global_input_permutation_mapping)[:num_dispatched_tokens]
    dispatched_lora_indices = expanded_lora_indices[perm]

    dispatched_expert_ids = torch.zeros(
        num_dispatched_tokens, dtype=torch.int32, device=lora_indices.device
    )
    start = 0
    for expert_id in range(min(group_list.shape[0], num_experts)):
        count = group_list[expert_id].item()
        end = start + count
        if count > 0 and start < num_dispatched_tokens:
            actual_end = min(end, num_dispatched_tokens)
            dispatched_expert_ids[start:actual_end] = expert_id
        start = end

    return dispatched_lora_indices * num_experts + dispatched_expert_ids


def apply_moe_lora_w13(
    gate_up_out: torch.Tensor,
    hidden_states: torch.Tensor,
    w13_lora_a_stacked: tuple[torch.Tensor, ...],
    w13_lora_b_stacked: tuple[torch.Tensor, ...],
    lora_expert_indices: torch.Tensor,
    scale: float,
    is_prefill: bool = False,
):
    r = w13_lora_b_stacked[0].shape[-1]
    num_dispatched_tokens = gate_up_out.shape[0]

    for slice_idx in range(len(w13_lora_a_stacked)):
        lora_a = w13_lora_a_stacked[slice_idx]
        lora_b = w13_lora_b_stacked[slice_idx]
        num_loras = lora_a.shape[0]
        num_experts = lora_a.shape[1]
        lora_a_merged = lora_a.reshape(num_loras * num_experts, r, -1)
        lora_b_merged = lora_b.reshape(num_loras * num_experts, -1, r)

        buffer = torch.zeros(
            (num_dispatched_tokens, r),
            dtype=torch.float32,
            device=gate_up_out.device,
        )

        if is_prefill:
            b_seq_start_loc, seq_len_tensor, sgmv_lora_indices, batches, max_seq_length, token_nums = \
                _compute_moe_sgmv_metadata(lora_expert_indices)
            sgmv_shrink(
                hidden_states, lora_a_merged, buffer,
                b_seq_start_loc, seq_len_tensor, sgmv_lora_indices,
                batches, max_seq_length, token_nums, scale,
            )
            sgmv_expand(
                buffer, lora_b_merged, gate_up_out,
                b_seq_start_loc, seq_len_tensor, sgmv_lora_indices,
                batches, max_seq_length, token_nums, add_inputs=True,
            )
        else:
            bgmv_shrink(hidden_states, lora_a_merged, buffer, lora_expert_indices, scale)
            bgmv_expand(buffer, lora_b_merged, gate_up_out, lora_expert_indices, add_inputs=True)


def apply_moe_lora_w2(
    activated_out: torch.Tensor,
    w2_output: torch.Tensor,
    w2_lora_a_stacked: tuple[torch.Tensor, ...],
    w2_lora_b_stacked: tuple[torch.Tensor, ...],
    lora_expert_indices: torch.Tensor,
    scale: float,
    is_prefill: bool = False,
):
    r = w2_lora_b_stacked[0].shape[-1]
    num_dispatched_tokens = activated_out.shape[0]

    lora_a = w2_lora_a_stacked[0]
    lora_b = w2_lora_b_stacked[0]
    num_loras = lora_a.shape[0]
    num_experts = lora_a.shape[1]
    lora_a_merged = lora_a.reshape(num_loras * num_experts, r, -1)
    lora_b_merged = lora_b.reshape(num_loras * num_experts, -1, r)

    buffer = torch.zeros(
        (num_dispatched_tokens, r),
        dtype=torch.float32,
        device=activated_out.device,
    )

    if is_prefill:
        b_seq_start_loc, seq_len_tensor, sgmv_lora_indices, batches, max_seq_length, token_nums = \
            _compute_moe_sgmv_metadata(lora_expert_indices)
        sgmv_shrink(
            activated_out, lora_a_merged, buffer,
            b_seq_start_loc, seq_len_tensor, sgmv_lora_indices,
            batches, max_seq_length, token_nums, scale,
        )
        sgmv_expand(
            buffer, lora_b_merged, w2_output,
            b_seq_start_loc, seq_len_tensor, sgmv_lora_indices,
            batches, max_seq_length, token_nums, add_inputs=True,
        )
    else:
        bgmv_shrink(activated_out, lora_a_merged, buffer, lora_expert_indices, scale)
        bgmv_expand(buffer, lora_b_merged, w2_output, lora_expert_indices, add_inputs=True)
