import torch

from vllm_ascend.lora.lora_ops import bgmv_expand, bgmv_expand_slice, bgmv_shrink

def _do_bgmv_shrink(hidden_states, lora_a_merged, buffer, lora_expert_indices, scale):
    bgmv_shrink(hidden_states, lora_a_merged, buffer, lora_expert_indices, scale)


def _do_bgmv_expand(buffer, lora_b_merged, output, lora_expert_indices,
                    slice_offset=0, slice_size=None, add_inputs=True):
    if slice_size is None:
        slice_size = output.shape[1]
    if slice_offset == 0 and slice_size == output.shape[1]:
        bgmv_expand(buffer, lora_b_merged, output, lora_expert_indices, add_inputs)
    else:
        bgmv_expand_slice(buffer, lora_b_merged, output, lora_expert_indices,
                            slice_offset, slice_size, add_inputs)

def _build_lora_expert_indices(
    lora_indices: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """为每个 dispatched token 构造 LoRA 权重矩阵中的行索引。

    LoRA 权重存储为 lora_a_merged = lora_a.reshape(num_loras * num_experts, r, -1)，
    其中行索引布局为:
      rows 0..(E-1)        → lora_0 × expert_0..E-1
      rows E..(2E-1)       → lora_1 × expert_0..E-1
      ...
      rows (L-1)*E..(L*E-1)→ lora_L-1 × expert_0..E-1

    本函数对每个 dispatched token j 计算:
      lora_expert_indices[j] = dispatched_lora_id * num_experts + dispatched_expert_id

    Args:
        lora_indices:       [num_tokens]         每个原始 token 的 LoRA 适配器 ID
                           (0..max_loras-1 或 -1 表示无 LoRA)
        expanded_row_idx:   [num_dispatched_tokens] dispatch 排序后的展平来源索引，
                           编码为 token_idx * top_k + k，其中 k 是 topk slot
                           (负值表示 padding token)
        topk_ids:           [num_tokens, top_k]   每个 token 选中的专家 ID
                           (经过 log2phy 映射后的物理 ID)
        num_experts:        int                   本地专家数 E

    Returns:
        lora_expert_indices: [num_dispatched_tokens] 每个 dispatched token 在
                             lora_a_merged 中的行索引 (-1 表示该 token 无 LoRA)

    Example (num_tokens=3, top_k=2, num_experts=64):
        lora_indices     = [0,  -1,   2]  # token_0→lora_0, token_1→无, token_2→lora_2
        topk_ids         = [[3, 7],        # token_0→expert_3, expert_7
                            [1, 5],        # token_1→expert_1, expert_5
                            [3, 0]]        # token_2→expert_3, expert_0

        expanded_lora_indices = [0, 0, -1, -1, 2, 2]  # 展平: flat_i = token_i*2 + slot
        flat_topk_ids         = [3, 7,  1,  5, 3, 0]  # 展平 expert IDs

        expanded_row_idx = [5, 2, 0, 4, 3, 1]  # dispatch 排序后:
            # 位置0→flat_5(expert_0), 位置1→flat_2(expert_1),
            # 位置2→flat_0(expert_3), 位置3→flat_4(expert_3),
            # 位置4→flat_3(expert_5), 位置5→flat_1(expert_7)

        perm                    = [5, 2, 0, 4, 3, 1]     # [num_dispatched_tokens]
        dispatched_lora_indices = [2, -1, 0, 2, -1, 0]   # expanded_lora_indices[perm]
        dispatched_expert_ids   = [0,  1, 3, 3,  5, 7]   # flat_topk_ids[perm]

        lora_expert_indices:
            = [2*64+0,  mask, 0*64+3, 2*64+3,  mask, 0*64+7]
            = [128,     -1,   3,      131,     -1,   7     ]
        # 128 → lora_2 × expert_0 的行
        #   3 → lora_0 × expert_3 的行
        # 131 → lora_2 × expert_3 的行
        #   7 → lora_0 × expert_7 的行
    """
    # top_k: int, 每个 token 选择的专家数
    top_k = topk_ids.shape[1]

    # [num_tokens] → [num_tokens * top_k], 将 lora_indices 按 top_k 复制展开
    expanded_lora_indices = lora_indices.repeat_interleave(top_k)

    # num_dispatched_tokens: int, dispatch 后实际被处理的 (token, expert) 对数量
    num_dispatched_tokens = expanded_row_idx.shape[0]

    # [num_dispatched_tokens], 取绝对值得到展平来源索引
    # (负值用于标记 padding token，abs 后变为合法索引)
    perm = torch.abs(expanded_row_idx)[:num_dispatched_tokens]

    # [num_dispatched_tokens], 用排序后的 perm 索引展平的 lora_indices
    dispatched_lora_indices = expanded_lora_indices[perm]

    # [num_tokens * top_k] → 展平所有 token 的 expert ID
    flat_topk_ids = topk_ids.reshape(-1)

    # [num_dispatched_tokens], 用同一个 perm 取出对应的 expert ID
    dispatched_expert_ids = flat_topk_ids[perm]

    # [num_dispatched_tokens], 计算最终索引: lora_id * E + expert_id
    lora_expert_indices = dispatched_lora_indices * num_experts + dispatched_expert_ids.to(lora_indices.dtype)

    # 将无 LoRA 的 token 标记为 -1，后续 bgmv kernel 会跳过这些行
    lora_expert_indices[dispatched_lora_indices < 0] = -1

    return lora_expert_indices


def apply_moe_lora_w13(
    gate_up_out: torch.Tensor,
    hidden_states: torch.Tensor,
    w13_lora_a_stacked: tuple[torch.Tensor, ...],
    w13_lora_b_stacked: tuple[torch.Tensor, ...],
    lora_expert_indices: torch.Tensor,
    scale: float,
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

        _do_bgmv_shrink(hidden_states, lora_a_merged, buffer, lora_expert_indices, scale)

        _do_bgmv_expand(buffer, lora_b_merged, gate_up_out, lora_expert_indices,
                        slice_offset=slice_idx * lora_b_merged.shape[1],
                        slice_size=lora_b_merged.shape[1],
                        add_inputs=True)


def apply_moe_lora_w2(
    activated_out: torch.Tensor,
    w2_output: torch.Tensor,
    w2_lora_a_stacked: tuple[torch.Tensor, ...],
    w2_lora_b_stacked: tuple[torch.Tensor, ...],
    lora_expert_indices: torch.Tensor,
    scale: float,
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

    _do_bgmv_shrink(activated_out, lora_a_merged, buffer, lora_expert_indices, scale)
    _do_bgmv_expand(buffer, lora_b_merged, w2_output, lora_expert_indices, add_inputs=True)
