# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


def _get_hccl_comm_name(group: dist.ProcessGroup, rank_id: int) -> str:
    backend = group._get_backend(torch.device("npu"))
    get_comm_name = backend.get_hccl_comm_name
    try:
        group_name = get_comm_name(rank_id, init_comm=False)
    except TypeError:
        group_name = get_comm_name(rank_id)
    if not group_name:
        raise RuntimeError(
            "Failed to get a non-empty HCCL comm name for mega_moe EP group. "
            "The group must be initialized before calling get_symm_buffer_for_mega_moe."
        )
    return group_name


def npu_get_mega_moe_ccl_buffer_size(
    ep_world_size: int,
    moe_expert_num: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    dispatch_quant_mode: int = 0,
    dispatch_quant_out_dtype: int = 28,
    combine_quant_mode: int = 0,
    comm_alg: str = "",
) -> int:
    """Calculate the required CCL buffer size in MB for the mega_moe operator.

    This is a pure-Python formula matching the ops-transformer implementation.
    The returned value is in MB (aligned to even MB).

    Args:
        ep_world_size: Expert parallelism world size [2, 768].
        moe_expert_num: Total number of MoE experts [1, 1024].
        num_max_tokens_per_rank: Max tokens per rank [1, 512].
        num_topk: Top-K routing value [1, 16].
        hidden: Hidden dimension [1024, 8192].
        dispatch_quant_mode: Dispatch quantization mode (4=MXP mode).
        dispatch_quant_out_dtype: Output dtype after dispatch quantization.
        combine_quant_mode: Combine quantization mode (reserved, must be 0).
        comm_alg: Communication algorithm (reserved, must be '').

    Returns:
        CCL buffer size in MB (aligned to even MB).
    """
    def inline_align(val: int, align: int) -> int:
        return (val + align - 1) // align * align

    torch._check(
        (ep_world_size >= 2) and (ep_world_size <= 1024),
        lambda: f"ep_world_size only support in [2, 1024], but got {ep_world_size=}.",
    )
    torch._check(
        (hidden >= 1024) and (hidden <= 8192),
        lambda: f"hidden only support in [1024, 8192], but got {hidden=}.",
    )
    total_ub_size = 256 * 1024
    tokens_per_rank_limit = (8 * (total_ub_size - 48 * 1024)) // (num_topk * (64 + ep_world_size))
    torch._check(
        (num_max_tokens_per_rank >= 1) and (num_max_tokens_per_rank <= tokens_per_rank_limit),
        lambda: (f"num_max_tokens_per_rank only support in [1, {tokens_per_rank_limit=}], "
                 f"tokens_per_rank_limit = (8 * (total_ub_size - 48 * 1024)) / (num_topk * (64 + ep_world_size)), "
                 f"but got {num_max_tokens_per_rank=}."),
    )
    torch._check(
        (moe_expert_num >= 1) and (moe_expert_num <= 2048),
        lambda: f"moe_expert_num only support in [1, 2048], but got {moe_expert_num=}.",
    )
    torch._check(
        (num_topk >= 1) and (num_topk <= 16),
        lambda: f"num_topk only support in [1, 16], but got {num_topk=}.",
    )

    local_moe_expert_num = moe_expert_num // ep_world_size
    align_32 = 32
    align_256 = 256
    align_512 = 512
    y_out_dtype_size = 2
    mb_conversion = 1024 * 1024

    # Soft-sync across all cards: 60 MB
    peermem_data_offset = 60 * 1024

    # mask_recv_size
    compare_count = inline_align(num_max_tokens_per_rank * num_topk * 4, align_256) // 4
    mask_align_size = inline_align(compare_count // 8, align_32)
    mask_slot_size = mask_align_size + align_32
    mask_recv_size = inline_align(local_moe_expert_num * ep_world_size * mask_slot_size, align_512)

    # quant_token_scale_size
    mx_scale_num = (hidden + align_32 - 1) // align_32
    data_bytes = inline_align(hidden, align_256)
    token_bytes = inline_align(data_bytes + mx_scale_num, align_32)
    quant_token_scale_size = inline_align(num_max_tokens_per_rank * token_bytes, align_512)

    # combine_send_size
    combine_out = inline_align(
        num_max_tokens_per_rank * hidden * num_topk * y_out_dtype_size, align_512
    )

    # Total required size
    ccl_buffer_size = peermem_data_offset + mask_recv_size + quant_token_scale_size + combine_out
    ccl_buffer_size = (
        inline_align(inline_align(ccl_buffer_size, mb_conversion) // mb_conversion, 2) // 2
    )

    return ccl_buffer_size


class SymmBuffer:
    """Wrapper for HCCL communication context and mega_moe configuration metadata.

    This is a pure-Python class that encapsulates the communication context
    tensor (created via CommContextManager) along with all the metadata
    parameters required by the mega_moe operator.

    Do not instantiate directly; use :func:`get_symm_buffer_for_mega_moe`.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        max_recv_token_num: int = 0,
        dispatch_quant_mode: int = 0,
        dispatch_quant_out_dtype: Optional[int] = None,
        combine_quant_mode: int = 0,
        comm_alg: str = "",
    ):
        # --- Communication metadata ---
        self.group = group
        self.rank_id = dist.get_rank(group)
        self.group_name = _get_hccl_comm_name(group, self.rank_id)
        self.ep_world_size = dist.get_world_size(group)

        # --- Create HCCL communication context ---
        self._ctx_manager = torch.classes._C_ascend.CommContextManager(
            self.group_name, self.ep_world_size, "auto"
        )
        self.context = self._ctx_manager.create_context()
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size

        # --- Configuration metadata ---
        self.num_experts = num_experts
        self.max_recv_token_num = max_recv_token_num
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.dispatch_quant_mode = dispatch_quant_mode
        self.dispatch_quant_out_dtype = dispatch_quant_out_dtype
        self.combine_quant_mode = combine_quant_mode
        self.comm_alg = comm_alg
        self.topo_type = self._ctx_manager.topo_type
        self.rank_num_per_server = self._ctx_manager.rank_num_per_server


def get_symm_buffer_for_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    *,
    max_recv_token_num: int = 0,
    dispatch_quant_mode: int = 0,
    dispatch_quant_out_dtype: Optional[int] = None,
    combine_quant_mode: int = 0,
    comm_alg: str = "",
) -> SymmBuffer:
    """Create a SymmBuffer for use with the mega_moe operator.

    This is a convenience factory that constructs an HCCL communication
    context tensor and bundles it with operator configuration metadata.

    Args:
        group: The expert-parallel (EP) distributed process group.
        num_experts: Total number of MoE experts.
        num_max_tokens_per_rank: Max tokens per rank (0 = auto).
        num_topk: Top-K routing value (reserved, not yet used internally).
        hidden: Hidden dimension (reserved, not yet used internally).
        intermediate_hidden: Intermediate projection dimension (reserved, not
            yet used internally).
        max_recv_token_num: Max tokens this rank can receive (0 = auto).
        dispatch_quant_mode: Quantization mode for dispatch (4 = MX mode).
        dispatch_quant_out_dtype: Output dtype after dispatch quantization
            (23 = float8_e5m2, 24 = float8_e4m3fn).
        combine_quant_mode: Combine quantization mode (reserved, must be 0).
        comm_alg: Communication algorithm (reserved, must be '').

    Returns:
        A SymmBuffer instance ready to be passed to :func:`mega_moe`.
    """
    return SymmBuffer(
        group=group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        max_recv_token_num=max_recv_token_num,
        dispatch_quant_mode=dispatch_quant_mode,
        dispatch_quant_out_dtype=dispatch_quant_out_dtype,
        combine_quant_mode=combine_quant_mode,
        comm_alg=comm_alg,
    )


def mega_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights: List[torch.Tensor],
    l2_weights: List[torch.Tensor],
    sym_buffer: SymmBuffer,
    *,
    l1_weights_sf: Optional[List[torch.Tensor]] = None,
    l2_weights_sf: Optional[List[torch.Tensor]] = None,
    l1_bias: Optional[List[torch.Tensor]] = None,
    l2_bias: Optional[List[torch.Tensor]] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    weight1_type: Optional[int] = None,
    weight2_type: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute the mega_moe fused operator (Dispatch + GMM1 + SwiGLU + GMM2 + Combine).

    Args:
        x: Input token data, shape (BS, H), dtype bfloat16.
        topk_ids: Expert indices for each token's top-K, shape (BS, K),
            dtype int32.
        topk_weights: Expert weights for each token's top-K, shape (BS, K),
            dtype float32.
        l1_weights: GroupMatmul1 right-hand weight matrices (one per expert).
            Each has shape (N, H), dtype float8_e5m2 or float8_e4m3fn.
        l2_weights: GroupMatmul2 right-hand weight matrices (one per expert).
            Each has shape (H, N/2), dtype float8_e5m2 or float8_e4m3fn.
        sym_buffer: Communication context and configuration, created by
            :func:`get_symm_buffer_for_mega_moe`.
        l1_weights_sf: Optional MX quantization scales for weight1.
            Shape (N, CeilDiv(H,64), 2), dtype float8_e8m0.
        l2_weights_sf: Optional MX quantization scales for weight2.
            Shape (H, CeilDiv(N/2,64), 2), dtype float8_e8m0.
        l1_bias: Optional bias for GroupMatmul1.
        l2_bias: Optional bias for GroupMatmul2.
        x_active_mask: Optional token participation mask (reserved).
        activation: Activation function name (default "swiglu").
        activation_clamp: Clamp value for activation.
        weight1_type: Override ACL data type for weight1.
        weight2_type: Override ACL data type for weight2.

    Returns:
        Tuple of (y, expert_token_nums):
        - y: Output tensor, shape (BS, H), dtype bfloat16.
        - expert_token_nums: Per-expert received token counts, shape
          (local_expert_num,), dtype int32.
    """
    import torch

    return torch.ops._C_ascend.npu_mega_moe(
        sym_buffer.context,
        x,
        topk_ids,
        topk_weights,
        l1_weights,
        l2_weights,
        sym_buffer.num_experts,
        sym_buffer.ep_world_size,
        sym_buffer.ccl_buffer_size,
        weight_scales1=l1_weights_sf,
        weight_scales2=l2_weights_sf,
        bias1=l1_bias,
        bias2=l2_bias,
        x_active_mask=x_active_mask,
        max_recv_token_num=sym_buffer.max_recv_token_num,
        dispatch_quant_mode=sym_buffer.dispatch_quant_mode,
        combine_quant_mode=sym_buffer.combine_quant_mode,
        comm_alg=sym_buffer.comm_alg,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        activation=activation,
        activation_clamp=activation_clamp,
        dispatch_quant_out_dtype=sym_buffer.dispatch_quant_out_dtype,
        weight1_type=weight1_type,
        weight2_type=weight2_type,
        topo_type=sym_buffer.topo_type,
        rank_num_per_server=sym_buffer.rank_num_per_server,
    )
