# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.eplb_map_flat import map_to_physical_triton as map_to_physical_triton


@triton.jit
def _record_expert_tokens_kernel(
    expert_tokens_ptr,
    expert_load_ptr,
    record_enabled_ptr,
    num_local_experts,
    local_expert_start,
    group_list_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if tl.load(record_enabled_ptr) != 0:
        mask = offsets < num_local_experts
        current = tl.load(expert_tokens_ptr + offsets, mask=mask, other=0)
        # Narrow before arithmetic when the destination is narrower: integer
        # add/subtract has the same result modulo the destination width.
        if current.dtype.primitive_bitwidth > expert_load_ptr.dtype.element_ty.primitive_bitwidth:
            current = current.to(expert_load_ptr.dtype.element_ty)
        if group_list_type == 1:
            local_load = current
        else:
            previous_offset = tl.maximum(offsets - 1, 0)
            previous = tl.load(expert_tokens_ptr + previous_offset, mask=mask & (offsets > 0), other=0)
            if previous.dtype.primitive_bitwidth > expert_load_ptr.dtype.element_ty.primitive_bitwidth:
                previous = previous.to(expert_load_ptr.dtype.element_ty)
            local_load = current - previous
        load_offsets = local_expert_start + offsets
        previous_load = tl.load(expert_load_ptr + load_offsets, mask=mask, other=0)
        tl.store(expert_load_ptr + load_offsets, previous_load + local_load, mask=mask)


def record_expert_tokens_triton(
    expert_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    group_list_type: int,
    local_expert_start: int,
) -> None:
    """Accumulate operator-provided local counts when collection is enabled."""
    num_local_experts = expert_tokens.numel()
    if num_local_experts == 0:
        return
    block_size = min(256, triton.next_power_of_2(num_local_experts))
    _record_expert_tokens_kernel[(triton.cdiv(num_local_experts, block_size),)](
        expert_tokens,
        expert_load_view,
        record_enabled,
        num_local_experts,
        local_expert_start,
        group_list_type=group_list_type,
        BLOCK_SIZE=block_size,
    )
