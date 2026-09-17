# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _collect_moe_load_kernel(
    tokens,
    loads,
    counter,
    load_enabled,
    counter_enabled,
    NUM_EXPERTS: tl.constexpr,
    WINDOW: tl.constexpr,
    MULTI_STAGE: tl.constexpr,
    HAS_GATE: tl.constexpr,
    CUMSUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.arange(0, BLOCK)
    count = tl.load(tokens + expert, expert < NUM_EXPERTS, other=0)
    if CUMSUM:
        previous = tl.load(tokens + expert - 1, (expert > 0) & (expert < NUM_EXPERTS), other=0)
        count -= previous
    if HAS_GATE:
        count *= tl.load(load_enabled)
    if MULTI_STAGE:
        iteration = tl.load(counter)
        offset = (iteration % WINDOW) * NUM_EXPERTS + expert
    else:
        offset = expert
    value = tl.load(loads + offset, expert < NUM_EXPERTS, other=0)
    value += count.to(loads.dtype.element_ty)
    tl.store(loads + offset, value, expert < NUM_EXPERTS)
    if MULTI_STAGE:
        advance = tl.load(counter_enabled) if HAS_GATE else 1
        tl.store(counter, iteration + advance)


def collect_moe_load(
    expert_tokens: torch.Tensor,
    moe_load: torch.Tensor,
    group_list_type: int,
    load_counter: torch.Tensor | None = None,
    load_enabled: torch.Tensor | None = None,
    counter_enabled: torch.Tensor | None = None,
) -> None:
    """Accumulate one layer's expert loads without temporary tensors."""
    num_experts = expert_tokens.numel()
    _collect_moe_load_kernel[(1,)](
        expert_tokens,
        moe_load,
        load_counter,
        load_enabled,
        counter_enabled,
        NUM_EXPERTS=num_experts,
        WINDOW=moe_load.shape[0] if load_counter is not None else 1,
        MULTI_STAGE=load_counter is not None,
        HAS_GATE=load_enabled is not None,
        CUMSUM=group_list_type == 0,
        BLOCK=triton.next_power_of_2(num_experts),
        multibuffer=False,
    )
