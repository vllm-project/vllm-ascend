#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context


def _build_or_get_topk(
    moe_comm_method,
    *,
    num_tokens: int,
    top_k: int,
    num_logical_experts: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cache = getattr(moe_comm_method, "_force_eplb_topk_cache", None)
    if cache is None:
        cache = {}
        moe_comm_method._force_eplb_topk_cache = cache
    key = (
        int(num_tokens),
        int(top_k),
        int(num_logical_experts),
        int(moe_comm_method.moe_config.ep_size),
        int(moe_comm_method.moe_config.ep_rank),
        dtype,
        device,
    )
    table = cache.get(key)
    if table is None:
        table = _build_round_robin_topk(
            num_tokens=num_tokens,
            top_k=top_k,
            num_logical_experts=num_logical_experts,
            ep_size=int(moe_comm_method.moe_config.ep_size),
            ep_rank=int(moe_comm_method.moe_config.ep_rank),
            device=device,
            dtype=dtype,
        )
        cache[key] = table
    return table


def build_force_eplb_topk(
    device: torch.device,
    max_num_tokens: int,
) -> None:
    """Build force-EPLB tables before ACLGraph capture."""
    forward_context = get_forward_context()
    moe_comm_method = forward_context.moe_comm_method
    if moe_comm_method is None:
        return

    vllm_config = get_current_vllm_config()
    capture_sizes = {int(size) for size in (vllm_config.compilation_config.cudagraph_capture_sizes or [])}
    if not capture_sizes:
        capture_sizes.add(int(max_num_tokens))

    moe_config = moe_comm_method.moe_config
    top_k = int(moe_config.experts_per_token)
    num_logical_experts = int(moe_config.num_logical_experts)
    for num_tokens in capture_sizes:
        for dtype in (torch.int32, torch.int64):
            _build_or_get_topk(
                moe_comm_method,
                num_tokens=int(num_tokens),
                top_k=top_k,
                num_logical_experts=num_logical_experts,
                dtype=dtype,
                device=device,
            )


def get_force_eplb_topk(
    topk_ids: torch.Tensor,
    num_logical_experts: int,
) -> torch.Tensor | None:
    """Return deterministic round-robin ids when the policy is enabled."""
    moe_comm_method = get_forward_context().moe_comm_method
    if moe_comm_method is None:
        return None
    top_k = int(topk_ids.shape[1])
    return _build_or_get_topk(
        moe_comm_method,
        num_tokens=int(topk_ids.shape[0]),
        top_k=int(top_k),
        num_logical_experts=int(num_logical_experts),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )


def _build_round_robin_topk(
    *,
    num_tokens: int,
    top_k: int,
    num_logical_experts: int,
    ep_size: int,
    ep_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if num_logical_experts % ep_size != 0:
        raise ValueError(
            "num_logical_experts must be divisible by ep_size: "
            f"num_logical_experts={num_logical_experts}, ep_size={ep_size}"
        )

    experts_per_rank = num_logical_experts // ep_size
    expanded_tokens = num_tokens * top_k
    expanded_offset = expanded_tokens * ep_rank + ep_rank

    idx = torch.arange(expanded_tokens, device=device, dtype=torch.int64)
    cursor = idx + expanded_offset
    col = torch.remainder(cursor, ep_size)
    row = torch.remainder(
        torch.div(cursor, ep_size, rounding_mode="floor"),
        experts_per_rank,
    )
    expert_ids = row + col * experts_per_rank
    return expert_ids.to(dtype=dtype).view(num_tokens, top_k)
