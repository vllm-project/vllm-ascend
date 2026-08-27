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
import os

import torch


def force_eplb_policy() -> str:
    return os.getenv("VLLM_ASCEND_FORCE_EPLB_POLICY", "").strip().lower()


def cann_round_robin_enabled() -> bool:
    return force_eplb_policy() == "cann_round_robin"


def build_cann_round_robin_topk(
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
