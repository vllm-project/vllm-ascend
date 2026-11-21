# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA input projection sharding with request-local recurrent state."""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_ascend.ascend_forward_context import _EXTRA_CTX

if TYPE_CHECKING:
    from .kda import Glm5NextLinearAttention


class KDAInputProjectionTPLayout:
    def __init__(self, group: GroupCoordinator, num_heads: int, head_dim: int):
        self.group = group
        self.size = group.world_size
        self.heads = num_heads // self.size
        self.head_dim = head_dim
        self.width = self.heads * head_dim

    def _all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor, group=self.group.device_group)
        return output


class KDAInputProjectionTP(KDAInputProjectionTPLayout):
    def project(
        self, layer: "Glm5NextLinearAttention", hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        local_tokens = hidden.shape[0]
        capacity = max(local_tokens, _EXTRA_CTX.max_tokens_across_dp or local_tokens)
        if capacity == 0:
            return None
        hidden = F.pad(hidden, (0, 0, 0, capacity - local_tokens))
        gathered = self.group.all_gather(hidden, dim=0)
        projected = layer.in_proj_qkvbfg_a(gathered)[0]
        qkv_beta, forget_a, gate_a = projected.split(
            (3 * self.width + self.heads, self.head_dim, self.head_dim), dim=-1
        )
        sent = qkv_beta.contiguous()
        received = torch.empty_like(sent)
        work = dist.all_to_all_single(received, sent, group=self.group.device_group, async_op=True)
        # Owner-local up-projections avoid sending their full-head outputs and
        # run while the QKV and beta exchange is in flight.
        start = self.group.rank_in_group * capacity
        forget = layer.f_b_proj(forget_a.narrow(0, start, capacity))[0]
        gate = layer.g_b_proj(gate_a.narrow(0, start, capacity))[0]
        assert work is not None
        work.wait()
        qkv, beta = received.view(self.size, capacity, -1).split((3 * self.width, self.heads), dim=-1)
        qkv = qkv.view(self.size, capacity, 3, self.heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3, 4).reshape(capacity, -1)
        beta = beta.permute(1, 0, 2).reshape(1, capacity, -1)
        forget = forget.view(1, capacity, layer.num_heads, self.head_dim)
        gate = gate.view(capacity, layer.num_heads, self.head_dim)
        return qkv, forget, beta, gate
