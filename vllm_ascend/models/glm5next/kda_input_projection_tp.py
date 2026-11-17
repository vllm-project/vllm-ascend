# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA input projection sharding with request-local recurrent state."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import npu_stream_switch

if TYPE_CHECKING:
    from .kda import Glm5NextLinearAttention


def projection_capacity(hidden: torch.Tensor) -> int:
    """Runner DP synchronization has already aligned graph buckets across peers."""
    synced = _EXTRA_CTX.max_tokens_across_dp
    return max(hidden.shape[0], synced or 0)


def uses_aiv_communication(group: GroupCoordinator | None) -> bool:
    """Resolve the configured communication mode once, without a device probe."""
    if group is None:
        return False
    backend = group.device_group._get_backend(torch.device("npu"))
    mode = backend.options.hccl_config.get("hccl_op_expansion_mode", 0)
    if mode:
        # HcclCommConfig: 3=AIV, 4=AIV_ONLY. All other explicit modes use
        # separate communication engines; a PG setting overrides the env.
        return mode in (3, 4)
    return envs.HCCL_OP_EXPANSION_MODE in ("AIV", "AIV_ONLY")


@dataclass
class PendingTensor:
    tensor: torch.Tensor
    work: dist.Work
    # Retain the send buffer until the corresponding receive has completed.
    source: torch.Tensor

    def wait(self) -> torch.Tensor:
        self.work.wait()
        consumer = torch.npu.current_stream()
        self.source.record_stream(consumer)
        self.tensor.record_stream(consumer)
        return self.tensor


class TPCommunication:
    """One ordered launch stream per executor; no collective is launched on aux."""

    def __init__(self):
        self.stream = None

    def _launch(self, source, output, group, op):
        main = torch.npu.current_stream()
        if self.stream is None:
            self.stream = torch.npu.Stream()
        ready = main.record_event()
        source.record_stream(self.stream)
        output.record_stream(main)
        with npu_stream_switch(self.stream):
            self.stream.wait_event(ready)
            work = op(output, source, group=group.device_group, async_op=True)
        return PendingTensor(output, work, source)

    def gather(self, hidden, group):
        shape = (hidden.shape[0] * group.world_size, *hidden.shape[1:])
        output = hidden.new_empty(shape)
        return self._launch(hidden.contiguous(), output, group, dist.all_gather_into_tensor)

    def exchange(self, source, group, *, overlap=True):
        source = source.contiguous()
        output = torch.empty_like(source)
        if overlap:
            return self._launch(source, output, group, dist.all_to_all_single)
        # Submit on the caller's stream. The KDA caller joins immediately,
        # before it can enqueue Vector state recovery or output preparation.
        work = dist.all_to_all_single(output, source, group=group.device_group, async_op=True)
        return PendingTensor(output, work, source)


class KDAInputProjectionTPLayout:
    def __init__(self, group: GroupCoordinator | None, num_heads: int, head_dim: int):
        self.group = group
        self.size = group.world_size if group is not None else 1
        self.num_heads = num_heads
        self.heads = num_heads // self.size
        self.head_dim = head_dim
        self.width = self.heads * head_dim

    @staticmethod
    def prepare_input(hidden: torch.Tensor) -> tuple[torch.Tensor, int]:
        capacity = projection_capacity(hidden)
        return F.pad(hidden, (0, 0, 0, capacity - hidden.shape[0])), capacity

    def restore_qkv_beta(self, received: torch.Tensor, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
        qkv, beta = received.view(self.size, capacity, -1).split((3 * self.width, self.heads), dim=-1)
        qkv = qkv.view(self.size, capacity, 3, self.heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3, 4).reshape(capacity, -1)
        beta = beta.permute(1, 0, 2).reshape(1, capacity, self.num_heads)
        return qkv, beta

    def restore_heads(self, received: torch.Tensor, capacity: int) -> torch.Tensor:
        return (
            received.view(self.size, capacity, self.heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .reshape(capacity, self.num_heads, self.head_dim)
        )

    def _all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor, group=self.group.device_group)
        return output


@dataclass
class PendingGate:
    pending: PendingTensor | None
    owner_gate: torch.Tensor | None
    layout: KDAInputProjectionTPLayout
    capacity: int

    def wait(self) -> torch.Tensor:
        if self.pending is None:
            assert self.owner_gate is not None
            return self.owner_gate.view(self.capacity, self.layout.num_heads, self.layout.head_dim)
        self.owner_gate = self.layout.restore_heads(self.pending.wait(), self.capacity)
        self.pending = None
        return self.owner_gate


class FullRankGateTP:
    def __init__(self, group: GroupCoordinator, heads: int, head_dim: int):
        self.layout = KDAInputProjectionTPLayout(group, heads, head_dim)
        self.group = group
        self.communication = TPCommunication()
        self.uses_aiv = uses_aiv_communication(group)

    def gather(self, hidden: torch.Tensor) -> PendingTensor:
        owner, _ = self.layout.prepare_input(hidden)
        return self.communication.gather(owner, self.group)

    def project(self, gathered: PendingTensor, layer) -> PendingGate:
        hidden = gathered.wait()
        gate = layer(hidden)[0]
        return PendingGate(
            self.communication.exchange(gate, self.group), None, self.layout, hidden.shape[0] // self.layout.size
        )


class KDAInputProjectionTP(KDAInputProjectionTPLayout):
    def project(
        self, layer: "Glm5NextLinearAttention", hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        hidden, capacity = self.prepare_input(hidden)
        if capacity == 0:
            return None
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
        qkv, beta = self.restore_qkv_beta(received, capacity)
        forget = forget.view(1, capacity, layer.num_heads, self.head_dim)
        gate = gate.view(capacity, layer.num_heads, self.head_dim)
        return qkv, forget, beta, gate
