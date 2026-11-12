# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP-axis projection exchanges; attention state remains with the DP owner.

The input-AG/head-A2A layout is adapted from vllm-ascend PR20574 at
66f79fede7980c1e1192796ef0af648fe9a13106. Kimi keeps MXFP8 QKV separate
from BF16 beta and full-rank G, rather than GLM's fused BF16 projection.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn import functional as F
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import is_forward_context_available
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.utils import npu_stream_switch


def projection_capacity(hidden: torch.Tensor) -> int:
    """Runner DP synchronization has already aligned graph buckets across peers."""
    synced = _EXTRA_CTX.max_tokens_across_dp if is_forward_context_available() else None
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


class ProjectionColumnParallelLinear(ColumnParallelLinear):
    """Use an explicit projection group without changing the model/state TP."""

    def __init__(self, input_size: int, output_size: int, *, group: GroupCoordinator | None, **kwargs):
        super().__init__(
            input_size,
            output_size,
            tp_rank=group.rank_in_group if group is not None else 0,
            tp_size=group.world_size if group is not None else 1,
            bias=False,
            gather_output=False,
            **kwargs,
        )

    def forward(self, hidden):
        return self.quant_method.apply(self, hidden, bias=None), None


class ProjectionMergedColumnParallelLinear(MergedColumnParallelLinear, ProjectionColumnParallelLinear):
    def __init__(self, input_size: int, output_sizes: list[int], **kwargs):
        self.output_sizes = output_sizes.copy()
        ProjectionColumnParallelLinear.__init__(self, input_size, sum(output_sizes), **kwargs)


@dataclass
class PendingTensor:
    tensor: torch.Tensor
    work: dist.Work
    # Retain the send buffer until the corresponding receive has completed.
    source: torch.Tensor

    def wait(self) -> torch.Tensor:
        self.work.wait()
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

    def scatter(self, source, group):
        shape = (source.shape[0] // group.world_size, *source.shape[1:])
        return self._launch(source.contiguous(), source.new_empty(shape), group, dist.reduce_scatter_tensor)


@dataclass
class PendingGate:
    pending: PendingTensor | None
    owner_gate: torch.Tensor | None
    size: int
    capacity: int
    heads: int
    head_dim: int

    def wait(self) -> torch.Tensor:
        if self.pending is None:
            assert self.owner_gate is not None
            return self.owner_gate.view(self.capacity, self.heads, self.head_dim)
        received = self.pending.wait().view(self.size, self.capacity, self.heads // self.size, self.head_dim)
        self.owner_gate = received.permute(1, 0, 2, 3).reshape(self.capacity, self.heads, self.head_dim)
        self.pending = None
        return self.owner_gate


@dataclass
class PendingKDAProjection:
    qkv: torch.Tensor
    beta: torch.Tensor
    forget: torch.Tensor
    gate: PendingGate
    pending_qkv: PendingTensor | None
    size: int
    capacity: int
    heads: int
    head_dim: int

    def wait(self):
        if self.pending_qkv is not None:
            local_heads = self.heads // self.size
            width = local_heads * self.head_dim
            received = self.pending_qkv.wait().view(self.size, self.capacity, -1)
            qkv, beta = received.split((3 * width, local_heads), dim=-1)
            qkv = qkv.reshape(self.size, self.capacity, 3, local_heads, self.head_dim)
            self.qkv = qkv.permute(1, 2, 0, 3, 4).reshape(self.capacity, -1)
            self.beta = beta.permute(1, 0, 2).reshape(self.capacity, self.heads)
            self.pending_qkv = None
        return (
            self.qkv,
            self.forget.view(1, self.capacity, self.heads, self.head_dim),
            self.beta.unsqueeze(0),
            self.gate,
        )


class FullRankGateTP:
    def __init__(self, group: GroupCoordinator, heads: int, head_dim: int):
        self.group, self.heads, self.head_dim = group, heads, head_dim
        self.communication = TPCommunication()
        self.uses_aiv = uses_aiv_communication(group)

    def gather(self, hidden: torch.Tensor) -> PendingTensor:
        capacity = projection_capacity(hidden)
        hidden = F.pad(hidden, (0, 0, 0, capacity - hidden.shape[0]))
        return self.communication.gather(hidden, self.group)

    def project(self, gathered: PendingTensor, layer) -> PendingGate:
        hidden = gathered.wait()
        gate = layer(hidden)[0]
        size = self.group.world_size
        return PendingGate(
            self.communication.exchange(gate, self.group),
            None,
            size,
            hidden.shape[0] // size,
            self.heads,
            self.head_dim,
        )


class KDAProjectionTP:
    def __init__(self, qkv_group, gate_group, heads: int, head_dim: int):
        self.qkv_group, self.gate_group = qkv_group, gate_group
        self.heads, self.head_dim = heads, head_dim
        self.communication = TPCommunication()
        self.qkv_uses_aiv = uses_aiv_communication(qkv_group)
        self.gate_uses_aiv = uses_aiv_communication(gate_group)

    def project(self, layer, hidden, *, aux_stream: torch.npu.Stream):
        capacity = projection_capacity(hidden)
        owner = F.pad(hidden, (0, 0, 0, capacity - hidden.shape[0]))
        if capacity == 0:
            raise RuntimeError("KDA projection TP requires the runner to skip globally empty steps uniformly.")
        q_input = self.communication.gather(owner, self.qkv_group) if self.qkv_group is not None else None
        if self.gate_group is self.qkv_group:
            g_input = q_input
        else:
            g_input = self.communication.gather(owner, self.gate_group) if self.gate_group is not None else None

        main = torch.npu.current_stream()
        owner_ready = main.record_event()
        owner.record_stream(aux_stream)
        # C1: owner F needs no gathered input and runs during input AG.
        with npu_stream_switch(aux_stream):
            aux_stream.wait_event(owner_ready)
            forget = layer.f_b_proj(layer.f_a_proj(owner)[0])[0]
            forget_ready = aux_stream.record_event()
        # G waits for input AG and owner F_b, avoiding concurrent Cube work.
        g_hidden = g_input.wait() if g_input is not None else owner
        main.wait_event(forget_ready)
        gate = layer.g_proj(g_hidden)[0]
        q_hidden = g_hidden if q_input is g_input else q_input.wait() if q_input is not None else owner
        input_ready = main.record_event()
        q_hidden.record_stream(aux_stream)
        # C2: raw BF16 beta Cube and MX input quant Vector have the same input.
        with npu_stream_switch(aux_stream):
            aux_stream.wait_event(input_ready)
            beta = layer.b_proj(q_hidden)[0]
            beta_ready = aux_stream.record_event()
        quantized = layer._quantize_fused_qkv(q_hidden, for_finegrained_tp=True)
        # This launch records a main-stream event after DynamicQuant, so G
        # communication cannot contend with that quantization on AIVs.
        pending_gate = self.exchange_gate(gate)
        main.wait_event(beta_ready)
        beta.record_stream(main)
        forget.record_stream(main)
        qkv = layer._matmul_fused_qkv(quantized)
        if self.gate_uses_aiv:
            # Hide AIV communication only behind QKV Cube, then join before
            # cat/reorder/zeros/cast/sigmoid/conv/recurrent Vector work.
            pending_gate.wait()
        q_pending = None
        if self.qkv_group is not None:
            if qkv.dtype != beta.dtype:
                raise RuntimeError("Kimi QKV/beta exchange requires equal raw projection dtypes.")
            q_pending = self.communication.exchange(
                torch.cat((qkv, beta), dim=-1), self.qkv_group, overlap=not self.qkv_uses_aiv
            )
        result = PendingKDAProjection(
            qkv,
            beta,
            forget,
            pending_gate,
            q_pending,
            self.qkv_group.world_size if self.qkv_group is not None else 1,
            capacity,
            self.heads,
            self.head_dim,
        )
        if self.qkv_uses_aiv:
            result.wait()
        # Non-AIV QKV communication can overlap independent output-buffer
        # initialization. Its consumers still wait before reading QKV/beta.
        return result

    def exchange_gate(self, gate: torch.Tensor) -> PendingGate:
        size = self.gate_group.world_size if self.gate_group is not None else 1
        return PendingGate(
            self.communication.exchange(gate, self.gate_group) if self.gate_group is not None else None,
            gate if self.gate_group is None else None,
            size,
            gate.shape[0] // size,
            self.heads,
            self.head_dim,
        )


class ChunkedMLPTP:
    def __init__(self, group):
        self.group = group
        self.communication = TPCommunication()

    def forward(self, layer, hidden):
        rows = hidden.shape[0]
        capacity = projection_capacity(hidden)
        owner = F.pad(hidden, (0, 0, 0, capacity - rows))
        chunks = owner.tensor_split(2) if capacity > 1 else (owner,)
        gathered = [self.communication.gather(chunk.contiguous(), self.group) for chunk in chunks]
        outputs = []
        for pending in gathered:
            up, down = layer.gate_up_proj, layer.down_proj
            projected = up.quant_method.apply(up, pending.wait(), bias=up.bias)
            down_input = layer._prepare_down_input(projected)
            bias = down.bias if self.group.rank_in_group == 0 else None
            partial = down.quant_method.apply(down, down_input, bias=bias)
            outputs.append(self.communication.scatter(partial, self.group))
        # AG0, AG1, RS0, RS1 are launched in the same order on every rank.
        return torch.cat([pending.wait() for pending in outputs], dim=0)[:rows]


class ChunkedLMHeadTP:
    def __init__(self, group):
        self.group = group
        self.communication = TPCommunication()

    def forward(self, processor, hidden, head, bias):
        # The runner has padded sampled rows to one common capacity already.
        chunks = hidden.tensor_split(2) if hidden.shape[0] > 1 else (hidden,)
        gathered = [self.communication.gather(chunk.contiguous(), self.group) for chunk in chunks]
        outputs = []
        for pending in gathered:
            logits = processor._apply_head(head, pending.wait(), bias)
            source = logits.contiguous().view(self.group.world_size, -1, logits.shape[-1])
            outputs.append(self.communication.exchange(source, self.group))
        # Each chunk independently restores owner rows and the full vocabulary.
        result = [
            p.wait().permute(1, 0, 2).reshape(p.tensor.shape[1], self.group.world_size * p.tensor.shape[2])
            for p in outputs
        ]
        return torch.cat(result, dim=0)[..., : processor.org_vocab_size]
