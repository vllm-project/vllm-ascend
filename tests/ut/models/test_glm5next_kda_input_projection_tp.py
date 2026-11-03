# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend import ascend_forward_context
from vllm_ascend.models.glm5next import kda as model_kda
from vllm_ascend.models.glm5next import kda_input_projection_tp
from vllm_ascend.ops import linear_op


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("explicit_group", [False, True])
@pytest.mark.parametrize("direction", ["column", "row"])
def test_projection_selects_explicit_group_without_forwarding_it(monkeypatch, size, explicit_group, direction):
    pcp_group = SimpleNamespace(world_size=1, rank_in_group=0)
    kda_group = SimpleNamespace(world_size=size, rank_in_group=size - 1)
    selected = kda_group if explicit_group else pcp_group
    seen: dict[str, int | bool] = {}

    def linear_init(self, input_size, output_size, **kwargs):
        torch.nn.Module.__init__(self)
        self.update_param_tp_status = lambda: None
        seen.update(input_size=input_size, output_size=output_size, **kwargs)

    monkeypatch.setattr(model_kda, "get_pcp_group", lambda: pcp_group)
    base = model_kda.ColumnParallelLinear if direction == "column" else model_kda.RowParallelLinear
    cls = model_kda.PCPColumnParallelLinear if direction == "column" else model_kda.PCPRowParallelLinear
    monkeypatch.setattr(base, "__init__", linear_init)
    layer = cls(4 * size, 8 * size, tp_group=kda_group if explicit_group else None)
    assert layer.tp_group is selected
    if direction == "column":
        assert seen == {
            "input_size": 4 * size,
            "output_size": 8 * size,
            "tp_rank": selected.rank_in_group,
            "tp_size": selected.world_size,
        }
    else:
        assert seen == {"input_size": 4 * size // selected.world_size, "output_size": 8 * size, "disable_tp": True}
        assert layer.tp_rank == selected.rank_in_group and layer.tp_size == selected.world_size


class CollectiveReference:
    def __init__(self, size):
        self.size = size
        self.barrier = Barrier(size, timeout=20)
        self.inputs = {}
        self.outputs = {}

    def exchange(self, rank, kind, tensor):
        self.inputs[rank] = tensor.clone()
        self.barrier.wait()
        if rank == 0:
            tensors = [self.inputs[i] for i in range(self.size)]
            for destination in range(self.size):
                if kind == "gather":
                    result = torch.cat(tensors)
                elif kind == "scatter":
                    result = torch.stack(tensors).sum(0).chunk(self.size)[destination]
                else:
                    result = torch.cat([source.chunk(self.size, dim=0)[destination] for source in tensors])
                self.outputs[destination] = result
        self.barrier.wait()
        result = self.outputs[rank].clone()
        self.barrier.wait()
        return result


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.parametrize("output_size", [1, 2, 4, 8])
@pytest.mark.parametrize("runner_v2", [False, True])
def test_projection_tp_matches_dense_weights_with_ragged_and_idle_owners(monkeypatch, size, output_size, runner_v2):
    """Different KDA and output groups must preserve each DP owner's recurrent state."""
    generator = torch.Generator().manual_seed(91)
    hidden_dim, heads, dim = 7, 8, 2
    width = heads * dim
    weights = [torch.randn(n, hidden_dim, generator=generator) * 0.1 for n in (width, width, width, heads, dim, dim)]
    forget_weight = torch.randn(width, dim, generator=generator) * 0.1
    gate_weight = torch.randn(width, dim, generator=generator) * 0.1
    output_weight = torch.randn(hidden_dim, width, generator=generator) * 0.1
    initial = torch.randn(3, heads, dim, generator=generator) * 0.1
    workers = max(size, output_size)
    states = [initial.clone() for _ in range(workers)]
    dense_states = [initial.clone() for _ in range(workers)]
    counts = ([3, 0, 1, 2, 2, 0, 1, 3][:workers], [1, 2, 0, 3, 0, 3, 2, 1][:workers], [0] * workers)
    inputs = [[torch.randn(n, hidden_dim, generator=generator) for n in step] for step in counts]
    coordinators = [CollectiveReference(size) for _ in range(workers // size)]
    output_coordinators = [CollectiveReference(output_size) for _ in range(workers // output_size)]
    thread = local()

    def core(qkv, forget, beta, output):
        count = thread.count
        q, k, v = qkv[:count].reshape(count, 3, heads, dim).unbind(1)
        state = thread.state
        state[:count].mul_(forget[0, :count].sigmoid()).add_(q * k + v * beta[0, :count, :, None].sigmoid())
        output[0, :count].copy_(state[:count])

    monkeypatch.setattr(ascend_forward_context, "get_forward_context", lambda: thread.context)
    monkeypatch.setattr(ascend_forward_context, "_USE_V2_EXTRA_KWARGS", runner_v2)
    monkeypatch.setattr(model_kda, "get_forward_context", lambda: thread.context)
    monkeypatch.setattr(linear_op, "oproj_tp_enable", lambda: output_size > 1)
    monkeypatch.setattr(linear_op, "get_otp_group", lambda: thread.output_group)
    monkeypatch.setattr(linear_op, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))

    def all_to_all(output, tensor, group, async_op=False):
        work = SimpleNamespace(wait=lambda: output.copy_(group.coordinator.exchange(group.rank, "all_to_all", tensor)))
        if async_op:
            return work
        work.wait()

    monkeypatch.setattr(kda_input_projection_tp.dist, "all_to_all_single", all_to_all)

    def linear(weight):
        return lambda tensor: (F.linear(tensor, weight), None)

    def worker(rank):
        local_rank = rank % size
        coordinator = coordinators[rank // size]
        group = SimpleNamespace(
            world_size=size,
            rank_in_group=local_rank,
            device_group=SimpleNamespace(rank=local_rank, coordinator=coordinator),
        )
        group.all_gather = lambda tensor, dim: coordinator.exchange(local_rank, "gather", tensor)
        output_rank = rank % output_size
        output_coordinator = output_coordinators[rank // output_size]
        thread.output_group = SimpleNamespace(
            world_size=output_size,
            rank_in_group=output_rank,
            device_group=SimpleNamespace(rank=output_rank, coordinator=output_coordinator),
            reduce_scatter=lambda tensor, dim: output_coordinator.exchange(output_rank, "scatter", tensor),
        )
        projection = SimpleNamespace(
            prefix="model.layers.0.linear_attn.o_proj",
            weight=output_weight.chunk(output_size, dim=1)[output_rank],
            quant_method=SimpleNamespace(apply=lambda layer, x, bias=None: F.linear(x, layer.weight, bias)),
            bias=None,
            skip_bias_add=False,
            return_bias=True,
            input_is_parallel=True,
            reduce_results=True,
            input_size_per_partition=width // output_size,
            output_size=hidden_dim,
        )
        output_op, _, actual_output_size = linear_op.get_parallel_op(False, projection.prefix, projection, "row")
        assert actual_output_size == output_size
        if output_op is not None:
            output_op.update_attrs()
        sharded = [w.chunk(size)[local_rank] if index < 4 else w for index, w in enumerate(weights)]
        layer = SimpleNamespace(
            prefix="kda",
            num_heads=heads,
            local_num_heads=heads,
            head_dim=dim,
            in_proj_qkvbfg_a=linear(torch.cat(sharded)),
            f_b_proj=linear(forget_weight),
            g_b_proj=linear(gate_weight),
            o_proj=linear(output_weight) if output_op is None else output_op.apply,
            o_norm=lambda output, gate: output * gate.unsqueeze(0).sigmoid(),
            _forward_core=lambda qkv_proj_states, g1, beta, core_attn_out: core(
                qkv_proj_states, g1, beta, core_attn_out
            ),
        )
        layer.projection_tp = kda_input_projection_tp.KDAInputProjectionTP(group, heads, dim)
        for step, batches in enumerate(inputs):
            hidden = batches[rank]
            extras = {"max_tokens_across_dp": max(counts[step])}
            thread.context = (
                SimpleNamespace(attn_metadata={}, additional_kwargs=extras)
                if runner_v2
                else SimpleNamespace(attn_metadata={}, **extras)
            )
            thread.count, thread.state = hidden.shape[0], states[rank]
            output = torch.empty_like(hidden)
            assert model_kda.Glm5NextLinearAttention._forward(layer, hidden, output) is None

            projected = F.linear(hidden, torch.cat(weights))
            qkv, beta, forget_a, gate_a = projected.split((3 * width, heads, dim, dim), dim=-1)
            dense_output = torch.zeros(1, hidden.shape[0], heads, dim)
            thread.state = dense_states[rank]
            core(
                qkv,
                F.linear(forget_a, forget_weight).reshape(1, -1, heads, dim),
                beta.unsqueeze(0),
                dense_output,
            )
            gated = dense_output * F.linear(gate_a, gate_weight).reshape(1, -1, heads, dim).sigmoid()
            expected = F.linear(gated.flatten(2).squeeze(0), output_weight)
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(states[rank], dense_states[rank])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(worker, range(workers)))
