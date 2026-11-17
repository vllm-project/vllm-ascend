# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_ascend.models.kimi_k3 import kimi_k3


@pytest.mark.parametrize("rank", range(16))
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_dense_adapter_preserves_token_rows(rank, sequence_parallel):
    x = torch.arange(16 * 8, dtype=torch.float64).view(16, 8)
    calls = []

    def init(module, **kwargs):
        nn.Module.__init__(module)
        module.gate_up_proj = SimpleNamespace(custom_op=None)
        assert kwargs["reduce_results"] is (not sequence_parallel)

    def upstream(module, value):
        calls.append("tp_mlp")
        result = value.square() + 2
        return result / 16 if sequence_parallel else result

    def gather(local):
        calls.append("gather")
        assert torch.equal(local, x[rank : rank + 1])
        return x

    def shard(value):
        calls.append("reduce_scatter")
        assert torch.equal(value * 16, x.square() + 2)
        return value[rank : rank + 1] * 16

    with (
        patch.object(kimi_k3.KimiMLP, "__init__", init),
        patch.object(kimi_k3.KimiMLP, "forward", upstream),
        patch.object(kimi_k3, "sp_all_gather", gather),
        patch.object(kimi_k3, "sp_reduce_scatter", shard),
    ):
        model = kimi_k3.AscendKimiMLP(
            hidden_size=8, intermediate_size=16, hidden_act="silu", use_sequence_parallel=sequence_parallel
        )
        given = x[rank : rank + 1] if sequence_parallel else x
        result = model(given)
        assert torch.equal(result, given.square() + 2)
        assert calls == (["gather", "tp_mlp", "reduce_scatter"] if sequence_parallel else ["tp_mlp"])


@pytest.mark.parametrize("aiv", [False, True])
def test_dense_finegrained_pipeline_uses_ccu_only(monkeypatch, aiv):
    group = SimpleNamespace(world_size=2, rank_in_group=0)
    context: dict[str, kimi_k3.AscendKimiMLP] = {}

    def init(module, **kwargs):
        nn.Module.__init__(module)
        module.gate_up_proj = SimpleNamespace(custom_op=kimi_k3.MLPColumnParallelOp(None))

    monkeypatch.setattr(kimi_k3.KimiMLP, "__init__", init)
    monkeypatch.setattr(kimi_k3.KimiMLP, "forward", lambda module, value: value * 2)
    monkeypatch.setattr(kimi_k3, "get_mlp_tp_group", lambda: group)
    monkeypatch.setattr(kimi_k3, "uses_aiv_communication", lambda group: aiv)
    monkeypatch.setattr(
        kimi_k3,
        "get_current_vllm_config",
        lambda: SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=context)),
    )
    model = kimi_k3.AscendKimiMLP(8, 16, "silu", prefix="mlp")
    if aiv:
        assert model._finegrained_mlp is None
        assert not context
        value = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        torch.testing.assert_close(model(value), value * 2)
    else:
        assert isinstance(model._finegrained_mlp, kimi_k3.ChunkedMLPTP)
        assert context["mlp"] is model
