# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import KimiK3MLP, KimiK3MoE, _move_module_to_device
from vllm_ascend.ops.activation import AscendSituAndMul, SituActivationConfig


def test_kimi_k3_skips_explicit_move_for_meta_modules():
    module = nn.Linear(4, 4, device="meta")

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.is_meta for parameter in module.parameters())


def test_kimi_k3_moves_non_meta_modules():
    module = nn.Linear(4, 4)

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())


def test_kimi_k3_passes_situ_parameters_through_activation_config(monkeypatch):
    class StubModule(nn.Module):
        pass

    fused_moe_kwargs = {}

    def fake_replicated_linear(*args, **kwargs):
        return StubModule()

    def fake_fused_moe(**kwargs):
        fused_moe_kwargs.update(kwargs)
        return StubModule()

    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", fake_replicated_linear)
    monkeypatch.setattr(kimi_k3, "FusedMoE", fake_fused_moe)
    config = SimpleNamespace(
        hidden_act="situ",
        hidden_size=32,
        routed_expert_hidden_size=16,
        num_shared_experts=0,
        num_experts=8,
        rms_norm_eps=1e-6,
        latent_moe_use_norm=False,
        moe_intermediate_size=12,
        num_experts_per_token=2,
        moe_renormalize=True,
        use_grouped_topk=True,
        num_expert_group=4,
        topk_group=2,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=2.5,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    KimiK3MoE(config, prefix="model.layers.1.block_sparse_moe")

    activation = fused_moe_kwargs["activation"]
    assert isinstance(activation, SituActivationConfig)
    assert activation.beta == 4.0
    assert activation.linear_beta == 25.0


def test_kimi_k3_dense_mlp_uses_callable_situ(monkeypatch):
    class StubLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    monkeypatch.setattr(kimi_k3, "MergedColumnParallelLinear", lambda *args, **kwargs: StubLinear())
    monkeypatch.setattr(kimi_k3, "RowParallelLinear", lambda *args, **kwargs: StubLinear())
    config = SimpleNamespace(
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    mlp = KimiK3MLP(config, hidden_size=4, intermediate_size=2)
    hidden_states = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    output = mlp(hidden_states)

    assert isinstance(mlp.act_fn, AscendSituAndMul)
    assert output.shape == (1, 2)
