# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark must load the same physical expert layout that EPLB initializes."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_ascend.models.deepseek_v4 import dspark as dspark_module
from vllm_ascend.models.deepseek_v4.dspark import DeepseekV4DSparkModel, DSparkDeepseekV4ForCausalLM


@pytest.fixture
def draft_factory(monkeypatch):
    class MoE(torch.nn.Module):
        def __init__(self, logical, redundant, local):
            super().__init__()
            self.n_logical_experts = logical
            self.n_physical_experts = logical + redundant
            self.n_local_physical_experts = local
            self.n_routed_experts = logical
            self.n_shared_experts = 0
            self.n_redundant_experts = redundant
            self.experts = torch.nn.Module()

    class Decoder(torch.nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp

    # Only layer allocation is replaced; MoE discovery, metadata extraction,
    # expert mapping and the draft checkpoint loader use production methods.
    monkeypatch.setattr(dspark_module, "DeepseekV4MoE", MoE)
    monkeypatch.setattr(dspark_module, "DeepseekV4DecoderLayer", Decoder)
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_rank", lambda: 0)

    def create(logical=256, redundant=8, ep_size=8, layers=1):
        draft = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
        torch.nn.Module.__init__(draft)
        draft.config = SimpleNamespace(n_routed_experts=logical, num_hidden_layers=43, num_attention_heads=8)
        draft.rotation_path = None
        draft.model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
        torch.nn.Module.__init__(draft.model)
        draft.model.config = draft.config
        draft.model.num_dspark_layers = layers
        local = (logical + redundant) // ep_size
        draft.model.layers = torch.nn.ModuleDict(
            {str(43 + i): Decoder(MoE(logical, redundant, local)) for i in range(layers)}
        )
        draft.set_moe_parameters()
        return draft

    return create


@pytest.mark.parametrize("num_moe_layers", [0, 1, 3])
def test_moe_discovery_counts_only_local_moe_layers(draft_factory, num_moe_layers):
    draft = draft_factory(layers=num_moe_layers)
    # Dense and pipeline-missing layers must not contribute to the count.
    draft.model.layers["dense"] = dspark_module.DeepseekV4DecoderLayer(torch.nn.Identity())
    draft.model.layers["missing"] = PPMissingLayer()
    draft.set_moe_parameters()

    assert draft.num_moe_layers == num_moe_layers
    assert len(draft.moe_layers) == len(draft.moe_mlp_layers) == num_moe_layers
    assert is_mixture_of_experts(draft) is (num_moe_layers > 0)
    assert draft.num_redundant_experts == (8 if num_moe_layers else 0)


@pytest.mark.parametrize("redundant", [0, 8])
@pytest.mark.parametrize("rank", range(8))
def test_checkpoint_loader_populates_every_local_physical_expert(draft_factory, rank, redundant):
    logical, ep_size = 256, 8
    draft = draft_factory(logical=logical, redundant=redundant, ep_size=ep_size)
    local = (logical + redundant) // ep_size
    begin = rank * local
    experts = draft.model.layers["43"].mlp.experts
    routed = experts.routed_experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed)
    routed.moe_config = SimpleNamespace(ep_size=ep_size)
    routed._use_v2_model_runner = True
    routed.shared_expert_online_loader = None
    routed.w13_weight = torch.nn.Parameter(torch.full((local, 2), -1.0), requires_grad=False)
    routed.w2_weight = torch.nn.Parameter(torch.full((local, 1), -1.0), requires_grad=False)

    def weight_loader(param, loaded, name, *, shard_id, expert_id, return_success):
        assert return_success
        if not begin <= expert_id < begin + local:
            return False
        column = 1 if shard_id == "w3" else 0
        param.data[expert_id - begin, column].copy_(loaded.squeeze())
        return True

    routed.w13_weight.weight_loader = weight_loader
    routed.w2_weight.weight_loader = weight_loader
    weights = [
        (f"mtp.0.ffn.experts.{expert}.w{projection}.weight", torch.tensor([float(expert)]))
        for expert in range(logical)
        for projection in (1, 2, 3)
    ]
    loaded = draft.load_weights(iter(weights))

    # Each rank owns its contiguous logical experts followed by one replica
    # from the next rank (the last rank wraps to expert 0). R=0 stays linear.
    expected = list(range(rank * (logical // ep_size), (rank + 1) * (logical // ep_size)))
    if redundant:
        expected.append(((rank + 1) * (logical // ep_size)) % logical)
    expected = torch.tensor(expected, dtype=torch.float32)
    torch.testing.assert_close(routed.w13_weight[:, 0], expected)
    torch.testing.assert_close(routed.w13_weight[:, 1], expected)
    torch.testing.assert_close(routed.w2_weight[:, 0], expected)
    assert loaded == {
        "model.layers.43.mlp.experts.routed_experts.w13_weight",
        "model.layers.43.mlp.experts.routed_experts.w2_weight",
    }
