# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.models import utils as model_utils
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_ascend.models import kimi_k3 as k3_module
from vllm_ascend.models import kimi_k3_mtp as mtp_module
from vllm_ascend.patch.platform import patch_eplb


class FakeMoE:
    """Attribute-compatible stand-in for AscendKimiMoE."""

    num_shared_experts = 2
    n_routed_experts = 256
    n_logical_experts = 256
    n_redundant_experts = 8
    n_physical_experts = 264
    n_local_physical_experts = 33
    experts = object()


class FakeDecoderLayer:
    def __init__(self, mlp):
        self.mlp = mlp


def test_mtp_moe_registration_metadata(monkeypatch):
    moe = FakeMoE()

    class FakePredictorLayer:
        mtp_block = FakeDecoderLayer(moe)

    monkeypatch.setattr(mtp_module, "AscendKimiMoE", FakeMoE)
    monkeypatch.setattr(mtp_module, "AscendKimiK3MultiTokenPredictorLayer", FakePredictorLayer)
    model: Any = mtp_module.AscendKimiK3MTP.__new__(mtp_module.AscendKimiK3MTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_expert_group=8, is_moe=True, num_experts=256, first_k_dense_replace=0, moe_layer_freq=1
    )
    model.model = SimpleNamespace(
        mtp_start_layer_idx=3,
        num_mtp_layers=3,
        layers={"3": FakePredictorLayer(), "4": PPMissingLayer(), "5": FakePredictorLayer()},
    )

    model.set_moe_parameters()

    assert model.num_moe_layers == 2
    assert model.moe_layers == [moe.experts, moe.experts]
    assert model.num_routed_experts == 256
    assert model.num_logical_experts == 256
    assert model.num_physical_experts == 264
    assert model.num_local_physical_experts == 33
    assert model.num_redundant_experts == 8
    assert model.num_shared_experts == 2
    assert model.num_expert_groups == 8
    assert is_mixture_of_experts(model)


def test_mtp_moe_registration_metadata_without_moe(monkeypatch):
    class FakePredictorLayer:
        mtp_block = FakeDecoderLayer(object())

    monkeypatch.setattr(mtp_module, "AscendKimiMoE", FakeMoE)
    monkeypatch.setattr(mtp_module, "AscendKimiK3MultiTokenPredictorLayer", FakePredictorLayer)
    model: Any = mtp_module.AscendKimiK3MTP.__new__(mtp_module.AscendKimiK3MTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_expert_group=None, is_moe=False, num_experts=None, first_k_dense_replace=0, moe_layer_freq=1
    )
    model.model = SimpleNamespace(mtp_start_layer_idx=3, num_mtp_layers=1, layers={"3": FakePredictorLayer()})

    model.set_moe_parameters()

    assert model.num_moe_layers == 0
    assert model.num_physical_experts == 0
    assert not is_mixture_of_experts(model)


def test_main_model_moe_registration_metadata(monkeypatch):
    moe = FakeMoE()

    monkeypatch.setattr(k3_module, "AscendKimiMoE", FakeMoE)
    monkeypatch.setattr(k3_module, "AscendKimiDecoderLayer", FakeDecoderLayer)
    model: Any = k3_module.AscendKimiLinearForCausalLM.__new__(k3_module.AscendKimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=3,
        is_moe=True,
        num_experts=256,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        num_expert_group=8,
    )
    model.model = SimpleNamespace(
        layers=[
            PPMissingLayer(),
            FakeDecoderLayer(moe),
            FakeDecoderLayer(moe),
        ]
    )

    model.set_moe_parameters()

    assert model.num_moe_layers == 2
    assert model.moe_layers == [moe.experts, moe.experts]
    assert model.num_redundant_experts == 8
    assert is_mixture_of_experts(model)


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_pp_stage_without_local_moe_is_not_registered(monkeypatch, draft, missing):
    monkeypatch.setattr(k3_module, "AscendKimiMoE", FakeMoE)
    monkeypatch.setattr(k3_module, "AscendKimiDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(mtp_module, "AscendKimiMoE", FakeMoE)

    class FakePredictorLayer:
        mtp_block = FakeDecoderLayer(object())

    monkeypatch.setattr(mtp_module, "AscendKimiK3MultiTokenPredictorLayer", FakePredictorLayer)
    model_cls = mtp_module.AscendKimiK3MTP if draft else k3_module.AscendKimiLinearForCausalLM
    model: Any = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=3,
        is_moe=True,
        num_experts=256,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        num_expert_group=8,
    )
    local_layer = FakePredictorLayer() if draft else FakeDecoderLayer(object())
    if missing:
        local_layer = PPMissingLayer()
    layers = {"3": local_layer} if draft else [local_layer, PPMissingLayer(), PPMissingLayer()]
    model.model = SimpleNamespace(mtp_start_layer_idx=3, num_mtp_layers=1, layers=layers)

    model.set_moe_parameters()

    assert model.num_moe_layers == 0
    assert model.moe_layers == []
    assert model.num_physical_experts == 0
    assert not is_mixture_of_experts(model)


@pytest.mark.parametrize("draft", [False, True])
def test_pp_eplb_state_has_one_weight_entry_per_local_layer(monkeypatch, draft):
    moe = FakeMoE()
    moe.experts = MagicMock()
    weights = [torch.ones((33, 2, 2))]
    moe.experts.get_expert_weights.return_value = weights
    monkeypatch.setattr(k3_module, "AscendKimiMoE", FakeMoE)
    monkeypatch.setattr(k3_module, "AscendKimiDecoderLayer", FakeDecoderLayer)
    monkeypatch.setattr(mtp_module, "AscendKimiMoE", FakeMoE)

    class FakePredictorLayer:
        mtp_block = FakeDecoderLayer(moe)

    monkeypatch.setattr(mtp_module, "AscendKimiK3MultiTokenPredictorLayer", FakePredictorLayer)
    model_cls = mtp_module.AscendKimiK3MTP if draft else k3_module.AscendKimiLinearForCausalLM
    model: Any = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=3,
        is_moe=True,
        num_experts=256,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        num_expert_group=8,
    )
    layers = (
        {"3": FakePredictorLayer(), "4": PPMissingLayer(), "5": FakePredictorLayer()}
        if draft
        else [PPMissingLayer(), FakeDecoderLayer(moe), FakeDecoderLayer(moe)]
    )
    model.model = SimpleNamespace(mtp_start_layer_idx=3, num_mtp_layers=3, layers=layers)
    model.set_moe_parameters()
    expert_load = torch.zeros((model.num_moe_layers, 264), dtype=torch.int32)
    logical_map = torch.zeros((model.num_moe_layers, 256, 2), dtype=torch.long)
    replica_count = torch.ones((model.num_moe_layers, 256), dtype=torch.long)

    model.set_eplb_state(expert_load, logical_map, replica_count)

    assert len(model.expert_weights) == expert_load.shape[0] == model.num_moe_layers == 2
    assert model.expert_weights == [weights, weights]
    assert [call.kwargs["moe_layer_idx"] for call in moe.experts.set_eplb_state.call_args_list] == [0, 1]


def test_update_physical_experts_metadata_propagates(monkeypatch):
    monkeypatch.setattr(k3_module, "AscendKimiMoE", FakeMoE)

    class FakeFusedMoE:
        def __init__(self):
            self.update_calls = 0

        def update_expert_map(self):
            self.update_calls += 1

    moe = FakeMoE()
    fused_moe = FakeFusedMoE()
    moe.experts = fused_moe

    model: Any = k3_module.AscendKimiLinearForCausalLM.__new__(k3_module.AscendKimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.num_logical_experts = 256
    model.num_physical_experts = 264
    model.num_local_physical_experts = 33
    model.num_redundant_experts = 8
    model.moe_mlp_layers = [moe]

    model.update_physical_experts_metadata(
        num_physical_experts=296,
        num_local_physical_experts=33,
    )

    assert model.num_redundant_experts == 40
    assert moe.n_physical_experts == 296
    assert moe.n_local_physical_experts == 33
    assert moe.n_redundant_experts == 40
    assert fused_moe.update_calls == 1


def test_conditional_generation_unwraps_language_model():
    wrapper = k3_module.AscendKimiK3ForConditionalGeneration.__new__(k3_module.AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)

    language_model = object()
    wrapper.language_model = language_model

    assert wrapper.get_language_model() is language_model


def test_moe_layer_predicate_matches_decoder_layer():
    config = SimpleNamespace(
        is_moe=True,
        num_experts=256,
        first_k_dense_replace=3,
        moe_layer_freq=1,
    )
    assert not k3_module.is_moe_layer_idx(config, 2)
    assert k3_module.is_moe_layer_idx(config, 3)

    config.moe_layer_freq = 2
    assert k3_module.is_moe_layer_idx(config, 4)
    assert not k3_module.is_moe_layer_idx(config, 5)

    config.num_experts = None
    assert not k3_module.is_moe_layer_idx(config, 5)

    config.is_moe = False
    assert not k3_module.is_moe_layer_idx(config, 5)


def test_update_physical_experts_metadata_rejects_resizing_local_weights():
    model: Any = k3_module.AscendKimiLinearForCausalLM.__new__(k3_module.AscendKimiLinearForCausalLM)
    nn.Module.__init__(model)
    model.num_local_physical_experts = 33
    with pytest.raises(AssertionError):
        model.update_physical_experts_metadata(296, 37)


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("reverse_weights", [False, True])
@pytest.mark.parametrize(
    "num_experts,num_redundant_experts,ep_size,ep_rank,use_v2,expected_experts",
    [
        (2, 0, 1, 0, False, [0, 1]),
        (2, 4, 1, 0, False, [0, 1, 0, 1, 0, 1]),
        (4, 0, 2, 1, True, [2, 3]),
        (4, 2, 2, 0, False, [0, 1, 2]),
        (4, 2, 2, 1, False, [3, 0, 1]),
        (4, 2, 2, 0, True, [0, 1, 2]),
        (4, 2, 2, 1, True, [2, 3, 0]),
        (5, 3, 2, 0, True, [0, 1, 2, 3]),
        (5, 3, 2, 1, True, [2, 3, 4, 0]),
        (2, 6, 2, 0, True, [0, 1, 0, 1]),
        (2, 6, 2, 1, True, [1, 0, 1, 0]),
    ],
)
def test_expert_loading_initializes_every_replica(
    monkeypatch,
    draft,
    packed,
    reverse_weights,
    num_experts,
    num_redundant_experts,
    ep_size,
    ep_rank,
    use_v2,
    expected_experts,
):
    # Upstream caches PP names by object ID; parameterized models can reuse IDs.
    monkeypatch.setattr(model_utils, "_model_to_pp_missing_layer_names", {})
    patch_eplb._patch_initial_expert_layout()
    model_cls = mtp_module.AscendKimiK3MTP if draft else k3_module.AscendKimiLinearModel
    model: Any = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        is_moe=True,
        num_experts=num_experts,
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        linear_attn_config={},
        q_lora_rank=None,
        is_linear_attn=False,
    )
    model.num_redundant_experts = model.n_redundant_experts = num_redundant_experts
    layer = nn.Module()
    layer.block_sparse_moe = nn.Module()
    layer.block_sparse_moe.experts = nn.Module()
    # Keep the real RoutedExperts type so the Ascend mapping patch discovers
    # the effective EP size instead of silently falling back to EP=1.
    routed = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(routed)
    routed.moe_config = SimpleNamespace(ep_size=ep_size)
    routed._use_v2_model_runner = use_v2
    layer.block_sparse_moe.experts.routed_experts = routed
    local_experts = (num_experts + num_redundant_experts) // ep_size
    local_start = ep_rank * local_experts

    def load_expert(param, weight, name, expert_id, shard_id):
        if not local_start <= expert_id < local_start + local_experts:
            return
        expert_id -= local_start
        if shard_id == "w2":
            param.data[expert_id].copy_(weight)
        else:
            param.data[expert_id, 0 if shard_id == "w1" else 1].copy_(weight)

    for name, shape in (
        ("w13_weight", (local_experts, 2, 1)),
        ("w2_weight", (local_experts, 1)),
        ("w13_weight_scale", (local_experts, 2, 1)),
    ):
        param = nn.Parameter(torch.full(shape, float("nan")), requires_grad=False)
        param.weight_loader = load_expert
        routed.register_parameter(name, param)
    if draft:
        model.model = nn.Module()
        model.model.layers = nn.ModuleDict({"2": nn.Module()})
        model.model.layers["2"].mtp_block = layer
        model.model.mtp_start_layer_idx = 2
        model.model.num_mtp_layers = 1
    else:
        model.layers = nn.ModuleList([layer, PPMissingLayer()])

    source = []
    for idx in (0, 1, 2):  # Main and MTP must skip each other's checkpoint layers.
        for expert in range(num_experts):
            for projection, value in (("w1", 1), ("w2", 2), ("w3", 3)):
                suffix = "weight_packed" if packed else "weight"
                name = f"model.layers.{idx}.block_sparse_moe.experts.{expert}.{projection}.{suffix}"
                name = f"language_model.{name}" if draft else name.removeprefix("model.")
                source.append((name, torch.tensor([10.0 * expert + value])))
                if projection != "w2":
                    source.append((name.replace(suffix, "weight_scale"), torch.tensor([0.1 * (expert + value)])))
    loaded = model.load_weights(iter(reversed(source) if reverse_weights else source))
    assert len(loaded) == 3
    for physical, logical in enumerate(expected_experts):
        torch.testing.assert_close(
            routed.w13_weight[physical], torch.tensor([[10.0 * logical + 1], [10.0 * logical + 3]])
        )
        torch.testing.assert_close(routed.w2_weight[physical], torch.tensor([10.0 * logical + 2]))
        torch.testing.assert_close(
            routed.w13_weight_scale[physical], torch.tensor([[0.1 * (logical + 1)], [0.1 * (logical + 3)]])
        )


def test_replica_loader_preserves_metadata_when_eplb_is_disabled():
    weight = ("unused.weight", torch.ones(1), {"loaded_shard_id": 2})
    model = nn.Module()
    model.named_parameters = MagicMock(side_effect=AssertionError("must reuse upstream directly"))
    assert list(k3_module.load_eplb_expert_weights(model, iter([weight]), 0))[0] is weight


def test_eplb_expert_remapping_preserves_weight_metadata():
    model = nn.Module()
    model.config = SimpleNamespace(is_moe=True, num_experts=4)
    model.experts = nn.Module()
    routed = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(routed)
    routed.moe_config = SimpleNamespace(ep_size=2)
    routed._use_v2_model_runner = True
    routed.register_parameter("w13_weight", nn.Parameter(torch.empty((3, 2, 1))))
    model.experts.routed_experts = routed
    metadata = {"loaded_shard_id": "w1", "sentinel": object()}
    weight = torch.ones(1)

    remapped = list(k3_module.load_eplb_expert_weights(model, [("experts.2.w1.weight", weight, metadata)], 2))

    assert [entry[0] for entry in remapped] == ["experts.2.w1.weight", "experts.3.w1.weight"]
    assert all(entry[1] is weight and entry[2] is metadata for entry in remapped)


@pytest.mark.parametrize("enable_eplb", [False, True])
def test_moe_uses_factory_expert_counts(monkeypatch, enable_eplb):
    config = SimpleNamespace(
        hidden_size=4,
        moe_intermediate_size=8,
        num_experts=256,
        num_experts_per_token=2,
        routed_expert_hidden_size=None,
        latent_moe_use_norm=False,
        routed_scaling_factor=1.0,
        num_shared_experts=None,
        hidden_act="silu",
        moe_renormalize=True,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
    )
    parallel = SimpleNamespace(enable_eplb=enable_eplb, eplb_config=SimpleNamespace(num_redundant_experts=8))
    runner = nn.Module()
    runner.moe_config = SimpleNamespace(
        num_logical_experts=256,
        num_experts=264 if enable_eplb else 256,
        num_local_experts=33 if enable_eplb else 256,
    )
    factory = MagicMock(return_value=runner)
    monkeypatch.setattr(k3_module, "FusedMoEFactory", factory)
    monkeypatch.setattr(
        k3_module, "GateLinear", lambda input_size, output_size, **kwargs: nn.Linear(input_size, output_size)
    )

    moe = k3_module.AscendKimiMoE(config=config, vllm_config=SimpleNamespace(parallel_config=parallel))

    assert factory.call_args.kwargs["enable_eplb"] == enable_eplb
    assert factory.call_args.kwargs["num_redundant_experts"] == (8 if enable_eplb else 0)
    assert moe.n_physical_experts == runner.moe_config.num_experts
    assert moe.n_local_physical_experts == runner.moe_config.num_local_experts
