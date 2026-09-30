# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

from torch import nn
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_ascend.models import kimi_k3 as k3_module
from vllm_ascend.models import kimi_k3_mtp as mtp_module


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
    model = mtp_module.AscendKimiK3MTP.__new__(mtp_module.AscendKimiK3MTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_expert_group=8)
    model.model = SimpleNamespace(layers={"0": FakePredictorLayer(), "1": FakePredictorLayer()})

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
    model = mtp_module.AscendKimiK3MTP.__new__(mtp_module.AscendKimiK3MTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_expert_group=None)
    model.model = SimpleNamespace(layers={"0": FakePredictorLayer()})

    model.set_moe_parameters()

    assert model.num_moe_layers == 0
    assert model.num_physical_experts == 0
    assert not is_mixture_of_experts(model)


def test_main_model_moe_registration_metadata(monkeypatch):
    moe = FakeMoE()

    monkeypatch.setattr(k3_module, "AscendKimiMoE", FakeMoE)
    monkeypatch.setattr(k3_module, "AscendKimiDecoderLayer", FakeDecoderLayer)
    model = k3_module.AscendKimiLinearForCausalLM.__new__(k3_module.AscendKimiLinearForCausalLM)
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
            FakeDecoderLayer(PPMissingLayer()),
            FakeDecoderLayer(moe),
            FakeDecoderLayer(moe),
        ]
    )

    model.set_moe_parameters()

    # The layer count is global (EPLB state is sized over the whole model)
    # while ``moe_layers`` holds only this rank's layers.
    assert model.num_moe_layers == 3
    assert model.moe_layers == [moe.experts, moe.experts]
    assert model.num_redundant_experts == 8
    assert is_mixture_of_experts(model)


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

    model = k3_module.AscendKimiLinearForCausalLM.__new__(k3_module.AscendKimiLinearForCausalLM)
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
