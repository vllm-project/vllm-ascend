# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.models.interfaces import get_mixture_of_experts_model
from vllm.v1.worker.gpu.eplb_utils import EPLBController

from vllm_ascend.models.glm5next import model as glm_model


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3, 4), requires_grad=False)
        self.eplb_state = EplbLayerState()

    def get_expert_weights(self):
        return [self.weight]

    def set_eplb_state(self, **kwargs):
        self.eplb_state.set_layer_state(**kwargs)


def _moe():
    moe = glm_model.Glm5NextMoE.__new__(glm_model.Glm5NextMoE)
    nn.Module.__init__(moe)
    moe.n_logical_experts = moe.n_routed_experts = 4
    moe.n_physical_experts = 6
    moe.n_local_physical_experts = 3
    moe.n_shared_experts = 1
    moe.n_redundant_experts = 2
    moe.experts = _Experts()
    return moe


def _model(monkeypatch, layers):
    text_model = nn.Module()
    text_model.layers = nn.ModuleList(layers)
    text_model.make_empty_intermediate_tensors = Mock()
    config = SimpleNamespace(vocab_size=8, n_group=1)
    monkeypatch.setattr(glm_model, "Glm5NextModel", lambda **kwargs: text_model)
    monkeypatch.setattr(glm_model, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False))
    monkeypatch.setattr(glm_model, "LogitsProcessor", Mock())
    return glm_model.Glm5NextForCausalLM(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config), quant_config=None)
    )


@pytest.mark.parametrize("multimodal", [False, True])
def test_target_registration_initializes_router_load_view(monkeypatch, multimodal):
    moe = _moe()
    model = _model(monkeypatch, [SimpleDecoder(moe)])
    served_model = model
    if multimodal:
        served_model = glm_model.Glm5NextForConditionalGeneration.__new__(glm_model.Glm5NextForConditionalGeneration)
        nn.Module.__init__(served_model)
        served_model.language_model = model
        served_model._language_model_names = ["language_model"]
    load = torch.zeros(1, 6, dtype=torch.int64)
    mapping = torch.arange(4).reshape(1, 4, 1)
    replicas = torch.ones(1, 4, dtype=torch.int64)

    def add_model(registered_model, model_config):
        registered_model.set_eplb_state(load, mapping, replicas)
        for layer in registered_model.moe_layers:
            layer.eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
            layer.eplb_state.num_unpadded_tokens_tensors = [torch.tensor(1)]

    controller = EPLBController(SimpleNamespace(enable_eplb=True), torch.device("cpu"))
    controller.state = SimpleNamespace(add_model=Mock(side_effect=add_model))
    added = controller.maybe_register_model(served_model, SimpleNamespace(model="glm5-next"))

    BaseRouter._validate_eplb_state(SimpleNamespace(eplb_state=moe.experts.eplb_state))
    assert added
    assert controller._has_registered_models
    assert get_mixture_of_experts_model(model) is model
    assert get_mixture_of_experts_model(served_model) is model
    assert moe.experts.eplb_state.expert_load_view.data_ptr() == load[0].data_ptr()
    assert model.expert_weights[0][0] is moe.experts.weight


class SimpleDecoder(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


@pytest.mark.parametrize("has_moe", [True, False])
def test_moe_metadata_skips_dense_and_missing_pipeline_layers(monkeypatch, has_moe):
    moes = [_moe(), _moe()] if has_moe else []
    layers = [glm_model.PPMissingLayer(), SimpleDecoder(nn.Identity())]
    layers.extend(SimpleDecoder(moe) for moe in moes)
    model = _model(monkeypatch, layers)

    assert model.num_moe_layers == len(moes)
    assert model.moe_mlp_layers == moes
    assert model.moe_layers == [moe.experts for moe in moes]
    assert (get_mixture_of_experts_model(model) is model) == has_moe
    if has_moe:
        assert model.num_logical_experts == 4
        assert model.num_physical_experts == 6
        assert model.num_local_physical_experts == 3
        assert model.num_redundant_experts == 2
