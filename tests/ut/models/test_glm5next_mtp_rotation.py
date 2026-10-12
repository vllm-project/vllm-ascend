# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

import vllm_ascend.models.glm5next.mtp as mtp_module
import vllm_ascend.utils as ascend_utils
from vllm_ascend.models.glm5next.mtp import Glm5NextMTP


def make_mtp(use_rotation):
    model = Glm5NextMTP.__new__(Glm5NextMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=1, n_routed_experts=0)
    model.is_rot_weight_used = use_rotation
    if use_rotation:
        model.rot = nn.Linear(4, 4, bias=False)
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = 45
    model.model.num_mtp_layers = 1
    layer = nn.Module()
    layer.enorm = nn.LayerNorm(4, bias=False)
    model.model.layers = nn.ModuleDict({"45": layer})
    return model


@pytest.mark.parametrize("use_rotation", [False, True])
def test_mtp_loads_exported_top_level_rotation(use_rotation):
    model = make_mtp(use_rotation)
    rotation = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    weights = [
        ("rot.weight", rotation),
        ("model.language_model.layers.45.enorm.weight", torch.full((4,), 2.0)),
    ]
    with patch.object(mtp_module, "fused_moe_make_expert_params_mapping", return_value=[]):
        loaded = model.load_weights(weights)
    assert loaded == {"model.layers.45.enorm.weight"} | ({"rot.weight"} if use_rotation else set())
    if use_rotation:
        torch.testing.assert_close(model.rot.weight, rotation)
    assert not model.has_own_lm_head


def test_mtp_rejects_missing_required_rotation():
    model = make_mtp(True)
    weights = [("model.language_model.layers.45.enorm.weight", torch.ones(4))]
    with (
        patch.object(mtp_module, "fused_moe_make_expert_params_mapping", return_value=[]),
        pytest.raises(ValueError, match="requires rot.weight"),
    ):
        model.load_weights(weights)


@pytest.mark.parametrize("use_rotation", [False, True])
def test_mtp_transforms_target_and_recycled_hidden_states(use_rotation):
    model = make_mtp(use_rotation)
    rotation = torch.tensor([[1.0, 0.25, 0.0, 0.0], [0.25, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.5]])
    if use_rotation:
        model.rot.weight.data.copy_(rotation)

    class Predictor(nn.Module):
        def forward(self, input_ids, positions, hidden_states, inputs_embeds, spec_step_idx):
            return hidden_states, hidden_states + 0.125

    model.model = Predictor()
    hidden = torch.tensor([[0.5, -1.0, 2.0, 3.0], [2.0, 1.0, -0.5, 4.0]])
    for step in range(3):
        original = hidden.clone()
        logits_hidden, feedback_hidden = model(
            torch.tensor([2, 3]), torch.tensor([10 + step, 20 + step]), hidden, spec_step_idx=step
        )
        expected = original @ rotation.T if use_rotation else original
        torch.testing.assert_close(logits_hidden, expected)
        torch.testing.assert_close(feedback_hidden, expected + 0.125)
        torch.testing.assert_close(hidden, original)
        hidden = feedback_hidden


@pytest.mark.parametrize(
    ("quant_description", "expected"),
    [(None, False), ({}, False), ({"is_rot_used": False}, False), ({"is_rot_used": True}, True)],
)
def test_rotation_is_gated_by_checkpoint(quant_description, expected):
    config = SimpleNamespace(hidden_size=4)
    quant_config = SimpleNamespace(quant_description=quant_description) if quant_description is not None else None
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=config), quant_config=quant_config)
    with (
        patch.object(mtp_module, "Glm5NextMultiTokenPredictor", return_value=nn.Module()),
        patch.object(Glm5NextMTP, "set_moe_parameters"),
        patch.object(ascend_utils, "_IS_ROT_WEIGHT_USED", None),
        patch.object(mtp_module, "is_rot_weight_used", wraps=ascend_utils.is_rot_weight_used) as read_flag,
    ):
        model = Glm5NextMTP(vllm_config=vllm_config)
    assert model.is_rot_weight_used is expected
    assert hasattr(model, "rot") is expected
    read_flag.assert_called_once_with(vllm_config)
