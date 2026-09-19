# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention
from vllm_ascend.models.glm5next.model import Glm5NextModel


class LinearWithBiasOutput(torch.nn.Linear):
    def forward(self, inputs):
        return super().forward(inputs), None


def make_layer(width):
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.head_dim = 128
    layer.f_b_proj = LinearWithBiasOutput(128, width, bias=False)
    layer.g_b_proj = LinearWithBiasOutput(128, width, bias=False)
    layer.register_buffer("_fg_b_weight", None, persistent=False)
    return layer


@pytest.mark.parametrize("num_tokens", [0, 1, 8, 64, 128, 129, 2048])
@pytest.mark.parametrize("width", [512, 1024])
def test_batched_projection_reads_strided_fused_input(num_tokens, width):
    torch.manual_seed(1024)
    layer = make_layer(width)
    projected = torch.randn(num_tokens, 3 * width + width // 128 + 256)
    fg_a = projected[:, -256:]
    fa, ga = fg_a.split(128, dim=-1)

    actual = layer._project_fg(fg_a)

    torch.testing.assert_close(actual[0], layer.f_b_proj(fa)[0])
    torch.testing.assert_close(actual[1], layer.g_b_proj(ga)[0])
    if num_tokens <= 128:
        assert not layer._fg_b_weight.requires_grad
    else:
        assert layer._fg_b_weight is None
    assert "_fg_b_weight" not in layer.state_dict()


def test_weight_reload_refreshes_captured_storage():
    layer = make_layer(512)
    model = Glm5NextModel.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(is_moe=False, is_linear_attn=True, num_nextn_predict_layers=0)
    model.add_module("linear_attention", layer)
    weights = [
        ("linear_attention.f_b_proj.weight", torch.ones_like(layer.f_b_proj.weight)),
        ("linear_attention.g_b_proj.weight", torch.ones_like(layer.g_b_proj.weight)),
    ]
    assert model.load_weights(weights) == {name for name, _ in weights}
    inputs = torch.ones(4, 256)
    layer._project_fg(inputs)
    address = layer._fg_b_weight.data_ptr()

    assert model.load_weights([(name, weight * factor) for (name, weight), factor in zip(weights, (2, 3))]) == {
        name for name, _ in weights
    }
    actual = layer._project_fg(inputs)

    assert layer._fg_b_weight.data_ptr() == address
    torch.testing.assert_close(actual[0], torch.full((4, 512), 256.0))
    torch.testing.assert_close(actual[1], torch.full((4, 512), 384.0))
