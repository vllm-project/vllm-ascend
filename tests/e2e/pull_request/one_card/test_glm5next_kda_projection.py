# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention


class LinearWithBiasOutput(torch.nn.Linear):
    def forward(self, inputs):
        return super().forward(inputs), None


def make_layer(width):
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.head_dim = 128
    for name in ("f_b_proj", "g_b_proj"):
        setattr(layer, name, LinearWithBiasOutput(128, width, bias=False, device="npu", dtype=torch.bfloat16))
    layer.register_buffer("_fg_b_weight", None, persistent=False)
    return layer


@torch.inference_mode()
@pytest.mark.parametrize("tokens", [0, 1, 7, 31, 64, 127, 128, 129, 255, 256, 257, 512, 2048])
@pytest.mark.parametrize("width", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("use_graph", [False, True])
def test_kda_projection_shape_boundaries(tokens, width, use_graph):
    """Use TP8/4/2/1 widths and token counts around the BMM fallback boundary."""
    torch.manual_seed(1024)
    layer = make_layer(width)
    projected = torch.randn(tokens, 3 * width + width // 128 + 256, device="npu", dtype=torch.bfloat16)
    fg_a = projected[:, -256:]
    if use_graph and tokens:
        layer._project_fg(fg_a)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = layer._project_fg(fg_a)
    for _ in range(2):
        projected.normal_()
        if use_graph and tokens:
            graph.replay()
        else:
            actual = layer._project_fg(fg_a)
        fa, ga = fg_a.split(128, dim=-1)
        assert torch.equal(actual[0], layer.f_b_proj(fa)[0])
        assert torch.equal(actual[1], layer.g_b_proj(ga)[0])


@torch.inference_mode()
@pytest.mark.parametrize("width", [1024, 2048, 4096, 8192])
def test_kda_projection_reload_after_graph_capture(width):
    layer = make_layer(width)
    inputs = torch.randn(128, 256, device="npu", dtype=torch.bfloat16)
    layer._project_fg(inputs)
    address = layer._fg_b_weight.data_ptr()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = layer._project_fg(inputs)
    for _ in range(2):
        layer.f_b_proj.weight.normal_()
        layer.g_b_proj.weight.normal_()
        layer.pack_fg_projection_weights()
        assert layer._fg_b_weight.data_ptr() == address
        graph.replay()
        fa, ga = inputs.split(128, dim=-1)
        assert torch.equal(actual[0], layer.f_b_proj(fa)[0])
        assert torch.equal(actual[1], layer.g_b_proj(ga)[0])
