# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.models.glm5next.kda import KDA_BATCHED_GATE_MAX_WIDTH, Glm5NextLinearAttention


def make_layer(width):
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.head_dim = 128
    layer.fg_b_proj = None
    with (
        torch.device("npu"),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        if width <= KDA_BATCHED_GATE_MAX_WIDTH:
            layer.fg_b_proj = torch.nn.Linear(128, 2 * width, bias=False, dtype=torch.bfloat16)
        else:
            layer.f_b_proj = ColumnParallelLinear(128, width, bias=False, disable_tp=True, params_dtype=torch.bfloat16)
            layer.g_b_proj = ColumnParallelLinear(128, width, bias=False, disable_tp=True, params_dtype=torch.bfloat16)
            layer.f_b_proj.weight.data.normal_()
            layer.g_b_proj.weight.data.normal_()
    return layer


def projection_weights(layer):
    if layer.fg_b_proj is None:
        return layer.f_b_proj.weight, layer.g_b_proj.weight
    return layer.fg_b_proj.weight.chunk(2)


@torch.inference_mode()
@pytest.mark.parametrize("tokens", [0, 1, 2, 4, 7, 8, 16, 31, 64, 127, 128, 129, 255, 256, 257, 512, 2048])
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
        f_weight, g_weight = projection_weights(layer)
        assert torch.equal(actual[0], torch.nn.functional.linear(fa, f_weight))
        assert torch.equal(actual[1], torch.nn.functional.linear(ga, g_weight))


@torch.inference_mode()
@pytest.mark.parametrize("width", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("tokens", [128, 129])
def test_kda_projection_reload_after_graph_capture(width, tokens):
    layer = make_layer(width)
    inputs = torch.randn(tokens, 256, device="npu", dtype=torch.bfloat16)
    layer._project_fg(inputs)
    addresses = [p.data_ptr() for p in layer.parameters()]
    runner = SimpleNamespace(
        get_model=lambda: layer,
        lora_config=None,
        model_config=SimpleNamespace(quantization=None),
        reset_lora_state=lambda: None,
        reset_encoder_cache=lambda: None,
        reset_mm_cache=lambda: None,
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = layer._project_fg(inputs)
    for _ in range(2):
        new_weights = [(name, torch.randn_like(param)) for name, param in layer.named_parameters()]
        GPUModelRunner.reload_weights(runner, weights_iterator=new_weights, is_checkpoint_format=False)
        assert [p.data_ptr() for p in layer.parameters()] == addresses
        graph.replay()
        fa, ga = inputs.split(128, dim=-1)
        if layer.fg_b_proj is None:
            f_weight, g_weight = (value for _, value in new_weights)
        else:
            f_weight, g_weight = new_weights[0][1].chunk(2)
        assert torch.equal(actual[0], torch.nn.functional.linear(fa, f_weight))
        assert torch.equal(actual[1], torch.nn.functional.linear(ga, g_weight))


@torch.inference_mode()
@pytest.mark.parametrize("tokens", [1, 4, 128, 129])
def test_wide_projection_preserves_nz_weights(tokens):
    layer = make_layer(8192)
    inputs = torch.randn(tokens, 256, device="npu", dtype=torch.bfloat16)
    f_weight, g_weight = (weight.clone() for weight in projection_weights(layer))
    torch.npu.config.allow_internal_format = True
    for projection in (layer.f_b_proj, layer.g_b_proj):
        projection.weight.data = torch_npu.npu_format_cast(projection.weight.data, 29)
        assert torch_npu.get_npu_format(projection.weight) == 29
    layer._project_fg(inputs)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = layer._project_fg(inputs)
    graph.replay()
    fa, ga = inputs.split(128, dim=-1)
    assert torch.equal(actual[0], torch.nn.functional.linear(fa, f_weight))
    assert torch.equal(actual[1], torch.nn.functional.linear(ga, g_weight))
