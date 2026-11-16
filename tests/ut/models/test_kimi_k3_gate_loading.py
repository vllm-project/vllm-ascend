# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_ascend.models.kimi_k3.kimi_k3 import AscendKimiLinearModel


def make_linear(cls, output_sizes, tp_size=1, tp_rank=0):
    # Exercise the real parameter loaders without initializing device kernels
    # or a distributed process group. Tensor contents verify TP slicing.
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.prefix = "layers.0.self_attn.g_proj" if cls is ColumnParallelLinear else "layers.0.self_attn.fused_qkv_a_proj"
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.output_sizes = output_sizes
    return layer


def add_parameter(layer, name, shape, dtype=torch.float32, output_dim=0):
    param = nn.Parameter(torch.full(shape, -1, dtype=dtype), requires_grad=False)
    if output_dim is not None:
        param.output_dim = output_dim
    param.weight_loader = layer.weight_loader
    layer.register_parameter(name, param)
    return param


def make_model(tp_size=1, tp_rank=0, fused_gate=False):
    model = AscendKimiLinearModel.__new__(AscendKimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        is_moe=False,
        is_linear_attn=True,
        linear_attn_config={"use_full_rank_gate": True},
        q_lora_rank=1536,
        mla_use_output_gate=True,
        num_hidden_layers=2,
        num_nextn_predict_layers=0,
    )
    model.n_redundant_experts = 0
    layer = nn.Module()
    layer.self_attn = nn.Module()
    sizes = [4, 2, 32] if fused_gate else [4, 2]
    fused = make_linear(MergedColumnParallelLinear, sizes)
    add_parameter(fused, "weight", (sum(sizes), 4))
    add_parameter(fused, "weight_scale", (sum(sizes), 1))
    layer.self_attn.fused_qkv_a_proj = fused
    if not fused_gate:
        layer.self_attn.g_proj = make_linear(ColumnParallelLinear, [32], tp_size, tp_rank)
    model.layers = nn.ModuleList([layer, PPMissingLayer()])
    return model


TP_RANKS = [(size, rank) for size in (1, 8, 16) for rank in range(size)]


@pytest.mark.parametrize("tp_size,tp_rank", TP_RANKS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8, torch.float8_e4m3fn])
def test_standalone_gate_loads_weights_and_quant_parameters(tp_size, tp_rank, dtype):
    model = make_model(tp_size, tp_rank)
    gate = model.layers[0].self_attn.g_proj
    rows = 32 // tp_size
    weights = []
    expected = {}
    for name, width, param_dtype in [
        ("weight", 4, dtype),
        ("weight_scale", 1, torch.float32),
        ("weight_offset", 1, torch.float32),
        ("bias", 1, torch.float32),
    ]:
        add_parameter(gate, name, (rows, width), param_dtype)
        source = ((torch.arange(32 * width).reshape(32, width) % 7) + 1).to(param_dtype)
        weights.append((f"layers.0.self_attn.g_proj.{name}", source))
        expected[name] = source.narrow(0, tp_rank * rows, rows)
    for name in ("input_scale", "input_offset"):
        add_parameter(gate, name, (1,), output_dim=None)
        weights.append((f"layers.0.self_attn.g_proj.{name}", torch.tensor(2.0)))
        expected[name] = torch.tensor([2.0])

    loaded = model.load_weights(iter(weights))

    assert loaded == {name for name, _ in weights}
    for name, value in expected.items():
        torch.testing.assert_close(getattr(gate, name).float(), value.float(), rtol=0, atol=0)
    fused = model.layers[0].self_attn.fused_qkv_a_proj
    assert torch.all(fused.weight == -1)
    assert torch.all(fused.weight_scale == -1)


def test_standalone_gate_preserves_loader_kwargs_and_default_loader():
    model = make_model()
    gate = model.layers[0].self_attn.g_proj
    param = add_parameter(gate, "weight", (32, 4))
    seen = []

    def weight_loader(param, value, *, marker):
        seen.append(marker)
        gate.weight_loader(param, value)

    param.weight_loader = weight_loader
    gate.register_parameter("input_scale", nn.Parameter(torch.zeros(1), requires_grad=False))
    kwargs = {"marker": "quant-loader-metadata"}
    loaded = model.load_weights(
        iter(
            [
                ("layers.0.self_attn.g_proj.weight", torch.ones(32, 4), kwargs),
                ("layers.0.self_attn.g_proj.input_scale", torch.tensor([3.0])),
            ]
        )
    )
    assert seen == ["quant-loader-metadata"]
    assert kwargs == {"marker": "quant-loader-metadata"}
    assert torch.all(param == 1)
    assert gate.input_scale.item() == 3
    assert loaded == {"layers.0.self_attn.g_proj.weight", "layers.0.self_attn.g_proj.input_scale"}


def test_gate_loading_retains_pp_skipping_and_qkv_loading():
    model = make_model()
    add_parameter(model.layers[0].self_attn.g_proj, "weight", (32, 4))
    loaded = model.load_weights(
        iter(
            [
                ("layers.1.self_attn.g_proj.weight", torch.ones(32, 4)),
                ("layers.1.self_attn.g_proj.weight_scale", torch.ones(32, 1)),
                ("layers.0.self_attn.q_a_proj.weight", torch.full((4, 4), 2.0)),
                ("layers.0.self_attn.kv_a_proj_with_mqa.weight", torch.full((2, 4), 3.0)),
                ("layers.0.self_attn.g_proj.weight", torch.full((32, 4), 4.0)),
            ]
        )
    )
    fused = model.layers[0].self_attn.fused_qkv_a_proj.weight
    torch.testing.assert_close(fused, torch.cat([torch.full((4, 4), 2.0), torch.full((2, 4), 3.0)]))
    assert torch.all(model.layers[0].self_attn.g_proj.weight == 4)
    assert loaded == {"layers.0.self_attn.fused_qkv_a_proj.weight", "layers.0.self_attn.g_proj.weight"}


def test_fused_mla_gate_keeps_upstream_shard_two_loading():
    model = make_model(fused_gate=True)
    loaded = model.load_weights(iter([("layers.0.self_attn.g_proj.weight", torch.full((32, 4), 5.0))]))
    fused = model.layers[0].self_attn.fused_qkv_a_proj.weight
    assert torch.all(fused[:6] == -1)
    assert torch.all(fused[6:] == 5)
    assert loaded == {"layers.0.self_attn.fused_qkv_a_proj.weight"}
