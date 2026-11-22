# SPDX-License-Identifier: Apache-2.0
"""CPU regression for the synthetic PD loader's legacy and fused KDA convs."""

import ast
import hashlib
from pathlib import Path

import pytest
import torch


@pytest.fixture
def loader():
    # Load the actual test loader without importing the NPU E2E server stack.
    path = Path(__file__).resolve().parents[2] / "e2e/common/glm53flash_pd.py"
    nodes = ast.parse(path.read_text()).body
    selected = [node for node in nodes if getattr(node, "name", None) in ("global_weight", "FlashPDLoader")]
    for node in selected:
        node.decorator_list = []
    base = type("FlashDummyLoader", (), {"load_weights": lambda self, model, config: None})
    scope = {"torch": torch, "hashlib": hashlib, "FlashDummyLoader": base}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope["FlashPDLoader"]()


def make_model(fused, size, rank):
    layer = type("Glm5NextLinearAttention", (torch.nn.Module,), {})()
    layer.tp_size, layer.tp_rank = size, rank
    layer.num_heads, layer.head_dim, layer.hidden_size = 4, 2, 6
    channels = layer.num_heads * layer.head_dim

    def parameter(name, shape):
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.full(shape, float("nan"))))
        layer.add_module(name, module)

    parameter("in_proj_qkvbfg_a", ((3 * channels + layer.num_heads) // size + 2 * layer.head_dim, layer.hidden_size))
    parameter("f_b_proj", (channels // size, layer.head_dim))
    parameter("g_b_proj", (channels // size, layer.head_dim))
    parameter("o_proj", (layer.hidden_size, channels // size))
    if fused:
        parameter("conv1d", (3 * channels // size, 1, 4))
    else:
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            parameter(name, (channels // size, 1, 4))
    model = torch.nn.Module()
    model.add_module("kda", layer)
    return model


@pytest.mark.parametrize("fused", [False, True], ids=["separate", "fused"])
@pytest.mark.parametrize("size", [1, 2], ids=["pcp1", "pcp2"])
def test_conv_weights_match_unsharded_legacy_model(loader, fused, size):
    baseline = make_model(False, 1, 0)
    loader.load_weights(baseline, None)
    local_convs = []
    for rank in range(size):
        model = make_model(fused, size, rank)
        pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
        loader.load_weights(model, None)
        for name, value in model.named_parameters():
            assert torch.isfinite(value).all()
            assert value.data_ptr() == pointers[name]
        if fused:
            assert not hasattr(model.kda, "q_conv1d")
            local_convs.append(model.kda.conv1d.weight.chunk(3, dim=0))
        else:
            local_convs.append(tuple(getattr(model.kda, f"{part}_conv1d").weight for part in ("q", "k", "v")))

    for index, part in enumerate(("q", "k", "v")):
        reconstructed = torch.cat([conv[index] for conv in local_convs], dim=0)
        expected = getattr(baseline.kda, f"{part}_conv1d").weight
        torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
