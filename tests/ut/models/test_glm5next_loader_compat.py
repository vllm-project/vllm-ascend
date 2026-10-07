# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real upstream weight mapping without loading model weights on NPU."""

import pytest
import torch
from torch import nn

from vllm_ascend.models.glm5next.model import Glm5NextForConditionalGeneration
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig


def _add_parameter(module, name):
    parts = name.split(".")
    for part in parts[:-1]:
        if part not in module._modules:
            module.add_module(part, nn.Module())
        module = module._modules[part]
    module.register_parameter(parts[-1], nn.Parameter(torch.zeros(2), requires_grad=False))


@pytest.mark.parametrize("quantized", [False, True])
def test_real_loader_drops_only_top_level_rotation_and_preserves_mapping(quantized):
    model_cls = Glm5NextForConditionalGeneration
    model = nn.Module()
    model.hf_to_vllm_mapper = model_cls.hf_to_vllm_mapper
    if quantized:
        model.quant_config = AscendModelSlimConfig({})
    names = [
        "lm_head.weight",
        "model.layers.0.rot.weight",
        "model.language_model.layers.0.self_attn.forget_gate.f_b_proj.weight",
        "model.visual.proj.weight",
    ]
    mapped = [model.hf_to_vllm_mapper._map_name(name) for name in names]
    for name in mapped:
        _add_parameter(model, name)
    weights = [(name, torch.full((2,), float(index + 1))) for index, name in enumerate(names)]
    weights += [("rot.weight", torch.full((2,), 99.0)), ("rot.weight.extra", torch.full((2,), 98.0))]
    weights.append(("rot.other", torch.full((2,), 97.0)))
    loaded = model_cls.load_weights(model, weights)
    assert loaded == set(mapped)
    parameters = dict(model.named_parameters())
    for index, name in enumerate(mapped):
        torch.testing.assert_close(parameters[name], torch.full((2,), float(index + 1)))
    # Skipping is local to this load; the class mapper remains unchanged.
    assert model_cls.hf_to_vllm_mapper._map_name("rot.weight") == "rot.weight"
