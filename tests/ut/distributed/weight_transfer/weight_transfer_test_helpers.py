# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small mocked HF model helpers shared by checkpoint-source unit tests."""

from unittest.mock import patch

import torch
from transformers import PretrainedConfig

from tests.e2e.pull_request.rlhf.weight_transfer import utils


def checkpoint_source(model, case):
    # Only supply num_experts when the case does not override an expert field.
    # Otherwise an incorrect fallback could pass by reading this default.
    overrides = (
        case.hf_overrides.get(case.meta_config_attribute, {}) if case.meta_config_attribute else case.hf_overrides
    )
    meta_config = PretrainedConfig()
    if not any(name in overrides for name in ("n_routed_experts", "num_local_experts", "num_experts")):
        meta_config.num_experts = 8
    config = (
        PretrainedConfig(**{case.meta_config_attribute: meta_config}) if case.meta_config_attribute else meta_config
    )
    with (
        patch.object(utils.AutoConfig, "from_pretrained", return_value=config),
        patch.object(utils.AutoModelForCausalLM, "from_config", return_value=model),
    ):
        source = utils.FixedRandomWeightSource(case, torch.device("cpu"))
    assert source._num_experts == 8
    return source


def router_meta_model(layer_index):
    model = torch.nn.Module()
    parent = model
    for component in ("model", "layers", str(layer_index), "mlp"):
        child = torch.nn.Module()
        parent.add_module(component, child)
        parent = child
    gate = torch.nn.Linear(16, 8, bias=False, device="meta")
    parent.add_module("gate", gate)
    return model, gate
