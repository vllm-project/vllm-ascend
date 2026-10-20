# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime reconstruction must not overwrite retained or newly loaded parameters."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.models.deepseek_v4.model import DeepseekV4Model


@pytest.mark.parametrize("value", [0.125, 0.375])
def test_runtime_hook_preserves_checkpoint_parameter_values_and_storage(value):
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.gate = nn.Module()
    layer.mlp.gate.tid2eid = nn.Parameter(torch.tensor([[2, 0], [1, 2]]), requires_grad=False)
    for name in ("bias_vl", "e_score_correction_bias"):
        layer.mlp.gate.register_parameter(name, nn.Parameter(torch.full((3,), value)))
    layer.self_attn = nn.Module()
    layer.self_attn.attn_sink = nn.Parameter(torch.full((2,), value))
    layer.self_attn.rotary_emb = SimpleNamespace(reinitialize_runtime_params=MagicMock())
    layer.self_attn.reinitialize_runtime_params = MagicMock()
    model.layers = nn.ModuleList([layer])
    model.config = SimpleNamespace(vocab_size=2, num_experts_per_tok=2, n_routed_experts=3)
    norm = nn.Module()
    norm.has_weight = False
    norm.weight = torch.zeros(3)
    model.runtime_norm = norm
    before = {n: (p.detach().clone(), p.data_ptr()) for n, p in model.named_parameters()}

    model.reinitialize_runtime_params()
    model.reinitialize_runtime_params()

    for name, param in model.named_parameters():
        expected, ptr = before[name]
        assert torch.equal(param, expected), name
        assert param.data_ptr() == ptr, name
    assert torch.equal(norm.weight, torch.ones(3))
    assert layer.self_attn.rotary_emb.reinitialize_runtime_params.call_count == 2
    assert layer.self_attn.reinitialize_runtime_params.call_count == 2
    assert not hasattr(model, "_runtime_tensor_backup")
