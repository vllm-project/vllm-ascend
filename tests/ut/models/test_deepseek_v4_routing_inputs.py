# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models.deepseek_v4 import model as deepseek_v4_module
from vllm_ascend.models.deepseek_v4.dspark import DeepseekV4DSparkModel
from vllm_ascend.models.deepseek_v4.model import DeepseekV4Model
from vllm_ascend.models.deepseek_v41 import model as deepseek_v41_module
from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkModel
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model


class _CaptureRoutingLayer:
    def __init__(self, index):
        self.layer_idx = index
        self.engram = None
        self.input_ids = []

    def __call__(self, positions, hidden_states, residual, llama_4_scaling=None, input_ids=None):
        self.input_ids.append(input_ids)
        return hidden_states, residual

    @staticmethod
    def hc_collapse(hidden_states, pre_mix):
        return hidden_states.mean(dim=1)


@pytest.mark.parametrize(
    "model_cls",
    [DeepseekV4Model, DeepseekV41Model, DeepseekV4DSparkModel, DeepseekV41DSparkModel],
)
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("needs_moe_input_ids", [False, True])
def test_model_prepares_routing_ids_once_per_forward(monkeypatch, model_cls, input_dtype, needs_moe_input_ids):
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(deepseek_v4_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(deepseek_v41_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(deepseek_v4_module, "get_pp_transport_tensors", lambda *_args: [])
    layers = [_CaptureRoutingLayer(i) for i in range(3)]
    is_draft = model_cls in (DeepseekV4DSparkModel, DeepseekV41DSparkModel)
    hidden_states = torch.randn(4, 8)
    embed = MagicMock(return_value=hidden_states)
    prepare_engram = MagicMock(return_value=({}, torch.empty(0, dtype=torch.bool)))
    model = SimpleNamespace(
        needs_moe_input_ids=needs_moe_input_ids,
        hc_mult=4,
        layers={str(i): layer for i, layer in enumerate(layers)} if is_draft else layers,
        start_layer=0,
        end_layer=len(layers),
        aux_hidden_state_layers=(),
        _mtp_hidden_buffer=None,
        embed_input_ids=embed,
        embed_tokens=embed,
        prepare_engram=prepare_engram,
        shared_attention_state=SimpleNamespace(reset=MagicMock()),
        hc_head=lambda hidden, *_args: hidden.mean(dim=1),
        hc_head_fn=None,
        hc_head_scale=None,
        hc_head_base=None,
        norm=lambda hidden: hidden,
    )
    positions = torch.arange(4)
    # A new draft substep must use the new IDs, including new placeholder rows.
    for step, values in enumerate(([-1, 0, 129259, 15], [13, -1, 129257, -1])):
        input_ids = torch.tensor(values, dtype=input_dtype)
        original = input_ids.clone()
        with patch.object(torch, "where", wraps=torch.where) as where:
            if is_draft:
                model_cls.forward(model, input_ids, positions)
            else:
                model_cls.forward(model, input_ids, positions, None)

        assert where.call_count == int(needs_moe_input_ids)
        routed_ids = layers[0].input_ids[step]
        for layer in layers:
            assert layer.input_ids[step] is routed_ids
        expected = torch.tensor([0 if value == -1 else value for value in values], dtype=input_dtype)
        if needs_moe_input_ids:
            torch.testing.assert_close(routed_ids, expected, rtol=0, atol=0)
            assert routed_ids is not input_ids
        else:
            assert routed_ids is input_ids
        torch.testing.assert_close(input_ids, original, rtol=0, atol=0)
        assert embed.call_args.args[0] is input_ids
        if model_cls is DeepseekV41Model:
            assert prepare_engram.call_args.args[0] is input_ids
