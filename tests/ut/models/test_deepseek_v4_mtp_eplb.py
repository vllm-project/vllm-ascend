# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

from torch import nn

from vllm_ascend.models.deepseek_v4 import mtp as mtp_module


def test_mtp_eplb_layer_count_follows_local_pp_layers(monkeypatch):
    class FakeMoE:
        experts = object()

    class FakeDecoder:
        mlp = FakeMoE()

    class FakePredictorLayer:
        mtp_block = FakeDecoder()

    monkeypatch.setattr(mtp_module, "DeepSeekMultiTokenPredictorLayer", FakePredictorLayer)
    monkeypatch.setattr(mtp_module, "DeepseekV4DecoderLayer", FakeDecoder)
    monkeypatch.setattr(mtp_module, "DeepseekV4MoE", FakeMoE)
    model = mtp_module.DeepSeekV4MTP.__new__(mtp_module.DeepSeekV4MTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_nextn_predict_layers=4)
    model.model = SimpleNamespace(layers={"0": FakePredictorLayer(), "1": FakePredictorLayer()})
    model.extract_moe_parameters = MagicMock()

    model.set_moe_parameters()

    assert model.num_moe_layers == 2
    assert len(model.moe_layers) == 2
