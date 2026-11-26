# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp import model


@pytest.mark.parametrize(
    ("dp", "pcp", "expected"),
    [(1, 1, ["base"]), (4, 1, ["base"]), (1, 2, ["PCP"]), (2, 2, ["PCP"]), (4, 2, ["PCP"]), (1, 4, ["PCP"])],
)
def test_ple_construction_and_registration(monkeypatch, dp, pcp, expected):
    calls = []
    registry: dict[str, object] = {}
    text = SimpleNamespace(linear_num_key_heads=16, linear_num_value_heads=24)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text),
        parallel_config=SimpleNamespace(
            use_ubatching=False,
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(static_forward_context=registry),
    )
    prefix = "model.layers.2.ple"

    class RecordingPLE:
        def __init__(self, cfg, vllm_config, *, prefix, layer_idx, ple_dense_layer_id, use_sequence_parallel):
            assert cfg is text and vllm_config is config
            assert (layer_idx, ple_dense_layer_id, use_sequence_parallel) == (2, 1, False)
            assert prefix not in registry
            self.prefix = prefix
            self.ple_dense_layer_id = ple_dense_layer_id
            registry[prefix] = self
            calls.append("base" if type(self) is RecordingPLE else "PCP")

    class RecordingPCPPLE(RecordingPLE):
        pass

    def init_decoder(self, *, vllm_config, prefix):
        torch.nn.Module.__init__(self)
        self.config = text
        ple = model.upstream_model.Qwen4ExpPLELayer(
            text, vllm_config=vllm_config, prefix=prefix, layer_idx=2, ple_dense_layer_id=1, use_sequence_parallel=False
        )
        self.layers = [SimpleNamespace(ple=ple, layer_idx=2, use_hc_sequence_parallel=False), SimpleNamespace(ple=None)]
        self.start_layer, self.end_layer = 0, len(self.layers)

    monkeypatch.setattr(model, "Qwen4ExpPLELayer", RecordingPLE)
    monkeypatch.setattr(model, "AscendPCPPLELayer", RecordingPCPPLE)
    monkeypatch.setattr(model.AscendQwen4ExpModel.__bases__[0], "__init__", init_decoder)
    backbone = model.AscendQwen4ExpModel(vllm_config=config, prefix=prefix)
    assert calls == expected
    assert registry[prefix] is backbone.layers[0].ple
    assert backbone.layers[1].ple is None
