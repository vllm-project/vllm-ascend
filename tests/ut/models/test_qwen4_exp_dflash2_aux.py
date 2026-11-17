# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm.model_executor.models.interfaces import SupportsEagle3, supports_eagle3

import vllm_ascend.models.qwen4_exp.model as qwen4_exp


class _Layer:
    def __init__(self):
        self.mlp_hyper_connection = SimpleNamespace(combine=self._combine)
        self.combine_calls = 0

    def __call__(self, *, hidden_states, prev_block_output, prev_injection, **kwargs):
        return hidden_states, torch.ones_like(hidden_states), None

    def _combine(self, hidden_states, block_output, injection):
        self.combine_calls += 1
        return hidden_states + block_output


def _model(monkeypatch, aux_layers):
    model = object.__new__(qwen4_exp.AscendQwen4ExpModel)
    model.config = SimpleNamespace(hc_count=4, hidden_size=8)
    model.use_hc_sequence_parallel = False
    model.start_layer = 0
    model.end_layer = 1
    model.layers = [_Layer()]
    model._mtp_hidden_buffer = None
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda hidden, block, injection: (
            hidden + block,
            torch.zeros(hidden.shape[0], 8),
            None,
        )
    )
    monkeypatch.setattr(
        qwen4_exp,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    SupportsEagle3.set_aux_hidden_state_layers(SimpleNamespace(model=model), aux_layers)
    return model


def _forward(model):
    return model.forward(
        input_ids=None,
        positions=torch.zeros(2, dtype=torch.long),
        inputs_embeds=torch.zeros(2, 8),
    )


def test_dflash2_aux_outputs_are_completed_layer_stream_means(monkeypatch):
    model = _model(monkeypatch, (1,))

    sample_hidden, aux_hidden = _forward(model)

    assert sample_hidden.shape == (2, 8)
    assert len(aux_hidden) == 1
    assert aux_hidden[0].shape == (2, 8)
    torch.testing.assert_close(aux_hidden[0], torch.ones(2, 8))
    assert model.layers[0].combine_calls == 1


def test_default_qwen4_exp_output_is_unchanged_without_aux_layers(monkeypatch):
    model = _model(monkeypatch, ())

    output = _forward(model)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 8)
    assert model.layers[0].combine_calls == 0


def test_dflash2_aux_outputs_keep_hidden_width_with_hc_sequence_parallel(monkeypatch):
    model = _model(monkeypatch, (1,))
    model.use_hc_sequence_parallel = True
    monkeypatch.setattr(qwen4_exp.upstream_model, "sp_shard", lambda hidden: hidden)
    monkeypatch.setattr(
        qwen4_exp.upstream_model,
        "sp_all_gather",
        lambda hidden: torch.cat((hidden, hidden), dim=0),
    )

    sample_hidden, aux_hidden = _forward(model)

    assert sample_hidden.shape == (2, 8)
    assert len(aux_hidden) == 1
    assert aux_hidden[0].shape == (2, 8)
    torch.testing.assert_close(aux_hidden[0], torch.ones(2, 8))


def test_text_and_multimodal_architectures_provide_eagle3_protocol():
    assert supports_eagle3(qwen4_exp.AscendQwen4ExpForCausalLM)
    assert supports_eagle3(qwen4_exp.AscendQwen4ExpForConditionalGeneration)
