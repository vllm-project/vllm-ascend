# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for preparing remote PLE rows with model state."""

from types import SimpleNamespace

import torch

from vllm_ascend.models.qwen4_exp import model_state, ple_rdma_embedding


class _RemoteEmbedding(ple_rdma_embedding.AscendRemotePLEEmbedding):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self._preparation_bound = False
        self.prepared = []
        self.cleared = 0

    def bind_preparation(self):
        self._preparation_bound = True

    def prepare_rows(self, ids):
        self.prepared.append(ids)

    def clear_prepared_rows(self):
        self.cleared += 1


def _state(monkeypatch):
    monkeypatch.setattr(model_state.Qwen4ExpModelState, "__init__", lambda *args, **kwargs: None)
    embedding = _RemoteEmbedding()
    layer = torch.nn.Module()
    layer.ngram_embedding = embedding
    layer.layer_multipliers = "multipliers"
    layer.ngram_heads_vocab_sizes = "vocab_sizes"
    layer.ngram_heads_offsets = "offsets"
    layer.eos_token_id = 9
    layer.heads_per_ngram = 2
    model = torch.nn.Module()
    model.layer = layer
    config = SimpleNamespace(additional_config={"shared_ple": {"backend": "rdma"}})
    return model_state.AscendQwen4ExpModelState(config, model, None, torch.device("cpu")), layer, embedding


def test_prepare_inputs_computes_and_prefetches_remote_ple_rows(monkeypatch):
    state, layer, embedding = _state(monkeypatch)
    batch = SimpleNamespace(num_reqs=1, input_ids=torch.tensor([[1, 2]]))
    model_inputs = {"query_start_loc": "query-start", "ngram_context": "context"}
    monkeypatch.setattr(model_state.Qwen4ExpModelState, "prepare_inputs", lambda *args: model_inputs)
    calls = []

    def make_ids(**kwargs):
        calls.append(kwargs)
        return torch.tensor([[4, 5]])

    monkeypatch.setattr("vllm.models.qwen4_exp.common.ops.ple.ple_ngram_ids", make_ids)
    result = state.prepare_inputs(batch, "request-states")

    assert result is model_inputs
    assert calls == [
        {
            "input_ids": batch.input_ids,
            "query_start_loc": "query-start",
            "ngram_context": "context",
            "layer_multipliers": "multipliers",
            "ngram_heads_vocab_sizes": "vocab_sizes",
            "ngram_heads_offsets": "offsets",
            "eos_token_id": 9,
            "heads_per_ngram": 2,
        }
    ]
    assert len(embedding.prepared) == 1
    assert torch.equal(embedding.prepared[0], torch.tensor([[4, 5]]))
    assert embedding.cleared == 0
    assert state._remote_ple_layers == [layer]


def test_empty_and_dummy_batches_clear_prepared_rows(monkeypatch):
    state, _, embedding = _state(monkeypatch)
    model_inputs = {"query_start_loc": torch.tensor([0])}
    monkeypatch.setattr(model_state.Qwen4ExpModelState, "prepare_inputs", lambda *args: model_inputs)
    monkeypatch.setattr(model_state.Qwen4ExpModelState, "prepare_dummy_inputs", lambda *args: {})
    monkeypatch.setattr(
        "vllm.models.qwen4_exp.common.ops.ple.ple_ngram_ids",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("empty batch must not compute PLE IDs")),
    )

    result = state.prepare_inputs(SimpleNamespace(num_reqs=0, input_ids=torch.empty((0, 0))), None)
    assert result is model_inputs
    state.prepare_dummy_inputs(1, 1)
    assert embedding.cleared == 2
    assert embedding.prepared == []
