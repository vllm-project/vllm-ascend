# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp import ple_rdma_embedding
from vllm_ascend.models.qwen4_exp.model_state import AscendQwen4ExpModelState, Qwen4ExpModelState


def test_remote_embedding_requires_state_binding_and_uses_prepared_rows(monkeypatch):
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    config = SimpleNamespace(
        additional_config={"shared_ple": {"backend": "rdma", "endpoint": "127.0.0.1:1", "checkpoint_prefix": prefix}},
        engram_config=SimpleNamespace(cpu_offload=True),
        speculative_config=None,
        model_config=SimpleNamespace(model="/model", hf_text_config=SimpleNamespace(split_ngram_parts=1)),
    )
    monkeypatch.setattr(ple_rdma_embedding, "get_current_vllm_config", lambda: config)

    def init_base(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.embedding_dim = 2
        self.etp_data_parallel_size = 1
        self.tp_size = 1
        self.reduce_results = True
        self.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=10)
        self.register_parameter("weight", torch.nn.Parameter(torch.empty((10, 2), device="meta")))

    class Reader:
        ple_external_checkpoint_names = frozenset({"table.shard_0.weight"})

        def fetch_into(self, ids, output, *, vocab_start, vocab_end):
            output.copy_(ids.unsqueeze(-1).expand_as(output))

        def close(self):
            pass

    monkeypatch.setattr(ple_rdma_embedding.Qwen4ExpPLEEmbedding, "__init__", init_base)
    monkeypatch.setattr(ple_rdma_embedding, "PLERemoteReader", lambda *args, **kwargs: Reader())
    embedding = ple_rdma_embedding.AscendRemotePLEEmbedding(
        prefix="model.layers.1.ple.ple_embedding.ngram_embedding", max_total_tokens=2, num_ngram_heads=2
    )
    ids = torch.tensor([[1, 2]], device="npu")
    with pytest.raises(RuntimeError, match="model-state"):
        embedding(ids)
    embedding.bind_preparation()
    embedding.prepare_rows(ids)
    assert torch.equal(embedding(ids).cpu(), torch.tensor([[[1, 1], [2, 2]]], dtype=torch.bfloat16))
    embedding.clear_prepared_rows()
    assert not embedding(ids).count_nonzero()
    embedding.close_host_offload()
    with pytest.raises(RuntimeError, match="model-state"):
        embedding(ids)


def test_model_state_binds_each_remote_ngram_layer(monkeypatch):
    monkeypatch.setattr(Qwen4ExpModelState, "__init__", lambda *args, **kwargs: None)
    embedding = ple_rdma_embedding.AscendRemotePLEEmbedding.__new__(ple_rdma_embedding.AscendRemotePLEEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding._preparation_bound = False
    layer = torch.nn.Module()
    layer.ngram_embedding = embedding
    model = torch.nn.Module()
    model.layer = layer
    config = SimpleNamespace(additional_config={"shared_ple": {"backend": "rdma"}})
    state = AscendQwen4ExpModelState(config, model, None, torch.device("cpu"))
    assert state._remote_ple_layers == [layer]
    assert embedding._preparation_bound
