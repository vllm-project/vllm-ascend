# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp import ple_bm_embedding


def test_shared_config_requires_explicit_bm_and_cpu_offload():
    config = SimpleNamespace(additional_config={}, engram_config=None)
    assert ple_bm_embedding.shared_ple_config(config) is None
    config.additional_config = {"shared_ple": {"backend": "bm", "root": "/pool", "checkpoint_prefix": "table"}}
    with pytest.raises(ValueError, match="cpu_offload"):
        ple_bm_embedding.shared_ple_config(config)
    config.engram_config = SimpleNamespace(cpu_offload=True)
    assert ple_bm_embedding.shared_ple_config(config)["root"] == "/pool"
    config.additional_config["shared_ple"]["backend"] = "unknown"
    with pytest.raises(ValueError, match="backend"):
        ple_bm_embedding.shared_ple_config(config)


def test_rdma_config_rejects_unvalidated_scheduling_modes():
    config = SimpleNamespace(
        additional_config={"shared_ple": {"backend": "rdma", "endpoint": "host:1", "checkpoint_prefix": "table"}},
        engram_config=SimpleNamespace(cpu_offload=True),
        speculative_config=object(),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
    )
    with pytest.raises(ValueError, match="speculative"):
        ple_bm_embedding.shared_ple_config(config)
    config.speculative_config = None
    assert ple_bm_embedding.shared_ple_config(config)["backend"] == "rdma"
    config.parallel_config.prefill_context_parallel_size = 2
    with pytest.raises(ValueError, match="PCP=DCP=1"):
        ple_bm_embedding.shared_ple_config(config)


def test_shared_embedding_preserves_etp_lookup_and_excludes_external_parameter(monkeypatch):
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.deepseek_v41.engram import common

    config = SimpleNamespace(
        additional_config={
            "shared_ple": {
                "backend": "bm",
                "root": "/pool",
                "checkpoint_prefix": "model.language_model.layers.1.ple.ple_embedding.ngram_embedding",
            }
        },
        engram_config=SimpleNamespace(cpu_offload=True),
        model_config=SimpleNamespace(model="/model", hf_text_config=SimpleNamespace(split_ngram_parts=2)),
    )
    monkeypatch.setattr(ple_bm_embedding, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)

    def init_base(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.embedding_dim = 2
        self.shard_indices = SimpleNamespace(org_vocab_start_index=2, org_vocab_end_index=4)
        self.tp_size = 2
        self.reduce_results = True
        self.parallel_group = SimpleNamespace(all_reduce=lambda tensor: tensor + 1)
        self.register_parameter("weight", torch.nn.Parameter(torch.empty((2, 2), device="meta")))

    class Reader(torch.nn.Module):
        ple_external_checkpoint_names = frozenset({"table.shard_0.weight"})

        def forward(self, ids):
            return ids.unsqueeze(-1).expand(*ids.shape, 2).to(torch.bfloat16)

    monkeypatch.setattr(ple_bm_embedding.Qwen4ExpPLEEmbedding, "__init__", init_base)
    events = []
    monkeypatch.setattr(common, "warmup_shared_engram_aicpu", lambda: events.append("warmup"), raising=False)
    monkeypatch.setattr(ple_bm_embedding, "open_bm_reader", lambda *args: events.append("attach") or Reader())
    monkeypatch.setattr(ple_bm_embedding.AscendSharedPLEEmbedding, "_get_dp_gather_slot", lambda self, n: (n, 0))
    monkeypatch.setattr(ple_bm_embedding.AscendSharedPLEEmbedding, "_gather_dp_ids", lambda self, ids, size: ids)
    monkeypatch.setattr(
        ple_bm_embedding.AscendSharedPLEEmbedding, "_select_embeddings", lambda self, value, n, offset: value
    )
    with pytest.raises(ValueError, match="does not match the model layer"):
        ple_bm_embedding.AscendSharedPLEEmbedding(prefix="model.layers.2.ple.ple_embedding.ngram_embedding")
    embedding = ple_bm_embedding.AscendSharedPLEEmbedding(prefix="model.layers.1.ple.ple_embedding.ngram_embedding")
    assert events == ["warmup", "attach"]
    assert "weight" not in dict(embedding.named_parameters())
    assert embedding.weight.device.type == "meta"
    assert embedding.reader.ple_external_checkpoint_names == frozenset({"table.shard_0.weight"})
    output = embedding(torch.tensor([[2, 3]]))
    assert torch.equal(output, torch.tensor([[[3, 3], [4, 4]]], dtype=torch.bfloat16))

    def fail_warmup():
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(common, "warmup_shared_engram_aicpu", fail_warmup)
    with pytest.raises(RuntimeError, match="warmup failed"):
        ple_bm_embedding.AscendSharedPLEEmbedding(prefix="model.layers.1.ple.ple_embedding.ngram_embedding")
    assert events == ["warmup", "attach"]
