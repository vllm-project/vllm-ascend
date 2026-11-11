# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.models.qwen4_exp import ple_rdma


def test_bf16_owner_loads_checkpoint_in_protocol_row_order(tmp_path, monkeypatch):
    prefix = "model.layers.1.ple.ple_embedding.ngram_embedding"
    weight = torch.arange(12, dtype=torch.float32).reshape(6, 2).to(torch.bfloat16)
    mapping = {}
    for index, data in enumerate((weight[:3], weight[3:])):
        name, filename = f"{prefix}.shard_{index}.weight", f"part-{index}.safetensors"
        save_file({name: data}, str(tmp_path / filename))
        mapping[name] = filename
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))

    class Engine:
        def __init__(self, endpoint, device, library_path):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(ple_rdma, "HixlEngine", Engine)
    owner = ple_rdma.PLERemoteOwner(tmp_path, prefix, 2, 2, "127.0.0.1:0", "127.0.0.1:12345")
    try:
        table = owner.tables["ple"]
        assert table.shape == (6, 4)
        assert torch.equal(table.reshape(-1).view(torch.bfloat16).reshape(6, 2), weight)
        _, metadata = ple_rdma.ple_metadata(tmp_path, prefix, 2, 2)
        assert owner.metadata == metadata
        assert metadata["tables"]["ple"] == {"rows": 6, "row_bytes": 4, "dtype": "BF16"}
    finally:
        owner.server_close()
    assert owner.engine.closed


def test_a5_ub_owner_uses_configured_engine_without_a2_alternate(monkeypatch):
    monkeypatch.setenv("ENGRAM_HIXL_FABRIC", "ub")
    owner = ple_rdma.PLERemoteOwner.__new__(ple_rdma.PLERemoteOwner)
    owner.engine = object()
    owner.engine_endpoint = "127.0.0.1:12345"
    owner.device = 0
    engine, endpoint = owner.engine_for_reader("127.0.0.1", 0)
    assert engine is owner.engine
    assert endpoint == owner.engine_endpoint


def test_rdma_fetch_rejects_graph_capture_before_network(monkeypatch):
    reader = ple_rdma.PLERemoteReader.__new__(ple_rdma.PLERemoteReader)
    calls = []
    reader._fetch_into = lambda *args, **kwargs: calls.append(True)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), raising=False)
    with pytest.raises(RuntimeError, match="outside graph capture"):
        reader.fetch_into(torch.tensor([0]), torch.empty((1, 2), dtype=torch.bfloat16), vocab_start=0, vocab_end=1)
    assert calls == []
