# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-model reference writing stays sharded and owns iterator outputs."""

import json

import torch
from safetensors.torch import load_file

from tests.e2e.common.rlhf.weight_transfer import weight_source


def test_reference_shards_preserve_reused_bf16_source(tmp_path, monkeypatch):
    monkeypatch.setattr(weight_source, "REFERENCE_SHARD_SIZE_BYTES", 8)

    def weights():
        tensor = torch.empty(3, dtype=torch.bfloat16)
        for index in range(3):
            yield str(index), tensor.fill_(index)

    weight_source._write_reference_shards(weights(), tmp_path)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 18
    assert len(set(index["weight_map"].values())) == 3
    for name, filename in index["weight_map"].items():
        tensor = load_file(tmp_path / filename)[name]
        assert torch.equal(tensor, torch.full((3,), int(name), dtype=torch.bfloat16))


def test_reference_shards_accept_one_oversized_parameter_and_empty_tensor(tmp_path, monkeypatch):
    monkeypatch.setattr(weight_source, "REFERENCE_SHARD_SIZE_BYTES", 8)
    payload = [("large", torch.arange(9, dtype=torch.bfloat16)), ("empty", torch.empty(0, dtype=torch.bfloat16))]
    weight_source._write_reference_shards(iter(payload), tmp_path)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 18
    for name, expected in payload:
        tensor = load_file(tmp_path / index["weight_map"][name])[name]
        assert torch.equal(tensor, expected)


def test_engram_tables_stay_fixed_while_other_weights_alternate():
    source = object.__new__(weight_source.DeepseekV41WeightSource)
    source._device = torch.device("cpu")
    source._num_experts = 8
    source._engram_shapes = {"layers.1.engram.embed.weight": (3, 4)}
    table = weight_source.TensorSpec("layers.1.engram.embed.weight", torch.bfloat16, (3, 4))
    projection = weight_source.TensorSpec("layers.1.attn.weight", torch.bfloat16, (3, 4))
    source.seed = weight_source.FIXED_WEIGHT_SEED
    table_a, projection_a = source._make_tensor(table), source._make_tensor(projection)
    source.seed += 1
    assert torch.equal(table_a, source._make_tensor(table))
    assert not torch.equal(projection_a, source._make_tensor(projection))
