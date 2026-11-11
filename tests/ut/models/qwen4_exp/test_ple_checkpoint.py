# SPDX-License-Identifier: Apache-2.0
"""The owner must load only its requested global PLE row range."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.models.qwen4_exp.ple_checkpoint import PLECheckpoint, checkpoint_identity


def test_checkpoint_identity_binds_model_metadata_and_table(tmp_path):
    (tmp_path / "config.json").write_text('{"model":"a"}')
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    original = checkpoint_identity(tmp_path, "layer1")
    assert original != checkpoint_identity(tmp_path, "layer2")
    (tmp_path / "config.json").write_text('{"model":"b"}')
    assert original != checkpoint_identity(tmp_path, "layer1")


def test_checkpoint_reads_cross_shard_range(tmp_path):
    prefix = "model.ple.ngram_embedding"
    expected = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
    weight_map = {}
    for i, rows in enumerate((expected[:3], expected[3:])):
        name = f"{prefix}.shard_{i}.weight"
        filename = f"weights-{i}.safetensors"
        save_file({name: rows}, str(tmp_path / filename))
        weight_map[name] = filename
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    checkpoint = PLECheckpoint(tmp_path, prefix, 2, 4)
    assert checkpoint.weight_names == frozenset(weight_map)
    assert f"{prefix}.other.weight" not in checkpoint.weight_names
    result = torch.empty((4, 4), dtype=torch.bfloat16)
    checkpoint.copy_rows(result, 1)
    assert torch.equal(result, expected[1:5])
    with pytest.raises(ValueError, match="exceeds checkpoint"):
        checkpoint.copy_rows(result, 3)


def test_checkpoint_rejects_missing_and_wrong_dtype(tmp_path):
    prefix = "model.ple.ngram_embedding"
    name = f"{prefix}.shard_0.weight"
    save_file({name: torch.zeros((3, 4), dtype=torch.float32)}, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "weights.safetensors"}}))
    with pytest.raises(ValueError, match="shard indexes"):
        PLECheckpoint(tmp_path, prefix, 2, 4)
    with pytest.raises(ValueError, match="Unsupported PLE shard layout"):
        PLECheckpoint(tmp_path, prefix, 1, 4)


def test_checkpoint_rejects_inconsistent_intermediate_shards(tmp_path):
    prefix = "model.ple.ngram_embedding"
    weight_map = {}
    for i, rows in enumerate((3, 2, 1)):
        name = f"{prefix}.shard_{i}.weight"
        filename = f"weights-{i}.safetensors"
        save_file({name: torch.zeros((rows, 4), dtype=torch.bfloat16)}, str(tmp_path / filename))
        weight_map[name] = filename
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="Inconsistent PLE shard row count"):
        PLECheckpoint(tmp_path, prefix, 3, 4)
