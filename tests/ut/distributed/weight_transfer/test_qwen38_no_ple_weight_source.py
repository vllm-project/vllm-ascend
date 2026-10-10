# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small real safetensors fixtures for preparation, integrity and iteration."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.e2e.pull_request.rlhf import prepare_qwen38_checkpoint as prep
from tests.e2e.pull_request.rlhf.qwen38_checkpoint_source import Qwen38CheckpointSource
from tests.e2e.pull_request.rlhf.qwen38_weight_transfer_utils import FAMILIES, NoPLEWorkerProbe


@pytest.fixture
def source_checkpoint(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    weights = {name: torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) for name in prep.GLOBAL_NAMES}
    for layer in range(5):
        weights[f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"] = torch.full(
            (2, 3), layer, dtype=torch.bfloat16
        )
    weights.update(
        {
            "model.language_model.layers.1.ple.embedding.weight": torch.ones(4, dtype=torch.bfloat16),
            "model.language_model.layers.1.ple.ngram.table": torch.arange(4, dtype=torch.int64),
            "model.visual.weight": torch.ones(4, dtype=torch.bfloat16),
            "mtp.layers.0.weight": torch.ones(4, dtype=torch.bfloat16),
        }
    )
    save_file(weights, str(directory / "model.safetensors"))
    prep.write_json(
        directory / "model.safetensors.index.json",
        {
            "metadata": {"total_size": sum(w.numel() * w.element_size() for w in weights.values())},
            "weight_map": {n: "model.safetensors" for n in weights},
        },
    )
    config = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 48,
            "layer_types": list(prep.LAYER_TYPES) * 12,
            "ple_layer_ids": [2],
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "hidden_size": 2560,
            "dtype": "bfloat16",
            "ple_embed_dim": 2560,
            "ngram_size": 3,
            "ngram_vocab_size_base": 20000000,
        },
    }
    prep.write_json(directory / "config.json", config)
    prep.write_json(directory / "tokenizer.json", {})
    return directory, weights, config


@pytest.fixture
def prepared(source_checkpoint, tmp_path, monkeypatch):
    source, weights, config = source_checkpoint
    directory = tmp_path / "output"
    digest = prep.prepare_checkpoint(source, directory, "fixture-revision", strict=False)
    retained = {n: w for n, w in weights.items() if prep.retain_name(n)}
    monkeypatch.setattr(prep, "EXPECTED_TENSORS", len(retained))
    monkeypatch.setattr(prep, "EXPECTED_BYTES", sum(w.numel() * w.element_size() for w in retained.values()))
    return directory, digest, retained, config


def test_preparation_preserves_values_and_only_patches_three_fields(prepared, source_checkpoint):
    directory, digest, retained, original_config = prepared
    manifest = prep.validate_checkpoint(directory, digest)
    expected_config = copy.deepcopy(original_config)
    expected_config["text_config"].update(num_hidden_layers=4, layer_types=list(prep.LAYER_TYPES), ple_layer_ids=[])
    assert prep.read_json(directory / "config.json") == expected_config
    assert set(manifest["tensors"]) == set(retained)
    assert len(manifest["excluded_ple"]) == 2
    assert prep.read_json(source_checkpoint[0] / "config.json") == original_config
    for name, entry in manifest["tensors"].items():
        assert torch.equal(load_file(str(directory / entry["file"]))[name], retained[name])
        assert entry["dtype"] == "BF16"


def test_source_repeats_original_fused_tensor_names_shapes_and_values(prepared):
    directory, digest, retained, _ = prepared
    source = Qwen38CheckpointSource(directory, digest, torch.device("cpu"))
    assert [m.name for m in source.metadata()] == sorted(retained)
    for _ in range(2):
        result = dict(source)
        assert set(result) == set(retained)
        for name in retained:
            assert torch.equal(result[name], retained[name])
    assert any(n.endswith("gate_up_proj") for n in result)
    assert not any(n.endswith("gate_up_proj.weight") for n in result)


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.4.mlp.weight",
        "model.language_model.layers.1.ple.ngram.weight",
        "model.visual.weight",
        "mtp.layers.0.weight",
    ],
)
def test_excludes_high_layers_ple_vision_mtp(name):
    assert not prep.retain_name(name)


@pytest.mark.parametrize("name", ["model.language_model.new_global.weight", "unknown.weight"])
def test_unknown_names_fail(name):
    with pytest.raises(ValueError, match="Unknown"):
        prep.retain_name(name)


def test_manifest_requires_external_digest(prepared):
    with pytest.raises(ValueError, match="Manifest SHA-256"):
        prep.validate_checkpoint(prepared[0], "0" * 64)


def test_corrupt_tensor_file_fails(prepared):
    directory, digest, _, _ = prepared
    shard = next(directory.glob("*.safetensors"))
    with shard.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="File SHA-256"):
        prep.validate_checkpoint(directory, digest)


def test_untracked_file_fails(prepared):
    directory, digest, _, _ = prepared
    (directory / "extra.safetensors").write_bytes(b"extra")
    with pytest.raises(ValueError, match="Untracked"):
        prep.validate_checkpoint(directory, digest)


@pytest.mark.parametrize("field,value", [("ple_layer_ids", [2]), ("num_experts", 8), ("num_experts_per_tok", 6)])
def test_config_drift_fails(prepared, field, value):
    config = prep.read_json(prepared[0] / "config.json")
    config["text_config"][field] = value
    with pytest.raises(ValueError, match="Invalid no-PLE config"):
        prep.validate_config(config)


def test_missing_index_weight_fails(source_checkpoint):
    directory, _, _ = source_checkpoint
    index = prep.read_json(directory / "model.safetensors.index.json")
    del index["weight_map"][next(iter(index["weight_map"]))]
    prep.write_json(directory / "model.safetensors.index.json", index)
    with pytest.raises(ValueError, match="unindexed"):
        prep.read_schema(directory)


@pytest.mark.parametrize("bad_dtype", ["I64", "F32"])
def test_retained_non_bf16_fails(prepared, bad_dtype):
    schema = prep.read_schema(prepared[0])
    next(iter(schema.values()))["dtype"] = bad_dtype
    with pytest.raises(ValueError, match="dtype"):
        prep.validate_schema(schema)


def test_production_totals_fail_on_small_fixture(source_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="Source profile changed"):
        prep.prepare_checkpoint(source_checkpoint[0], tmp_path / "output", "fixture-revision")
    assert not (tmp_path / "output").exists()


def test_refuses_existing_output(source_checkpoint):
    directory = source_checkpoint[0]
    with pytest.raises(ValueError, match="new directory"):
        prep.prepare_checkpoint(directory, directory, "fixture-revision", strict=False)


def test_path_traversal_fails(source_checkpoint):
    with pytest.raises(ValueError, match="Invalid checkpoint file"):
        prep.local_file(source_checkpoint[0], "../outside.safetensors")


def test_duplicate_json_keys_fail(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text('{"weight_map": {}, "weight_map": {}}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        prep.read_json(file)


def test_unresolved_lfs_tokenizer_fails_before_writing_weights(source_checkpoint, tmp_path):
    directory = source_checkpoint[0]
    (directory / "tokenizer.json").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Unresolved Git LFS pointer"):
        prep.prepare_checkpoint(directory, output, "fixture-revision", strict=False)
    assert not output.exists()


def test_exact_lfs_tokenizer_resolution_preserves_source(source_checkpoint, tmp_path):
    directory = source_checkpoint[0]
    resolved = tmp_path / "resolved-tokenizer.json"
    resolved.write_text('{"version": "1.0"}')
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{prep.file_sha256(resolved)}\nsize {resolved.stat().st_size}\n"
    )
    (directory / "tokenizer.json").write_text(pointer)
    output = tmp_path / "output"
    digest = prep.prepare_checkpoint(directory, output, "fixture-revision", strict=False, resolved_tokenizer=resolved)
    prep.validate_checkpoint(output, digest, strict=False)
    assert (output / "tokenizer.json").read_bytes() == resolved.read_bytes()
    assert (directory / "tokenizer.json").read_text() == pointer


def test_wrong_tokenizer_resolution_fails(source_checkpoint, tmp_path):
    resolved = tmp_path / "wrong-tokenizer.json"
    resolved.write_text('{"wrong": true}')
    with pytest.raises(ValueError, match="Resolved tokenizer SHA-256"):
        prep.prepare_checkpoint(
            source_checkpoint[0], tmp_path / "output", "fixture-revision", strict=False, resolved_tokenizer=resolved
        )


def test_probe_counts_direct_hyperconnection_methods_and_restores_them():
    decoder_class = type("Qwen4ExpDecoderLayer", (torch.nn.Module,), {})
    hyperconnection_class = type(
        "GatedResidual",
        (torch.nn.Module,),
        {
            "mix": lambda self, value: value + 1,
            "combine_and_mix": lambda self, value: value + 2,
        },
    )
    model = torch.nn.Module()
    layers = [decoder_class() for _ in range(4)]
    model.layers = torch.nn.ModuleList(layers)
    for layer in layers:
        layer.ple = None
    for family in FAMILIES[:-1]:
        layers[0].add_module(family, type(family, (torch.nn.Identity,), {})())
    hc = hyperconnection_class()
    layers[0].add_module("hyperconnection", hc)
    original = hc.mix
    probe = NoPLEWorkerProbe()
    probe.model_runner = SimpleNamespace(
        get_model=lambda: model,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                text_config=SimpleNamespace(
                    num_hidden_layers=4,
                    ple_layer_ids=[],
                    num_experts=512,
                    num_experts_per_tok=10,
                )
            ),
        ),
    )
    with patch("vllm_ascend.distributed.weight_transfer.npu_ipc_engine.npu_generate_uuid", return_value="same-device"):
        assert probe.install_no_ple_probe() == "same-device"
    for family in FAMILIES[:-1]:
        getattr(layers[0], family)(torch.ones(1))
    assert hc.mix(3) == 4 and hc.combine_and_mix(3) == 5
    counts = probe.check_no_ple_execution()
    assert counts["GatedResidual"] == 2
    assert hc.mix == original
    assert not probe._probe_hooks and not probe._probe_methods
