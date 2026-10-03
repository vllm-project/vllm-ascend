# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import struct

import pytest

from tests.e2e.glm53_flash.checkpoint import (
    INDEX_NAME,
    file_hash,
    keep_tensor,
    make_source_manifest,
    prepare_checkpoint,
    read_header,
    read_json,
    reduced_config,
    write_json,
)


def write_safetensors(path, tensors):
    header = {"__metadata__": {"format": "pt"}}
    payload = b""
    for name, (dtype, data) in tensors.items():
        width = 2 if dtype == "BF16" else 1
        header[name] = {
            "dtype": dtype,
            "shape": [len(data) // width],
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload += data
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    layer_types = ["deepseek_sparse_attention" if index % 4 == 3 else "linear_attention" for index in range(45)]
    text = {
        "num_hidden_layers": 45,
        "num_nextn_predict_layers": 1,
        "n_routed_experts": 288,
        "hidden_size": 4096,
        "layer_types": layer_types,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
        "indexer_types": ["full"] * 45,
        "eos_token_id": [1, 2, 3],
        "linear_attn_config": {
            "kda_layers": [i for i in range(45) if i % 4 != 3],
            "full_attn_layers": list(range(3, 45, 4)),
            "num_heads": 64,
        },
    }
    write_json(root / "config.json", {"architectures": ["Glm5NextForConditionalGeneration"], "text_config": text})
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "processor_config.json"):
        write_json(root / name, {"unchanged": name})
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    prefix = "model.language_model.layers."
    tensors = {f"{prefix}{layer}.mlp.weight": ("I8", bytes([layer])) for layer in (0, 1, 2, 3, 4, 44, 45)}
    tensors.update(
        {
            f"{prefix}3.mlp.weight_scale": ("BF16", b"\x80\x3f"),
            f"{prefix}3.mlp.weight_offset": ("I8", b"\x00"),
            "model.visual.layers.9.weight": ("BF16", b"\x00\x40"),
            "lm_head.weight": ("BF16", b"\x80\x3f"),
            "model.language_model.embed_tokens.weight": ("BF16", b"\x80\x3f"),
        }
    )
    shard = "weights.safetensors"
    write_safetensors(root / shard, tensors)
    write_json(root / INDEX_NAME, {"metadata": {}, "weight_map": dict.fromkeys(tensors, shard)})
    write_json(
        root / "quant_model_description.json", {**dict.fromkeys(tensors, "W8A8"), "metadata": {}, "group_size": 0}
    )
    return root


def test_config_projection_is_nonmutating(source):
    config = read_json(source / "config.json")
    original = copy.deepcopy(config)
    projected = reduced_config(config)["text_config"]
    assert config == original
    assert projected["num_hidden_layers"] == 5
    assert projected["num_nextn_predict_layers"] == 0
    assert projected["linear_attn_config"]["kda_layers"] == [0, 1, 2, 4]
    assert projected["linear_attn_config"]["full_attn_layers"] == [3]
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        assert len(projected[key]) == 5
    assert projected["eos_token_id"] == [1, 2, 3]
    assert projected["hidden_size"] == 4096 and projected["n_routed_experts"] == 288


def test_projection_preserves_payloads_quantization_and_source(source, tmp_path):
    manifest = make_source_manifest(source)
    original = {path.name: file_hash(path) for path in source.iterdir()}
    output = tmp_path / "derived"
    result = prepare_checkpoint(source, output, manifest)
    original_header, original_offset = read_header(source / "weights.safetensors")
    weights = read_json(output / INDEX_NAME)["weight_map"]
    assert set(weights) == {key for key in original_header if key != "__metadata__" and keep_tensor(key)}
    assert "model.visual.layers.9.weight" in weights
    assert not any("layers.44." in key or "layers.45." in key for key in weights)
    total_bytes = 0
    for name, shard in weights.items():
        header, offset = read_header(output / shard)
        actual = header[name]
        expected = original_header[name]
        assert actual["dtype"] == expected["dtype"] and actual["shape"] == expected["shape"]
        start, end = actual["data_offsets"]
        base_start, base_end = expected["data_offsets"]
        assert (output / shard).read_bytes()[offset + start : offset + end] == (
            source / "weights.safetensors"
        ).read_bytes()[original_offset + base_start : original_offset + base_end]
        total_bytes += end - start
    assert result["tensor_bytes"] == read_json(output / INDEX_NAME)["metadata"]["total_size"] == total_bytes
    quant = read_json(output / "quant_model_description.json")
    assert set(quant) == {*weights, "metadata", "group_size"}
    assert quant["model.language_model.layers.3.mlp.weight_scale"] == "W8A8"
    assert {path.name: file_hash(path) for path in source.iterdir()} == original
    assert prepare_checkpoint(source, output, manifest) == result


@pytest.mark.parametrize("name", ["tokenizer.json", "weights.safetensors"])
def test_source_checksum_failure(source, tmp_path, name):
    manifest = make_source_manifest(source)
    with (source / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_checkpoint(source, tmp_path / "derived", manifest)
    assert not (tmp_path / "derived").exists()


def test_derived_corruption_is_not_silently_rebuilt(source, tmp_path):
    manifest = make_source_manifest(source)
    output = tmp_path / "derived"
    prepare_checkpoint(source, output, manifest)
    (output / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_checkpoint(source, output, manifest)


def test_wrong_manifest_and_unsafe_output(source, tmp_path):
    manifest = make_source_manifest(source)
    manifest["source_id"] = "wrong"
    with pytest.raises(ValueError, match="manifest checksum"):
        prepare_checkpoint(source, tmp_path / "derived", manifest)
    with pytest.raises(ValueError, match="disjoint"):
        prepare_checkpoint(source, source / "derived", manifest)
    with pytest.raises(ValueError, match="disjoint"):
        prepare_checkpoint(source, tmp_path, manifest)


def test_index_traversal_rejected(source):
    index = read_json(source / INDEX_NAME)
    index["weight_map"]["lm_head.weight"] = "../outside.safetensors"
    write_json(source / INDEX_NAME, index)
    with pytest.raises(ValueError, match="Unsafe checkpoint filename"):
        make_source_manifest(source)


def test_truncated_tensor_payload(source):
    shard = source / "weights.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated/trailing"):
        read_header(shard)


def test_index_references_missing_tensor(source, tmp_path):
    index = read_json(source / INDEX_NAME)
    index["weight_map"]["lm_head.absent"] = "weights.safetensors"
    write_json(source / INDEX_NAME, index)
    manifest = make_source_manifest(source)
    with pytest.raises(ValueError, match="absent tensors"):
        prepare_checkpoint(source, tmp_path / "derived", manifest)
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("field,value", [("num_hidden_layers", 5), ("n_routed_experts", 16), ("indexer_types", [])])
def test_wrong_source_layout(source, field, value):
    config = read_json(source / "config.json")
    config["text_config"][field] = value
    with pytest.raises(ValueError):
        reduced_config(config)
