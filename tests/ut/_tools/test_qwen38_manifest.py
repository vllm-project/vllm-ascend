# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for reduced PLE/MRoPE checkpoint geometry."""

import copy
import json
import struct

import pytest

from tests.e2e.common.rlhf.weight_transfer.qwen38_manifest import (
    checkpoint_manifest,
    ple_buffer_values,
    prepare_descriptor,
    reduced_config,
    resized_tensor,
    selected_tensor,
    validate_ci_config,
)


@pytest.fixture
def original_config():
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "language_model_only": False,
        "text_config": {
            "num_hidden_layers": 8,
            "layer_types": ["linear_attention"] * 3
            + ["full_attention"]
            + ["linear_attention"] * 3
            + ["full_attention"],
            "ple_layer_ids": [2, 6],
            "ngram_vocab_size_base": 20_000_000,
            "head_dim": 256,
            "hidden_size": 2560,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "indexer_head_dim": 128,
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.25,
            },
            "partial_rotary_factor": 0.25,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "ple_embed_dim": 2560,
            "make_ngram_vocab_size_divisible_by": 128,
            "split_ngram_parts": 128,
            "vocab_size": 248320,
            "mtp": {"num_hidden_layers": 1},
        },
    }


def test_reduction_keeps_ple_mrope_and_original_source(original_config):
    snapshot = copy.deepcopy(original_config)
    config = reduced_config(original_config)
    text = config["text_config"]
    assert original_config == snapshot
    assert text["ple_layer_ids"] == [2]
    assert text["head_dim"] * text["rope_parameters"]["partial_rotary_factor"] == 64
    assert text["vocab_size"] == 248320
    assert text["mtp"]["num_hidden_layers"] == 0
    assert config["language_model_only"] is False


@pytest.mark.parametrize(
    "suffix,old,expected",
    [
        ("q_proj.weight", [12288, 2560], [6144, 2560]),
        ("k_proj.weight", [512, 2560], [256, 2560]),
        ("v_proj.weight", [512, 2560], [256, 2560]),
        ("o_proj.weight", [2560, 6144], [2560, 3072]),
        ("q_norm.weight", [256], [128]),
        ("k_norm.weight", [256], [128]),
        ("indexer.index_qk_proj.weight", [640, 2560], [640, 2560]),
    ],
)
def test_attention_checkpoint_shapes_follow_reduced_config(original_config, suffix, old, expected):
    config = reduced_config(original_config)
    shape, values = resized_tensor("model.language_model.layers.3.self_attn." + suffix, old, config)
    assert shape == expected
    assert values is None


def test_ple_shards_cover_padded_table_without_gaps(original_config):
    config = reduced_config(original_config)
    text = config["text_config"]
    buffers = ple_buffer_values(text, 0)
    sizes, offsets = buffers["ngram_heads_vocab_sizes"], buffers["ngram_heads_offsets"]
    assert len(sizes) == 16
    assert sizes[:4] == [4099, 4111, 4127, 4129]
    assert offsets[0] == 0
    assert all(offsets[i + 1] == offsets[i] + sizes[i] for i in range(len(sizes) - 1))
    assert all(x > 0 and x % 2 == 1 and x * text["vocab_size"] < 2**63 for x in buffers["layer_multipliers"])
    rows = []
    for i in range(128):
        shape, values = resized_tensor(
            f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight", [2500012, 160], config
        )
        rows.append(shape[0])
        assert shape[1] == 160 and values is None
    assert sum(rows) % 128 == 0
    assert 0 <= sum(rows) - sum(sizes) < 128
    assert sum(rows) * 160 * 2 < 32 * 2**20


def test_integer_buffers_use_new_vocabulary_bounds(original_config):
    config = reduced_config(original_config)
    for key, expected in ple_buffer_values(config["text_config"], 0).items():
        shape, values = resized_tensor(
            "model.language_model.layers.1.ple.ple_embedding." + key, [len(expected)], config
        )
        assert shape == [len(expected)]
        assert values == expected
    assert ple_buffer_values(config["text_config"], 0) != ple_buffer_values(config["text_config"], 1)


@pytest.mark.parametrize("mutation", ["no_ple", "text_only", "wrong_head", "wrong_rope", "duplicate_ple"])
def test_invalid_runtime_contract_is_rejected(original_config, mutation):
    config = reduced_config(original_config)
    text = config["text_config"]
    if mutation == "no_ple":
        text["ple_layer_ids"] = []
    elif mutation == "text_only":
        config["architectures"] = ["Qwen4ExpForCausalLM"]
    elif mutation == "wrong_head":
        text["head_dim"] = 256
    elif mutation == "wrong_rope":
        text["rope_parameters"]["partial_rotary_factor"] = 0.25
    else:
        text["ple_layer_ids"] = [2, 2]
    with pytest.raises(ValueError):
        validate_ci_config(config)


def test_manifest_reads_headers_not_pretrained_tensor_bytes(tmp_path, original_config):
    names = [f"model.language_model.layers.{i}.dummy.weight" for i in range(4)]
    names += ["model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"]
    header = {
        name: {
            "dtype": "I64" if name.endswith("sizes") else "BF16",
            "shape": [16] if name.endswith("sizes") else [1],
            "data_offsets": [0, 128] if name.endswith("sizes") else [0, 2],
        }
        for name in names
    }
    encoded = json.dumps(header).encode()
    # Deliberately omit all tensor data. Only header reads can succeed.
    (tmp_path / "headers.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    (tmp_path / "config.json").write_text(json.dumps(original_config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names, "headers.safetensors")})
    )
    manifest = checkpoint_manifest(tmp_path)
    assert manifest["ple_enabled"] is True
    assert manifest["head_dim"] == 128
    buffer = next(x for x in manifest["tensors"] if x["dtype"] == "I64")
    assert buffer["values"][:4] == [4099, 4111, 4127, 4129]
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    index["weight_map"][names[0]] = "../outside.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="escapes"):
        checkpoint_manifest(tmp_path)


def test_selection_retains_ple_and_rejects_vision_mtp_and_later_layers():
    assert selected_tensor("model.language_model.layers.1.ple.conv1d.weight")
    assert not selected_tensor("visual.patch_embed.proj.weight")
    assert not selected_tensor("model.language_model.layers.4.mlp.weight")
    assert not selected_tensor("model.language_model.mtp.layers.0.weight")


def test_prepare_refuses_existing_or_nested_outputs_before_writes(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        prepare_descriptor(tmp_path, tmp_path / "nested")
    target = tmp_path.parent / (tmp_path.name + "_existing")
    target.mkdir()
    with pytest.raises(FileExistsError):
        prepare_descriptor(tmp_path, target)
