# SPDX-License-Identifier: Apache-2.0
"""Reduced materials keep real tensor bytes while excluding MTP from reduced backbones."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.e2e.pull_request.pd_materials.prepare import digest, prepare


@pytest.mark.parametrize("kind", ["glm", "k3", "dsv4"])
def test_reduced_material_preserves_selected_weights_and_metadata(tmp_path, kind):
    source = tmp_path / "source"
    source.mkdir()
    if kind == "glm":
        config = {
            "num_hidden_layers": 9,
            "indexer_types": ["full"] * 9,
            "mlp_layer_types": ["dense", *(["sparse"] * 8)],
            "num_nextn_predict_layers": 1,
        }
        tensors = {
            "model.layers.0.weight": torch.arange(6).reshape(2, 3),
            "model.layers.8.weight": torch.ones(2),
            "model.layers.9.enorm.weight": torch.arange(4),
        }
        expected = {
            "model.layers.0.weight": tensors["model.layers.0.weight"],
        }
    elif kind == "dsv4":
        config = {"num_hidden_layers": 6, "num_nextn_predict_layers": 1, "compress_ratios": [0, 0, 4, 128, 4, 128, 0]}
        tensors = {
            "model.layers.0.weight": torch.arange(6),
            "model.layers.5.weight": torch.ones(2),
            "mtp.0.weight": torch.ones(3),
        }
        expected = {"model.layers.0.weight": tensors["model.layers.0.weight"]}
    else:
        config = {"text_config": {"num_hidden_layers": 4, "num_experts": 32, "linear_attn_config": {}}}
        tensors = {
            "language_model.model.layers.1.block_sparse_moe.gate.weight": torch.arange(64).reshape(32, 2),
            "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight": torch.arange(6).reshape(2, 3),
            "language_model.model.layers.1.block_sparse_moe.experts.31.w1.weight": torch.ones(2),
            "mm_projector.rot_proj.weight": torch.ones(3),
        }
        expected = {name: tensor[:16] if name.endswith("gate.weight") else tensor for name, tensor in tensors.items()}
        del expected["language_model.model.layers.1.block_sparse_moe.experts.31.w1.weight"]
        del expected["mm_projector.rot_proj.weight"]
        expected = {name.removeprefix("language_model."): tensor for name, tensor in expected.items()}
    (source / "config.json").write_text(json.dumps(config))
    index = {"weight_map": dict.fromkeys(tensors, "source.safetensors")}
    (source / "quant_model_weights.safetensors.index.json").write_text(json.dumps(index))
    quant = {"model_quant_type": "W8A8_DYNAMIC", **dict.fromkeys(tensors, "FLOAT")}
    (source / "quant_model_description.json").write_text(json.dumps(quant))
    (source / "tokenizer_config.json").write_text('{"fixture": true}\n')
    save_file(tensors, source / "source.safetensors")
    output = tmp_path / "output"
    prepare(source, output, kind)
    reduced_config = json.loads((output / "config.json").read_text())
    if kind == "k3":
        assert reduced_config["architectures"] == ["KimiK3ForCausalLM"]
        assert "text_config" not in reduced_config
        assert reduced_config["model_type"] == "kimi_linear"
        assert reduced_config["mla_use_rope"] is False
    if kind == "glm":
        assert reduced_config["mlp_layer_types"] == ["dense", *(["sparse"] * 6)]
    assert reduced_config.get("text_config", reduced_config)["num_nextn_predict_layers"] == 0
    if kind == "dsv4":
        assert reduced_config["compress_ratios"] == [0, 0, 4, 128]
    actual = load_file(output / "model-00000.safetensors")
    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor)
    material_quant = json.loads((output / "quant_model_description.json").read_text())
    assert set(material_quant) == {"model_quant_type", *expected}
    assert (output / "tokenizer_config.json").read_bytes() == (source / "tokenizer_config.json").read_bytes()
    manifest = json.loads((output / "pd_material_manifest.json").read_text())
    assert manifest["files"]["model-00000.safetensors"] == digest(output / "model-00000.safetensors")
    assert manifest["tensor_count"] == len(expected)
    with pytest.raises(FileExistsError):
        prepare(source, output, kind)
