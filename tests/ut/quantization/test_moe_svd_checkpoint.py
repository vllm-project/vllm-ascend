import json
import os
import shutil

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_ascend.quantization.moe_svd import PROJECTIONS, quantize_factor
from vllm_ascend.quantization.moe_svd_checkpoint import convert


def make_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    config = {
        "model_type": "deepseek_v3",
        "num_hidden_layers": 1,
        "first_k_dense_replace": 0,
        "n_routed_experts": 2,
        "moe_intermediate_size": 128,
        "hidden_size": 256,
    }
    (source / "config.json").write_text(json.dumps(config))
    description = {"version": "1.0.0", "group_size": 0}
    shards = [{}, {}]
    for expert in range(2):
        for projection in PROJECTIONS:
            shape = (256, 128) if projection == "down_proj" else (128, 256)
            factor = quantize_factor(torch.randn(shape))
            prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
            shards[0][prefix + ".weight"] = factor.weight
            shards[1][prefix + ".weight_scale"] = factor.scale
            shards[1][prefix + ".weight_offset"] = torch.zeros_like(factor.scale)
            shards[1][prefix + ".scale_bias"] = torch.zeros(shape[0])
            description[prefix + ".weight"] = "W4A8_DYNAMIC"
    retained = torch.randn(16, 32).bfloat16()
    shards[0]["model.embed_tokens.weight"] = retained
    shared = {}
    for projection in PROJECTIONS:
        shape = (256, 128) if projection == "down_proj" else (128, 256)
        prefix = f"model.layers.0.mlp.shared_experts.{projection}"
        shared[prefix + ".weight"] = torch.randint(-127, 128, shape, dtype=torch.int8)
        shared[prefix + ".weight_scale"] = torch.rand(shape[0], 1).clamp_min(1e-6)
        shared[prefix + ".weight_offset"] = torch.zeros(shape[0], 1)
    shards[1].update(shared)
    description.update({key: "W8A8_DYNAMIC" for key in shared})
    shards[1]["model.layers.1.mlp.experts.0.gate_proj.weight"] = torch.ones(8, 8, dtype=torch.int8)
    description["model.layers.1.mlp.experts.0.gate_proj.weight"] = "W8A8_DYNAMIC"
    weight_map = {}
    for index, shard in enumerate(shards):
        filename = f"original-{index}.safetensors"
        save_file(shard, source / filename)
        weight_map.update({key: filename for key in shard})
    (source / "quant_model_description.json").write_text(json.dumps(description))
    (source / "quant_model_weights.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return source, output, retained, shared


@pytest.fixture(scope="module")
def converted_checkpoint(tmp_path_factory):
    source, output, retained, shared = make_source(tmp_path_factory.mktemp("moe_svd"))
    convert(source, output, rank=32, workers=1, threads=1)
    return source, output, retained, shared


def test_sharded_conversion_resume_and_actual_size(converted_checkpoint):
    source, output, retained, shared = converted_checkpoint
    before = {p.name: p.read_bytes() for p in output.glob("*.safetensors")}
    result = json.loads((output / "quant_model_weights.safetensors.index.json").read_text())
    assert result["metadata"]["total_size"] < sum(p.stat().st_size for p in source.glob("*.safetensors"))
    assert len([key for key in result["weight_map"] if ".left_weight" in key]) == 6
    assert not any("layers.0.mlp.experts." in key and key.endswith(".weight") for key in result["weight_map"])
    name = "model.embed_tokens.weight"
    with safe_open(output / result["weight_map"][name], framework="pt") as handle:
        torch.testing.assert_close(handle.get_tensor(name), retained, atol=0, rtol=0)
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in result["weight_map"]
    converted_description = json.loads((output / "quant_model_description.json").read_text())
    for key, original in shared.items():
        assert converted_description[key] == "W8A8_DYNAMIC"
        with safe_open(output / result["weight_map"][key], framework="pt") as handle:
            actual = handle.get_tensor(key)
            assert actual.dtype == original.dtype
            torch.testing.assert_close(actual, original, atol=0, rtol=0)
    convert(source, output, rank=32, workers=1, threads=1)
    assert before == {p.name: p.read_bytes() for p in output.glob("*.safetensors")}
    with pytest.raises(ValueError, match="different conversion"):
        convert(source, output, rank=64, workers=1, threads=1)


@pytest.mark.parametrize("corruption", ["weight", "shape", "dtype", "retained"])
def test_resume_rejects_corrupt_shards(converted_checkpoint, tmp_path, corruption):
    source, original, _, _ = converted_checkpoint
    output = tmp_path / "output"
    shutil.copytree(original, output)
    path = output / ("retained-00000.safetensors" if corruption == "retained" else "svd-layer000-expert000.safetensors")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}  # noqa: SIM118 - safe_open is not iterable
    key = next(iter(tensors)) if corruption == "retained" else "model.layers.0.mlp.experts.0.gate_proj.left_weight"
    if corruption == "shape":
        tensors[key] = tensors[key][:-1].contiguous()
    elif corruption == "dtype":
        tensors[key] = tensors[key].float()
    else:
        tensors[key].reshape(-1)[0] += 1
    save_file(tensors, path, metadata=metadata)

    with pytest.raises(ValueError, match="checksum|shape or dtype"):
        convert(source, output, rank=32, workers=1, threads=1)
    assert not (output / "moe_svd_ready.json").exists()


def test_resume_detects_source_change_with_same_size_and_timestamp(converted_checkpoint, tmp_path):
    source, original, _, _ = converted_checkpoint
    output = tmp_path / "output"
    shutil.copytree(original, output)
    path = source / "original-0.safetensors"
    stat = path.stat()
    content = path.read_bytes()
    changed = bytearray(content)
    changed[-1] ^= 1
    try:
        path.write_bytes(changed)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with pytest.raises(ValueError, match="different conversion"):
            convert(source, output, rank=32, workers=1, threads=1)
    finally:
        path.write_bytes(content)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


@pytest.mark.parametrize("rank", [16, 128, 256])
def test_conversion_rejects_invalid_or_noncompressing_rank(converted_checkpoint, tmp_path, rank):
    source, _, _, _ = converted_checkpoint
    with pytest.raises(ValueError, match="rank|Rank"):
        convert(source, tmp_path / "output", rank=rank, workers=1, threads=1)
    assert not (tmp_path / "output").exists()
