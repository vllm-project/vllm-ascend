"""Convert ModelSlim DeepSeek routed experts to packed INT4 SVD factors.

Each completed expert is atomically installed and independently resumable.
The model index and ready marker are published only after all shards finish.
"""

import argparse
import hashlib
import json
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_ascend.quantization.moe_svd import (
    FORMAT_VERSION,
    PROJECTIONS,
    dequantize_modelslim,
    quantize_factor,
    rotate_factors,
    svd_factors,
    validate_factor_dimensions,
)

EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    digest = hashlib.sha256(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    digest.update(memoryview(tensor.contiguous().reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def validate_shard(path, metadata, expected_keys, specs=None):
    with safe_open(path, framework="pt", device="cpu") as handle:
        saved = handle.metadata() or {}
        if any(saved.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Resume metadata mismatch: {path}")
        if set(handle.keys()) != set(expected_keys):
            raise ValueError(f"Incomplete shard: {path}")
        checksums = json.loads(saved.get("tensor_sha256", "{}"))
        for key in expected_keys:
            tensor = handle.get_tensor(key)
            if specs is not None and (tuple(tensor.shape), tensor.dtype) != specs[key]:
                raise ValueError(f"Invalid tensor shape or dtype: {path}: {key}")
            if checksums.get(key) != tensor_sha256(tensor):
                raise ValueError(f"Tensor checksum mismatch: {path}: {key}")
        return saved


def expert_tensor_specs(layer, expert, rank, hidden_size, intermediate_size):
    specs = {}
    for projection in PROJECTIONS:
        out_size, in_size = (
            (hidden_size, intermediate_size) if projection == "down_proj" else (intermediate_size, hidden_size)
        )
        prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for side, out_width, in_width in (("left", out_size, rank), ("right", rank, in_size)):
            specs[f"{prefix}.{side}_weight"] = ((out_width // 2, in_width), torch.int8)
            specs[f"{prefix}.{side}_scale"] = ((out_width, 1), torch.float32)
            specs[f"{prefix}.{side}_bias"] = ((out_width,), torch.float32)
    return specs


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_save(path, tensors, metadata):
    temporary = path.with_suffix(".partial")
    checksums = {key: tensor_sha256(tensor) for key, tensor in tensors.items()}
    save_file(tensors, temporary, metadata={**metadata, "tensor_sha256": json.dumps(checksums, sort_keys=True)})
    temporary.replace(path)


def convert_expert(task):
    source, output, files, layer, expert, rank, threads, fingerprint, rotation, hidden_size, intermediate_size = task
    torch.set_num_threads(threads)
    filename = f"svd-layer{layer:03d}-expert{expert:03d}.safetensors"
    path = Path(output) / filename
    metadata = {"format": "pt", "moe_svd_source": fingerprint, "rank": str(rank)}
    specs = expert_tensor_specs(layer, expert, rank, hidden_size, intermediate_size)
    if path.exists():
        saved = validate_shard(path, metadata, specs, specs)
        return filename, list(specs), json.loads(saved["energy"])
    tensors = {}
    energy = {}
    for projection in PROJECTIONS:
        prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        values = {}
        for suffix in ("weight", "weight_scale", "weight_offset"):
            key = f"{prefix}.{suffix}"
            with safe_open(Path(source) / files[key], framework="pt", device="cpu") as handle:
                values[suffix] = handle.get_tensor(key)
        dense = dequantize_modelslim(values["weight"], values["weight_scale"], values["weight_offset"])
        expected_shape = (
            (hidden_size, intermediate_size) if projection == "down_proj" else (intermediate_size, hidden_size)
        )
        if tuple(dense.shape) != expected_shape:
            raise ValueError(f"Source expert shape differs from model configuration: {prefix}")
        left, right, retained = svd_factors(dense, rank)
        if rotation == "hadamard":
            left, right = rotate_factors(left, right)
        energy[projection] = retained
        for name, matrix in (("left", left), ("right", right)):
            factor = quantize_factor(matrix)
            tensors[f"{prefix}.{name}_weight"] = factor.weight
            tensors[f"{prefix}.{name}_scale"] = factor.scale
            # Ascend A8W4 uses an activation offset internally. Compensation
            # must be computed from the quantized factor actually being used.
            tensors[f"{prefix}.{name}_bias"] = 8 * factor.dequantize().sum(dim=1)
    atomic_save(path, tensors, {**metadata, "energy": json.dumps(energy)})
    return filename, list(tensors), energy


def convert(source, output, rank=1024, workers=16, threads=4, max_layers=None, rotation="hadamard"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Input and output directories must be separate")
    config = json.loads((source / "config.json").read_text())
    description = json.loads((source / "quant_model_description.json").read_text())
    index = json.loads((source / "quant_model_weights.safetensors.index.json").read_text())
    if config["model_type"] != "deepseek_v3" or description.get("version") != "1.0.0":
        raise ValueError("Expected a ModelSlim 1.0.0 DeepSeek V3 checkpoint")
    if description.get("group_size") != 0:
        raise ValueError("Only per-channel source quantization is supported")
    hidden_size, intermediate_size = config["hidden_size"], config["moe_intermediate_size"]
    validate_factor_dimensions(rank, hidden_size, intermediate_size)
    if rank * (hidden_size + intermediate_size) >= hidden_size * intermediate_size:
        raise ValueError("Rank must reduce the number of packed expert weight elements")
    if config.get("hidden_act", "silu") != "silu" or config.get("moe_layer_freq", 1) != 1:
        raise ValueError("Expected consecutive SiLU MoE layers")
    if workers < 1 or threads < 1:
        raise ValueError("Workers and threads must be positive")
    if rotation not in ("none", "hadamard"):
        raise ValueError("Rotation must be none or hadamard")
    depth = config["num_hidden_layers"]
    if max_layers is not None:
        if not config["first_k_dense_replace"] < max_layers <= depth:
            raise ValueError("Reduced depth must include at least one MoE layer")
        depth = max_layers
    files = index["weight_map"]
    stamp = {
        "source": str(source),
        "rank": rank,
        "max_layers": max_layers,
        "format_version": FORMAT_VERSION,
        "algorithm": "fp32_gram_eigh",
        "config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "description_sha256": hashlib.sha256((source / "quant_model_description.json").read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(
            (source / "quant_model_weights.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "source_files": {
            name: {
                "size": (source / name).stat().st_size,
                "sha256": file_sha256(source / name),
            }
            for name in sorted(set(files.values()))
        },
    }
    if rotation == "hadamard":
        stamp["latent_rotation"] = {"method": "signed_block_hadamard", "seed": 0}
    fingerprint = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "moe_svd_conversion.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != stamp:
            raise ValueError("Output belongs to a different conversion; choose a new directory")
    elif any(output.iterdir()):
        raise ValueError("Output must be empty or contain a matching conversion manifest")
    else:
        atomic_json(manifest, stamp)
    (output / "moe_svd_ready.json").unlink(missing_ok=True)

    experts = set()
    retained_files = {}
    retained_description = {k: v for k, v in description.items() if not k.startswith("model.layers.")}
    for key, filename in files.items():
        match = LAYER_RE.match(key)
        if max_layers is not None and match and int(match[1]) >= depth:
            continue
        expert = EXPERT_RE.match(key)
        if expert and int(expert[1]) < depth:
            experts.add((int(expert[1]), int(expert[2])))
            continue
        retained_files.setdefault(filename, []).append(key)
        if key in description:
            retained_description[key] = description[key]
    expected = {
        (layer, expert)
        for layer in range(config["first_k_dense_replace"], depth)
        for expert in range(config["n_routed_experts"])
    }
    if experts != expected:
        raise ValueError("Source expert coverage differs from the model configuration")
    for layer, expert in experts:
        for projection in PROJECTIONS:
            key = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            if description.get(key) != "W4A8_DYNAMIC":
                raise ValueError(f"Unsupported source expert quantization: {key}")

    new_map = {}
    started = time.monotonic()
    atomic_json(output / "conversion_status.json", {"phase": "converting", "experts": len(experts), "completed": 0})
    tasks = []
    for layer, expert in sorted(experts):
        prefix = f"model.layers.{layer}.mlp.experts.{expert}."
        selected = {
            key: files[key]
            for projection in PROJECTIONS
            for suffix in ("weight", "weight_scale", "weight_offset")
            for key in (f"{prefix}{projection}.{suffix}",)
        }
        tasks.append(
            (
                str(source),
                str(output),
                selected,
                layer,
                expert,
                rank,
                threads,
                fingerprint,
                rotation,
                hidden_size,
                intermediate_size,
            )
        )
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for completed, (filename, keys, energy) in enumerate(pool.map(convert_expert, tasks), 1):
            new_map.update({key: filename for key in keys})
            retained_description.update({key: "W4A8_SVD" for key in keys})
            status = {
                "phase": "converting",
                "completed": completed,
                "experts": len(experts),
                "seconds": time.monotonic() - started,
                "last_shard": filename,
                "energy": energy,
            }
            atomic_json(output / "conversion_status.json", status)
            print(json.dumps(status), flush=True)

    # Copy only retained tensors. Mixed original shards must not sneak dense
    # expert weights back into the output checkpoint.
    for position, (filename, keys) in enumerate(sorted(retained_files.items())):
        destination = f"retained-{position:05d}.safetensors"
        path = output / destination
        metadata = {"format": "pt", "moe_svd_source": fingerprint}
        if path.exists():
            validate_shard(path, metadata, keys)
        else:
            with safe_open(source / filename, framework="pt", device="cpu") as handle:
                tensors = {key: handle.get_tensor(key).contiguous() for key in keys}
                atomic_save(path, tensors, metadata)
        new_map.update({key: destination for key in keys})

    for path in source.iterdir():
        if path.is_file() and (
            path.suffix in (".json", ".py", ".jinja") or path.name in ("README.md", "LICENSE", "NOTICE")
        ):
            if path.name in ("config.json", "quant_model_description.json") or path.name.endswith(".index.json"):
                continue
            shutil.copy2(path, output / path.name)
    config["moe_svd"] = {
        "format_version": FORMAT_VERSION,
        "rank": rank,
        "weight_bits": 4,
        "activation_bits": 8,
        "group_size": 0,
        "latent_rotation": rotation,
        "layers": sorted({layer for layer, _ in experts}),
    }
    if max_layers is not None:
        config["num_hidden_layers"] = depth
        config["num_nextn_predict_layers"] = 0
    atomic_json(output / "config.json", config)
    atomic_json(output / "quant_model_description.json", retained_description)
    total_size = 0
    for filename in set(new_map.values()):
        with safe_open(output / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                tensor = handle.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()
    atomic_json(
        output / "quant_model_weights.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": new_map},
    )
    atomic_json(
        output / "conversion_status.json",
        {
            "phase": "completed",
            "experts": len(experts),
            "total_size": total_size,
            "seconds": time.monotonic() - started,
        },
    )
    atomic_json(output / "moe_svd_ready.json", {"source_fingerprint": fingerprint, "total_size": total_size})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--rotation", choices=("none", "hadamard"), default="hadamard")
    args = parser.parse_args()
    convert(args.input, args.output, args.rank, args.workers, args.threads, args.max_layers, args.rotation)


if __name__ == "__main__":
    main()
