# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-preserving, bounded-memory projection of the real 0916 checkpoint.

No torch/vLLM import is needed. Safetensors payloads are copied, not converted.
The reviewed source manifest hashes every file consumed by this projection;
discarded-only shards are not downloaded or read. Never modify the source.
"""

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import struct
import tempfile
from pathlib import Path

NUM_LAYERS = 5
RECIPE_VERSION = 1
COPY_BUFFER_BYTES = 8 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024 * 1024
INDEX_NAME = "quant_model_weights.safetensors.index.json"
DERIVED_MANIFEST = "glm53_projection.json"
TEXT_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
ASSETS = (
    "config.json",
    "quant_model_description.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "image_processing_glm5_next.py",
)
REQUIRED_ASSETS = ("config.json", "quant_model_description.json", "tokenizer.json", "tokenizer_config.json")
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_file(root: Path, name: str) -> Path:
    """Index and manifest entries must be simple local files, never traversal."""
    if not name or Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"Unsafe checkpoint filename: {name!r}")
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular checkpoint file: {path}")
    return path


def keep_tensor(name: str) -> bool:
    match = TEXT_LAYER.match(name)
    return match is None or int(match[1]) < NUM_LAYERS


def reduced_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    if result.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        raise ValueError("Expected the GLM-5.3-Flash conditional-generation checkpoint")
    text = result["text_config"]
    expected_attention = ["linear_attention"] * 3 + ["deepseek_sparse_attention", "linear_attention"]
    if text["layer_types"][:NUM_LAYERS] != expected_attention:
        raise ValueError("Unexpected GLM-5.3-Flash attention layout")
    if text["mlp_layer_types"][:NUM_LAYERS] != ["dense"] * 3 + ["sparse"] * 2:
        raise ValueError("Unexpected GLM-5.3-Flash MLP layout")
    if text.get("n_routed_experts") != 288 or text.get("num_hidden_layers") != 45:
        raise ValueError("Expected the unmodified 45-layer, 288-expert 0916 checkpoint")
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        if len(text[key]) != text["num_hidden_layers"]:
            raise ValueError(f"Invalid per-layer config field: {key}")
        text[key] = text[key][:NUM_LAYERS]
    linear = text["linear_attn_config"]
    for key in ("kda_layers", "full_attn_layers"):
        linear[key] = [index for index in linear[key] if index < NUM_LAYERS]
    text["num_hidden_layers"] = NUM_LAYERS
    text["num_nextn_predict_layers"] = 0
    return result


def source_files(source: Path) -> tuple[list[str], dict[str, str]]:
    reduced_config(read_json(safe_file(source, "config.json")))
    for name in REQUIRED_ASSETS:
        safe_file(source, name)
    weights = read_json(safe_file(source, INDEX_NAME))["weight_map"]
    selected = {name: shard for name, shard in weights.items() if keep_tensor(name)}
    layers = {int(match[1]) for name in selected if (match := TEXT_LAYER.match(name))}
    if layers != set(range(NUM_LAYERS)):
        raise ValueError(f"Missing retained layers: {layers}")
    assets = [name for name in ASSETS if (source / name).is_file()]
    return sorted({INDEX_NAME, *assets, *selected.values()}), selected


def make_source_manifest(source: Path) -> dict:
    names, _ = source_files(source)
    files = {}
    for name in names:
        path = safe_file(source, name)
        print(f"Hashing source {name}", flush=True)
        files[name] = {"size": path.stat().st_size, "sha256": file_hash(path)}
    body = {
        "schema_version": 1,
        "model": "GLM-5.3-Flash-W8A8-0916",
        "recipe_version": RECIPE_VERSION,
        "retained_layers": list(range(NUM_LAYERS)),
        "files": files,
    }
    return {**body, "source_id": object_hash(body)}


def validate_manifest(manifest: dict) -> None:
    body = {key: value for key, value in manifest.items() if key != "source_id"}
    if manifest.get("source_id") != object_hash(body):
        raise ValueError("Source manifest checksum mismatch")
    if manifest.get("recipe_version") != RECIPE_VERSION or manifest.get("retained_layers") != list(range(NUM_LAYERS)):
        raise ValueError("Source manifest uses a different projection recipe")


def verify_files(root: Path, files: dict) -> None:
    for name, expected in files.items():
        path = safe_file(root, name)
        if path.stat().st_size != expected["size"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= MAX_HEADER_BYTES:
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(stream.read(size))
    offset = size + 8
    cursor = 0
    entries = sorted(
        (value for key, value in header.items() if key != "__metadata__"), key=lambda value: value["data_offsets"]
    )
    for entry in entries:
        start, end = entry["data_offsets"]
        dtype = entry["dtype"]
        if dtype not in DTYPE_BYTES or any(not isinstance(dim, int) or dim < 0 for dim in entry["shape"]):
            raise ValueError(f"Unsupported safetensors dtype/shape: {entry}")
        expected = math.prod(entry["shape"]) * DTYPE_BYTES[dtype]
        if start != cursor or end - start != expected:
            raise ValueError(f"Invalid safetensors tensor offsets: {path}")
        cursor = end
    if offset + cursor != path.stat().st_size:
        raise ValueError(f"Truncated/trailing safetensors payload: {path}")
    return header, offset


def project_shard(source: Path, output: Path, names: set[str]) -> int:
    header, data_start = read_header(source)
    if not names <= header.keys():
        raise ValueError(f"Index references absent tensors in {source}")
    ordered = sorted(names, key=lambda name: header[name]["data_offsets"][0])
    projected = {"__metadata__": header.get("__metadata__", {"format": "pt"})}
    total = 0
    for name in ordered:
        entry = copy.deepcopy(header[name])
        size = entry["data_offsets"][1] - entry["data_offsets"][0]
        entry["data_offsets"] = [total, total + size]
        projected[name] = entry
        total += size
    encoded = json.dumps(projected, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with source.open("rb") as src, output.open("xb") as dst:
        dst.write(struct.pack("<Q", len(encoded)))
        dst.write(encoded)
        for name in ordered:
            begin, end = header[name]["data_offsets"]
            src.seek(data_start + begin)
            remaining = end - begin
            while remaining:
                data = src.read(min(remaining, COPY_BUFFER_BYTES))
                if not data:
                    raise ValueError(f"Source changed/truncated while copying {name}")
                dst.write(data)
                remaining -= len(data)
    return total


def prepare_checkpoint(source: Path, output: Path, manifest: dict) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be disjoint directories")
    validate_manifest(manifest)
    if output.exists():
        derived = read_json(output / DERIVED_MANIFEST)
        if derived["source_id"] != manifest["source_id"] or derived["recipe_version"] != RECIPE_VERSION:
            raise ValueError("Existing derived cache has different provenance; refusing to overwrite")
        verify_files(output, derived["files"])
        return derived
    names, weights = source_files(source)
    if set(names) != set(manifest["files"]):
        raise ValueError("Source file set differs from the reviewed manifest")
    verify_files(source, manifest["files"])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete checkpoint. A failed build never becomes a cache hit.
    with tempfile.TemporaryDirectory(prefix=".glm53-building-", dir=output.parent) as staging:
        stage = Path(staging)
        for name in ASSETS:
            if name in manifest["files"]:
                shutil.copyfile(safe_file(source, name), stage / name)
        write_json(stage / "config.json", reduced_config(read_json(source / "config.json")))
        quant = read_json(source / "quant_model_description.json")
        write_json(
            stage / "quant_model_description.json", {key: value for key, value in quant.items() if keep_tensor(key)}
        )
        shards = sorted(set(weights.values()))
        output_index = {}
        total = 0
        for index, shard in enumerate(shards, 1):
            selected = {name for name, filename in weights.items() if filename == shard}
            target = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            print(f"Projecting {shard} -> {target}", flush=True)
            total += project_shard(safe_file(source, shard), stage / target, selected)
            output_index.update(dict.fromkeys(sorted(selected), target))
        write_json(stage / INDEX_NAME, {"metadata": {"total_size": total}, "weight_map": output_index})
        files = {
            path.name: {"size": path.stat().st_size, "sha256": file_hash(path)} for path in sorted(stage.iterdir())
        }
        derived = {
            "source_id": manifest["source_id"],
            "recipe_version": RECIPE_VERSION,
            "tensor_bytes": total,
            "tensor_count": len(weights),
            "files": files,
        }
        write_json(stage / DERIVED_MANIFEST, derived)
        # Another task must not replace a concurrently published cache.
        stage.rename(output)
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--create-source-manifest",
        action="store_true",
        help="Offline provisioning only; never called by the regression test",
    )
    args = parser.parse_args()
    if args.create_source_manifest:
        if args.manifest.exists():
            parser.error("Refusing to replace an existing source manifest")
        write_json(args.manifest, make_source_manifest(args.source_dir))
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required for projection")
        print(json.dumps(prepare_checkpoint(args.source_dir, args.output_dir, read_json(args.manifest)), indent=2))


if __name__ == "__main__":
    main()
