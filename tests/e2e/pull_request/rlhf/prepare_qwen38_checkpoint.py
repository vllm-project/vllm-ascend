# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare the four-layer no-PLE profile without materializing a state dict.

Copy tensor byte ranges into independent one-tensor safetensors shards. Neither
excluded tensors nor entire source shards are read into memory. The output is
published only after its index, schema and content hashes have been verified.
"""

import argparse
import copy
import hashlib
import json
import math
import shutil
import struct
import tempfile
from pathlib import Path

import regex as re

PROFILE = "qwen38-flash-next-4layer-no-ple-bf16"
NUM_LAYERS = 4
EXPECTED_TENSORS = 101
EXPECTED_BYTES = 23_294_733_120
EXPECTED_PLE_TENSORS = 137
EXPECTED_PLE_BYTES = 102_466_171_160
COPY_CHUNK_BYTES = 8 * 2**20
MAX_HEADER_BYTES = 16 * 2**20
LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention", "full_attention")
GLOBAL_NAMES = frozenset(
    {
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    }
)
LAYER_PATTERN = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
AUXILIARY_NAMES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.jinja",
    }
)


def read_json(path: Path) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique_pairs)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(COPY_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def validate_auxiliary_files(directory: Path, resolved_tokenizer: Path | None = None) -> None:
    if resolved_tokenizer is not None:
        original = directory / "tokenizer.json"
        with original.open("rb") as stream:
            prefix = stream.read(256)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            pointer = prefix.decode().splitlines()
            expected = next(line.removeprefix("oid sha256:") for line in pointer if line.startswith("oid sha256:"))
            size = int(next(line.removeprefix("size ") for line in pointer if line.startswith("size ")))
            if resolved_tokenizer.stat().st_size != size:
                raise ValueError("Resolved tokenizer size differs from source LFS pointer")
        else:
            expected = file_sha256(original)
        if file_sha256(resolved_tokenizer) != expected:
            raise ValueError("Resolved tokenizer SHA-256 differs from source")
    for filename in sorted(AUXILIARY_NAMES):
        path = (
            resolved_tokenizer
            if filename == "tokenizer.json" and resolved_tokenizer is not None
            else directory / filename
        )
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Unresolved Git LFS pointer: {filename}")
        if filename.endswith(".json"):
            read_json(path)
    if not (directory / "tokenizer.json").is_file():
        raise ValueError("Missing tokenizer.json")


def local_file(directory: Path, filename: str) -> Path:
    """Index and manifest entries must resolve inside the checkpoint."""
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Invalid checkpoint file: {filename}")
    return path


def retain_name(name: str) -> bool:
    if "ple" in name.split("."):
        return False
    layer = LAYER_PATTERN.match(name)
    if layer:
        return int(layer[1]) < NUM_LAYERS
    if name in GLOBAL_NAMES:
        return True
    if name.startswith("model.language_model.") and not name.startswith("model.language_model.mtp."):
        raise ValueError(f"Unknown language-model global weight: {name}")
    if not name.startswith(("model.visual.", "model.mtp.", "mtp.")):
        raise ValueError(f"Unknown checkpoint namespace: {name}")
    return False


def read_schema(directory: Path) -> dict[str, dict]:
    """Check the actual shards, including unindexed extras and byte ranges."""
    index = read_json(directory / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    schema = {}
    for filename in sorted(set(weight_map.values())):
        path = local_file(directory, filename)
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated header: {filename}")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > MAX_HEADER_BYTES:
                raise ValueError(f"Header too large: {filename}")
            header = json.loads(stream.read(length))
        data_start = 8 + length
        expected_start = 0
        for name, entry in sorted(
            ((n, e) for n, e in header.items() if n != "__metadata__"),
            key=lambda pair: pair[1]["data_offsets"][0],
        ):
            start, end = entry["data_offsets"]
            if start != expected_start or end < start or data_start + end > path.stat().st_size:
                raise ValueError(f"Invalid tensor byte range: {name}")
            expected_start = end
            if name in schema or weight_map.get(name) != filename:
                raise ValueError(f"Duplicate or unindexed tensor: {name}")
            schema[name] = {**entry, "file": filename, "data_start": data_start, "bytes": end - start}
        if data_start + expected_start != path.stat().st_size:
            raise ValueError(f"Trailing shard bytes: {filename}")
    if set(schema) != set(weight_map):
        raise ValueError("Index and shard tensor sets differ")
    return schema


def validate_config(config: dict) -> None:
    text = config["text_config"]
    if config["model_type"] != "qwen4_exp" or config["architectures"] != ["Qwen4ExpForConditionalGeneration"]:
        raise ValueError("Expected original Qwen4Exp architecture")
    required = {
        "num_hidden_layers": NUM_LAYERS,
        "layer_types": list(LAYER_TYPES),
        "ple_layer_ids": [],
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "hidden_size": 2560,
        "dtype": "bfloat16",
    }
    for name, value in required.items():
        if text.get(name) != value:
            raise ValueError(f"Invalid no-PLE config: {name}")


def validate_schema(schema: dict, *, strict: bool = True) -> None:
    if not schema or not schema.keys() >= GLOBAL_NAMES:
        raise ValueError("Missing global weights")
    layers = set()
    for name, entry in schema.items():
        if not retain_name(name) or entry["dtype"] != "BF16":
            raise ValueError(f"Unexpected tensor or dtype: {name}")
        if any(not isinstance(size, int) or size < 0 for size in entry["shape"]):
            raise ValueError(f"Invalid tensor shape: {name}")
        if entry["bytes"] != 2 * math.prod(entry["shape"]):
            raise ValueError(f"Invalid BF16 tensor size: {name}")
        if layer := LAYER_PATTERN.match(name):
            layers.add(int(layer[1]))
    if layers != set(range(NUM_LAYERS)):
        raise ValueError("Missing decoder layers")
    if strict and (len(schema) != EXPECTED_TENSORS or sum(e["bytes"] for e in schema.values()) != EXPECTED_BYTES):
        raise ValueError("Source profile changed: expected 101 BF16 tensors and 23,294,733,120 bytes")


def validate_checkpoint(directory: Path, manifest_sha256: str, *, strict: bool = True) -> dict:
    manifest_path = directory / "qwen38-no-ple-manifest.json"
    if file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("Manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    if manifest["profile"] != PROFILE:
        raise ValueError("Wrong checkpoint profile")
    tracked = manifest["files"]
    actual_files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual_files != set(tracked) | {manifest_path.name}:
        raise ValueError("Untracked or missing checkpoint files")
    for filename, digest in tracked.items():
        if file_sha256(local_file(directory, filename)) != digest:
            raise ValueError(f"File SHA-256 mismatch: {filename}")
    validate_config(read_json(directory / "config.json"))
    validate_auxiliary_files(directory)
    expected_config = copy.deepcopy(manifest["source_config"])
    expected_config["text_config"].update(num_hidden_layers=NUM_LAYERS, layer_types=list(LAYER_TYPES), ple_layer_ids=[])
    if read_json(directory / "config.json") != expected_config:
        raise ValueError("Config changed outside the three permitted fields")
    schema = read_schema(directory)
    validate_schema(schema, strict=strict)
    recorded = manifest["tensors"]
    if set(recorded) != set(schema):
        raise ValueError("Manifest and checkpoint tensor sets differ")
    for name, entry in schema.items():
        for field in ("shape", "dtype", "file", "bytes"):
            if recorded[name][field] != entry[field]:
                raise ValueError(f"Manifest schema mismatch: {name}.{field}")
    total = sum(e["bytes"] for e in schema.values())
    if (
        manifest["total_size"] != total
        or read_json(directory / "model.safetensors.index.json")["metadata"]["total_size"] != total
    ):
        raise ValueError("Incorrect total_size")
    return manifest


def copy_tensor(source: Path, target: Path, name: str, entry: dict) -> str:
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            name: {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": [0, entry["bytes"]],
            },
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    digest = hashlib.sha256()
    with source.open("rb") as reader, target.open("wb") as writer:
        writer.write(struct.pack("<Q", len(header)))
        writer.write(header)
        reader.seek(entry["data_start"] + entry["data_offsets"][0])
        remaining = entry["bytes"]
        while remaining:
            block = reader.read(min(COPY_CHUNK_BYTES, remaining))
            if not block:
                raise ValueError(f"Truncated source tensor: {name}")
            writer.write(block)
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def prepare_checkpoint(
    source: Path,
    output: Path,
    source_revision: str,
    *,
    strict: bool = True,
    resolved_tokenizer: Path | None = None,
) -> str:
    """Fail closed on schema drift; leave source files and existing output intact."""
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be a new directory separate from the source")
    validate_auxiliary_files(source, resolved_tokenizer)
    schema = read_schema(source)
    retained = {n: e for n, e in sorted(schema.items()) if retain_name(n)}
    validate_schema(retained, strict=strict)
    excluded_ple = {
        n: e
        for n, e in schema.items()
        if "ple" in n.split(".") and (m := LAYER_PATTERN.match(n)) and int(m[1]) < NUM_LAYERS
    }
    if strict and (
        len(excluded_ple) != EXPECTED_PLE_TENSORS
        or sum(e["bytes"] for e in excluded_ple.values()) != EXPECTED_PLE_BYTES
    ):
        raise ValueError("Source PLE profile changed")
    source_config = read_json(source / "config.json")
    config = copy.deepcopy(source_config)
    config["text_config"].update(num_hidden_layers=NUM_LAYERS, layer_types=list(LAYER_TYPES), ple_layer_ids=[])
    validate_config(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        tensors, weight_map = {}, {}
        for i, (name, entry) in enumerate(retained.items(), 1):
            filename = f"model-{i:05d}-of-{len(retained):05d}.safetensors"
            payload_hash = copy_tensor(local_file(source, entry["file"]), staging / filename, name, entry)
            tensors[name] = {key: entry[key] for key in ("shape", "dtype", "bytes")}
            tensors[name].update(file=filename, source_file=entry["file"], source_tensor_sha256=payload_hash)
            weight_map[name] = filename
        write_json(staging / "config.json", config)
        total = sum(e["bytes"] for e in retained.values())
        write_json(
            staging / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": weight_map}
        )
        for filename in sorted(AUXILIARY_NAMES):
            if (source / filename).is_file():
                asset = (
                    resolved_tokenizer
                    if filename == "tokenizer.json" and resolved_tokenizer is not None
                    else source / filename
                )
                shutil.copyfile(asset, staging / filename)
        if not (staging / "tokenizer.json").is_file():
            raise ValueError("Missing tokenizer.json")
        manifest = {
            "version": 1,
            "profile": PROFILE,
            "source_revision": source_revision,
            "source_config_sha256": file_sha256(source / "config.json"),
            "source_index_sha256": file_sha256(source / "model.safetensors.index.json"),
            "source_tokenizer_file_sha256": file_sha256(source / "tokenizer.json"),
            "source_config": source_config,
            "tensors": tensors,
            "total_size": total,
            "excluded_ple": excluded_ple,
            "files": {p.name: file_sha256(p) for p in sorted(staging.iterdir())},
        }
        write_json(staging / "qwen38-no-ple-manifest.json", manifest)
        digest = file_sha256(staging / "qwen38-no-ple-manifest.json")
        validate_checkpoint(staging, digest, strict=strict)
        staging.rename(output)
        return digest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True, help="Immutable source model revision")
    parser.add_argument("--resolved-tokenizer", type=Path, help="Resolve tokenizer LFS pointer with an exact-hash file")
    args = parser.parse_args()
    digest = prepare_checkpoint(
        args.source, args.output, args.source_revision, resolved_tokenizer=args.resolved_tokenizer
    )
    print(f"manifest_sha256={digest}")


if __name__ == "__main__":
    main()
