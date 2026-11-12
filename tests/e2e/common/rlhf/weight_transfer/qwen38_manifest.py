# SPDX-License-Identifier: Apache-2.0
"""Prepare a small PLE/MRoPE Qwen3.8 descriptor without loading tensors."""

import argparse
import copy
import json
import math
import shutil
import struct
from pathlib import Path

import regex as re

NUM_LAYERS = 4
CI_HEAD_DIM = 128
CI_PLE_VOCAB_BASE = 4096
CI_ROTARY_DIM = 64
TEXT_PREFIX = "model.language_model."
LAYER_PATTERN = re.compile(r"model\.language_model\.layers\.(\d+)\.")
PLE_SHARD_PATTERN = re.compile(r"\.ngram_embedding\.shard_(\d+)\.weight$")
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "BOOL": 1}
ASSET_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def reduced_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    if result.get("model_type") != "qwen4_exp":
        raise ValueError("Expected the local Qwen3.8-Flash-Next qwen4_exp config")
    text = result["text_config"]
    if text["num_hidden_layers"] < NUM_LAYERS:
        raise ValueError("The source must contain at least four layers")
    text["num_hidden_layers"] = NUM_LAYERS
    text["layer_types"] = text["layer_types"][:NUM_LAYERS]
    if text["layer_types"] != ["linear_attention"] * 3 + ["full_attention"]:
        raise ValueError("Expected three linear-attention layers and one full-attention layer")
    # PLE IDs are one-based. Keep the original PLE path inside the retained layers.
    text["ple_layer_ids"] = [i for i in text["ple_layer_ids"] if 1 <= i <= NUM_LAYERS]
    if not text["ple_layer_ids"]:
        raise ValueError("The retained layers must include PLE")
    text["ngram_vocab_size_base"] = CI_PLE_VOCAB_BASE
    text["head_dim"] = CI_HEAD_DIM
    # Keep the original 64 rotary channels required by the existing QSA indexer.
    text["partial_rotary_factor"] = CI_ROTARY_DIM / CI_HEAD_DIM
    rope = text["rope_parameters"]
    if rope.get("mrope_section") != [11, 11, 10] or not rope.get("mrope_interleaved"):
        raise ValueError("Expected the original interleaved MRoPE layout")
    rope["partial_rotary_factor"] = CI_ROTARY_DIM / CI_HEAD_DIM
    text["mtp_num_hidden_layers"] = 0
    if isinstance(text.get("mtp"), dict):
        text["mtp"]["num_hidden_layers"] = 0
        text["mtp"]["layer_types"] = []
    result["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    result["language_model_only"] = False
    validate_ci_config(result)
    return result


def validate_ci_config(config: dict) -> None:
    """Reject configurations that silently drop PLE/MRoPE or mismatch shapes."""
    text = config["text_config"]
    rope = text.get("rope_parameters", {})
    if text.get("num_hidden_layers") != NUM_LAYERS:
        raise ValueError("The CI case requires exactly four layers")
    if not text.get("ple_layer_ids") or any(not 1 <= i <= NUM_LAYERS for i in text["ple_layer_ids"]):
        raise ValueError("The CI case must retain valid one-based PLE layer IDs")
    if len(set(text["ple_layer_ids"])) != len(text["ple_layer_ids"]):
        raise ValueError("PLE layer IDs must be unique")
    if text.get("head_dim") != CI_HEAD_DIM or text.get("ngram_vocab_size_base") != CI_PLE_VOCAB_BASE:
        raise ValueError("Unexpected CI attention dimension or PLE vocabulary base")
    if config.get("architectures") != ["Qwen4ExpForConditionalGeneration"] or config.get("language_model_only"):
        raise ValueError("Use the conditional-generation architecture to preserve MRoPE")
    if rope.get("mrope_section") != [11, 11, 10] or not rope.get("mrope_interleaved"):
        raise ValueError("The CI case requires interleaved MRoPE")
    if (
        text.get("partial_rotary_factor") != CI_ROTARY_DIM / CI_HEAD_DIM
        or rope.get("partial_rotary_factor") != CI_ROTARY_DIM / CI_HEAD_DIM
    ):
        raise ValueError("Both rotary factors must preserve 64 rotary channels")
    if (
        text.get("indexer_head_dim") != 128
        or text.get("split_ngram_parts", 0) <= 0
        or text.get("make_ngram_vocab_size_divisible_by", 0) <= 0
    ):
        raise ValueError("Invalid QSA indexer or PLE shard geometry")
    if text.get("layer_types") != ["linear_attention"] * 3 + ["full_attention"]:
        raise ValueError("Unexpected reduced attention layout")


def ple_buffer_values(text: dict, dense_layer_id: int) -> dict[str, list[int]]:
    """Match Qwen4Exp's prime vocabulary layout and SplitMix64 hash buffers."""
    heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
    if heads <= 0 or text["ple_embed_dim"] % heads:
        raise ValueError("PLE embedding dimension must be divisible by its ngram head count")
    primes = []
    candidate = text["ngram_vocab_size_base"]
    while len(primes) < (dense_layer_id + 1) * heads:
        if candidate >= 2 and all(candidate % d for d in range(2, math.isqrt(candidate) + 1)):
            primes.append(candidate)
        candidate += 1
    sizes = primes[dense_layer_id * heads :]
    offsets = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += size
    mask = (1 << 64) - 1
    gamma = 0x9E3779B97F4A7C15
    half_bound = max(1, (((1 << 63) - 1) // text["vocab_size"]) // 2)
    seed = text.get("seed", 1234) + 10007 * dense_layer_id
    multipliers = []
    for i in range(text["ngram_size"]):
        value = (seed + gamma * (i + 2)) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value = (value ^ (value >> 31)) & mask
        multipliers.append(2 * (value % half_bound) + 1)
    return dict(layer_multipliers=multipliers, ngram_heads_vocab_sizes=sizes, ngram_heads_offsets=offsets)


def selected_tensor(name: str) -> bool:
    if name == "lm_head.weight":
        return True
    if not name.startswith(TEXT_PREFIX) or ".mtp." in name:
        return False
    match = LAYER_PATTERN.match(name)
    return match is None or int(match[1]) < NUM_LAYERS


def resized_tensor(name: str, shape: list[int], config: dict) -> tuple[list[int], list[int] | None]:
    text = config["text_config"]
    values = None
    if ".ple.ple_embedding." in name:
        match = LAYER_PATTERN.match(name)
        dense_id = text["ple_layer_ids"].index(int(match[1]) + 1)
        buffers = ple_buffer_values(text, dense_id)
        leaf = name.rsplit(".", 1)[-1]
        if leaf in buffers:
            values = buffers[leaf]
            shape = [len(values)]
        shard = PLE_SHARD_PATTERN.search(name)
        if shard:
            divisor = text["make_ngram_vocab_size_divisible_by"]
            rows = math.ceil(sum(buffers["ngram_heads_vocab_sizes"]) / divisor) * divisor
            parts = text["split_ngram_parts"]
            index = int(shard[1])
            if index >= parts:
                raise ValueError("PLE shard index exceeds split_ngram_parts")
            shard_rows = math.ceil(rows / parts)
            shape = [
                max(0, min(shard_rows, rows - index * shard_rows)),
                text["ple_embed_dim"] // len(buffers["ngram_heads_vocab_sizes"]),
            ]
    elif ".self_attn." in name and ".indexer." not in name:
        suffix = name.rsplit(".self_attn.", 1)[1]
        hidden, head = text["hidden_size"], text["head_dim"]
        shapes = {
            "q_proj.weight": [2 * text["num_attention_heads"] * head, hidden],
            "k_proj.weight": [text["num_key_value_heads"] * head, hidden],
            "v_proj.weight": [text["num_key_value_heads"] * head, hidden],
            "o_proj.weight": [hidden, text["num_attention_heads"] * head],
            "q_norm.weight": [head],
            "k_norm.weight": [head],
        }
        shape = shapes.get(suffix, shape)
    return shape, values


def checkpoint_manifest(source: Path) -> dict:
    config = reduced_config(json.loads((source / "config.json").read_text()))
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}
    tensors = []
    for name, shard in sorted(weight_map.items()):
        if not selected_tensor(name):
            continue
        shard_path = (source / shard).resolve()
        if not shard_path.is_relative_to(source.resolve()):
            raise ValueError(f"Shard escapes model directory: {shard}")
        if shard not in headers:
            with shard_path.open("rb") as stream:
                raw = stream.read(8)
                if len(raw) != 8:
                    raise ValueError(f"Truncated safetensors file: {shard}")
                length = struct.unpack("<Q", raw)[0]
                if not 0 < length <= 64 * 2**20:
                    raise ValueError(f"Invalid safetensors header length: {length}")
                headers[shard] = json.loads(stream.read(length))
        entry = headers[shard][name]
        dtype, original = entry["dtype"], entry["shape"]
        if dtype not in DTYPE_BYTES or not original or not all(isinstance(s, int) and s > 0 for s in original):
            raise ValueError(f"Unsupported tensor layout: {name}, {entry}")
        if entry["data_offsets"][1] - entry["data_offsets"][0] != math.prod(original) * DTYPE_BYTES[dtype]:
            raise ValueError(f"Inconsistent tensor header: {name}")
        shape, values = resized_tensor(name, original, config)
        if not all(s > 0 for s in shape):
            raise ValueError(f"Empty reduced tensor: {name}")
        tensor = dict(name=name, shape=shape, dtype=dtype, size_bytes=math.prod(shape) * DTYPE_BYTES[dtype])
        if values is not None:
            if dtype != "I64":
                raise ValueError(f"Expected int64 PLE buffer: {name}")
            tensor["values"] = values
        tensors.append(tensor)
    layers = {int(m[1]) for tensor in tensors if (m := LAYER_PATTERN.match(tensor["name"]))}
    if layers != set(range(NUM_LAYERS)) or not any(".ple." in x["name"] for x in tensors):
        raise ValueError("Missing retained layers or PLE parameters")
    return dict(
        schema_version=2,
        source=str(source.resolve()),
        weight_values="deterministic_generated_not_pretrained",
        num_hidden_layers=NUM_LAYERS,
        ple_enabled=True,
        head_dim=CI_HEAD_DIM,
        mrope=True,
        total_tensor_bytes=sum(x["size_bytes"] for x in tensors),
        config=config,
        tensors=tensors,
    )


def prepare_descriptor(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate directories")
    if output.exists():
        raise FileExistsError("Output already exists; choose a new descriptor directory")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing processor/tokenizer asset: {name}")
    manifest = checkpoint_manifest(source)
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(manifest["config"], indent=2) + "\n")
    (output / "weight_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ASSET_NAMES:
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_descriptor(args.source, args.output)
    print(
        json.dumps(
            {
                "descriptor": str(args.output),
                "payload_gib": manifest["total_tensor_bytes"] / 2**30,
                "ple_enabled": True,
                "head_dim": CI_HEAD_DIM,
                "mrope": True,
            },
            indent=2,
        )
    )
