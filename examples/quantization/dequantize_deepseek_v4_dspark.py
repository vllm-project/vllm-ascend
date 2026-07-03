#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dequantize DeepSeek V4 DSpark MXFP weights to bf16 safetensors.

This helper is intended for DeepSeek V4 DSpark checkpoints whose dense linear
weights use block FP8 scales and whose MoE expert weights use packed FP4 plus
per-32 scales. The output checkpoint keeps normal sidecar files, consumes the
paired ``*.scale`` tensors, and writes bf16 model shards plus a fresh HF index.

Example:

    python examples/quantization/dequantize_deepseek_v4_dspark.py \
        --source /path/to/DeepSeek-V4-Flash-DSpark \
        --output /path/to/DeepSeek-V4-Flash-DSpark-bf16
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
FP4_PAIR_TABLE = torch.stack(
    (
        FP4_TABLE[torch.arange(256, dtype=torch.uint8).bitwise_and(0x0F).long()],
        FP4_TABLE[torch.arange(256, dtype=torch.uint8).bitwise_right_shift(4).long()],
    ),
    dim=-1,
)


def _float8_weight_dtypes() -> set[torch.dtype]:
    dtypes = set()
    for name in ("float8_e4m3fn", "float8_e5m2"):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            dtypes.add(dtype)
    return dtypes


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def paired_scale_name(weight_name: str) -> str:
    return f"{weight_name.removesuffix('.weight')}.scale"


def paired_weight_name(scale_name: str) -> str:
    return f"{scale_name.removesuffix('.scale')}.weight"


def dequantize_expert_fp4(
    fp4_weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if fp4_weight.dtype != torch.int8:
        raise ValueError(f"expected packed fp4 int8 weight, got {fp4_weight.dtype}")
    if fp4_weight.ndim != 2:
        raise ValueError(f"expected 2D packed fp4 weight, got {fp4_weight.ndim}D")

    out_dim, packed_in_dim = fp4_weight.shape
    in_dim = packed_in_dim * 2
    expected_scale_shape = (out_dim, in_dim // 32)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            "unexpected fp4 scale shape: "
            f"weight={tuple(fp4_weight.shape)} "
            f"scale={tuple(weight_scale.shape)} "
            f"expected={expected_scale_shape}"
        )

    values = FP4_PAIR_TABLE[fp4_weight.view(torch.uint8).long()].reshape(out_dim, in_dim)
    expanded_scale = weight_scale.to(torch.float32).repeat_interleave(32, dim=1)
    return (values * expanded_scale[:, :in_dim]).to(dtype=torch.bfloat16)


def dequantize_dense_fp8(
    fp8_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    block_n, block_k = block_size
    if fp8_weight.ndim != 2:
        raise ValueError(f"expected 2D fp8 weight, got {fp8_weight.ndim}D")

    n, k = fp8_weight.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    if tuple(weight_scale.shape) != (n_tiles, k_tiles):
        raise ValueError(
            "unexpected fp8 scale shape: "
            f"weight={tuple(fp8_weight.shape)} "
            f"scale={tuple(weight_scale.shape)} "
            f"block_size={(block_n, block_k)}"
        )

    expanded_scale = weight_scale.to(torch.float32).repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
    expanded_scale = expanded_scale[:n, :k]
    return (fp8_weight.to(torch.float32) * expanded_scale).to(dtype=torch.bfloat16)


def copy_sidecar_files(source: Path, output: Path) -> None:
    skip_names = {"model.safetensors.index.json"}
    for item in source.iterdir():
        if item.name in skip_names or item.name.startswith("."):
            continue
        if item.suffix == ".safetensors":
            continue

        dst = output / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, dst)


def write_config(source: Path, output: Path) -> None:
    with (source / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)

    config["quantization_config"] = None
    config["expert_dtype"] = None
    config["torch_dtype"] = "bfloat16"
    config["dspark_mtp_dequantized_to_bf16"] = True
    config["dspark_full_dequantized_to_bf16"] = True

    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def save_current_shard(
    output: Path,
    shard_tensors: OrderedDict[str, torch.Tensor],
    shard_index: int,
    weight_map: dict[str, str],
) -> int:
    if not shard_tensors:
        return shard_index

    filename = f"model-{shard_index:05d}.safetensors"
    save_file(dict(shard_tensors), output / filename, metadata={"format": "pt"})
    for name in shard_tensors:
        weight_map[name] = filename
    shard_tensors.clear()
    return shard_index + 1


def get_block_size(source: Path, source_index: dict) -> tuple[int, int]:
    block_size = tuple(source_index.get("metadata", {}).get("weight_block_size", []))
    if len(block_size) != 2:
        with (source / "config.json").open(encoding="utf-8") as f:
            config = json.load(f)
        block_size = tuple(config.get("quantization_config", {}).get("weight_block_size", [128, 128]))
    return int(block_size[0]), int(block_size[1])


def convert(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if source == output or source in output.parents:
        raise ValueError("--output must not be the source directory or a child of it")

    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing source index: {index_path}")
    with index_path.open(encoding="utf-8") as f:
        source_index = json.load(f)

    source_weight_map: dict[str, str] = source_index["weight_map"]
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copy_sidecar_files(source, output)
    write_config(source, output)

    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in source_weight_map.items():
        keys_by_shard[shard].append(name)

    scale_keys_to_skip = {
        name for name in source_weight_map if name.endswith(".scale") and paired_weight_name(name) in source_weight_map
    }
    paired_weights = {
        name for name in source_weight_map if name.endswith(".weight") and paired_scale_name(name) in source_weight_map
    }

    cross_shard = [
        (name, paired_scale_name(name))
        for name in paired_weights
        if source_weight_map[name] != source_weight_map[paired_scale_name(name)]
    ]
    if cross_shard:
        raise ValueError(f"cross-shard weight/scale pairs are not supported: {cross_shard[:5]}")

    fp8_weight_dtypes = _float8_weight_dtypes()
    block_size = get_block_size(source, source_index)
    max_shard_bytes = int(args.max_shard_gb * 1024**3)
    output_weight_map: dict[str, str] = {}
    shard_tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
    current_bytes = 0
    shard_index = 1
    total_tensor_bytes = 0
    counters: Counter[str] = Counter()
    start = time.time()

    source_shards = list(OrderedDict.fromkeys(source_weight_map.values()))
    for source_shard_idx, shard_name in enumerate(source_shards, start=1):
        if args.limit_shards and source_shard_idx > args.limit_shards:
            break
        shard_path = source / shard_name
        shard_start = time.time()
        converted_in_shard = 0
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for name in keys_by_shard[shard_name]:
                if args.max_tensors and counters["converted_tensors"] >= args.max_tensors:
                    break
                if name in scale_keys_to_skip:
                    counters["skip_quant_scale"] += 1
                    continue

                weight = shard.get_tensor(name)
                if name in paired_weights:
                    scale = shard.get_tensor(paired_scale_name(name))
                    if ".experts." in name and weight.dtype == torch.int8:
                        output_tensor = dequantize_expert_fp4(weight, scale)
                        counters["mxfp4_to_bf16"] += 1
                    elif weight.dtype in fp8_weight_dtypes:
                        output_tensor = dequantize_dense_fp8(weight, scale, block_size)
                        counters["mxfp8_to_bf16"] += 1
                    else:
                        raise ValueError(f"paired quant tensor {name} has unsupported dtype {weight.dtype}")
                else:
                    counters[f"preserve_{str(weight.dtype).replace('torch.', '')}"] += 1
                    output_tensor = weight.contiguous()

                output_tensor = output_tensor.contiguous()
                nbytes = tensor_nbytes(output_tensor)
                if shard_tensors and current_bytes + nbytes > max_shard_bytes:
                    shard_index = save_current_shard(output, shard_tensors, shard_index, output_weight_map)
                    current_bytes = 0

                shard_tensors[name] = output_tensor
                current_bytes += nbytes
                total_tensor_bytes += nbytes
                counters["converted_tensors"] += 1
                converted_in_shard += 1

        print(
            f"[{source_shard_idx}/{len(source_shards)}] {shard_name}: "
            f"converted={converted_in_shard} "
            f"elapsed={time.time() - shard_start:.1f}s "
            f"total={counters['converted_tensors']}",
            flush=True,
        )
        if args.max_tensors and counters["converted_tensors"] >= args.max_tensors:
            break

    shard_index = save_current_shard(output, shard_tensors, shard_index, output_weight_map)
    num_output_shards = shard_index - 1

    renamed_weight_map: dict[str, str] = {}
    for old_idx in range(1, num_output_shards + 1):
        old_name = f"model-{old_idx:05d}.safetensors"
        new_name = f"model-{old_idx:05d}-of-{num_output_shards:05d}.safetensors"
        (output / old_name).rename(output / new_name)
        for tensor_name, filename in output_weight_map.items():
            if filename == old_name:
                renamed_weight_map[tensor_name] = new_name

    index = {
        "metadata": {
            "format": "dspark_full_bf16",
            "source_model": source.name,
            "total_size": str(total_tensor_bytes),
            "converted_tensors": str(counters["converted_tensors"]),
            "skipped_quant_scales": str(counters["skip_quant_scale"]),
        },
        "weight_map": renamed_weight_map,
    }
    with (output / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    report = {
        "source": str(source),
        "output": str(output),
        "elapsed_seconds": round(time.time() - start, 3),
        "output_shards": num_output_shards,
        "output_tensor_bytes": total_tensor_bytes,
        "output_file_bytes": sum(p.stat().st_size for p in output.glob("model-*.safetensors")),
        "actions": dict(counters),
    }
    with (output / "conversion_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source DeepSeek V4 DSpark checkpoint directory.")
    parser.add_argument("--output", required=True, help="Output bf16 checkpoint directory.")
    parser.add_argument("--max-shard-gb", type=float, default=4.0, help="Maximum output shard size in GiB.")
    parser.add_argument("--overwrite", action="store_true", help="Remove the output directory if it already exists.")
    parser.add_argument(
        "--limit-shards", type=int, default=0, help="Only convert the first N source shards. For debug only."
    )
    parser.add_argument("--max-tensors", type=int, default=0, help="Stop after N output tensors. For debug only.")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
