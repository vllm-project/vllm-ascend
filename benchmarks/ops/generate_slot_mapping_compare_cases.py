#!/usr/bin/env python3
"""Generate varied single-op cases for slot-mapping comparisons.

Example:
    python benchmarks/ops/generate_slot_mapping_compare_cases.py \
        --output /home/lingmutian/tmp/slot_mapping_compare_cases.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BLOCK_SIZE = 128
TRITON_BLOCK_SIZE = 1024
PHYSICAL_BLOCK_UPPER_BOUND = 320


def tensor(
    shape: list[int],
    dtype: str,
    initializer: str | None = None,
    **kwargs: int,
) -> dict[str, object]:
    spec: dict[str, object] = {"shape": shape, "dtype": dtype, "device": "npu:0"}
    if initializer is not None:
        spec["initializer"] = initializer
    spec.update(kwargs)
    return spec


def next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def make_case(
    name: str,
    num_reqs: int,
    tokens_per_request: int,
    *,
    max_num_tokens: int | None = None,
    pos_start: int = 0,
    cp_size: int = 1,
    cp_rank: int = 0,
    cp_interleave: int = 1,
) -> dict[str, object]:
    total_tokens = num_reqs * tokens_per_request
    max_num_tokens = total_tokens if max_num_tokens is None else max_num_tokens
    max_position = pos_start + total_tokens - 1
    block_table_width = max_position // BLOCK_SIZE + 1
    window_size = next_power_of_2((TRITON_BLOCK_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)

    pointee = tensor(
        [num_reqs, block_table_width],
        "torch.int32",
        "randint",
        low=0,
        high=PHYSICAL_BLOCK_UPPER_BOUND,
    )
    return {
        "name": name,
        "kernel": "_compute_slot_mappings_kernel",
        "grid": [1, num_reqs + 1],
        "launch_config": {
            "num_reqs": num_reqs,
            "tokens_per_request": tokens_per_request,
            "max_num_tokens": max_num_tokens,
            "pos_start": pos_start,
            "block_table_width": block_table_width,
            "cp_size": cp_size,
        },
        "arguments": {
            "max_num_tokens": max_num_tokens,
            "idx_mapping": tensor([num_reqs], "torch.int32", "arange"),
            "query_start_loc": tensor(
                [num_reqs + 1],
                "torch.int32",
                "arange",
                start=0,
                step=tokens_per_request,
            ),
            "pos": tensor([total_tokens], "torch.int64", "arange", start=pos_start),
            "block_table_ptrs": {
                "shape": [1],
                "dtype": "torch.uint64",
                "device": "npu:0",
                "initializer": "data_ptrs",
                "pointees": [pointee],
            },
            "block_table_strides": tensor([1], "torch.int64", "full", value=block_table_width),
            "block_sizes": tensor([1], "torch.int32", "full", value=BLOCK_SIZE),
            # These are consumed only by the current upstream V2 reference
            # kernel. The Ascend wrapper ignores them.
            "kernel_block_sizes": tensor([1], "torch.int32", "full", value=BLOCK_SIZE),
            "slot_mapping_enabled": tensor([1], "torch.bool", "ones"),
            "slot_mappings": tensor([1, max_num_tokens], "torch.int32"),
            "slot_mappings_stride": max_num_tokens,
            "cp_rank": cp_rank,
            "CP_SIZE": cp_size,
            "CP_INTERLEAVE": cp_interleave,
            "PAD_ID": -1,
            "TRITON_BLOCK_SIZE": TRITON_BLOCK_SIZE,
            "BLOCK_TABLE_WINDOW_SIZE": window_size,
        },
    }


def make_cases() -> list[dict[str, object]]:
    cases = [
        make_case("decode-1", 1, 1),
        make_case("decode-40", 40, 1),
        make_case("decode-64", 64, 1),
        make_case("decode-256", 256, 1),
        make_case("decode-1024", 1024, 1),
        make_case("tile-1", 1, TRITON_BLOCK_SIZE),
        make_case("tile-8", 8, TRITON_BLOCK_SIZE),
        make_case("tile-64", 64, TRITON_BLOCK_SIZE),
        make_case("tile-256", 256, TRITON_BLOCK_SIZE),
        make_case("tile-1024", 1024, TRITON_BLOCK_SIZE),
        make_case("cross-block-1x1024", 1, TRITON_BLOCK_SIZE, pos_start=BLOCK_SIZE - 1),
        make_case("cross-tile-1x1025", 1, TRITON_BLOCK_SIZE + 1, pos_start=BLOCK_SIZE - 1),
        make_case("long-1x8192", 1, 8192),
        make_case("long-64x8192", 64, 8192),
        make_case("long-64x32768", 64, 32768),
        make_case("captured-tail", 8, 9, max_num_tokens=8192),
        make_case("cp2-tile-64-rank0", 64, TRITON_BLOCK_SIZE, cp_size=2),
        make_case("cp4-tile-64-rank2", 64, TRITON_BLOCK_SIZE, cp_size=4, cp_rank=2),
    ]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file")
    args = parser.parse_args()

    cases = make_cases()
    args.output.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
