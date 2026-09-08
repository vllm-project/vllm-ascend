# SPDX-License-Identifier: Apache-2.0

"""Convert an NPU KV cache profile and token trace into a replay scenario."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _page_count(group: dict[str, Any], num_tokens: int) -> int:
    block_size = int(group.get("experimental_block_size_tokens", group["block_size_tokens"]))
    token_blocks = math.ceil(num_tokens / block_size)
    if group["contains_mamba"]:
        speculative = int(group.get("num_speculative_blocks", 0))
        # In align mode, old position-indexed states are released. Upstream's
        # MambaSpec budgets at most 2 + speculative blocks per live request.
        return min(token_blocks + speculative, 2 + speculative)
    return token_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--token-lengths",
        default="2048,4096,8192,16384",
        help="Comma-separated peak sequence lengths, one per synthetic request.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the token-length pattern to build a larger concurrent workload.",
    )
    parser.add_argument("--arrival-gap", type=int, default=1)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument(
        "--bounded-superpage-mib",
        default="2,8,32,64",
        help="Comma-separated superpage sizes to sweep.",
    )
    args = parser.parse_args()

    with args.profile.open(encoding="utf-8") as file:
        profile: dict[str, Any] = json.load(file)

    groups = profile["groups"]
    layer_counts = {int(group["num_layers"]) for group in groups}
    if len(layer_counts) != 1:
        raise ValueError(f"this replay expects HMA groups with the same number of layers; got {sorted(layer_counts)}")
    replicated_layer_slices = layer_counts.pop()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    lengths = [int(value) for value in args.token_lengths.split(",") if value] * args.repeat
    superpage_sizes = [int(float(value) * 1024**2) for value in args.bounded_superpage_mib.split(",") if value]
    requests = []
    for index, num_tokens in enumerate(lengths):
        requests.append(
            {
                "request_id": f"r{index}_len{num_tokens}",
                "start_step": index * args.arrival_gap,
                "duration_steps": args.duration,
                "page_counts": {group["name"]: _page_count(group, num_tokens) for group in groups},
            }
        )

    scenario = {
        "description": "Generated from an NPU KVCacheConfig profile. Mamba demand assumes align mode.",
        "source_profile": str(args.profile),
        # HMA applies the same BlockPool ID to identical raw-tensor slices for
        # every layer in a group. Model one slice: both memory and object sizes
        # are then per-layer, while the resulting admissible request count is
        # unchanged for all replicated slices.
        "memory_bytes": int(profile["total_raw_tensor_bytes"]) // replicated_layer_slices,
        "replicated_layer_slices": replicated_layer_slices,
        "uniform_page_size_bytes": max(int(group["page_size_bytes"]) for group in groups),
        "uniform_slots_per_page": {group["name"]: int(group.get("uniform_slots_per_page", 1)) for group in groups},
        "page_types": [
            {
                "name": group["name"],
                "page_size_bytes": int(group.get("experimental_small_page_size_bytes", group["page_size_bytes"])),
            }
            for group in groups
        ],
        "bounded_superpage_sizes_bytes": superpage_sizes,
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(scenario, file, indent=2)
    print(f"Replay scenario written to {args.output}")


if __name__ == "__main__":
    main()
