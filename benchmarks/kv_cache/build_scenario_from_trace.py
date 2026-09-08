#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Convert a real-text JSONL serving trace into an allocator replay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-bytes", type=int)
    parser.add_argument("--arrival-gap", type=int, default=0)
    parser.add_argument("--duration", type=int, default=16)
    parser.add_argument("--attention-block-size", type=int)
    parser.add_argument("--mamba-block-size", type=int)
    parser.add_argument("--target-bucket", type=int)
    parser.add_argument("--bounded-superpage-mib", default="4,8,16,32,64")
    args = parser.parse_args()

    profile: dict[str, Any] = json.loads(args.profile.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.target_bucket is not None:
        records = [record for record in records if int(record["target_prompt_tokens"]) == args.target_bucket]
    groups = profile["groups"]
    layer_counts = {int(group["num_layers"]) for group in groups}
    if len(layer_counts) != 1:
        raise ValueError("all HMA groups must have the same layer count")
    replicated_layer_slices = layer_counts.pop()
    memory_bytes = (
        args.memory_bytes
        if args.memory_bytes is not None
        else int(profile["total_raw_tensor_bytes"]) // replicated_layer_slices
    )

    requests = []
    for index, record in enumerate(records):
        num_tokens = int(record["prompt_tokens"]) + int(record.get("max_output_tokens", 0))
        page_counts = {}
        for group in groups:
            if group["contains_mamba"]:
                block_size = args.mamba_block_size or int(
                    group.get(
                        "experimental_block_size_tokens",
                        group["block_size_tokens"],
                    )
                )
                count = min(
                    math.ceil(num_tokens / block_size) + int(group.get("num_speculative_blocks", 0)),
                    2 + int(group.get("num_speculative_blocks", 0)),
                )
            else:
                block_size = args.attention_block_size or int(
                    group.get(
                        "experimental_block_size_tokens",
                        group["block_size_tokens"],
                    )
                )
                count = math.ceil(num_tokens / block_size)
            page_counts[group["name"]] = count
        requests.append(
            {
                "request_id": record["request_id"],
                "source_dataset": record.get("source_dataset"),
                "target_prompt_tokens": record.get("target_prompt_tokens"),
                "start_step": index * args.arrival_gap,
                "duration_steps": args.duration,
                "page_counts": page_counts,
            }
        )

    scenario = {
        "description": "Real-text allocator replay derived from LongBench/LongBench-v2.",
        "source_profile": str(args.profile),
        "source_trace": str(args.trace),
        "memory_bytes": memory_bytes,
        "replicated_layer_slices": replicated_layer_slices,
        "uniform_page_size_bytes": max(int(group["page_size_bytes"]) for group in groups),
        "uniform_slots_per_page": {group["name"]: int(group.get("uniform_slots_per_page", 1)) for group in groups},
        "page_types": [
            {
                "name": group["name"],
                "page_size_bytes": int(
                    group.get(
                        "experimental_small_page_size_bytes",
                        group["page_size_bytes"],
                    )
                ),
            }
            for group in groups
        ],
        "bounded_superpage_sizes_bytes": [
            int(float(value) * 1024**2) for value in args.bounded_superpage_mib.split(",") if value
        ],
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(scenario, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Replay scenario written to {args.output}")


if __name__ == "__main__":
    main()
