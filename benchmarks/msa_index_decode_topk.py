# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MSA decode TopK with fixed model configuration and dynamic shapes."""

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.msa import msa_m3_index_postprocess as op

BATCH_SIZES = (1, 8, 16, 32, 64, 128, 256, 512)
BLOCK_COUNTS = (256, 512, 1024, 2048, 4096)
HEADS = 4
TOPK = 16
BLOCK_SIZE = 128
INIT_BLOCKS = 0
LOCAL_BLOCKS = 1


def cache_entries() -> dict[str, int | None]:
    """Read this triton-ascend version's in-process specialization cache."""
    result = {}
    for name, kernel in (
        ("prepare", op._ascendc_decode_score_prepare_kernel),
        ("postprocess", op._ascendc_decode_topk_postprocess_kernel),
    ):
        cache = getattr(kernel, "cache", None)
        result[name] = sum(len(entries) for entries in cache.values()) if cache is not None else None
    return result


def measure(fn, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.npu.synchronize()
    elapsed = sorted(start.elapsed_time(end) * 1000 for start, end in zip(starts, ends))
    return {"median_us": statistics.median(elapsed), "p95_us": elapsed[math.ceil(repeats * 0.95) - 1]}


def make_case(batch: int, blocks: int):
    generator = torch.Generator().manual_seed(42)
    reference = torch.rand((HEADS, batch, blocks), generator=generator)
    # Exercise different visible lengths within the allocated block-table width.
    visible = torch.linspace(max(TOPK, blocks // 2), blocks, batch).to(torch.int32)
    context_cpu = visible * BLOCK_SIZE - 1
    score = reference.to("npu")
    context = context_cpu.to("npu")
    prepared = torch.empty_like(score)
    raw = torch.empty((HEADS, batch, TOPK), dtype=torch.int64, device="npu")
    values = torch.empty((HEADS, batch, TOPK), dtype=torch.float32, device="npu")
    indices = torch.empty_like(raw, dtype=torch.int32)
    counts = torch.empty((HEADS, batch), dtype=torch.int32, device="npu")

    def prepare():
        op._ascendc_decode_score_prepare_kernel[(batch, HEADS, op.triton.cdiv(blocks, op._PREPARE_TILE))](
            score,
            prepared,
            context,
            *score.stride(),
            *prepared.stride(),
            context.stride(0),
            blocks,
            blocks,
            1,
            INIT_BLOCKS=INIT_BLOCKS,
            LOCAL_BLOCKS=LOCAL_BLOCKS,
            BLOCK_SIZE=BLOCK_SIZE,
            TILE=op._PREPARE_TILE,
        )

    def topk():
        torch.topk(prepared, TOPK, dim=-1, out=(values, raw))

    def postprocess():
        op._ascendc_decode_topk_postprocess_kernel[(batch, HEADS)](
            raw,
            indices,
            counts,
            context,
            *raw.stride(),
            *indices.stride(),
            *counts.stride(),
            context.stride(0),
            blocks,
            1,
            TOPK=TOPK,
            BLOCK_SIZE=BLOCK_SIZE,
            TILE=op.triton.next_power_of_2(TOPK),
        )

    def full():
        return op.index_decode_topk_fused(score, context, 1, TOPK, blocks, INIT_BLOCKS, LOCAL_BLOCKS)

    def validate(actual_indices, actual_counts):
        columns = torch.arange(blocks).view(1, 1, -1)
        expected = reference.masked_fill(columns >= visible.view(1, -1, 1), -float("inf"))
        expected.masked_fill_(columns == (visible - 1).view(1, -1, 1), 1.0e29)
        actual_indices, actual_counts = actual_indices.cpu(), actual_counts.cpu()
        assert torch.all(actual_counts == TOPK)
        assert torch.all((actual_indices >= 0) & (actual_indices < visible.view(1, -1, 1)))
        sorted_ids = actual_indices.sort(dim=-1).values
        assert torch.all(sorted_ids[..., 1:] != sorted_ids[..., :-1])
        selected = expected.gather(-1, actual_indices.long()).sort(dim=-1, descending=True).values
        torch.testing.assert_close(selected, expected.topk(TOPK, dim=-1).values, rtol=0, atol=0)

    return full, prepare, topk, postprocess, validate, indices, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    records = []
    for batch in BATCH_SIZES:
        for blocks in BLOCK_COUNTS:
            full, prepare, topk, postprocess, validate, indices, counts = make_case(batch, blocks)
            torch.npu.synchronize()
            before = cache_entries()
            start = time.perf_counter()
            output = full()
            torch.npu.synchronize()
            first_ms = (time.perf_counter() - start) * 1000
            validate(*output)
            record = {"bs": batch, "num_blocks": blocks, "first_call_wall_ms": first_ms}
            record.update({f"cache_before_{k}": v for k, v in before.items()})
            for name, fn in (("full", full), ("prepare", prepare), ("topk", topk), ("postprocess", postprocess)):
                record.update({f"{name}_{k}": v for k, v in measure(fn, args.warmup, args.repeats).items()})
            validate(indices, counts)
            record.update({f"cache_after_{k}": v for k, v in cache_entries().items()})
            records.append(record)
            print(json.dumps(record), flush=True)
    warmed_cache = cache_entries()
    # Interleave shapes after the full sweep; a new cache entry indicates specialization.
    replay = []
    for batch, blocks in ((512, 4096), (1, 256), (256, 1024), (8, 2048), (32, 512), (1, 4096)):
        full, _, _, _, validate, _, _ = make_case(batch, blocks)
        output = full()
        validate(*output)
        replay.append({"bs": batch, "num_blocks": blocks, **measure(full, args.warmup, args.repeats)})
    summary = {
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "triton": op.triton.__version__,
        "heads": HEADS,
        "topk": TOPK,
        "block_size": BLOCK_SIZE,
        "init_blocks": INIT_BLOCKS,
        "local_blocks": LOCAL_BLOCKS,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "correctness_cases": len(records),
        "warmed_cache": warmed_cache,
        "replay_cache": cache_entries(),
        "dynamic_replay": replay,
        "timing": (
            "NPU Event elapsed time in eager mode; includes launch gaps. "
            "First-call wall time includes compilation/cache loading. "
            "Stage TopK uses preallocated outputs; full interface allocates outputs."
        ),
    }
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
