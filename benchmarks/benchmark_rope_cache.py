# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Compare strided and contiguous MLA RoPE lookups on one Ascend NPU.

Run on an idle NPU. Reports lookup latency, not end-to-end model throughput.
Example: python benchmarks/benchmark_rope_cache.py --rows 1048576 --dim 64
"""

import argparse
import json
import statistics

import torch
import torch_npu

from vllm_ascend.ops import rotary_embedding as rope


def measure(positions, iterations):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        rope.get_cos_and_sin_mla(positions, use_cache=True)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    """Alternate measurement order and compare identical persistent-buffer paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1048576)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 48])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.rows, args.dim, args.iterations, args.repeats, *args.batches) <= 0 or args.dim % 2:
        parser.error("dimensions/counts must be positive, and --dim must be even")
    torch.npu.set_device(args.device)
    torch.manual_seed(42)
    table = torch.randn(args.rows, args.dim, dtype=torch.bfloat16, device="npu")
    rope._cos_cache = rope._sin_cache = None
    rope._record_cos_and_sin_cache_interleaved(table)
    contiguous = (rope._cos_cache, rope._sin_cache)
    halves = table.view(-1, 2, args.dim // 2).repeat(1, 1, 2).chunk(2, dim=1)
    strided = tuple(half.squeeze(1) for half in halves)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "device": torch.npu.get_device_name(args.device),
                "config": vars(args),
                "old_stride": strided[0].stride(),
                "new_stride": contiguous[0].stride(),
            }
        ),
        flush=True,
    )
    for batch in args.batches:
        positions = torch.linspace(0, args.rows - 1, batch, device="npu").to(torch.int64)
        rope._cos_mla = torch.empty(batch, 1, 1, args.dim, dtype=table.dtype, device="npu")
        rope._sin_mla = torch.empty_like(rope._cos_mla)
        reference = [half.repeat(1, 2)[positions, None, None] for half in table.chunk(2, dim=-1)]
        layouts = {"strided": strided, "contiguous": contiguous}
        for caches in layouts.values():
            rope._cos_cache, rope._sin_cache = caches
            result = rope.get_cos_and_sin_mla(positions, use_cache=True)
            for actual, expected in zip(result, reference):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for _ in range(5):
                rope.get_cos_and_sin_mla(positions, use_cache=True)
        torch.npu.synchronize()
        samples = {name: [] for name in layouts}
        for repeat in range(args.repeats):
            order = list(layouts) if repeat % 2 == 0 else list(reversed(layouts))
            for name in order:
                rope._cos_cache, rope._sin_cache = layouts[name]
                samples[name].append(measure(positions, args.iterations))
        medians = {name: statistics.median(values) for name, values in samples.items()}
        print(
            json.dumps(
                {
                    "batch": batch,
                    "exact_equality": True,
                    "samples_ms": samples,
                    "median_ms": medians,
                    "speedup": medians["strided"] / medians["contiguous"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
