# SPDX-License-Identifier: Apache-2.0
"""Break-even payload of the int8 quantized all-reduce, per TP degree.

This is the harness that produced the per-TP defaults in
``vllm_ascend.distributed.device_communicators.quantized_all_reduce``
(``_DEFAULT_MIN_NUMEL_A2``). Re-run it when the hardware, the HCCL stack or
vLLM's activation layout changes, because the numbers are machine specific.

Why a dependent chain: int8 issues four ops per collective (per-channel absmax,
an 8 KiB fp32 amax all-reduce, quantize, dequantize) where bf16 issues one, so
its host-side cost is 2-4x larger. Timing one collective between two events with
``synchronize()`` around it charges that host cost to every call, which a real
forward pass does not pay. Queuing many *independent* collectives hides it
completely, which is optimistic in the other direction. Linking the calls (the
output of call k is the input of call k+1, as layer k+1 depends on layer k)
keeps the dependency and amortizes the fill/drain over the whole window, and
that is the number which reproduces serving TTFT (see the PR discussion: it
predicts -2.09 s at TP=2 / 8K against -2.07 s measured, while the single-call
convention predicts -1.87 s).

Payloads are given in MiB because that is the only unit in which the shapes are
integers: ``tokens = MiB * 1024 * 1024 / (hidden_size * 2)`` for a bf16 tensor,
and ``min_numel = MiB * 524,288`` elements.

Usage (one process per NPU, the group is the whole world):
    torchrun --standalone --nproc_per_node=8 benchmarks/benchmark_quantized_all_reduce.py
    torchrun --standalone --nproc_per_node=2 benchmarks/benchmark_quantized_all_reduce.py --mib 8 10 12.5 16
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.distributed.device_communicators.quantized_all_reduce import (
    QuantizedAllReduce,
)

DEFAULT_MIB = [8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0, 64.0, 80.0, 100.0, 120.0, 160.0]


def chain_ms(fn, chain: int, rounds: int) -> float:
    """Median time of one call inside a window of ``chain`` dependent calls."""
    samples = []
    for _ in range(rounds):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        start.record()
        for _ in range(chain):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / chain)
    return statistics.median(samples)


def single_ms(fn, iters: int, rounds: int) -> float:
    """Median time of one call with the device drained before and after it."""
    samples = []
    for _ in range(rounds):
        for _ in range(iters):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def rel_l2(qar: QuantizedAllReduce, x: torch.Tensor, group) -> float:
    """Relative L2 error of one quantized all-reduce against bf16."""
    ref = x.clone()
    dist.all_reduce(ref, group=group)
    got = qar.all_reduce(x.clone()).float()
    return ((got - ref.float()).norm() / ref.float().norm()).item()


def break_even(rows: list[dict]) -> float | None:
    """Linear interpolation of the first payload where int8 beats bf16."""
    for prev, cur in zip(rows, rows[1:]):
        if prev["speedup"] < 1.0 <= cur["speedup"]:
            span = cur["speedup"] - prev["speedup"]
            frac = (1.0 - prev["speedup"]) / span
            return prev["mib"] + frac * (cur["mib"] - prev["mib"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mib", type=float, nargs="+", default=DEFAULT_MIB, help="payload sizes to sweep")
    parser.add_argument("--hidden-size", type=int, default=5120, help="last dimension of the payload")
    parser.add_argument("--chain", type=int, default=16, help="dependent calls per timing window")
    parser.add_argument("--rounds", type=int, default=8, help="windows per measurement (median taken)")
    parser.add_argument("--warmup", type=int, default=5, help="untimed calls before each payload")
    parser.add_argument("--single", action="store_true", help="also report the drain-charged convention")
    parser.add_argument("--json", type=str, default="", help="write the table to this path (rank 0)")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="hccl")
    group = dist.group.WORLD
    # min_numel=1 quantizes every payload: this measures cost, not the gate.
    qar = QuantizedAllReduce(group, min_numel=1)
    if not qar.supported:
        if rank == 0:
            print("quantized all-reduce is not supported on this device family; nothing to measure.")
        dist.destroy_process_group()
        return

    if rank == 0:
        print(
            f"ranks={dist.get_world_size()} hidden={args.hidden_size} max_q={qar.max_q} "
            f"chain={args.chain} rounds={args.rounds}",
            flush=True,
        )
        head = f"{'MiB':>7} {'tokens':>8} {'bf16 ms':>8} {'int8 ms':>8} {'speedup':>8} {'rel-L2':>7}"
        if args.single:
            head += f" {'bf16 s':>7} {'int8 s':>7} {'sp_s':>6}"
        print(head, flush=True)

    rows = []
    for mib in args.mib:
        tokens = int(round(mib * 1024 * 1024 / (args.hidden_size * 2)))
        real_mib = tokens * args.hidden_size * 2 / 1024 / 1024
        x = torch.randn(tokens, args.hidden_size, dtype=torch.bfloat16, device="npu")
        ref = x.clone()
        buf = x.clone()  # bf16 in-place chain: every call waits for the previous one

        def bf16_fn(buf=buf):
            dist.all_reduce(buf, group=group)

        def int8_fn():
            nonlocal x
            x = qar.all_reduce(x)

        for _ in range(args.warmup):
            bf16_fn()
            int8_fn()
        torch.npu.synchronize()

        x = ref.clone()
        bf16_ms = chain_ms(bf16_fn, args.chain, args.rounds)
        x = ref.clone()
        int8_ms = chain_ms(int8_fn, args.chain, args.rounds)
        err = rel_l2(qar, ref, group)

        row = {
            "mib": real_mib,
            "tokens": tokens,
            "numel": tokens * args.hidden_size,
            "bf16_ms": bf16_ms,
            "int8_ms": int8_ms,
            "speedup": bf16_ms / int8_ms,
            "rel_l2": err,
        }
        if args.single:
            x = ref.clone()
            bf16_single = single_ms(bf16_fn, 10, args.rounds)
            x = ref.clone()
            int8_single = single_ms(int8_fn, 10, args.rounds)
            row |= {
                "bf16_ms_single": bf16_single,
                "int8_ms_single": int8_single,
                "speedup_single": bf16_single / int8_single,
            }
        rows.append(row)

        if rank == 0:
            line = (
                f"{real_mib:>7.1f} {tokens:>8} {bf16_ms:>8.3f} {int8_ms:>8.3f} "
                f"{row['speedup']:>8.2f} {100 * err:>6.1f}%"
            )
            if args.single:
                line += f" {row['bf16_ms_single']:>7.3f} {row['int8_ms_single']:>7.3f} {row['speedup_single']:>6.2f}"
            print(line, flush=True)

        x = ref
        del buf, ref
        torch.npu.empty_cache()

    if rank == 0:
        crossing = break_even(rows)
        if crossing is None:
            print("no break-even inside the swept range; extend --mib", flush=True)
        else:
            print(f"break-even: {crossing:.1f} MiB = {int(round(crossing * 524288))} elements", flush=True)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(rows, handle, indent=2)
                handle.write("\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
