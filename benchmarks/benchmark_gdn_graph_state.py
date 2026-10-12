# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Compare per-group copies/padding with batched GDN graph input writes.

Run on an NPU: python benchmarks/benchmark_gdn_graph_state.py --num-groups 23
Reports synchronized wall time including dispatch and validation overhead.
This microbenchmark does not measure model TPOT or DFlash acceptance.
"""

import argparse
from time import perf_counter

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.v2.mamba.graph_state import GDNGraphStateUpdater


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-groups", type=int, default=23)
    parser.add_argument("--num-reqs", type=int, default=8)
    parser.add_argument("--graph-reqs", type=int, default=16)
    parser.add_argument("--state-slots", type=int, default=4)
    parser.add_argument("--clear-spec", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    if min(args.num_groups, args.graph_reqs, args.state_slots, args.iterations) < 1 or args.warmup < 0:
        parser.error("Group, graph, slot and iteration counts must be positive; warmup must be non-negative")
    if not 0 <= args.num_reqs <= args.graph_reqs:
        parser.error("num-reqs must be between zero and graph-reqs")
    torch.npu.set_device(0)
    columns = args.state_slots + 2
    tables = torch.arange(args.num_groups * args.graph_reqs * columns, dtype=torch.int32, device="npu").view(
        args.num_groups, args.graph_reqs, columns
    )
    sources = [table[:, : args.state_slots] for table in tables]
    outputs = [torch.empty((args.graph_reqs, args.state_slots), dtype=torch.int32, device="npu") for _ in sources]
    resets = [torch.empty_like(dst) if args.clear_spec else None for dst in outputs]
    updates = [
        (src, dst, args.num_reqs, args.graph_reqs, 0, reset) for src, dst, reset in zip(sources, outputs, resets)
    ]
    updater = GDNGraphStateUpdater()

    def per_group():
        for src, dst, reset in zip(sources, outputs, resets):
            dst[: args.num_reqs].copy_(src[: args.num_reqs])
            dst[args.num_reqs :].fill_(0)
            if reset is not None:
                reset.fill_(-1)

    def batched():
        updater.apply(updates)

    per_group()
    expected = [dst.clone() for dst in outputs]
    for dst in outputs:
        dst.fill_(-99)
    for reset in resets:
        if reset is not None:
            reset.fill_(99)
    batched()
    for dst, reference, reset in zip(outputs, expected, resets):
        torch.testing.assert_close(dst, reference, rtol=0, atol=0)
        if reset is not None:
            torch.testing.assert_close(reset, torch.full_like(reset, -1), rtol=0, atol=0)
    for name, run in (("per-group Torch", per_group), ("batched GDN graph state", batched)):
        for _ in range(args.warmup):
            run()
        torch.npu.synchronize()
        start = perf_counter()
        for _ in range(args.iterations):
            run()
        torch.npu.synchronize()
        elapsed_us = (perf_counter() - start) * 1e6 / args.iterations
        print(
            f"{name}: {elapsed_us:.2f} us/step ({args.num_groups} groups, {args.num_reqs}/{args.graph_reqs} requests)"
        )


if __name__ == "__main__":
    main()
