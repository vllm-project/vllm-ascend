# SPDX-License-Identifier: Apache-2.0
"""Lightweight screen for grid count and token tile; final claims use msprof.

Run on one explicitly selected NPU, after copying this script, the candidate,
and case_generator.py into the same isolated validation environment.
"""

import argparse
import json
import statistics

import torch

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

from case_generator import Case, load_candidate, make_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--e", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--scoring", choices=("softmax", "sigmoid"), default="softmax")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--tokens-per-grid", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--grid-factor", type=int, choices=(1, 2, 4))
    parser.add_argument("--block-t", type=int, choices=(2, 4, 8, 16, 32, 64))
    parser.add_argument("--profile-only", action="store_true", help="Launch once; msprof op owns warmup and replay")
    args = parser.parse_args()
    import torch_npu  # noqa: F401

    init_device_properties_triton()
    vector_cores = get_vectorcore_num()
    module = load_candidate(args.candidate).__globals__
    route = module["_moe_gating_topk_map_record_kernel"]
    reduce = module["_reduce_grid_records_kernel"]
    triton = module["triton"]
    case = Case(args.t, args.e, args.k, args.scoring)
    logits, _, table, _, enabled, valid, _ = make_inputs(case, "npu")
    weights = torch.empty((args.t, args.k), dtype=logits.dtype, device="npu")
    ids = torch.empty((args.t, args.k), dtype=torch.int32, device="npu")
    load = torch.zeros(args.e, dtype=torch.int32, device="npu")

    if args.t <= vector_cores + 1:
        per_grid_values = (args.tokens_per_grid,) if args.tokens_per_grid else (1, 2, 4, 8)
        configurations = [(triton.cdiv(args.t, per_grid), args.block_t or per_grid) for per_grid in per_grid_values]
    else:
        tile = args.block_t or (8 if args.t <= 512 else 32)
        factors = (args.grid_factor,) if args.grid_factor else (1, 2, 4)
        configurations = [(min(args.t, vector_cores * factor), tile) for factor in factors]

    for num_grids, block_t in configurations:
        records = torch.empty((num_grids, args.e), dtype=torch.int32, device="npu")

        def launch(num_grids=num_grids, block_t=block_t, records=records):
            route[(num_grids,)](
                logits,
                logits,
                table,
                enabled,
                valid,
                weights,
                ids,
                records,
                args.t,
                args.e,
                table.shape[0],
                0,
                args.e,
                1.0,
                K=args.k,
                BLOCK_T=block_t,
                BLOCK_E=triton.next_power_of_2(args.e),
                BLOCK_K=triton.next_power_of_2(args.k),
                BLOCK_P=triton.next_power_of_2(args.e),
                HAS_BIAS=False,
                SOFTMAX=args.scoring == "softmax",
                VALID_IS_TENSOR=True,
                num_warps=4,
            )
            reduce[(1,)](
                records,
                load,
                enabled,
                num_grids,
                0,
                args.e,
                BLOCK_GRID=triton.next_power_of_2(num_grids),
                BLOCK_P=triton.next_power_of_2(args.e),
                num_warps=4,
            )

        if args.profile_only:
            launch()
            torch.npu.synchronize()
            print(json.dumps({"t": args.t, "e": args.e, "num_grids": num_grids, "block_t": block_t}), flush=True)
            continue

        for _ in range(3):
            launch()
        torch.npu.synchronize()
        elapsed_us = []
        for _ in range(args.repetitions):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            launch()
            end.record()
            torch.npu.synchronize()
            elapsed_us.append(start.elapsed_time(end) * 1000)
        print(
            json.dumps(
                {
                    "t": args.t,
                    "e": args.e,
                    "k": args.k,
                    "scoring": args.scoring,
                    "vector_cores": vector_cores,
                    "num_grids": num_grids,
                    "block_t": block_t,
                    "event_median_us": statistics.median(elapsed_us),
                    "event_min_us": min(elapsed_us),
                    "event_max_us": max(elapsed_us),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
