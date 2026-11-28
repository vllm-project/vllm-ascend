# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise split Gather writes and zero fill with a B10 fixture."""

import argparse
import json
from dataclasses import replace

import benchmark
import torch


@torch.inference_mode()
def run(args):
    device = "npu:0"
    torch.npu.set_device(device)
    batch, context, heads, topk, rope = 10, 131072, 64, 2048, 64
    storage = benchmark.prepare_storage(batch * context // benchmark.PAGE_SIZE, device, args.seed, rope)
    inputs, metadata = benchmark.make_inputs(batch, context, heads, topk, args.seed, storage, "token")
    indices, block_table, query_ends, kv_lengths = metadata
    first_visible = kv_lengths - benchmark.QUERY_TOKENS + 1
    causal_limits = (first_visible[:, None] + torch.arange(benchmark.QUERY_TOKENS)).flatten()
    if args.partial_cast:
        # Preserve the panel split in lane 0 of each 512-token tile. Tails in
        # the other lanes exercise the vector-cast limits around 16 rows.
        indices[:, 0, ::512] = causal_limits[:, None]
        filtered_indices_per_row = topk // 512
        for stop, rows_in_tail in zip(range(512, topk + 1, 512), (15, 16, 17, 31), strict=True):
            indices[:, 0, stop - 32 + rows_in_tail : stop] = causal_limits[:, None]
            filtered_indices_per_row += 32 - rows_in_tail
    else:
        # One future index per 256-token AIV lane gives an initial 31-row copy,
        # then 32-row copies straddling each 128-token KV panel, and a zero tail.
        indices[:, 0, ::256] = causal_limits[:, None]
        filtered_indices_per_row = topk // 256
    if args.native_baseline:
        # The stock operator expects a valid prefix followed by -1 padding.
        # Keep the same surviving indices and their order as the stress fixture.
        valid = (indices >= 0) & (indices < causal_limits[:, None, None])
        prefix_length = topk - filtered_indices_per_row
        indices[..., :prefix_length] = indices[valid].reshape(*indices.shape[:-1], prefix_length)
        indices[..., prefix_length:] = -1
    inputs = replace(inputs, indices=indices.to(device))
    rows, golden = benchmark.selected_golden(inputs, (indices, block_table, query_ends, kv_lengths), batch, args.scale)
    candidate = benchmark.load_candidate(args.candidate, 512, True)
    if args.native_baseline:
        baseline_output = benchmark.native(inputs, args.scale, implementation="torch-npu")
    else:
        baseline = benchmark.load_candidate(args.baseline_candidate, 512, True)
        baseline_output = baseline(inputs, args.scale)
    base = benchmark.output_tensor(baseline_output, inputs).cpu()
    actual = benchmark.output_tensor(candidate(inputs, args.scale), inputs).cpu()
    print(
        json.dumps(
            {
                "batch": batch,
                "seed": args.seed,
                "filtered_indices_per_row": filtered_indices_per_row,
                "baseline_golden_max_abs": benchmark.check(base[rows], golden, 0.015, 0.02)["max_abs"],
                "candidate_golden_max_abs": benchmark.check(actual[rows], golden, 0.015, 0.02)["max_abs"],
                "candidate_baseline_max_abs": benchmark.check(actual, base, 0.015, 0.02)["max_abs"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=93211)
    parser.add_argument("--scale", type=float, default=benchmark.DEFAULT_SCALE)
    parser.add_argument("--candidate", default="qsfa_native_reference:qsfa_glm54_mtp5_blocked_kv")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--baseline-candidate", default="qsfa_native_reference:qsfa_glm54_mtp5")
    baseline.add_argument("--native-baseline", action="store_true", help="compare against the torch-npu operator")
    parser.add_argument(
        "--partial-cast",
        action="store_true",
        help="filter 53 entries per row, exercising 15-, 16- and 17-row cast tails in the TileLang stress case",
    )
    run(parser.parse_args())
