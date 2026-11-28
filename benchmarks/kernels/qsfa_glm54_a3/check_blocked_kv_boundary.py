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
    # One future index per 256-token AIV lane gives an initial 31-row copy,
    # then 32-row copies straddling each 128-token KV panel, and a zero tail.
    indices[:, 0, ::256] = causal_limits[:, None]
    inputs = replace(inputs, indices=indices.to(device))
    rows, golden = benchmark.selected_golden(inputs, (indices, block_table, query_ends, kv_lengths), batch, args.scale)
    baseline = benchmark.load_candidate("qsfa_native_reference:qsfa_glm54_mtp5", 512, True)
    candidate = benchmark.load_candidate(args.candidate, 512, True)
    base = benchmark.output_tensor(baseline(inputs, args.scale), inputs).cpu()
    actual = benchmark.output_tensor(candidate(inputs, args.scale), inputs).cpu()
    print(
        json.dumps(
            {
                "batch": batch,
                "seed": args.seed,
                "filtered_indices_per_row": topk // 256,
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
    run(parser.parse_args())
