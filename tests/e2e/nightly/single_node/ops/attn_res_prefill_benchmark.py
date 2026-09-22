# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native AttnRes binaries with identical inputs and graph timing.

Run the baseline with --write-reference, then each candidate with
--reference pointing to that same directory. Token/head shapes match the
PP4/TP8, 16K scheduled-token workload when sequence parallelism is enabled.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401


def make_inputs(tokens, hidden, valid, no_add, pp):
    generator = torch.Generator().manual_seed(9121 + valid)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).bfloat16().npu()

    prefix = randn(tokens, hidden)
    addend = None if no_add else randn(tokens, hidden)
    storage = randn(tokens, 10, hidden)
    bank = storage[:, 1:9]
    bank[:, valid:] = float("nan")
    proj = (randn(1, hidden).float() / hidden**0.5).bfloat16()
    norm = randn(hidden)
    output_norm = None if pp else randn(hidden)
    return prefix, addend, bank, proj, norm, output_norm


@torch.inference_mode()
def run_case(args, valid, first=False, pp=False, no_add=False, write_idx=-1):
    if first:
        name, no_add, write_idx = "first", True, 0
    elif pp:
        name = "pp_add"
    elif no_add:
        name = f"post_valid{valid}"
    elif write_idx >= 0:
        name = f"boundary_valid{valid}"
    else:
        name = f"valid{valid}"
    prefix, addend, bank, proj, norm, output_norm = make_inputs(args.tokens, args.hidden, valid, no_add, pp)

    native = torch.ops._C_ascend.attn_res_fwd
    fused = native.fused_prefill if args.prefill_kernel else native.fused

    def call():
        return fused(
            prefix,
            addend,
            bank,
            proj,
            norm,
            1e-5,
            valid,
            output_norm,
            1e-5,
            write_idx,
            True,
            not pp,
        )

    result = call()
    torch.npu.synchronize()
    reference_path = args.reference / f"{name}.pt"
    if args.write_reference:
        torch.save(tuple(x.cpu() for x in result), reference_path)
    else:
        baseline = torch.load(reference_path, map_location="cpu", weights_only=True)
        # No reduction/rounding change is allowed in this optimization.
        for observed, expected in zip(result, baseline):
            torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
    expected_prefix = prefix if addend is None else prefix + addend
    torch.testing.assert_close(result[1], expected_prefix, rtol=0, atol=0)
    if write_idx >= 0:
        torch.testing.assert_close(bank[:, write_idx], expected_prefix, rtol=0, atol=0)

    # Time the production save_materialized=False path. The above extra
    # materialized output is used only to validate its bitwise contract.
    def timed_call():
        return fused(
            prefix,
            addend,
            bank,
            proj,
            norm,
            1e-5,
            valid,
            output_norm,
            1e-5,
            write_idx,
            False,
            not pp,
        )

    for _ in range(args.warmup):
        timed_call()
    torch.npu.synchronize()
    captured = torch.npu.NPUGraph()
    with torch.npu.graph(captured):
        # Capture a burst, so Python graph-replay submission gaps do not
        # dominate these sub-200us kernels. A trial launches this graph once.
        for _ in range(args.repeats):
            graph_result = timed_call()
    for _ in range(args.warmup):
        captured.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(args.trials):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        captured.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / args.repeats)
    # Graph replay must load current weights, not persist UB state between calls.
    output_norm.zero_() if output_norm is not None else prefix.zero_()
    captured.replay()
    expected = timed_call()
    torch.npu.synchronize()
    for actual, golden in zip(graph_result, expected):
        torch.testing.assert_close(actual, golden, rtol=0, atol=0)
    return {"case": name, "median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--prefill-kernel", action="store_true", help="Use the isolated prefill-only CANN op")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=64)
    args = parser.parse_args()
    args.reference.mkdir(parents=True, exist_ok=True)
    results = [run_case(args, 0, first=True)]
    results += [run_case(args, valid) for valid in (1, 2, 4, 8)]
    results += [run_case(args, valid, no_add=True) for valid in (1, 2)]
    results.append(run_case(args, 1, write_idx=1))
    results.append(run_case(args, 0, pp=True))
    by_name = {r["case"]: r["median_us"] for r in results}
    modeled_chain = (
        by_name["first"]
        + 22 * by_name["valid1"]
        + by_name["post_valid1"]
        + by_name["boundary_valid1"]
        + 20 * by_name["valid2"]
        + by_name["post_valid2"]
        + by_name["pp_add"]
    )
    report = {
        "label": args.label,
        "prefill_kernel": args.prefill_kernel,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "graph": True,
        "numerical_contract": "bitwise_vs_saved_baseline",
        "modeled_47_call_chain_us": modeled_chain,
        "scope": "sum of isolated medians; not a measured model chain or PP4 latency",
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
