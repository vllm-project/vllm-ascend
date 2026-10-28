# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired sampling-mask benchmark; run manually on an idle Ascend device.

Supply the original sampling_mask.py with --baseline-file. Single-row cases
compare the original and changed operators. Multirow cases compare kernels on
the same pre-expanded rows: the original API does not support request-major
outputs, so its flattened-row reference is not an end-to-end MTP baseline.
Eager Event time includes host dispatch gaps; graph measurements amortize a
captured sequence of launches. All correctness copies precede timed regions.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch_npu  # noqa: F401
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.sample import sampling_mask as changed


class LegacySamplingMaskTensors(NamedTuple):
    token_ids: torch.Tensor
    packed_mask: torch.Tensor
    counts: torch.Tensor
    vocab_size: int


def _inputs(num_reqs, rows_per_request, vocab_size):
    num_rows = num_reqs * rows_per_request
    logits = torch.full((num_rows, vocab_size), -float("inf"), device="npu")
    logits[:, :20] = 0.0
    logits[:, 20] = float("nan")
    logits[:, 21] = float("inf")
    sampled_cpu = torch.arange(num_reqs, dtype=torch.int32) % (rows_per_request + 1)
    sampled_cpu[0] = rows_per_request
    sampled = sampled_cpu.to("npu")
    offsets = torch.arange(num_reqs + 1, dtype=torch.int32, device="npu") * rows_per_request
    flags = (torch.arange(rows_per_request)[None, :] < sampled_cpu[:, None]).flatten().to(torch.int32).to("npu")
    expected = np.zeros((num_rows, vocab_size), dtype=bool)
    expected[flags.cpu().numpy().astype(bool), :20] = True
    return logits, offsets, sampled, flags, expected


def _paths(baseline, logits, offsets, sampled, flags, rows_per_request):
    num_rows, vocab_size = logits.shape
    packed_old = torch.empty((num_rows, (vocab_size + 7) // 8), dtype=torch.uint8, device="npu")
    packed_new = torch.empty_like(packed_old)
    counts_old = torch.empty(num_rows, dtype=torch.int32, device="npu")
    counts_new = torch.empty_like(counts_old)

    def baseline_kernel():
        baseline._pack_sampling_mask_kernel_ascend[(num_rows,)](
            logits,
            logits.stride(0),
            logits.stride(1),
            flags,
            packed_old,
            packed_old.stride(0),
            counts_old,
            vocab_size,
            BLOCK_SIZE=baseline._BLOCK_SIZE,
        )
        return packed_old, counts_old

    def changed_kernel():
        changed._pack_sampling_mask_kernel_ascend[(num_rows,)](
            logits,
            logits.stride(0),
            logits.stride(1),
            offsets,
            sampled,
            packed_new,
            packed_new.stride(0),
            counts_new,
            vocab_size,
            ROWS_PER_REQUEST=rows_per_request,
            BLOCK_SIZE=changed._BLOCK_SIZE,
        )
        return packed_new, counts_new

    def changed_api():
        result = changed.sampling_mask_from_logits_npu(
            SamplingMaskTensors, logits, offsets, sampled, 20, rows_per_request
        )
        return result.packed_mask, result.counts

    paths = {"baseline_kernel": baseline_kernel, "changed_kernel": changed_kernel, "changed_api": changed_api}
    if rows_per_request == 1:

        def baseline_api():
            result = baseline.sampling_mask_from_logits_npu(LegacySamplingMaskTensors, logits, flags, 20)
            return result.packed_mask, result.counts

        paths["baseline_api"] = baseline_api
    return paths


def _correctness(paths, expected, vocab_size):
    for fn in paths.values():
        packed, counts = fn()
        actual = np.unpackbits(packed.cpu().numpy(), axis=1, count=vocab_size, bitorder="little")
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(counts.cpu().numpy(), expected.sum(axis=1))


def _measure(paths, mode, trials, repeats, rng):
    callables = {}
    graphs = []
    for name, fn in paths.items():
        if mode == "graph":
            if not name.endswith("_kernel"):
                continue
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(repeats):
                    fn()
            graphs.append(graph)
            callables[name] = graph.replay
        else:
            callables[name] = fn
    records = {name: [] for name in callables}
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    for _ in range(trials):
        order = list(callables)
        rng.shuffle(order)
        for name in order:
            fn = callables[name]
            torch.npu.synchronize()
            wall_start = time.perf_counter()
            start.record()
            for _ in range(1 if mode == "graph" else repeats):
                fn()
            end.record()
            end.synchronize()
            wall_us = (time.perf_counter() - wall_start) * 1e6 / repeats
            records[name].append({"event_us": start.elapsed_time(end) * 1000 / repeats, "wall_us": wall_us})
    return {
        name: {
            "median_event_us": statistics.median(s["event_us"] for s in samples),
            "median_wall_us": statistics.median(s["wall_us"] for s in samples),
            "samples": samples,
        }
        for name, samples in records.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-graph-ratio", type=float, help="Optional regression gate against the original row kernel"
    )
    args = parser.parse_args()
    if args.trials < 1 or args.repeats < 1:
        parser.error("trials and repeats must be positive")
    if args.output.exists():
        parser.error("output must not exist; preserve earlier measurements")
    init_device_properties_triton()
    spec = importlib.util.spec_from_file_location("baseline_sampling_mask", args.baseline_file)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    rng = random.Random(args.seed)
    report = {
        "trials": args.trials,
        "repeats": args.repeats,
        "seed": args.seed,
        "baseline_sha256": hashlib.sha256(args.baseline_file.read_bytes()).hexdigest(),
        "changed_sha256": hashlib.sha256(Path(changed.__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "device": torch.npu.get_device_name(),
        "cases": [],
    }
    for num_reqs in (1, 8, 64):
        for vocab_size in (4097, 248320):
            for rows_per_request in (1, 3):
                logits, offsets, sampled, flags, expected = _inputs(num_reqs, rows_per_request, vocab_size)
                paths = _paths(baseline, logits, offsets, sampled, flags, rows_per_request)
                _correctness(paths, expected, vocab_size)
                for fn in paths.values():
                    for _ in range(5):
                        fn()
                torch.npu.synchronize()
                case = {
                    "num_reqs": num_reqs,
                    "rows_per_request": rows_per_request,
                    "vocab_size": vocab_size,
                    "active_rows": int(flags.sum().cpu()),
                    "correctness_passed": True,
                }
                for mode in ("eager", "graph"):
                    case[mode] = _measure(paths, mode, args.trials, args.repeats, rng)
                case["graph_kernel_ratio"] = (
                    case["graph"]["changed_kernel"]["median_event_us"]
                    / case["graph"]["baseline_kernel"]["median_event_us"]
                )
                case["comparison"] = (
                    "single_row_baseline" if rows_per_request == 1 else "preexpanded_row_kernel_reference"
                )
                report["cases"].append(case)
                print(json.dumps({k: v for k, v in case.items() if k not in ("eager", "graph")}), flush=True)
                del paths, logits, offsets, sampled, flags, expected
    report["max_graph_kernel_ratio"] = max(c["graph_kernel_ratio"] for c in report["cases"])
    report["max_graph_ratio_limit"] = args.max_graph_ratio
    report["performance_gate_passed"] = (
        report["max_graph_kernel_ratio"] <= args.max_graph_ratio if args.max_graph_ratio is not None else None
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    return 1 if report["performance_gate_passed"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
