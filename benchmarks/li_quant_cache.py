# SPDX-License-Identifier: Apache-2.0
"""Paired complete-chain LI timings against the current main fused store.

Run the one-card test_li_quant_cache.py correctness/graph tests first.
Each reported sample is a device-event batch mean, not a serving ITL.
Graph replay removes Python launch gaps; eager results include those gaps.
"""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.ops.triton.sfa_indexer_store import store_indexer_key_scale
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def native_chain(x, slots, key_cache, scale_cache):
    key, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    store_indexer_key_scale(key, scale, slots, key_cache, scale_cache)
    return key, scale


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def measure(fn, repeats):
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=1_127_168)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--unroll", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.capacity < 6 or min(args.samples, args.repeats, args.unroll) < 1:
        parser.error("capacity must be >=6 and all sample counts must be positive")
    torch.npu.set_device(0)
    torch.manual_seed(42)
    init_device_properties_triton()
    records = []
    for dtype in (torch.bfloat16, torch.float16):
        for slot_dtype in (torch.int32, torch.int64):
            x = torch.randn(6, 128, dtype=dtype, device="npu")
            slots = torch.arange(args.capacity - 6, args.capacity, dtype=slot_dtype, device="npu")
            caches = [
                (
                    torch.zeros(args.capacity, 128, dtype=torch.int8, device="npu"),
                    torch.zeros(args.capacity, 1, dtype=torch.float16, device="npu"),
                )
                for _ in range(2)
            ]
            functions = [
                partial(native_chain, x, slots, *caches[0]),
                partial(torch.ops._C_ascend.li_quant_cache, x, slots, *caches[1]),
            ]
            for fn in functions:
                for _ in range(5):
                    fn()
            for left, right in zip(caches[0], caches[1]):
                assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
            graphs, outputs = [], []
            for fn in functions:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                    for _ in range(args.unroll):
                        output = fn()
                graphs.append(graph)
                outputs.append(output)
            for mode in ("eager", "graph"):
                calls = functions if mode == "eager" else [g.replay for g in graphs]
                divisor = 1 if mode == "eager" else args.unroll
                samples = []
                for index in range(args.samples):
                    order = (0, 1) if index % 2 == 0 else (1, 0)
                    timings = {}
                    for variant in order:
                        timings[variant] = measure(calls[variant], args.repeats) / divisor
                    samples.append({"A_us": timings[0], "B_us": timings[1], "order": order})
                summary = {
                    v: {f"p{p}_us": percentile([s[f"{v}_us"] for s in samples], p / 100) for p in (50, 75, 90, 99)}
                    for v in ("A", "B")
                }
                records.append(
                    dict(
                        dtype=str(dtype),
                        slot_dtype=str(slot_dtype),
                        mode=mode,
                        summary=summary,
                        samples=samples,
                        paired_median_saving_percent=statistics.median(
                            100 * (1 - s["B_us"] / s["A_us"]) for s in samples
                        ),
                    )
                )
    result = dict(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        capacity=args.capacity,
        baseline="main native dynamic_quant + fused key/scale store",
        candidate="AscendC quant/scale/cache fusion",
        sample_unit="device-event batch mean in microseconds",
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "samples"} for row in records], indent=2))


if __name__ == "__main__":
    main()
