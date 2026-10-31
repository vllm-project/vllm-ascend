# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the complete fused HcPre operator in one installed revision.

Run each revision in a fresh process with identical arguments. Every sample,
including warmup samples, is retained. Results do not measure model inference.
"""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


def make_inputs(tokens, hidden, signed):
    generator = torch.Generator().manual_seed(1024)
    fan_in = 4 * hidden
    x = (torch.rand(tokens, 4, hidden, generator=generator) * 2).bfloat16()
    fn = torch.rand(24, fan_in, generator=generator) / fan_in
    scale = torch.rand(3, generator=generator) * 2
    base = torch.rand(24, generator=generator) * 2
    if signed:
        x = (x.float() - 1).bfloat16()
        fn = (fn - 0.5 / fan_in) * fan_in**0.5
        base = torch.linspace(-3, 3, 24)
    return tuple(t.npu() for t in (x, fn, scale, base))


def measure(inputs, iterations, samples, warmup, timing, replays):
    def call():
        return torch.ops._C_ascend.npu_hc_pre_v3(
            *inputs, None, hc_mult=4, hc_sinkhorn_iters=iterations, norm_eps=1e-6, hc_eps=1e-6
        )

    if timing == "graph":
        for _ in range(3):
            outputs = call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(replays):
                outputs = call()
        invoke = graph.replay
    else:
        invoke = call
        replays = 1
    events = []
    walls = []
    for _ in range(warmup + samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        begin = time.perf_counter_ns()
        start.record()
        if timing == "graph":
            invoke()
        else:
            outputs = invoke()
        end.record()
        torch.npu.synchronize()
        walls.append((time.perf_counter_ns() - begin) / (1000 * replays))
        events.append(start.elapsed_time(end) * 1000 / replays)
    return tuple(t.cpu() for t in outputs), events, walls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 17, 128, 257, 512])
    parser.add_argument("--hidden", type=int, nargs="+", default=[4096, 7168])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timing", choices=["event", "graph"], default="event")
    parser.add_argument("--replays", type=int, default=20, help="HcPre calls per captured graph")
    args = parser.parse_args()
    if (
        min(*args.tokens, *args.hidden, args.iterations, args.rounds, args.samples, args.replays) <= 0
        or args.warmup < 0
    ):
        parser.error("sizes, iterations, rounds and samples must be positive; warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    if not enable_custom_op():
        raise RuntimeError("The custom HcPre operator must be installed")
    torch_npu.npu.config.allow_internal_format = True
    torch.set_num_threads(1)
    report = {
        "scope": "Complete HcPre operator; NPU events and synchronized host wall time; no model speedup claim",
        "label": args.label,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(0),
        "platform": platform.platform(),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "samples": args.samples,
        "timing": args.timing,
        "calls_per_sample": args.replays if args.timing == "graph" else 1,
        "cases": [],
    }
    with torch.inference_mode():
        for hidden in args.hidden:
            for tokens in args.tokens:
                for signed in (False, True):
                    key = f"t{tokens}-d{hidden}-signed{int(signed)}"
                    inputs = make_inputs(tokens, hidden, signed)
                    original = tuple(t.clone() for t in inputs)
                    records = []
                    expected = None
                    if args.reference is not None:
                        expected = torch.load(args.reference / f"{key}.pt", weights_only=True)
                    for round_index in range(args.rounds):
                        outputs, events, walls = measure(
                            inputs, args.iterations, args.samples, args.warmup, args.timing, args.replays
                        )
                        if expected is not None:
                            for actual, reference in zip(outputs, expected):
                                assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)), key
                        records.append(
                            {
                                "round": round_index,
                                "event_us": events,
                                "wall_us": walls,
                                "event_median_us": statistics.median(events[args.warmup :]),
                                "wall_median_us": statistics.median(walls[args.warmup :]),
                            }
                        )
                    for value, reference in zip(inputs, original):
                        assert torch.equal(value, reference), key
                    destination = args.output / f"{key}.pt"
                    torch.save(outputs, destination)
                    report["cases"].append(
                        {
                            "case": key,
                            "shape": [tokens, 4, hidden],
                            "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                            "bitwise_reference_checked": expected is not None,
                            "rounds": records,
                        }
                    )
                    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(key, [round(r["event_median_us"], 3) for r in records], flush=True)


if __name__ == "__main__":
    main()
