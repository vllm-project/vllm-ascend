# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare A3 fused SiTU with stock GMM + SiTU using alternating graph replays.

Run from a checkout with rebuilt OPP and framework extensions. Both paths clone
input because stock GMM may mutate it. Timings include that common clone.
Use --profile separately: profiler-instrumented times are not benchmark results.
The adjacent native correctness suite provides the fixed-seed INT4 test fixture.
"""

import argparse
import json
import os
import runpy
import statistics
from pathlib import Path

import torch
import torch_npu


def main():
    # Build stable inputs, capture each path, then time warmed ABBA replays.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts", default="3,0,2")
    p.add_argument("--k", type=int, default=7168)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--capacity", type=int, default=0)
    p.add_argument("--as-list", action="store_true")
    p.add_argument("--profile")
    p.add_argument("--metric", default="PipeUtilization")
    p.add_argument("--out")
    p.add_argument("--impl", choices=["both", "fused", "split"], default="both")
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--repeats", type=int, default=6)
    args = p.parse_args()
    torch.manual_seed(721)
    torch.npu.config.allow_internal_format = True
    m = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_gmm_situ_quant_msd.py"
        )
    )
    counts = [int(v) for v in args.counts.split(",")]
    rows = sum(counts)
    capacity = max(rows, args.capacity)
    if rows <= 0 or min(counts) < 0 or min(args.batch, args.repeats) <= 0:
        p.error("Use nonnegative counts with at least one live row, and positive batch/repeats")
    data = m["_layer"](len(counts), args.k, args.n, True, 721)
    if args.as_list:
        data = (
            [w.view(torch.int8).clone().view(torch.int32) for w in data[0]],
            [s.clone() for s in data[1]],
            *data[2:],
        )
    weights, scales = data[:2] if args.as_list else ([data[4][0]], [data[4][1]])
    assist = torch.stack([8 * w.float().sum(0) * s for w, s in zip(data[2], data[3])]).npu()
    assists = [a.clone() for a in assist.unbind()] if args.as_list else [assist]
    x = torch.randint(-128, 128, (capacity, args.k), dtype=torch.int8, device="npu")
    xscale = torch.full((capacity,), 0.01, device="npu")
    groups = torch.tensor(counts, dtype=torch.int64, device="npu")

    def fused():
        return m["_call"](
            x.clone(), xscale, data, groups, 1, True, assists if args.as_list else assist, not args.as_list
        )

    def split():
        h = torch_npu.npu_grouped_matmul(
            x=[x.clone()],
            weight=weights,
            scale=[s.unsqueeze(-2) for s in scales],
            bias=assists,
            per_token_scale=[xscale],
            split_item=2,
            group_type=0,
            group_list=groups,
            group_list_type=1,
            output_dtype=torch.bfloat16,
        )[0]
        return torch.ops._C_ascend.dequant_situ_quant(
            x=h,
            weight_scale=None,
            activation_scale=None,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=groups,
            beta=4.0,
            linear_beta=25.0,
            activate_left=True,
            quant_mode="dynamic",
        )

    fs = {"fused": fused, "split": split}
    if args.impl != "both":
        fs = {args.impl: fs[args.impl]}
    graphs = {}
    graph_memory = {}
    for name, fn in fs.items():
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        memory_before = torch.npu.memory_allocated()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = [fn() for _ in range(args.batch)]
        for _ in range(5):
            graph.replay()
        torch.npu.synchronize()
        graphs[name] = graph, outputs
        graph_memory[name] = torch.npu.memory_allocated() - memory_before

    if args.profile:
        prof = torch_npu.profiler
        with prof.profile(
            activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
            record_shapes=True,
            experimental_config=prof._ExperimentalConfig(
                profiler_level=prof.ProfilerLevel.Level1,
                aic_metrics=getattr(prof.AiCMetrics, args.metric),
                data_simplification=False,
            ),
            on_trace_ready=prof.tensorboard_trace_handler(args.profile),
        ):
            for name, (graph, _) in graphs.items():
                with torch.profiler.record_function(name):
                    for _ in range(3):
                        graph.replay()
                    torch.npu.synchronize()
    else:
        times = {n: [] for n in fs}
        order = list(fs) + list(reversed(fs))
        for name in order * args.repeats:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000 / args.batch)
        result = dict(
            counts=counts,
            k=args.k,
            n=args.n,
            capacity=capacity,
            as_list=args.as_list,
            samples_us=times,
            graph_memory_bytes=graph_memory,
            environment=dict(
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                soc=torch_npu.npu.get_soc_version(),
                opp=os.environ.get("ASCEND_CUSTOM_OPP_PATH", "default"),
            ),
            median_us={n: statistics.median(t) for n, t in times.items()},
        )
        if len(fs) == 2:
            f = [t[:rows].cpu() for t in graphs["fused"][1][-1]]
            s = [t[:rows].cpu() for t in graphs["split"][1][-1]]
            error = (f[0].short() - s[0].short()).abs()
            result["max_lsb"] = error.max().item()
            result["max_scale_relative_error"] = ((f[1] - s[1]).abs() / s[1].abs().clamp_min(1e-12)).max().item()
            result["fraction_gt_one"] = (error > 1).float().mean().item()
            assert result["max_lsb"] <= 2 and result["fraction_gt_one"] < 0.005, result
            torch.testing.assert_close(f[1], s[1], rtol=0.02, atol=1e-5)
            torch.testing.assert_close(f[0].float() * f[1][:, None], s[0].float() * s[1][:, None], rtol=0.03, atol=0.02)
        print(json.dumps(result), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
