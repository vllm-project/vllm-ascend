# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired K3.1 A3 MSD SiTU operator timing, with eager/graph accuracy checks.

Run from a source checkout with its compiled extension and A3 custom OPP:
    python benchmarks/benchmark_gmm_situ_quant_msd.py --output /tmp/situ-benchmark

Both graphs include an identical activation reset because the CANN split GMM
rewrites its INT8 input. Results include that reset, warm up graph replay, and
retain every alternating AB/BA round. This measures operator latency only.
"""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 250 <= torch_npu.npu.get_soc_version() <= 256:
        raise RuntimeError("This benchmark requires Ascend A3")
    source = (
        Path(__file__).resolve().parents[1]
        / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_gmm_situ_quant_msd.py"
    )
    spec = importlib.util.spec_from_file_location("situ_test", source)
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    torch.npu.config.allow_internal_format = True
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "E": 14,
        "K": 7168,
        "N": 7168,
        "seed": 674,
        "dtype": "A8W4",
        "layout": "native NZ stacked",
        "cases": [(1, 7168), (16, 7168), (128, 7168), (2048, 32768)],
        "warmup": 10,
        "rounds": 6,
        "replays_per_round": 50,
        "order": "AB,BA,AB,BA,AB,BA",
        "metric": "mean NPU event microseconds per graph replay",
        "max_cv": 0.10,
        "regression_ratio": 1.05,
        "copy_policy": "Both graphs reset the same sized activation buffer; reset included.",
        "scope": "operator only; not serving throughput or latency",
    }
    (root / "config.json").write_text(json.dumps(config, indent=2))
    layer = t._layer(14, 7168, 7168, True, 675)
    results = []
    for rows, capacity in config["cases"]:
        g = torch.Generator().manual_seed(674)
        source_x = torch.randint(-128, 128, (capacity, 7168), dtype=torch.int8, generator=g).npu()
        x = source_x.clone()
        xs = (torch.rand(capacity, generator=g) * 0.01 + 0.001).npu()
        counts = [rows // 14 + int(i < rows % 14) for i in range(14)]
        groups = torch.tensor(counts, dtype=torch.int64, device="npu")
        channel_scale = layer[5].unsqueeze(-2)

        def split(x=x, source_x=source_x, channel_scale=channel_scale, xs=xs, groups=groups):
            x.copy_(source_x)
            intermediate = torch_npu.npu_grouped_matmul(
                x=[x],
                weight=[layer[4]],
                scale=[channel_scale],
                bias=[layer[6]],
                per_token_scale=[xs],
                split_item=2,
                group_type=0,
                group_list=groups,
                group_list_type=1,
                output_dtype=torch.bfloat16,
            )[0]
            return torch.ops._C_ascend.dequant_situ_quant(
                x=intermediate,
                weight_scale=None,
                activation_scale=None,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=groups,
                beta=t.BETA,
                linear_beta=t.LINEAR_BETA,
                activate_left=True,
                quant_mode="dynamic",
            )

        def fused(x=x, source_x=source_x, xs=xs, groups=groups):
            x.copy_(source_x)
            return t._call(x, xs, layer, groups, 1, nz=True, stacked=True, external_assist=True)

        reference = split()
        candidate = fused()
        av = candidate[0][:rows].float() * candidate[1][:rows, None]
        rv = reference[0][:rows].float() * reference[1][:rows, None]
        excess = ((av - rv).abs() - (3 * reference[1][:rows, None] + 0.02 * rv.abs() + 1e-5)).max().item()
        assert excess <= 0, (rows, excess)
        graphs = {}
        for name, fn in [("split", split), ("fused", fused)]:
            for _ in range(config["warmup"]):
                fn()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, capture_error_mode="global"):
                output = fn()
            graphs[name] = (graph, output)
        measurements = {"split": [], "fused": []}
        # Capture does not warm up graph replay itself. Exclude first-replay setup
        # from measured rounds for both paths.
        for _ in range(config["warmup"]):
            for graph, _output in graphs.values():
                graph.replay()
        torch.npu.synchronize()
        for name, expected in [("split", reference), ("fused", candidate)]:
            observed = graphs[name][1]
            assert torch.equal(observed[0][:rows], expected[0][:rows]), ("graph codes", name, rows)
            torch.testing.assert_close(observed[1][:rows], expected[1][:rows], rtol=1e-5, atol=1e-8)
        for index in range(config["rounds"]):
            order = ["split", "fused"] if index % 2 == 0 else ["fused", "split"]
            for name in order:
                graph = graphs[name][0]
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                torch.npu.synchronize()
                start.record()
                for _ in range(config["replays_per_round"]):
                    graph.replay()
                end.record()
                end.synchronize()
                measurements[name].append(start.elapsed_time(end) * 1000 / config["replays_per_round"])
        summary = {
            name: {
                "mean_us": statistics.mean(v),
                "stdev_us": statistics.stdev(v),
                "cv": statistics.stdev(v) / statistics.mean(v),
            }
            for name, v in measurements.items()
        }
        ratio = summary["fused"]["mean_us"] / summary["split"]["mean_us"]
        case = {
            "rows": rows,
            "capacity": capacity,
            "measurements_us": measurements,
            "summary": summary,
            "fused_over_split": ratio,
            "speedup": 1 / ratio,
            "accuracy_max_excess": excess,
            "status": "inconclusive"
            if max(s["cv"] for s in summary.values()) > 0.10
            else ("regression" if ratio > 1.05 else "passed"),
        }
        results.append(case)
        (root / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(case), flush=True)
        del graphs, source_x, x, xs, reference, candidate, av, rv
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
