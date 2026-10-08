# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile mHC replication on a real NPU; preserve raw traces and a Markdown report.

Run after building the custom operators for Ascend A2::

    python benchmarks/mhc_expand.py --output /tmp/mhc-profile

The output directory must not exist, preventing results from different runs
from being mixed. Cases are independently designed, not testcase-gen output.
"""

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.glm5next.ops.mhc_ops import hc_expand
from vllm_ascend.ops.mhc import mhc_expand
from vllm_ascend.utils import enable_custom_op

WARMUP = 5
ACTIVE = 5


def profile_case(fn, trace_dir):
    trace_dir.mkdir(parents=True)
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=WARMUP, active=ACTIVE, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
    ) as prof:
        for _ in range(WARMUP + ACTIVE):
            result = fn()
            # Drain queued launches before moving the profiler boundary. A
            # warmup kernel must not spill into the five active invocations.
            torch.npu.synchronize()
            prof.step()
    torch.npu.synchronize()
    del result
    paths = list(trace_dir.glob("**/ASCEND_PROFILER_OUTPUT/op_statistic.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one op_statistic.csv below {trace_dir}, found {paths}")
    with paths[0].open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        key = next((key for key in reader.fieldnames or [] if key.replace(" ", "").lower() == "totaltime(us)"), None)
        if key is None:
            raise RuntimeError(f"No Total Time(us) column in {paths[0]}")
        rows = list(reader)
        # These benchmark paths each launch one device kernel per invocation.
        # Reject incomplete or contaminated traces instead of normalizing them
        # with a denominator that does not match the recorded work.
        count = sum(int(row["Count"]) for row in rows)
        if count != ACTIVE:
            raise RuntimeError(f"Expected {ACTIVE} device calls in {paths[0]}, recorded {count}")
        times = [float(row[key].replace(",", "")) for row in rows]
    if not times or sum(times) <= 0:
        raise RuntimeError(f"No device time recorded in {paths[0]}")
    return sum(times) / ACTIVE


def wall_case(fn, iterations):
    """Amortized wall time including Python enqueue and final device completion."""
    for _ in range(WARMUP):
        result = fn()
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        result = fn()
    torch.npu.synchronize()
    elapsed_us = (time.perf_counter_ns() - start) / iterations / 1000
    del result
    return elapsed_us


def capture_case(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        result = fn()

    def replay():
        graph.replay()
        return result

    return replay


def summarize(samples):
    lines = [
        "\n## Repeated-round summary\n",
        "| Case | Dtype | Custom median (us) | Expand median (us) | Repeat median (us) | "
        "Expand/custom median | Ratio min-max |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    ratios_by_dtype = {}
    for index, (dtype, rounds) in samples.items():
        medians = {name: statistics.median(row[name] for row in rounds) for name in rounds[0]}
        ratios = [row["expand"] / row["custom"] for row in rounds]
        ratio = statistics.median(ratios)
        ratios_by_dtype.setdefault(dtype, []).append(ratio)
        lines.append(
            f"| {index} | {dtype} | {medians['custom']:.3f} | {medians['expand']:.3f} | "
            f"{medians['repeat']:.3f} | {ratio:.3f} | {min(ratios):.3f}-{max(ratios):.3f} |"
        )
    all_ratios = [ratio for ratios in ratios_by_dtype.values() for ratio in ratios]
    lines += [
        "\n## Summary\n",
        "| Dtype | Cases | Mean speedup | Geomean speedup | Custom faster | Native faster |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dtype, ratios in [*ratios_by_dtype.items(), ("all", all_ratios)]:
        lines.append(
            f"| {dtype} | {len(ratios)} | {statistics.mean(ratios):.3f} | "
            f"{statistics.geometric_mean(ratios):.3f} | "
            f"{sum(r > 1 for r in ratios)} | {sum(r < 1 for r in ratios)} |"
        )
    lines += [
        "\n## Observations\n",
        "- Ratios compare the same case/round; medians and min-max expose run variation.",
        "- Profiler measures device time; wall and graph measure amortized enqueue-to-completion time.",
        "- Small per-forward savings do not establish full-model throughput gains.",
    ]
    return lines


def main():
    # Keep source selection, correctness and timing together so each reported
    # row is tied to the exact production entry point and baseline used.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("mhc_expand_cases.jsonl"))
    parser.add_argument("--path", choices=["raw", "helper", "glm"], default="raw")
    parser.add_argument("--timing", choices=["profiler", "wall", "graph"], default="profiler")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100, help="Calls per wall/graph measurement")
    args = parser.parse_args()
    if args.rounds < 1 or args.iterations < 1:
        parser.error("rounds and iterations must be positive")
    if not torch.npu.is_available():
        raise RuntimeError("A real NPU is required")
    if not get_current_hardware_profile().supports(HardwareCapability.MHC_EXPAND):
        raise RuntimeError("VllmMhcExpand is currently built for A2")
    if not enable_custom_op():
        raise RuntimeError("The compiled vLLM Ascend extension is required")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    custom = {"raw": torch.ops._C_ascend.npu_mhc_expand, "helper": mhc_expand, "glm": hc_expand}[args.path]
    report = [
        "# mHC Expand NPU measurements\n",
        f"- Device: {torch.npu.get_device_name(0)}",
        f"- Host: {platform.platform()}",
        f"- PyTorch: {torch.__version__}; torch_npu: {torch_npu.__version__}",
        f"- Path: {args.path}; timing: {args.timing}; independent rounds: {args.rounds}.",
        f"- Profiler: warmup={WARMUP}, active={ACTIVE}, repeat=1; separate trace per implementation.",
        f"- Wall/graph: {args.iterations} calls, with synchronization before and after the timed batch.",
        "- Profiler metric: sum of all Total Time(us) rows / five active invocations.",
        "- Alternate implementation order each round; graph capture and correctness are outside timing.",
        "- Cases are independently designed (用例为自行设计，非 testcase-gen 产出).",
        "- No end-to-end model speedup is inferred.\n",
        "| Round | Case | Shape | Dtype | Custom (us) | Expand (us) | Repeat (us) | Expand/custom |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    samples = {}
    with torch.inference_mode():
        for round_index in range(args.rounds):
            for index, case in enumerate(cases):
                inputs = {item["name"]: item for item in case["inputs"]}
                dtype_name = inputs["x"]["dtype"]
                x = torch.randn(*inputs["x"]["shape"], dtype=getattr(torch, dtype_name), device="npu")
                mult = inputs["mult"]["value"]
                functions = {
                    "custom": lambda x=x, mult=mult: custom(x, mult),
                    "expand": lambda x=x, mult=mult: x.unsqueeze(1).expand(-1, mult, -1).contiguous(),
                    "repeat": lambda x=x, mult=mult: x.unsqueeze(1).repeat(1, mult, 1),
                }
                expected = functions["repeat"]().cpu().view(torch.int16)
                for fn in functions.values():
                    assert torch.equal(fn().cpu().view(torch.int16), expected)
                order = list(functions) if round_index % 2 == 0 else list(reversed(functions))
                times = {}
                for name in order:
                    fn = functions[name]
                    if args.timing == "graph":
                        fn = capture_case(fn)
                        x.fill_(-2)
                        assert torch.equal(fn().cpu().view(torch.int16), functions["repeat"]().cpu().view(torch.int16))
                    if args.timing == "profiler":
                        trace_dir = args.output / f"round_{round_index:02d}" / f"case_{index:03d}" / name
                        times[name] = profile_case(fn, trace_dir)
                    else:
                        times[name] = wall_case(fn, args.iterations)
                samples.setdefault(index, (dtype_name, []))[1].append(times)
                report.append(
                    f"| {round_index} | {index} | {tuple(x.shape)}, mult={mult} | {dtype_name} | "
                    f"{times['custom']:.3f} | {times['expand']:.3f} | {times['repeat']:.3f} | "
                    f"{times['expand'] / times['custom']:.3f} |"
                )
                print(report[-1], flush=True)
                (args.output / "report.md").write_text("\n".join(report) + "\n")
    report += summarize(samples)
    (args.output / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
