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
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op

WARMUP = 5
ACTIVE = 5


def profile_case(fn, trace_dir):
    trace_dir.mkdir(parents=True)
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
        times = [float(row[key].replace(",", "")) for row in reader]
    if not times or sum(times) <= 0:
        raise RuntimeError(f"No device time recorded in {paths[0]}")
    return sum(times) / ACTIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("mhc_expand_cases.jsonl"))
    args = parser.parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("A real NPU is required")
    if not get_current_hardware_profile().supports(HardwareCapability.MHC_EXPAND):
        raise RuntimeError("VllmMhcExpand is currently built for A2")
    if not enable_custom_op():
        raise RuntimeError("The compiled vLLM Ascend extension is required")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    report = [
        "# mHC Expand NPU measurements\n",
        f"- Device: {torch.npu.get_device_name(0)}",
        f"- Host: {platform.platform()}",
        f"- PyTorch: {torch.__version__}; torch_npu: {torch_npu.__version__}",
        f"- warmup={WARMUP}, active={ACTIVE}, repeat=1; separate trace per case and implementation.",
        "- Metric: sum of every Total Time(us) row divided by five active invocations.",
        "- Cases are independently designed (用例为自行设计，非 testcase-gen 产出).",
        "- Device operator latency only; no end-to-end model speedup is inferred.\n",
        "| Case | Shape | Dtype | Custom (us) | Expand contiguous (us) | Repeat (us) | Expand/custom | Repeat/custom |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ratios = []
    for index, case in enumerate(cases):
        inputs = {item["name"]: item for item in case["inputs"]}
        dtype_name = inputs["x"]["dtype"]
        dtype = getattr(torch, dtype_name)
        x = torch.randn(*inputs["x"]["shape"], dtype=dtype, device="npu")
        mult = inputs["mult"]["value"]
        functions = {
            "custom": lambda x=x, mult=mult: torch.ops._C_ascend.npu_mhc_expand(x, mult),
            "expand": lambda x=x, mult=mult: x.unsqueeze(1).expand(-1, mult, -1).contiguous(),
            "repeat": lambda x=x, mult=mult: x.unsqueeze(1).repeat(1, mult, 1),
        }
        expected = functions["repeat"]().cpu().view(torch.int16)
        for fn in functions.values():
            assert torch.equal(fn().cpu().view(torch.int16), expected)
        times = {name: profile_case(fn, args.output / f"case_{index:03d}" / name) for name, fn in functions.items()}
        expand_ratio, repeat_ratio = times["expand"] / times["custom"], times["repeat"] / times["custom"]
        ratios.append((dtype_name, expand_ratio, repeat_ratio))
        row = (
            f"| {index} | {tuple(x.shape)}, mult={mult} | {dtype_name} | {times['custom']:.3f} | "
            f"{times['expand']:.3f} | {times['repeat']:.3f} | {expand_ratio:.3f} | {repeat_ratio:.3f} |"
        )
        report.append(row)
        print(row, flush=True)
        (args.output / "report.md").write_text("\n".join(report) + "\n")
    report += [
        "\n## Summary\n",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Cases | {len(ratios)} |",
        f"| Mean expand/custom | {statistics.mean(item[1] for item in ratios):.3f} |",
        f"| Custom faster | {sum(item[1] > 1 for item in ratios)} |",
        f"| Baseline faster | {sum(item[1] < 1 for item in ratios)} |",
        "\n### By dtype\n",
        "| Dtype | Cases | Mean expand/custom | Geomean expand/custom | "
        "Geomean repeat/custom | Custom faster | Baseline faster |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dtype in ["float16", "bfloat16", "all"]:
        selected = [item for item in ratios if dtype == "all" or item[0] == dtype]
        if selected:
            report.append(
                f"| {dtype} | {len(selected)} | "
                f"{statistics.mean(item[1] for item in selected):.3f} | "
                f"{statistics.geometric_mean(item[1] for item in selected):.3f} | "
                f"{statistics.geometric_mean(item[2] for item in selected):.3f} | "
                f"{sum(item[1] > 1 for item in selected)} | {sum(item[1] < 1 for item in selected)} |"
            )
    wins = sum(item[1] > 1 for item in ratios)
    report += [
        "\n## Observations\n",
        f"- Custom is faster than expand/contiguous in {wins}/{len(ratios)} measured cases.",
        f"- Expand/custom ratios range from {min(item[1] for item in ratios):.3f} "
        f"to {max(item[1] for item in ratios):.3f}; inspect decode and prefill separately.",
        "- Expansion is an initialization operation; these kernel results do not establish model throughput gains.",
        "- Raw traces are stored in case_NNN/{custom,expand,repeat}/ below this report.\n",
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
