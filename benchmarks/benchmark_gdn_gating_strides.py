# SPDX-License-Identifier: Apache-2.0
"""Compare BA contiguous copies plus gating with direct strided gating.

This measures the indicated local operation chain, not model throughput.
Run in a matched vllm-ascend NPU environment. No speedup threshold is enforced.
Export the pre-change kernel from the PR base, then pass it explicitly:

    git show <base-sha>:vllm_ascend/ops/triton/fused_gdn_gating.py > /tmp/gdn_baseline.py
    python benchmarks/benchmark_gdn_gating_strides.py --baseline-source /tmp/gdn_baseline.py

Event timing covers eagerly submitted calls, including device idle time between
submissions; it is not isolated kernel or warmed graph-replay latency.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.ops.triton import fused_gdn_gating as candidate
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def load_baseline(source: Path):
    """Load an explicit pre-change kernel instead of benchmarking against itself."""
    source = source.resolve()
    if source.read_bytes() == Path(candidate.__file__).read_bytes():
        raise ValueError("baseline source must be the pre-change kernel, not the candidate")
    spec = importlib.util.spec_from_file_location("gdn_gating_baseline", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline Python source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.fused_gdn_gating_patch


def measure(fn, iterations):
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    begin = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations, (time.perf_counter() - begin) * 1e6 / iterations


def cases(args, baseline_gating):
    init_device_properties_triton()
    for rows in args.rows:
        a, b = torch.randn(rows, args.width * 2, device="npu", dtype=torch.bfloat16).chunk(2, -1)
        log = torch.randn(args.width, device="npu", dtype=torch.float32)
        bias = torch.randn_like(log)

        def baseline(a=a, b=b, log=log, bias=bias):
            return baseline_gating(log, a.contiguous(), b.contiguous(), bias)

        def optimized(a=a, b=b, log=log, bias=bias):
            return candidate.fused_gdn_gating_patch(log, a, b, bias)

        avoided_copy_bytes = sum(x.numel() * x.element_size() for x in (a, b) if not x.is_contiguous())
        yield dict(rows=rows, avoided_copy_bytes=avoided_copy_bytes), baseline, optimized


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-source", type=Path, required=True, help="Python kernel file exported from the PR base"
    )
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 7, 128, 4868])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if min(*args.rows, args.width, args.rounds, args.iterations) <= 0:
        parser.error("all sizes and iteration counts must be positive")
    baseline_gating = load_baseline(args.baseline_source)
    torch.manual_seed(17)
    print(
        json.dumps(
            dict(
                device=torch.npu.get_device_name(),
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                dtype="bfloat16",
                width=args.width,
                baseline_source=str(args.baseline_source.resolve()),
                baseline_sha256=hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
                timing="eager submissions; not isolated kernel latency",
            )
        )
    )
    for label, baseline, optimized in cases(args, baseline_gating):
        expected = baseline()
        expected = expected.clone() if isinstance(expected, torch.Tensor) else tuple(x.clone() for x in expected)
        torch.testing.assert_close(expected, optimized(), rtol=0, atol=0)
        for _ in range(10):
            baseline()
            optimized()
        times = {"baseline": [], "optimized": []}
        for index in range(args.rounds):
            order = [("baseline", baseline), ("optimized", optimized)]
            if index % 2:
                order.reverse()
            for name, fn in order:
                times[name].append(measure(fn, args.iterations))
        print(
            json.dumps(
                dict(
                    case=label,
                    event_median_us={k: statistics.median(t[0] for t in v) for k, v in times.items()},
                    wall_median_us={k: statistics.median(t[1] for t in v) for k, v in times.items()},
                    rounds_event_wall_us=times,
                )
            )
        )


if __name__ == "__main__":
    main()
