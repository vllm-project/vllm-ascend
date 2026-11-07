# SPDX-License-Identifier: Apache-2.0
"""Correctness-gated graph benchmark for the six exported compressor shapes.

Use --iteration to record one real optimization attempt. This program does not
manufacture an iteration history or count environment/build failures as rounds.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from tests.ut.ops.compressor_serial_reference import compressor_reference


def load_serial_module():
    # A standalone extension build does not generate vLLM's _build_info.
    # Load the real module source without initializing the model plugin.
    source = Path(__file__).parents[1] / "vllm_ascend/ops/compressor_serial.py"
    spec = importlib.util.spec_from_file_location("compressor_serial_benchmark", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_case(case, device):
    """Preserve exported shapes; metadata values are synthetic, not trace data."""
    shapes = copy.deepcopy(case["inputs"])
    # The installed public operator validates TND output capacity from X,
    # independently of valid-token metadata. Preserve fixture shapes and record
    # this runtime adaptation in the result rather than changing the export.
    capacity = min(shapes[0][0], shapes[0][0] // case["ratio"] + shapes[11][0])
    shapes[6][0] = shapes[7][0] = capacity
    ratio = case["ratio"]
    batch = shapes[11][0]
    gen = torch.Generator().manual_seed(20261026)

    def random(shape, dtype=torch.float32, scale=0.1):
        return (torch.randn(shape, generator=gen) * scale).to(dtype).to(device)

    x, wk, wg = [random(s, torch.bfloat16) for s in shapes[:3]]
    state, ape, gamma, sin, cos = [random(s) for s in shapes[3:8]]
    # TND may contain padded tokens. Preserve the exported valid-row workload.
    if case["phase"] == "prefill":
        total = min(shapes[0][0], case["inputs"][6][0] * ratio)
        lengths = [total // batch // ratio * ratio] * batch
        lengths[-1] += total - sum(lengths)
        starts = [0] * batch
    else:
        lengths = [1] * batch
        starts = [ratio - 1] * batch
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    if cu[-1] > shapes[0][0]:
        raise ValueError("synthetic valid tokens exceed traced X capacity")
    table = torch.zeros(shapes[8], dtype=torch.int32)
    next_page = 1
    for b, (start, length) in enumerate(zip(starts, lengths)):
        # Map all pages touched by either history reads or cache-tail writes.
        lo = max(0, start // ratio * ratio - (case["coff"] - 1) * ratio)
        hi = start + length
        for page in range(lo // shapes[3][1], (hi + shapes[3][1] - 1) // shapes[3][1]):
            if next_page >= shapes[3][0] or page >= table.shape[1]:
                raise ValueError("synthetic block mapping exceeds traced shape")
            table[b, page] = next_page
            next_page += 1
    meta = [table, torch.tensor(cu, dtype=torch.int32), None, torch.tensor(starts, dtype=torch.int32)]
    return (x, wk, wg, state, ape, gamma, sin, cos, *(None if t is None else t.to(device) for t in meta))


def fused_call(args, case):
    return torch.ops._C_ascend.compressor(*args, 64, case["ratio"], case["coff"], 1e-6, 2, 1)


def capture(fn):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = fn()
    torch.npu.synchronize()
    return graph, result


def graph_times(graph, samples=31, replays=20):
    times = []
    for _ in range(samples):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / replays)
    return times


def check_case(args, case, plan):
    initial = args[3].clone()
    cpu_args = tuple(None if t is None else t.cpu() for t in args)
    golden, cache_golden = compressor_reference(cpu_args, ratio=case["ratio"], coff=case["coff"])
    actual = plan(args[0])
    torch.npu.synchronize()
    torch.testing.assert_close(actual.float().cpu(), golden, rtol=0.02, atol=0.02)
    torch.testing.assert_close(args[3].cpu(), cache_golden, rtol=1e-4, atol=3e-4)
    serial_out, serial_cache = actual.clone(), args[3].clone()
    args[3].copy_(initial)
    fused = fused_call(args, case)
    torch.npu.synchronize()
    # Padded output rows are outside the public valid-token contract.
    valid = golden.abs().sum(dim=-1) != 0
    torch.testing.assert_close(serial_out[valid].float(), fused[valid].float(), rtol=0.02, atol=0.02)
    torch.testing.assert_close(serial_cache, args[3], rtol=1e-4, atol=3e-4)
    args[3].copy_(initial)
    return initial, valid


def profile_sequence(graph):
    # Ten adjacent warmup -> complete target sequences; retain every sample.
    marker = torch.empty((128, 1024, 1024), device="npu", dtype=torch.float16)
    assert marker.numel() * marker.element_size() == 256 * 1024 * 1024
    for _ in range(10):
        torch.argmax(marker)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-library", type=Path, required=True)
    parser.add_argument("--fused-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--case", default="all")
    parser.add_argument("--cast-blocks", type=int, default=48)
    parser.add_argument("--pool-blocks", type=int, default=48)
    parser.add_argument("--final-blocks", type=int, default=48)
    parser.add_argument("--pool-threads", type=int, default=256)
    parser.add_argument("--weight-layout", choices=("kn", "nk"), default="nk")
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--cube-math-type", choices=("KEEP_DTYPE", "USE_HF32"))
    parser.add_argument("--hf32", action="store_true")
    parser.add_argument("--profile", choices=("fused", "serial"))
    opts = parser.parse_args()
    if opts.output.exists():
        raise FileExistsError("choose a fresh result path; iteration evidence is immutable")
    if opts.profile and opts.case == "all":
        raise ValueError("profile exactly one case per application for unambiguous task mapping")
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = opts.hf32
    if opts.cube_math_type:
        torch.npu.matmul.cube_math_type = getattr(torch.npu.CubeMathType, opts.cube_math_type)
    torch.ops.load_library(str(opts.fused_library.resolve()))
    torch.ops.load_library(str(opts.serial_library.resolve()))
    fixture = Path(__file__).parents[1] / "tests/ut/ops/compressor_serial_shapes.json"
    cases = json.loads(fixture.read_text())["cases"]
    serial_module = load_serial_module()
    config = serial_module.CompressorSerialConfig(
        opts.cast_blocks, opts.pool_blocks, opts.final_blocks, opts.pool_threads, opts.weight_layout
    )
    result = {
        "iteration": opts.iteration,
        "hypothesis": opts.hypothesis,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "library_sha256": hashlib.sha256(opts.serial_library.read_bytes()).hexdigest(),
        "fused_library_sha256": hashlib.sha256(opts.fused_library.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "hf32": opts.hf32,
        "cube_math_type": str(torch.npu.matmul.cube_math_type),
        "launch_policy": "adaptive" if opts.auto_launch else "explicit",
        "config": asdict(config),
        "metadata": "synthetic; original export retained; runtime TND RoPE capacity adapted",
        "cases": [],
    }
    for case in cases:
        if opts.case not in ("all", case["id"]):
            continue
        args = make_case(case, "npu")
        plan = serial_module.CompressorSerialPlan.prepare(
            *args, cmp_ratio=case["ratio"], coff=case["coff"], config=None if opts.auto_launch else config
        )
        print(f"Checking {case['id']}: runtime RoPE shape {tuple(args[6].shape)}", flush=True)
        initial, valid = check_case(args, case, plan)
        fused_graph, fused_graph_output = capture(partial(fused_call, args, case))
        args[3].copy_(initial)
        serial_graph, _ = capture(partial(plan, args[0]))
        args[3].copy_(initial)
        # A second correctness check must exercise the captured graph itself.
        serial_graph.replay()
        torch.npu.synchronize()
        graph_out, graph_state = plan.output.clone(), args[3].clone()
        args[3].copy_(initial)
        fused_graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(graph_out[valid].float(), fused_graph_output[valid].float(), rtol=0.02, atol=0.02)
        torch.testing.assert_close(graph_state, args[3], rtol=1e-4, atol=3e-4)
        if opts.profile:
            profile_sequence(serial_graph if opts.profile == "serial" else fused_graph)
            result["cases"].append(
                {"id": case["id"], "profile": opts.profile, "expected_target_samples": 10, "precision_passed": True}
            )
            continue
        fused_us = graph_times(fused_graph)
        serial_us = graph_times(serial_graph)
        speed = statistics.median(fused_us) / statistics.median(serial_us)
        result["cases"].append(
            {
                "id": case["id"],
                "fused_us": fused_us,
                "serial_us": serial_us,
                "ratio": speed,
                "target_met": speed >= 0.8,
                "runtime_shapes": [None if t is None else list(t.shape) for t in args],
                "launch_config": asdict(plan.config),
                "reference_shapes": case["inputs"],
            }
        )
        print(json.dumps(result["cases"][-1]), flush=True)
    if not result["cases"]:
        raise ValueError("no case selected")
    if not opts.profile:
        result["all_cases_meet_target"] = all(c["target_met"] for c in result["cases"])
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
