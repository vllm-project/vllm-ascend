# SPDX-License-Identifier: Apache-2.0
"""Paired production/preallocated KDA microbenchmark against PR #17301.

Flow: validate native reference -> prepare -> exact checks -> disable JIT ->
randomized paired eager measurements and graph replays -> exclusive JSON write.
Requires a separately built PR #17301 native reference; this is not a CI test.
The eager Event interval includes host dispatch gaps, not just device execution.
"""

import argparse
import hashlib
import importlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops import kda_state_copy as production
from vllm_ascend.ops.triton import kda_state_copy as kernels


def _paths(state, indices, flags, packed, plan, to_cache):
    """Keep output allocation in both production paths, outside both raw paths."""

    def native_raw():
        torch.ops._C_ascend.kda_state_copy(state, packed, indices, None if to_cache else flags, to_cache)
        return packed

    def native_without_predicate():
        if to_cache:
            torch.ops._C_ascend.kda_state_copy(state, packed.to(state.dtype).contiguous(), indices, None, True)
            return packed
        out = state.new_empty((indices.numel(), *state.shape[1:]))
        torch.ops._C_ascend.kda_state_copy(state, out, indices, flags, False)
        return out

    def native_production():
        if not to_cache:
            # The original PR checks eligibility in the gather branch. This
            # includes the native-op presence check absent from Triton startup.
            assert hasattr(torch.ops._C_ascend, "kda_state_copy")
            assert production.supports_kda_state_copy(state)
        return native_without_predicate()

    def triton_production():
        if to_cache:
            plan.scatter(state, packed, indices)
            return packed
        return plan.gather(state, indices, flags)

    def triton_raw():
        plan._launch(state, packed, indices, None if to_cache else flags, to_cache=to_cache)
        return packed

    return {
        "native_production": native_production,
        "native_no_predicate": native_without_predicate,
        "triton_production": triton_production,
        "native_preallocated": native_raw,
        "triton_preallocated": triton_raw,
        # Identical callable under two labels detects order/measurement bias.
        "native_control_duplicate": native_without_predicate,
    }


def _measure(paths, mode, trials, rng):
    """Randomize every trial; do not subtract Event time from wall/host time."""
    graphs = {}
    graph_repeats = 100
    if mode == "graph":
        for name, fn in paths.items():
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(graph_repeats):
                    fn()
            for _ in range(10):
                graph.replay()
            graphs[name] = graph
    samples = {name: [] for name in paths}
    for _ in range(trials):
        names = list(paths)
        rng.shuffle(names)
        for name in names:
            torch.npu.synchronize()
            if mode in ("wall", "host_enqueue"):
                start = time.perf_counter_ns()
                result = paths[name]()
                if mode == "wall":
                    torch.npu.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1000
            else:
                begin = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                begin.record()
                result = graphs[name].replay() if mode == "graph" else paths[name]()
                end.record()
                end.synchronize()
                elapsed = begin.elapsed_time(end) * 1000 / (graph_repeats if mode == "graph" else 1)
            samples[name].append(elapsed)
            # Keep allocation lifetime/deallocation outside every measured
            # interval, including the transition between different paths.
            del result
    return {"samples_us": samples, "medians_us": {key: statistics.median(value) for key, value in samples.items()}}


@torch.inference_mode()
def main(output: Path, trials: int, seed: int):
    """Measure fixed representative FP32/INT32 gapped layouts in one process."""
    if trials <= 0 or output.exists():
        raise ValueError("trials must be positive and output must not exist")
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    if not hasattr(torch.ops._C_ascend, "kda_state_copy"):
        raise RuntimeError("benchmark requires a separately built PR #17301 native reference")
    torch.npu.set_device(0)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    result = {"trials": trials, "seed": seed, "records": [], "correctness": [], "compiler_entries": 0}
    result["source_sha256"] = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in (("production", production), ("kernel", kernels))
    }
    workloads = []
    for count in (1, 8):
        payload, stride, offset, rows = 196608, 5308416, 393216, max(8, count + 1)
        backing = torch.full(((rows - 1) * stride + offset + payload,), -23.0, device="npu")
        state = backing.as_strided((rows, 12, 128, 128), (stride, 16384, 128, 1), offset)
        indices = torch.arange(rows - count, rows, device="npu", dtype=torch.int32)
        plan = production.KDAStateCopyPlan.prepare(state, 128)
        plan.seal()
        for case in ("gather", "clear", "scatter"):
            flags = torch.full((count,), case == "gather", dtype=torch.bool, device="npu")
            packed = torch.full((count, 12, 128, 128), 5.0, device="npu")
            paths = _paths(state, indices, flags, packed, plan, case == "scatter")
            for name, fn in paths.items():
                state.fill_(3)
                out = fn()
                actual = state[indices] if case == "scatter" else out
                expected = 5 if case == "scatter" else (3 if case == "gather" else 0)
                torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=0, atol=0)
                result["correctness"].append([count, case, name, "PASS"])
                for _ in range(50):
                    fn()
            workloads.append((count, case, paths))
    torch.npu.synchronize()

    def forbidden(*args, **kwargs):
        result["compiler_entries"] += 1
        raise AssertionError("steady-state measurement entered a Triton compiler")

    kernels._kda_state_copy_kernel.run = forbidden
    for name in ("triton", "triton.compiler", "triton.compiler.compiler", "triton.runtime.jit"):
        module = importlib.import_module(name)
        if callable(getattr(module, "compile", None)):
            module.compile = forbidden
    for count, case, paths in workloads:
        for mode in ("host_enqueue", "wall", "event", "graph"):
            record = {"selected": count, "case": case, "mode": mode, **_measure(paths, mode, trials, rng)}
            result["records"].append(record)
            print(count, case, mode, record["medians_us"], flush=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.output, args.trials, args.seed)
