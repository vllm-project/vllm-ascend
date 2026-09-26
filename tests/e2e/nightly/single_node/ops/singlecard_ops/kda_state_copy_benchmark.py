#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the profiled KDA state-copy subchain with fused AscendC copies.

The observed 16K/H12 prefill profile contains 144 gathers and 144 scatters of
one FP32[12,128,128] payload. Report each operation and their profile-count
estimate separately; these are not whole-KDA-module or whole-model timings.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
from vllm_ascend.ops.triton.fla.utils import clear_ssm_states


def legacy_copy_function():
    # Exercise the preserved production baseline without importing model
    # construction/weight-loading dependencies into this operator benchmark.
    path = Path(__file__).resolve().parents[6] / "vllm_ascend/ops/kimi_kda.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if getattr(node, "name", None) == "_copy_strided_recurrent_states"]
    scope = {"torch": torch, "batch_memcpy_kernel": batch_memcpy_kernel}
    exec(compile(tree, str(path), "exec"), scope)
    return scope["_copy_strided_recurrent_states"]


def measure(functions, *, warmup, trials, graph, graph_repeats):
    functions = dict(functions)
    graphs = []
    for name, function in functions.items():
        for _ in range(warmup):
            function()
        torch.npu.synchronize()
        if graph:
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                for _ in range(graph_repeats):
                    function()
            graphs.append(captured)
            functions[name] = captured.replay
            for _ in range(warmup):
                captured.replay()
    torch.npu.synchronize()
    names = list(functions)
    samples = {name: [] for name in names}
    for trial in range(trials):
        order = names[trial % len(names) :] + names[: trial % len(names)]
        for name in order:
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record()
            functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / (graph_repeats if graph else 1))
    return {
        name: {
            "median_us": statistics.median(values),
            "minimum_us": min(values),
            "samples_us": values,
            "mode": "graph" if graph else "eager",
            "warmup_calls": warmup,
            "calls_per_graph_replay": graph_repeats if graph else None,
            "timing_scope": (
                "device events around one graph replay containing repeated subchains; divide by captured call count"
                if graph
                else "device events around named state-copy subchain; includes eager host dispatch gaps"
            ),
        }
        for name, values in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--graph-repeats", type=int, default=100)
    parser.add_argument("--states", type=int, default=1)
    args = parser.parse_args()
    if min(args.warmup, args.trials, args.graph_repeats, args.states) < 1:
        parser.error("warmup, trials, graph-repeats and states must be positive")
    torch.npu.set_device(args.device)
    old_copy = legacy_copy_function()
    heads, value_dim, key_dim = 12, 128, 128
    payload = heads * value_dim * key_dim
    cache_rows, stride, offset = max(8, args.states + 1), 5308416, 2 * payload
    with torch.inference_mode():
        backing = torch.empty((cache_rows - 1) * stride + offset + payload, device="npu")
        state = backing.as_strided(
            (cache_rows, heads, value_dim, key_dim), (stride, value_dim * key_dim, key_dim, 1), offset
        )
        for row in range(cache_rows):
            state[row].fill_(row + 1)
        indices = torch.arange(cache_rows - args.states, cache_rows, dtype=torch.int32, device="npu")
        native_packed = torch.empty((args.states, heads, value_dim, key_dim), device="npu")
        legacy_packed = torch.empty_like(native_packed)
        final_state = torch.randn_like(native_packed)
        results = {
            "scope": "KDA state-copy subchain only; excludes convolution, projections, recurrence and output norm",
            "associated_prefill_tokens": 16384,
            "state_shape": list(state.shape),
            "selected_shape": list(native_packed.shape),
            "cache_stride_bytes": stride * state.element_size(),
            "payload_bytes": payload * state.element_size(),
            "profile_gather_count": 144,
            "profile_scatter_count": 144,
            "records": {},
        }
        for has_initial in (True, False):
            flags = torch.full((args.states,), has_initial, dtype=torch.bool, device="npu")

            def legacy_gather(flags=flags):
                old_copy(state, legacy_packed, indices, to_cache=False)
                clear_ssm_states(legacy_packed, flags)

            def native_gather(flags=flags):
                torch.ops._C_ascend.kda_state_copy(state, native_packed, indices, flags, False)

            legacy_gather()
            native_gather()
            torch.testing.assert_close(native_packed.cpu(), legacy_packed.cpu(), rtol=0, atol=0)
            measurements = measure(
                {"legacy_gather_and_clear": legacy_gather, "native_gather_and_clear": native_gather},
                warmup=args.warmup,
                trials=args.trials,
                graph=args.graph,
                graph_repeats=args.graph_repeats,
            )
            results["records"][f"has_initial_state_{has_initial}"] = measurements
        results["records"]["scatter"] = measure(
            {
                "legacy_scatter": lambda: old_copy(state, final_state, indices, to_cache=True),
                "native_scatter": lambda: torch.ops._C_ascend.kda_state_copy(state, final_state, indices, None, True),
            },
            warmup=args.warmup,
            trials=args.trials,
            graph=args.graph,
            graph_repeats=args.graph_repeats,
        )
        # Do not infer how many of the original gathers had valid initial
        # states: report both endpoints of the profile-count estimate.
        for has_initial in (True, False):
            gather = results["records"][f"has_initial_state_{has_initial}"]
            scatter = results["records"]["scatter"]
            results[f"estimated_144_gather_plus_144_scatter_us_all_initial_{has_initial}"] = {
                route: 144
                * (gather[f"{route}_gather_and_clear"]["median_us"] + scatter[f"{route}_scatter"]["median_us"])
                for route in ("legacy", "native")
            }
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
