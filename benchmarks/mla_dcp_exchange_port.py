# SPDX-License-Identifier: Apache-2.0
"""Distributed correctness and A-B graph timing for the Kimi PR #16349 port.

torchrun --standalone --nproc_per_node=16 benchmarks/mla_dcp_exchange_port.py
Loads the unchanged baseline helpers from this checkout without vLLM startup.
"""

import ast
import importlib.util
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

ROOT = Path(__file__).resolve().parents[1]


def load_helpers():
    path = ROOT / "vllm_ascend/ops/triton/mla_dcp_exchange.py"
    spec = importlib.util.spec_from_file_location("mla_dcp_exchange", path)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    source = ROOT / "vllm_ascend/attention/context_parallel/common_cp.py"
    tree = ast.parse(source.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_process_attn_out_lse", "_npu_attention_update")
    ]
    assert len(nodes) == 2
    scope = {"torch": torch, "torch_npu": torch_npu, "dist": dist}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    ranks = dist.get_world_size()

    def baseline(output, lse):
        packed = scope["_process_attn_out_lse"](output, lse, dcp_size=ranks, dcp_device_group=dist.group.WORLD)
        return scope["_npu_attention_update"](512, packed, dcp_size=ranks)

    return baseline, candidate


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    ranks = dist.get_world_size()
    baseline, candidate = load_helpers()
    torch.manual_seed(16349 + rank)
    results = []
    checks = 0
    # Keep every graph and its external buffers alive: releasing earlier
    # captures while reusing allocations invalidated this runtime's replay test.
    captured = []
    token_cases = (
        [] if os.environ.get("CHECK_LAYERS_ONLY") else map(int, os.environ.get("TEST_TOKENS", "1,8,12").split(","))
    )
    for tokens in token_cases:
        # Non-contiguous head slices model FIA trimming; offset is odd on rank 0.
        backing = torch.randn(tokens * (ranks * 6 + 2) * 512 + 1, dtype=torch.bfloat16, device="npu")
        output = backing[1:].view(tokens, ranks * 6 + 2, 512)[:, : ranks * 6]
        lse = torch.randn(tokens, ranks * 6, 2, dtype=torch.float32, device="npu")[..., :1]
        assert candidate.can_exchange(output, lse, ranks)
        for case in ("finite", "mixed_empty", "all_empty"):
            output.normal_()
            lse.normal_()
            if case == "all_empty" or (case == "mixed_empty" and rank % 2 == 0):
                output.zero_()
                lse.fill_(float("-inf"))
            expected = baseline(output, lse)
            actual = candidate.exchange(output, lse, dist.group.WORLD)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            baseline_nonfinite = (~torch.isfinite(expected)).sum().item()
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "tokens": tokens,
                            "case": case,
                            "baseline_nonfinite": baseline_nonfinite,
                            "candidate_nonfinite": (~torch.isfinite(actual)).sum().item(),
                        }
                    ),
                    flush=True,
                )
            assert torch.isfinite(actual).all()
            if case == "all_empty":
                assert torch.count_nonzero(actual).item() == 0
            checks += 1

        # Graph replay must consume new values rather than cached capture data.
        output = output.contiguous()
        lse = lse.contiguous()
        output.normal_()
        lse.normal_()
        for _ in range(3):
            candidate.exchange(output, lse, dist.group.WORLD)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replay_output = candidate.exchange(output, lse, dist.group.WORLD)
        captured.append((graph, output, lse, replay_output))
        graph.replay()
        torch.npu.synchronize()
        static_output = replay_output.clone()
        torch.testing.assert_close(static_output, baseline(output, lse), atol=0, rtol=0)
        for _ in range(10):
            output.normal_()
            lse.normal_()
            graph.replay()
            torch.npu.synchronize()
            saved_output = replay_output.clone()
            expected = baseline(output, lse)
            torch.testing.assert_close(saved_output, expected, atol=0, rtol=0)
            checks += 1

        timings = {}
        for name, fn in (
            ("native", baseline),
            ("packed", lambda o, stats: candidate.exchange(o, stats, dist.group.WORLD)),
        ):
            for _ in range(20):
                fn(output, lse)
            torch.npu.synchronize()
            timed_graph = torch.npu.NPUGraph()
            with torch.npu.graph(timed_graph):
                _timed_output = fn(output, lse)
            captured.append((timed_graph, output, lse, _timed_output))
            samples = []
            for _ in range(5):
                dist.barrier()
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(200):
                    timed_graph.replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(start.elapsed_time(end) * 1000 / 200, device="npu")
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                samples.append(elapsed.item())
            timings[name] = {"median_us": statistics.median(samples), "samples_us": samples}
        results.append({"tokens": tokens, "timings": timings})

    # Kimi has 24 MLA layers. One graph must handle distinct buffers for each.
    layer_inputs = [
        (torch.randn(1, ranks * 6, 512, device="npu", dtype=torch.bfloat16), torch.randn(1, ranks * 6, 1, device="npu"))
        for _ in range(24)
    ]
    for output, lse in layer_inputs:
        candidate.exchange(output, lse, dist.group.WORLD)
    torch.npu.synchronize()
    layer_graph = torch.npu.NPUGraph()
    with torch.npu.graph(layer_graph):
        layer_outputs = [candidate.exchange(output, lse, dist.group.WORLD) for output, lse in layer_inputs]
    for _ in range(3):
        for output, lse in layer_inputs:
            output.add_(0.125)
            lse.add_(0.0625)
        layer_graph.replay()
        torch.npu.synchronize()
        for actual, (output, lse) in zip(layer_outputs, layer_inputs):
            torch.testing.assert_close(actual, baseline(output, lse), atol=0, rtol=0)
            checks += 1
    result = {"rank": rank, "ranks": ranks, "exact_checks": checks, "results": results}
    target = Path(os.environ.get("RESULT_DIR", "/tmp/mla-dcp-port-results"))
    target.mkdir(parents=True, exist_ok=True)
    (target / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    if rank == 0:
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
