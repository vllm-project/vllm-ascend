# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[Role] Check raw-ID Hash TopK against normalized IDs and measure graph latency.
[Invariants] Same logits/table for both paths; exact A/B IDs and weights required.
[Timeouts & Locks] Bounded shapes/replays; caller reserves one idle NPU.
[Side-Effect Policy] Only creates requested JSON/profiler output; no package changes.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu


def make_case(rows, with_hash, with_vision, id_dtype):
    generator = torch.Generator().manual_seed(20261110 + rows)
    logits = torch.randn(rows, 384, generator=generator)
    text_bias = torch.randn(384, generator=generator) * 0.2
    vision_bias = torch.randn(384, generator=generator) * 0.2 if with_vision else None
    ids = torch.arange(rows, dtype=id_dtype) % 4096
    ids[::3] = -1
    if with_vision and rows > 1:
        ids[1::7] = 129257
    table = (torch.arange(4096)[:, None] * 7 + torch.arange(6)[None, :] * 17) % 384 if with_hash else None
    return logits, text_bias, vision_bias, ids, table


def reference(logits, text_bias, vision_bias, ids, table):
    ids = torch.where(ids == -1, 0, ids).long()
    image = (ids >= 129257) & (ids < 129262) if vision_bias is not None else torch.zeros_like(ids, dtype=torch.bool)
    scores = torch.nn.functional.softplus(logits).sqrt()
    bias = torch.where(image[:, None], vision_bias, text_bias) if vision_bias is not None else text_bias
    expert_ids = torch.topk(scores + bias, 6, dim=-1).indices
    if table is not None:
        hash_ids = table[torch.where(image, 0, ids)]
        expert_ids = torch.where(image[:, None], expert_ids, hash_ids)
    weights = scores.gather(1, expert_ids)
    weights = weights / weights.sum(-1, keepdim=True) * 1.5
    return weights, expert_ids.int()


def functions(op, case):
    logits, bias, vision_bias, ids, table = case
    tensors = [x.npu() if x is not None else None for x in [logits, bias, vision_bias, ids, table]]
    x, b, vb, token_ids, hash_table = tensors
    if hash_table is not None:
        hash_table = hash_table.int()

    def run(values):
        return op(
            x=x,
            k=6,
            bias=b,
            input_ids=values,
            tid2eid=hash_table,
            k_group=1,
            group_count=1,
            routed_scaling_factor=1.5,
            eps=1e-20,
            group_select_mode=1,
            renorm=0,
            norm_type=2,
            out_flag=False,
            bias_vl=vb,
            image_sentinel_lo=129257,
            image_sentinel_count=5,
        )

    def baseline():
        values = token_ids.to(torch.int64)
        return run(torch.where(values == -1, 0, values))

    def candidate():
        return run(token_ids)

    return baseline, candidate


def capture(fn, inner):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        for _ in range(inner):
            outputs = fn()
    return graph, outputs


def paired_latency(baseline, candidate, inner=32, replays=10):
    graphs = [capture(fn, inner) for fn in [baseline, candidate]]
    for graph, _ in graphs:
        graph.replay()
    torch.npu.synchronize()
    samples = [[], []]
    for trial in range(6):
        for index in [0, 1] if trial % 2 == 0 else [1, 0]:
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graphs[index][0].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000 / (inner * replays))
    medians = [statistics.median(x) for x in samples]
    for (_, outputs), fn in zip(graphs, [baseline, candidate]):
        expected = fn()
        torch.testing.assert_close(outputs[1], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0], expected[0], rtol=0, atol=0)
    return {
        "baseline_us": medians[0],
        "candidate_us": medians[1],
        "saved_us": medians[0] - medians[1],
        "reduction_pct": (1 - medians[1] / medians[0]) * 100,
        "paired_samples_us": samples,
        "inner_calls": inner,
        "replays": replays,
    }


def profile(fn, path):
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(path)),
        record_shapes=True,
    ) as profiler:
        for _ in range(4):
            fn()
            profiler.step()
        torch.npu.synchronize()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="Optional isolated extension implementing the same 0.32 schema")
    parser.add_argument("--namespace", default="_C_ascend")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 23, 96, 132, 432])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if args.library:
        torch.ops.load_library(args.library)
    else:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Hash TopK custom op is unavailable")
    op = getattr(torch.ops, args.namespace).moe_gating_top_k_hash
    torch.npu.set_device(0)
    results = {
        "torch": torch.__version__,
        "device": torch.npu.get_device_name(0),
        "namespace": args.namespace,
        "precision": [],
        "graph_latency": [],
    }
    for rows in args.rows:
        for with_hash, with_vision in [(True, False), (True, True), (False, True)]:
            for id_dtype in [torch.int32, torch.int64]:
                case = make_case(rows, with_hash, with_vision, id_dtype)
                baseline, candidate = functions(op, case)
                expected_weights, expected_ids = reference(*case)
                old, new = baseline(), candidate()
                torch.testing.assert_close(new[1].cpu(), expected_ids, rtol=0, atol=0)
                torch.testing.assert_close(new[0].cpu(), expected_weights, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(new[1], old[1], rtol=0, atol=0)
                torch.testing.assert_close(new[0], old[0], rtol=0, atol=0)
                for fn in [baseline, candidate]:
                    graph, outputs = capture(fn, 1)
                    graph.replay()
                    torch.npu.synchronize()
                    torch.testing.assert_close(outputs[1], old[1], rtol=0, atol=0)
                    torch.testing.assert_close(outputs[0], old[0], rtol=0, atol=0)
                record = {
                    "rows": rows,
                    "hash": with_hash,
                    "vision": with_vision,
                    "id_dtype": str(id_dtype),
                    "eager_and_graph": "passed",
                    "baseline_candidate_bit_exact": True,
                    "max_weight_abs_error": float((new[0].cpu() - expected_weights).abs().max()),
                }
                results["precision"].append(record)
                print(json.dumps(record), flush=True)
        case = make_case(rows, True, False, torch.int32)
        baseline, candidate = functions(op, case)
        result = {"rows": rows, **paired_latency(baseline, candidate)}
        results["graph_latency"].append(result)
        print(json.dumps(result), flush=True)
    if args.profile_dir:
        baseline, candidate = functions(op, make_case(132, True, False, torch.int32))
        for name, fn in [("baseline", baseline), ("candidate", candidate)]:
            profile(fn, args.profile_dir / name)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
