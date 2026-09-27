# SPDX-License-Identifier: Apache-2.0
"""Reproducible standalone cases for gating TopK + EPLB map + record.

Run inside an Ascend container, without installing the local checkout:

    python case_generator.py --task function --t 64 --e 16 --k 8
    python case_generator.py --task performance --matrix

The baseline is the current two-launch path: the project CANN operator and
the extracted, installed Triton map-and-record function. A candidate module
can be passed with --candidate; it must export ``moe_gating_topk_map_record``.
"""

import argparse
import importlib.util
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

DECODE_TOKENS = (64, 128, 256, 512)
PREFILL_TOKENS = (65536, 131072, 262144, 524288)
GRADIENT_TOKENS = (1024, 2048, 4096, 8192, 16384, 32768)
EXPERT_COUNTS = (8, 16, 32)
TOP_K_VALUES = (6, 8)
SCORING_KINDS = ("softmax", "sigmoid")
TABLE_ROWS = 1024


@dataclass(frozen=True)
class Case:
    tokens: int
    experts: int
    top_k: int
    scoring: str
    bias: bool = False
    record: bool = True
    valid_tokens: int | None = None
    dtype: str = "float32"
    pattern: str = "random"
    map_kind: str = "permutation"
    valid_as_int: bool = False

    def label(self) -> str:
        return (
            f"T={self.tokens},E={self.experts},K={self.top_k},"
            f"score={self.scoring},bias={int(self.bias)},record={int(self.record)},"
            f"dtype={self.dtype},pattern={self.pattern},map={self.map_kind},valid_int={int(self.valid_as_int)}"
        )


def make_inputs(case: Case, device: str):
    generator = torch.Generator(device="cpu").manual_seed(20260927)
    dtype = getattr(torch, case.dtype)
    logits = torch.randn(case.tokens, case.experts, generator=generator, dtype=torch.float32).to(device, dtype=dtype)
    if case.pattern == "ties":
        logits = (torch.arange(case.experts, device=device) % 3).to(dtype).expand(case.tokens, -1).contiguous()
    elif case.pattern == "all-zero":
        logits.zero_()
    bias = (
        torch.randn(case.experts, generator=generator, dtype=torch.float32).to(device, dtype=dtype)
        if case.bias
        else None
    )
    rows = torch.arange(TABLE_ROWS, device=device, dtype=torch.int32)[:, None]
    expert = torch.arange(case.experts, device=device, dtype=torch.int32)[None, :]
    table_expert = expert if case.map_kind == "permutation" else expert // 2
    table = ((rows + table_expert) % case.experts).contiguous()
    load = torch.zeros(case.experts, dtype=torch.int32, device=device)
    enabled = torch.tensor(case.record, dtype=torch.bool, device=device)
    valid = torch.tensor(
        case.tokens if case.valid_tokens is None else case.valid_tokens,
        dtype=torch.int32,
        device=device,
    )
    # The production record kernel consumes local counts from the downstream
    # MoE operator. Keep a stable representative count vector for profiling
    # that kernel without including the MoE computation in the timing region.
    expert_tokens = torch.full(
        (case.experts,),
        case.tokens * case.top_k // case.experts,
        dtype=torch.int32,
        device=device,
    )
    return logits, bias, table, load, enabled, valid, expert_tokens


def load_baseline_ops(path: str):
    spec = importlib.util.spec_from_file_location("upstream_eplb_kernels", Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load upstream EPLB kernels: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline(case: Case, tensors, baseline_ops):
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    logits, bias, table, load, enabled, _, expert_tokens = tensors
    weights, logical_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
        logits,
        k=case.top_k,
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=1,
        norm_type=0 if case.scoring == "softmax" else 1,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        bias_opt=bias,
    )
    physical_ids = baseline_ops.map_to_physical_triton(logical_ids, table)
    baseline_ops.record_expert_tokens_triton(expert_tokens, load, enabled, 1, 0)
    return weights, physical_ids


def independent_reference(case: Case, tensors):
    logits, bias, table, _, _, _, _ = tensors
    x = logits.cpu().float()
    scores = x.softmax(dim=-1) if case.scoring == "softmax" else x.sigmoid()
    keys = scores if bias is None else scores + bias.cpu().float()
    logical_ids = torch.argsort(keys, dim=-1, descending=True, stable=True)[:, : case.top_k]
    weights = scores.gather(1, logical_ids)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    row = torch.arange(case.tokens)[:, None] % TABLE_ROWS
    physical_ids = table.cpu()[row, logical_ids]
    valid = case.tokens if case.valid_tokens is None else case.valid_tokens
    expected_load = torch.zeros(case.experts, dtype=torch.int32)
    if case.record:
        expected_load += torch.bincount(physical_ids[:valid].reshape(-1).to(torch.int64), minlength=case.experts).to(
            torch.int32
        )
    return weights.to(logits.dtype), physical_ids, expected_load


def load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("moe_gating_candidate", Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.moe_gating_topk_map_record


def run_one(case: Case, baseline_ops, candidate=None, check=True):
    tensors = make_inputs(case, "npu")
    weights, physical_ids = baseline(case, tensors, baseline_ops)
    torch.npu.synchronize()
    if check:
        ref_weights, ref_ids, _ = independent_reference(case, tensors)
        torch.testing.assert_close(physical_ids.cpu(), ref_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights.cpu(), ref_weights, rtol=1e-4, atol=1e-5)
    if candidate is not None:
        candidate_valid = (
            (case.tokens if case.valid_tokens is None else case.valid_tokens) if case.valid_as_int else tensors[5]
        )
        candidate_tensors = (*tensors[:3], torch.zeros_like(tensors[3]), tensors[4], candidate_valid)
        out_weights, out_ids = candidate(
            *candidate_tensors,
            k=case.top_k,
            scoring=case.scoring,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(out_ids.cpu(), physical_ids.cpu(), rtol=0, atol=0)
        expected_load = torch.zeros(case.experts, dtype=torch.int32)
        if case.record:
            valid_rows = case.tokens if case.valid_tokens is None else case.valid_tokens
            expected_load += torch.bincount(
                physical_ids[:valid_rows].cpu().reshape(-1).to(torch.int64),
                minlength=case.experts,
            ).to(torch.int32)
        torch.testing.assert_close(candidate_tensors[3].cpu(), expected_load, rtol=0, atol=0)
        torch.testing.assert_close(out_weights.cpu(), weights.cpu(), rtol=1e-4, atol=1e-5)
    return {"case": case.label(), "status": "pass"}


def benchmark(case: Case, baseline_ops, candidate=None, repetitions=20):
    tensors = make_inputs(case, "npu")
    implementations = [("baseline", lambda c, t: baseline(c, t, baseline_ops))]
    if candidate is not None:
        implementations.append(("candidate", lambda c, t: candidate(*t[:6], k=c.top_k, scoring=c.scoring)))
    results = []
    for name, op in implementations:
        for _ in range(5):
            tensors[3].zero_()
            op(case, tensors)
        torch.npu.synchronize()
        stream_elapsed_us = []
        host_us = []
        dispatch_us = []
        for _ in range(repetitions):
            tensors[3].zero_()
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            host_start = time.perf_counter_ns()
            start.record()
            dispatch_start = time.perf_counter_ns()
            op(case, tensors)
            dispatch_us.append((time.perf_counter_ns() - dispatch_start) / 1000)
            end.record()
            torch.npu.synchronize()
            host_us.append((time.perf_counter_ns() - host_start) / 1000)
            stream_elapsed_us.append(start.elapsed_time(end) * 1000)
        results.append(
            {
                "case": case.label(),
                "implementation": name,
                # Stream events include host enqueue gaps between kernels.
                # msprof op provides each kernel's device execution time.
                "stream_elapsed_us_median": round(statistics.median(stream_elapsed_us), 3),
                "host_sync_us_median": round(statistics.median(host_us), 3),
                "host_dispatch_us_median": round(statistics.median(dispatch_us), 3),
                "samples": repetitions,
            }
        )
    return results


def check_graph_replay(candidate):
    case = Case(64, 16, 8, "softmax", record=False, valid_as_int=True)
    logits, _, table, load, enabled, _, _ = make_inputs(case, "npu")
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    expected_weights, logical_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
        logits,
        k=8,
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=1,
        norm_type=0,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        bias_opt=None,
    )
    candidate(logits, None, table, load, enabled, 64, k=8, scoring="softmax")
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        weights, ids = candidate(logits, None, table, load, enabled, 64, k=8, scoring="softmax")
    row = torch.arange(TABLE_ROWS, dtype=torch.int32, device="npu")[:, None]
    expert = torch.arange(case.experts, dtype=torch.int32, device="npu")[None, :]
    for shift in (1, 3):
        table.copy_((row + expert + shift) % case.experts)
        graph.replay()
        torch.npu.synchronize()
        expected_ids = table[torch.arange(case.tokens, device="npu")[:, None], logical_ids.long()]
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(load, torch.zeros_like(load), rtol=0, atol=0)
    return {"case": case.label(), "status": "graph-replay-pass"}


def cases_from_args(args):
    if args.edge_suite:
        yield Case(64, 8, 6, "sigmoid", True, False, 64, "float32", "random", "permutation", True)
        yield Case(64, 16, 8, "softmax", False, True, 48, "float32", "all-zero", "redundant", True)
        yield Case(128, 32, 6, "sigmoid", True, True, 96, "float32", "ties", "redundant", False)
        yield Case(256, 16, 8, "softmax", True, True, 256, "float16", "random", "redundant", False)
        yield Case(512, 32, 8, "sigmoid", True, True, 500, "bfloat16", "random", "permutation", False)
    elif args.matrix or args.gradient:
        tokens_to_test = GRADIENT_TOKENS if args.gradient else (*DECODE_TOKENS, *PREFILL_TOKENS)
        for tokens in tokens_to_test:
            for experts in EXPERT_COUNTS:
                for scoring in SCORING_KINDS:
                    for top_k in TOP_K_VALUES:
                        yield Case(tokens, experts, top_k, scoring, args.bias, args.record, dtype=args.dtype)
    else:
        yield Case(
            args.t,
            args.e,
            args.k,
            args.scoring,
            args.bias,
            args.record,
            args.valid_tokens,
            args.dtype,
            args.pattern,
            args.map_kind,
            args.valid_as_int,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("function", "accuracy", "performance", "profile", "graph", "simulator"), required=True
    )
    parser.add_argument("--candidate")
    parser.add_argument("--profile-implementation", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--eplb-source")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--gradient", action="store_true")
    parser.add_argument("--edge-suite", action="store_true")
    parser.add_argument("--t", type=int, default=64)
    parser.add_argument("--e", type=int, default=16)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--scoring", choices=SCORING_KINDS, default="softmax")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--pattern", choices=("random", "ties", "all-zero"), default="random")
    parser.add_argument("--map-kind", choices=("permutation", "redundant"), default="permutation")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--valid-tokens", type=int)
    parser.add_argument("--valid-as-int", action="store_true")
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if args.task == "simulator":
        for case in cases_from_args(args):
            cpu = make_inputs(case, "cpu")
            weights, ids, load = independent_reference(case, cpu)
            print(
                json.dumps(
                    {"case": case.label(), "weights": weights.tolist(), "ids": ids.tolist(), "load": load.tolist()}
                )
            )
        return
    import torch_npu  # noqa: F401  (registers the NPU device)

    candidate = load_candidate(args.candidate) if args.candidate else None
    if args.task == "graph":
        if candidate is None:
            raise ValueError("--candidate is required for graph replay")
        print(json.dumps(check_graph_replay(candidate), sort_keys=True))
        return
    eplb_source = args.eplb_source or Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/eplb.py"
    baseline_ops = load_baseline_ops(str(eplb_source))
    for case in cases_from_args(args):
        if case.top_k > case.experts:
            continue
        if args.task == "profile":
            tensors = make_inputs(case, "npu")
            if args.profile_implementation == "candidate":
                if candidate is None:
                    raise ValueError("--candidate is required for candidate profiling")
                candidate(*tensors[:6], k=case.top_k, scoring=case.scoring)
            else:
                baseline(case, tensors, baseline_ops)
            torch.npu.synchronize()
            result = {"case": case.label(), "implementation": args.profile_implementation, "status": "profiled"}
        elif args.task == "performance":
            result = benchmark(case, baseline_ops, candidate, args.repetitions)
        else:
            result = run_one(
                case,
                baseline_ops,
                candidate,
                check=case.tokens <= max(DECODE_TOKENS) and case.pattern == "random" and case.dtype == "float32",
            )
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
