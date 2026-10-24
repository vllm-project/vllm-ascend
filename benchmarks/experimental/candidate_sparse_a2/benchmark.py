# SPDX-License-Identifier: Apache-2.0
"""Run from this directory inside an A2 vLLM-Ascend image with native QLI V3."""

import argparse
import hashlib
import importlib
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_npu
import triton
from kernels import allocate, metadata, pack, score
from reference import BLOCK_SIZE, NEG_HUGE, block_plan

from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
importlib.import_module("vllm_ascend.vllm_ascend_C")

HEADS = 32
WIDTH = 128
PAGE = 128
TOPK = 512
CANDIDATES = 2048


@dataclass
class Case:
    query: list
    weights: list
    query_scale: list
    key: torch.Tensor
    scale: torch.Tensor
    table: torch.Tensor
    candidates: torch.Tensor
    positions: torch.Tensor
    starts: torch.Tensor
    lengths: torch.Tensor
    residual: torch.Tensor | None
    ratio: int
    max_queries: int
    max_length: int


def make_case(length, requests, queries, ratio, seed, pattern="random", hybrid=True):
    """Unequal requests, shuffled physical pages and real Hybrid page strides."""
    torch.manual_seed(seed)
    lengths = [max(0, length - request * 17) for request in range(requests)]
    page_counts = [max(1, math.ceil(value / PAGE)) for value in lengths]
    total_pages = sum(page_counts)
    physical = torch.randperm(total_pages, dtype=torch.int32)
    table = torch.zeros((requests, max(page_counts)), dtype=torch.int32)
    begin = 0
    for request, pages in enumerate(page_counts):
        table[request, :pages] = physical[begin : begin + pages]
        begin += pages
    key_stride = 147712 if hybrid else PAGE * WIDTH
    scale_stride = 147712 // 2 if hybrid else PAGE
    if hybrid:
        # A C1 Hybrid page shares one allocation: latent KV, index K, scales.
        latent_bytes = PAGE * 512 * 2
        index_bytes = PAGE * WIDTH
        storage = torch.empty(total_pages * key_stride, dtype=torch.int8, device="npu")
        key = storage.as_strided((total_pages, PAGE, 1, WIDTH), (key_stride, WIDTH, WIDTH, 1), latent_bytes)
        scale = storage.view(torch.float16).as_strided(
            (total_pages, PAGE, 1),
            (scale_stride, 1, 1),
            (latent_bytes + index_bytes) // 2,
        )
    else:
        key = torch.empty((total_pages, PAGE, 1, WIDTH), dtype=torch.int8, device="npu")
        scale = torch.empty((total_pages, PAGE, 1), dtype=torch.float16, device="npu")
    key.copy_(torch.randint(-90, 91, key.shape, dtype=torch.int8))
    scale.copy_(torch.rand(scale.shape, dtype=torch.float16) * 0.01 + 0.005)
    query_counts = [max(1, queries - request % queries) for request in range(requests)]
    starts = [0]
    positions = []
    candidates = []
    for request, count in enumerate(query_counts):
        starts.append(starts[-1] + count)
        for query in range(count):
            position = lengths[request] * ratio - count + query
            positions.append(position)
            visible = max(0, (position + 1) // ratio)
            blocks = math.ceil(visible / BLOCK_SIZE)
            ids = torch.randperm(blocks).tolist()
            if pattern == "empty":
                ids = []
            elif pattern == "one":
                ids = [blocks - 1] if blocks else []
            elif pattern == "prefix":
                ids = list(range(min(CANDIDATES, blocks)))
            else:
                ids = ids[:CANDIDATES]
                # Match the source's pinned final visible block.
                if blocks and blocks - 1 not in ids:
                    ids[-1] = blocks - 1
            candidates.append(ids + [-1] * (CANDIDATES - len(ids)))
    rows = sum(query_counts)
    query = [torch.randint(-90, 91, (rows, HEADS, WIDTH), dtype=torch.int8).npu() for _ in range(4)]
    weights = [torch.randn((rows, HEADS), dtype=torch.float16).npu() for _ in range(4)]
    query_scale = [(torch.rand((rows, HEADS), dtype=torch.float16) * 0.01 + 0.005).npu() for _ in range(4)]
    if pattern == "ties":
        for value in query:
            value.zero_()
    return Case(
        query,
        weights,
        query_scale,
        key,
        scale,
        table.npu(),
        torch.tensor(candidates, dtype=torch.int32).unsqueeze(1).npu(),
        torch.tensor(positions, dtype=torch.int64).npu(),
        torch.tensor(starts, dtype=torch.int32).npu(),
        torch.tensor(lengths, dtype=torch.int32).npu(),
        torch.zeros(requests, dtype=torch.int32, device="npu") if ratio == 2 else None,
        ratio,
        queries,
        length,
    )


def native_metadata(case):
    return torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        HEADS,
        1,
        WIDTH,
        TOPK,
        2,
        cu_seqlens_q=case.starts,
        seqused_k=case.lengths,
        cmp_residual_k=case.residual,
        batch_size=case.lengths.numel(),
        max_seqlen_q=case.max_queries,
        max_seqlen_k=max(1, case.max_length),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=case.ratio,
        device=str(case.key.device),
    )


def native_score(case, index, schedule, output):
    selected, _, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        case.query[index],
        case.key,
        case.weights[index],
        case.query_scale[index],
        case.scale,
        TOPK,
        2,
        cu_seqlens_q=case.starts,
        seqused_k=case.lengths,
        cmp_residual_k=case.residual,
        block_table=case.table,
        metadata=schedule,
        max_seqlen_q=case.max_queries,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=case.ratio,
        candidate_topk_index=case.candidates,
        candidate_mode=2,
        candidate_topk_blocks=CANDIDATES,
        candidate_block_size=BLOCK_SIZE,
    )
    return prepare_indexer_indices(selected.squeeze(1), case.positions, case.ratio, indices_output=output)


def pack_case(case, buffers):
    pack(
        case.key,
        case.scale,
        case.table,
        case.candidates,
        case.positions,
        case.starts,
        case.lengths,
        case.ratio,
        buffers,
    )


def outputs(case):
    return [torch.empty((case.query[0].shape[0], TOPK), dtype=torch.int32, device="npu") for _ in range(4)]


def sparse_group(case, buffers, mapped, output, consumers, *, schedule=None):
    if schedule is None:
        pack_case(case, buffers)
        schedule = metadata(case.query[0], buffers)
    for index in range(consumers):
        selected = score(
            case.query[index],
            case.weights[index],
            case.query_scale[index],
            schedule,
            buffers,
            mapped[index],
        )
        prepare_indexer_indices(selected, case.positions, case.ratio, indices_output=output[index])
    return output[:consumers]


def check_pack(case, buffers):
    """Every gathered INT8 byte and FP16 scale must match its original page."""
    key, scale, table = case.key.cpu(), case.scale.cpu(), case.table.cpu()
    packed_key = buffers.key.cpu().reshape(case.query[0].shape[0], buffers.capacity, WIDTH)
    packed_scale = buffers.scale.cpu().reshape(case.query[0].shape[0], buffers.capacity)
    candidates, positions, starts, lengths = (
        case.candidates.cpu(),
        case.positions.cpu(),
        case.starts.cpu(),
        case.lengths.cpu(),
    )
    requests, counts = buffers.requests.cpu(), buffers.lengths.cpu()
    mapped = buffers.candidates.cpu()
    for row in range(case.query[0].shape[0]):
        request = int(torch.searchsorted(starts[1:], row, right=True))
        visible = max(0, min(int(positions[row] + 1) // case.ratio, int(lengths[request])))
        _, expected_candidates, expected_positions = block_plan(candidates[row, 0].tolist(), visible)
        assert int(requests[row]) == request
        assert int(counts[row]) == len(expected_positions)
        assert mapped[row, 0].tolist() == expected_candidates + [-1] * (CANDIDATES - len(expected_candidates))
        ids = torch.tensor(expected_positions, dtype=torch.int64)
        pages = table[request, ids // PAGE].long()
        torch.testing.assert_close(packed_key[row, : len(ids)], key[pages, ids % PAGE, 0], rtol=0, atol=0)
        torch.testing.assert_close(packed_scale[row, : len(ids)], scale[pages, ids % PAGE, 0], rtol=0, atol=0)


def selected_scores(case, consumer, row, indices):
    """Recompute only selected scores using the existing INT8 CPU golden steps."""
    starts = case.starts.cpu()
    request = int(torch.searchsorted(starts[1:], row, right=True))
    query = case.query[consumer][row].cpu().int()
    ids = indices[indices >= 0].long()
    table = case.table[request].cpu().long()
    pages = table[ids // PAGE]
    keys = case.key.cpu()[pages, ids % PAGE, 0].int()
    scales = case.scale.cpu()[pages, ids % PAGE, 0].float()
    qk = ((query @ keys.T).float() / 1024).clamp_min(0).half()
    weights = (case.weights[consumer][row].cpu() * case.query_scale[consumer][row].cpu()).half()
    values = (weights.float() @ qk.float()) * scales
    candidates = case.candidates[row, 0].cpu()
    return torch.where(torch.isin(ids // BLOCK_SIZE, candidates), values, NEG_HUGE)


def check_selection(case, dense, sparse):
    # Validation runs off device after the measured call, never in a hot path.
    host_case = Case(
        *[[tensor.cpu() for tensor in getattr(case, name)] for name in ("query", "weights", "query_scale")],
        *[
            getattr(case, name).cpu()
            for name in (
                "key",
                "scale",
                "table",
                "candidates",
                "positions",
                "starts",
                "lengths",
            )
        ],
        case.residual,
        case.ratio,
        case.max_queries,
        case.max_length,
    )
    exact = True
    for consumer, (expected, actual) in enumerate(zip(dense, sparse)):
        expected, actual = expected.cpu(), actual.cpu()
        exact &= torch.equal(expected, actual)
        for row in range(actual.shape[0]):
            request = int(torch.searchsorted(case.starts.cpu()[1:], row, right=True))
            visible = max(
                0,
                min(
                    int(case.positions[row].cpu() + 1) // case.ratio,
                    int(case.lengths[request].cpu()),
                ),
            )
            live = actual[row][actual[row] >= 0]
            assert live.numel() == min(visible, TOPK)
            assert live.unique().numel() == live.numel()
            assert (live < visible).all()
            assert torch.equal(live, live.sort().values)
            assert torch.equal(actual[row, : live.numel()], live)
            assert (actual[row, live.numel() :] == -1).all()
            if not torch.equal(expected[row], actual[row]):
                # Equal-score alternatives are reported, never called exact agreement.
                expected_scores = selected_scores(host_case, consumer, row, expected[row]).sort().values
                actual_scores = selected_scores(host_case, consumer, row, actual[row]).sort().values
                torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)
    return {
        "valid_topk": True,
        "exact_indices": exact,
        "equal_scores_for_different_indices": not exact,
    }


def capture(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        output = fn()
    graph.replay()
    torch.npu.synchronize()
    return graph, output


def measure(functions, samples, repeats):
    raw = {name: [] for name in functions}
    names = list(functions)
    for sample in range(samples):
        for name in names[sample % len(names) :] + names[: sample % len(names)]:
            start, end = (
                torch.npu.Event(enable_timing=True),
                torch.npu.Event(enable_timing=True),
            )
            torch.npu.synchronize()
            wall = time.perf_counter_ns()
            start.record()
            for _ in range(repeats):
                functions[name]()
            end.record()
            end.synchronize()
            raw[name].append(
                {
                    "device_us": start.elapsed_time(end) * 1000 / repeats,
                    "wall_us": (time.perf_counter_ns() - wall) / (1000 * repeats),
                }
            )
    return {
        name: {
            "median_device_us": statistics.median(row["device_us"] for row in rows),
            "median_wall_us": statistics.median(row["wall_us"] for row in rows),
            "samples": rows,
        }
        for name, rows in raw.items()
    }


def refresh(case, replacement):
    """Refresh graph inputs in place, including pages, candidates and visibility."""
    for name in (
        "key",
        "scale",
        "table",
        "candidates",
        "positions",
        "starts",
        "lengths",
    ):
        getattr(case, name).copy_(getattr(replacement, name))
    for name in ("query", "weights", "query_scale"):
        for destination, source in zip(getattr(case, name), getattr(replacement, name)):
            destination.copy_(source)


def record_phase(args, tag, phase, data):
    # Keep completed phases if the device disappears during a later phase.
    with args.output.with_suffix(".jsonl").open("a") as stream:
        stream.write(json.dumps({"case": tag, "phase": phase, "data": data}) + "\n")


def run_case(case, consumers, args, tag):
    rows = case.query[0].shape[0]
    buffers = allocate(rows, CANDIDATES, TOPK, PAGE, case.key.device)
    dense_output, sparse_output, mapped = outputs(case), outputs(case), outputs(case)
    dense_schedule = native_metadata(case)

    def dense():
        return [native_score(case, index, dense_schedule, dense_output[index]) for index in range(consumers)]

    def sparse():
        return sparse_group(case, buffers, mapped, sparse_output, consumers)

    expected = [value.cpu().clone() for value in dense()]
    actual = sparse()
    check_pack(case, buffers)
    checks = {"eager": check_selection(case, expected, actual)}
    record_phase(args, tag, "eager_validation", checks["eager"])
    schedules = metadata(case.query[0], buffers)

    def score_only():
        return sparse_group(case, buffers, mapped, sparse_output, consumers, schedule=schedules)

    def sparse_pack_each_consumer():
        for index in range(consumers):
            pack_case(case, buffers)
            schedule = metadata(case.query[0], buffers)
            selected = score(
                case.query[index],
                case.weights[index],
                case.query_scale[index],
                schedule,
                buffers,
                mapped[index],
            )
            prepare_indexer_indices(
                selected,
                case.positions,
                case.ratio,
                indices_output=sparse_output[index],
            )
        return sparse_output[:consumers]

    functions = {
        "dense_existing_metadata": dense,
        "sparse_full_including_pack_and_metadata": sparse,
        "sparse_score_and_restore_only": score_only,
    }
    if consumers > 1:
        functions["sparse_pack_each_consumer"] = sparse_pack_each_consumer
    # All real extra work is in sparse_full; score_only is a diagnostic ceiling.
    timings = {"eager": measure(functions, args.samples, args.repeats)}
    record_phase(args, tag, "eager_timings", timings["eager"])
    if args.profile_dir is not None:
        profile_path = args.profile_dir / "-".join(str(value) for value in tag.values())
        config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        )
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_path)),
        ):
            for _ in range(3):
                for name, function in functions.items():
                    with torch.profiler.record_function(name):
                        function()
            torch.npu.synchronize()
        record_phase(args, tag, "profiler", {"path": str(profile_path)})
    if args.graph:
        dense_graph, _ = capture(dense)
        sparse_graph, graph_output = capture(sparse)
        checks["graph"] = check_selection(case, expected, graph_output)
        record_phase(args, tag, "graph_validation", checks["graph"])
        timings["graph"] = measure(
            {
                "dense_existing_metadata": dense_graph.replay,
                "sparse_full_including_pack_and_metadata": sparse_graph.replay,
            },
            args.samples,
            args.repeats,
        )
        record_phase(args, tag, "graph_timings", timings["graph"])
        replacement = make_case(
            case.max_length,
            case.lengths.numel(),
            case.max_queries,
            case.ratio,
            args.seed + 1,
            pattern="one",
            hybrid=args.hybrid,
        )
        # Same allocation shapes; change both length and position to exercise dynamic schedules.
        replacement.lengths.sub_(min(3, case.max_length)).clamp_(min=0)
        replacement.positions.sub_(min(3, case.max_length) * case.ratio)
        refresh(case, replacement)
        # The dense schedule must be regenerated when lengths change.
        dense_schedule = native_metadata(case)
        refreshed_expected = [value.cpu().clone() for value in dense()]
        sparse_graph.replay()
        torch.npu.synchronize()
        check_pack(case, buffers)
        checks["graph_refreshed_inputs"] = check_selection(case, refreshed_expected, graph_output)
        record_phase(args, tag, "graph_refreshed_inputs", checks["graph_refreshed_inputs"])
    return {
        "checks": checks,
        "capacity": buffers.capacity,
        "packed_cache_bytes": buffers.key.numel() * buffers.key.element_size()
        + buffers.scale.numel() * buffers.scale.element_size(),
        "actual_key_stride": case.key.stride(0),
        "actual_scale_stride": case.scale.stride(0),
        "timings": timings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--consumers", type=int, choices=[1, 4], nargs="+", default=[1, 4])
    parser.add_argument("--ratio", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["random", "empty", "one", "prefix", "ties"],
        default=["random"],
    )
    parser.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    result = {
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "triton": triton.__version__,
            "python": platform.python_version(),
            "device": torch.npu.get_device_name(0),
        },
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("kernels.py", "benchmark.py", "reference.py")
        },
        "arguments": {
            **vars(args),
            "output": str(args.output),
            "profile_dir": str(args.profile_dir) if args.profile_dir is not None else None,
        },
        "cases": [],
    }
    args.output.write_text(json.dumps(result, indent=2))
    args.output.with_suffix(".jsonl").write_text(
        json.dumps(
            {
                "environment": result["environment"],
                "source_sha256": result["source_sha256"],
            }
        )
        + "\n"
    )
    for length in args.lengths:
        for requests in args.requests:
            for consumers in args.consumers:
                for pattern in args.patterns:
                    tag = {
                        "length": length,
                        "requests": requests,
                        "queries": args.queries,
                        "consumers": consumers,
                        "pattern": pattern,
                        "ratio": args.ratio,
                    }
                    print(json.dumps({"starting": tag}), flush=True)
                    record_phase(args, tag, "starting", {})
                    case = make_case(
                        length,
                        requests,
                        args.queries,
                        args.ratio,
                        args.seed,
                        pattern,
                        args.hybrid,
                    )
                    data = run_case(case, consumers, args, tag)
                    result["cases"].append({**tag, **data})
                    args.output.write_text(json.dumps(result, indent=2))
                    print(json.dumps(result["cases"][-1]), flush=True)
                    del case
                    torch.npu.empty_cache()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
