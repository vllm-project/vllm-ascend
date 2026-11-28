# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark GLM-5.3 and GLM-5.4 MTP5 QSFA on synthetic, paged INT8 KV."""

import argparse
import csv
import functools
import importlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

QUERY_TOKENS = 6
LATENT_DIM = 512
PAGE_SIZE = 128
QUANT_TILE = 128
POOL_SIZE = 4
DEFAULT_SCALE = 256**-0.5
SCALE_BYTES = 4 * (LATENT_DIM // QUANT_TILE)
PEAK_BF16_TFLOPS = 280.0
PEAK_HBM_GB_S = 1600.0
WORKLOAD_COLUMNS = ("valid_sparse_pairs", "useful_flops", "logical_io_bytes")
UTILIZATION_COLUMNS = tuple(
    f"{implementation}_{metric}"
    for implementation in ("native", "candidate")
    for metric in ("useful_mfu_pct", "logical_mbu_pct")
)


@dataclass(frozen=True)
class Inputs:
    query: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor
    block_table: torch.Tensor
    query_ends: torch.Tensor
    kv_lengths: torch.Tensor
    candidate_query: torch.Tensor | None = None
    query_rope: torch.Tensor | None = None

    @property
    def rope_dim(self):
        return self.query.shape[-1] - LATENT_DIM


def prepare_storage(max_pages, device, seed, rope_dim=0):
    """Allocate each physical page once even when sweeping many batch sizes."""
    generator = torch.Generator().manual_seed(seed)
    packed_dim = LATENT_DIM + 2 * rope_dim + SCALE_BYTES
    storage = torch.empty((max_pages, PAGE_SIZE, 1, packed_dim), dtype=torch.int8, device=device)
    for start in range(0, max_pages, 128):
        pages = min(128, max_pages - start)
        prefix = (pages, PAGE_SIZE, 1)
        data = torch.randint(-127, 128, (*prefix, LATENT_DIM), dtype=torch.int8, generator=generator)
        scales = torch.rand((*prefix, LATENT_DIM // QUANT_TILE), generator=generator) * 0.02 + 0.005
        parts = [data]
        if rope_dim:
            rope = torch.randn((*prefix, rope_dim), generator=generator, dtype=torch.bfloat16)
            parts.append(rope.view(torch.int8))
        parts.append(scales.contiguous().view(torch.int8))
        storage[start : start + pages].copy_(torch.cat(parts, dim=-1))
    return storage


def make_inputs(batch, context_length, heads, topk, seed, storage, index_pattern="kpool"):
    generator = torch.Generator().manual_seed(seed + batch)
    rng = random.Random(seed + batch)
    page_count = math.ceil(context_length / PAGE_SIZE)
    rows = batch * QUERY_TOKENS
    query = (torch.randn((rows, heads, LATENT_DIM), generator=generator) * 0.5).to(torch.bfloat16)
    block_table = torch.randperm(batch * page_count, dtype=torch.int32, generator=generator).reshape(batch, page_count)
    rope_dim = (storage.shape[-1] - LATENT_DIM - SCALE_BYTES) // 2
    candidate_query = query_rope = None
    if rope_dim:
        rope = (torch.randn((rows, heads, rope_dim), generator=generator) * 0.5).to(torch.bfloat16)
        candidate_query = query.to(storage.device)
        query_rope = rope.to(storage.device)
        query = torch.cat((query, rope), dim=-1)
    group_size = POOL_SIZE if index_pattern == "kpool" else 1
    indices = torch.full((rows, 1, topk + group_size - 1), -1, dtype=torch.int32)
    query_ends = torch.arange(1, batch + 1, dtype=torch.int32) * QUERY_TOKENS
    kv_lengths = torch.tensor(
        [context_length - request % min(31, context_length - QUERY_TOKENS + 1) for request in range(batch)],
        dtype=torch.int32,
    )
    group_offsets = torch.arange(group_size, dtype=torch.int32)
    for request, kv_length in enumerate(kv_lengths.tolist()):
        first_visible = kv_length - QUERY_TOKENS + 1
        first_groups = first_visible // group_size
        shared = rng.sample(range(first_groups), min(first_groups, topk // group_size) // 2)
        shared_set = set(shared)
        for token in range(QUERY_TOKENS):
            visible = first_visible + token
            groups = visible // group_size
            needed = min(groups, topk // group_size)
            sampled = [group for group in rng.sample(range(groups), needed) if group not in shared_set]
            group_ids = shared + sampled[: needed - len(shared)]
            rng.shuffle(group_ids)
            history = (torch.tensor(group_ids, dtype=torch.int32)[:, None] * group_size + group_offsets).flatten()
            tail = torch.arange(groups * group_size, visible, dtype=torch.int32)
            values = torch.cat((history, tail))
            indices[request * QUERY_TOKENS + token, 0, : values.numel()] = values
    cpu_metadata = (indices, block_table, query_ends, kv_lengths)
    device_tensors = (tensor.to(storage.device) for tensor in (query, *cpu_metadata))
    q, idx, table, q_ends, lengths = device_tensors
    inputs = Inputs(q, storage[: batch * page_count], idx, table, q_ends, lengths, candidate_query, query_rope)
    return inputs, cpu_metadata


def selected_golden(inputs, cpu_metadata, batch, scale):
    """Compute FP32 attention for the six rows of one request in each batch."""
    indices, block_table, query_ends, kv_lengths = cpu_metadata
    request = batch // 2
    start = request * QUERY_TOKENS
    end = int(query_ends[request])
    query = inputs.query[start:end].cpu().float()
    result = torch.zeros_like(query[..., :LATENT_DIM])
    packed_rows = inputs.kv.view(-1, inputs.kv.shape[-1])
    scale_offset = LATENT_DIM + 2 * inputs.rope_dim
    for local_row, row in enumerate(range(start, end)):
        row_indices = indices[row, 0, : min(indices.shape[-1], int(kv_lengths[request]) - (end - row - 1))]
        invalid = (row_indices < 0).nonzero()
        if invalid.numel():
            row_indices = row_indices[: int(invalid[0, 0])]
        causal_limit = int(kv_lengths[request]) - (end - row - 1)
        tokens = row_indices[(row_indices >= 0) & (row_indices < causal_limit)].long()
        if tokens.numel() == 0:
            continue
        physical = block_table[request, tokens // PAGE_SIZE].long() * PAGE_SIZE + tokens % PAGE_SIZE
        packed = packed_rows.index_select(0, physical.to(inputs.kv.device)).cpu()
        scales = packed[:, scale_offset:].contiguous().view(torch.float32).repeat_interleave(QUANT_TILE, dim=-1)
        latent = (packed[:, :LATENT_DIM].float() * scales).to(torch.bfloat16).float()
        logits = query[local_row, :, :LATENT_DIM] @ latent.T
        if inputs.rope_dim:
            rope = packed[:, LATENT_DIM:scale_offset].contiguous().view(torch.bfloat16).float()
            logits += query[local_row, :, LATENT_DIM:] @ rope.T
        result[local_row] = (logits * scale).softmax(dim=-1) @ latent
    return slice(start, end), result


def check_fixture(cpu_metadata, topk, index_pattern="kpool"):
    indices, block_table, query_ends, kv_lengths = cpu_metadata
    if not torch.equal(block_table.flatten().sort().values, torch.arange(block_table.numel(), dtype=torch.int32)):
        raise AssertionError("Block table must uniquely assign each physical page")
    first_visible = kv_lengths - QUERY_TOKENS + 1
    visible = (first_visible[:, None] + torch.arange(QUERY_TOKENS)).flatten()[:, None]
    group_size = POOL_SIZE if index_pattern == "kpool" else 1
    history = (visible // group_size * group_size).clamp_max(topk)
    expected_count = history + visible % group_size
    valid = torch.arange(indices.shape[-1])[None, :] < expected_count
    flat_indices = indices[:, 0, :]
    if not torch.all((flat_indices[valid] >= 0) & (flat_indices[valid] < visible.expand_as(valid)[valid])):
        raise AssertionError("Valid sparse IDs must fall inside the causal window")
    if not torch.all(flat_indices[~valid] == -1) or int(query_ends[-1]) != indices.shape[0]:
        raise AssertionError("Invalid sparse slots or inconsistent cumulative query length")
    return int(expected_count.sum())


def workload_counts(inputs, cpu_metadata, topk, index_pattern):
    pairs = check_fixture(cpu_metadata, topk, index_pattern)
    tokens, heads, qk_dim = inputs.query.shape
    # Useful QK and PV only. One multiply-add counts as two FLOPs.
    useful_flops = 2 * pairs * heads * (qk_dim + LATENT_DIM)
    # Count one complete packed KV row per selected (query, KV) pair.
    # Query/output and metadata are counted once, without cache/reload assumptions.
    logical_io_bytes = pairs * inputs.kv.shape[-1] * inputs.kv.element_size()
    logical_io_bytes += inputs.query.numel() * inputs.query.element_size()
    logical_io_bytes += tokens * heads * LATENT_DIM * inputs.query.element_size()
    logical_io_bytes += sum(tensor.numel() * tensor.element_size() for tensor in cpu_metadata)
    return dict(zip(WORKLOAD_COLUMNS, (pairs, useful_flops, logical_io_bytes), strict=True))


def add_utilization(record, peak_bf16_tflops, peak_hbm_gb_s):
    for implementation in ("native", "candidate"):
        latency_us = record.get(f"{implementation}_us")
        if latency_us is None:
            continue
        record[f"{implementation}_useful_mfu_pct"] = (
            100 * record["useful_flops"] / (latency_us * peak_bf16_tflops * 1e6)
        )
        record[f"{implementation}_logical_mbu_pct"] = (
            100 * record["logical_io_bytes"] / (latency_us * peak_hbm_gb_s * 1e3)
        )


def native(inputs, scale, *, implementation="vllm-ascend"):
    kwargs = dict(
        query=inputs.query,
        key=inputs.kv,
        value=inputs.kv,
        sparse_indices=inputs.indices,
        scale_value=scale,
        sparse_block_size=1,
        block_table=inputs.block_table,
        actual_seq_lengths_query=inputs.query_ends,
        actual_seq_lengths_kv=inputs.kv_lengths,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=QUANT_TILE,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=inputs.rope_dim,
    )
    if implementation == "torch-npu":
        import torch_npu

        return torch_npu.npu_kv_quant_sparse_flash_attention(**kwargs)
    return torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm(**kwargs, return_softmax_lse=False)


def load_candidate(specification, block_i, packed_gather):
    module_name, function_name = specification.split(":", maxsplit=1)
    function = getattr(importlib.import_module(module_name), function_name)
    combined_query = getattr(function, "uses_combined_query", False)
    if block_i is not None:
        function = functools.partial(function, block_i=block_i)
    if packed_gather:
        function = functools.partial(function, packed_gather=True)

    def call(inputs, scale):
        query = inputs.query if combined_query or inputs.candidate_query is None else inputs.candidate_query
        rope = {} if combined_query or inputs.query_rope is None else {"query_rope": inputs.query_rope}
        return function(
            query,
            inputs.kv,
            inputs.indices,
            inputs.block_table,
            inputs.query_ends,
            inputs.kv_lengths,
            scale_value=scale,
            **rope,
        )

    return call


def load_baseline(args):
    specification = getattr(args, "baseline_candidate", None)
    if specification is None:
        return functools.partial(native, implementation=args.native_impl)
    return load_candidate(specification, args.block_i[0] if args.block_i else None, args.packed_gather)


def candidate_label(specification, block_i, packed_gather):
    suffix = "/packed-gather" if packed_gather else ""
    tiled = f"{specification}[block_i={block_i}]" if block_i is not None else specification
    return tiled + suffix


def output_tensor(result, inputs):
    output = result[0] if isinstance(result, (tuple, list)) else result
    expected_shape = (*inputs.query.shape[:-1], LATENT_DIM)
    if output.shape != expected_shape or output.dtype != torch.bfloat16 or output.device != inputs.query.device:
        raise ValueError(f"Invalid output metadata: {tuple(output.shape)}, {output.dtype}, {output.device}")
    return output


def check(actual, expected, atol, rtol):
    actual = actual.float()
    expected = expected.float()
    if not actual.isfinite().all() or not expected.isfinite().all():
        raise ValueError("Non-finite output in accuracy check")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    diff = actual - expected
    return {"max_abs": float(diff.abs().max()), "rmse": float(diff.square().mean().sqrt())}


class Graph:
    def __init__(self, function, inputs, scale, warmup, forwards):
        self.forwards = forwards
        self.graph = torch.npu.NPUGraph()
        self.start = torch.npu.Event(enable_timing=True)
        self.end = torch.npu.Event(enable_timing=True)
        for _ in range(warmup):
            function(inputs, scale)
        torch.npu.synchronize()
        with torch.npu.graph(self.graph):
            for _ in range(forwards):
                self.output = output_tensor(function(inputs, scale), inputs)
        for _ in range(warmup):
            self.graph.replay()
        torch.npu.synchronize()

    def time_us(self):
        self.start.record()
        self.graph.replay()
        self.end.record()
        self.end.synchronize()
        elapsed = self.start.elapsed_time(self.end) * 1000 / self.forwards
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Invalid NPU event time: {elapsed} us")
        return elapsed


def paired_timings(native_graph, candidate_graph, samples):
    measurements = []
    for sample in range(samples):
        switched = sample % 2 == 1
        graphs = (candidate_graph, native_graph) if switched else (native_graph, candidate_graph)
        times = [graph.time_us() for graph in (*graphs, *reversed(graphs))]
        outer = (times[0] + times[3]) / 2
        inner = (times[1] + times[2]) / 2
        baseline, candidate = (inner, outer) if switched else (outer, inner)
        measurements.append(
            {
                "order": "BAAB" if switched else "ABBA",
                "replays_us": times,
                "native_us": baseline,
                "candidate_us": candidate,
                "speedup": baseline / candidate,
            }
        )
    return measurements


def summarize(measurements, seed):
    speedups = [pair["speedup"] for pair in measurements]
    rng = random.Random(seed)
    boot = sorted(statistics.median(rng.choices(speedups, k=len(speedups))) for _ in range(2000))
    return {
        "native_us": statistics.median(pair["native_us"] for pair in measurements),
        "candidate_us": statistics.median(pair["candidate_us"] for pair in measurements),
        "paired_speedup": statistics.median(speedups),
        "speedup_ci95_low": boot[50],
        "speedup_ci95_high": boot[1949],
    }


def arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        default="qsfa_native_reference:qsfa_glm54_mtp5_blocked_kv",
        help="module:function implementing QSFA",
    )
    parser.add_argument(
        "--baseline-candidate", help="module:function to use as the paired timing and accuracy baseline"
    )
    parser.add_argument("--block-i", type=int, action="append", help="candidate tile per run; repeat to tune variants")
    parser.add_argument("--packed-gather", action="store_true", help="enable the candidate's fused packed gather")
    parser.add_argument("--batch", type=int, nargs="+", default=list(range(1, 81)))
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--topk", type=int, default=2048, help="sparse token budget before any KPool tail")
    parser.add_argument("--rope-dim", type=int, choices=(0, 64), default=64, help="GLM-5.3 uses 0, GLM-5.4 uses 64")
    parser.add_argument(
        "--index-pattern", choices=("kpool", "token"), help="defaults to kpool for 0 RoPE, token for 64"
    )
    parser.add_argument(
        "--native-impl",
        choices=("vllm-ascend", "torch-npu"),
        help="defaults to the production dispatch for the RoPE dimension",
    )
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--forwards", type=int, default=9)
    parser.add_argument("--samples", type=int, default=23)
    parser.add_argument("--tuning-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=93211)
    parser.add_argument("--atol", type=float, default=0.015)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument(
        "--peak-bf16-tflops", type=float, default=PEAK_BF16_TFLOPS, help="dense BF16 TFLOP/s per single A3 die"
    )
    parser.add_argument("--peak-hbm-gb-s", type=float, default=PEAK_HBM_GB_S, help="decimal GB/s per single A3 die")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, default=Path("qsfa-results"))
    parser.add_argument("--cpu-fixture", action="store_true", help="validate small synthetic fixtures without NPU")
    parser.add_argument("--native-only", action="store_true", help="validate and time only native QSFA")
    parser.add_argument("--accuracy-only", action="store_true", help="run eager checks and skip graph capture/timing")
    args = parser.parse_args()
    if args.index_pattern is None:
        args.index_pattern = "token" if args.rope_dim else "kpool"
    if args.native_impl is None:
        args.native_impl = "torch-npu" if args.rope_dim else "vllm-ascend"
    if min(args.batch) < 1 or len(set(args.batch)) != len(args.batch):
        parser.error("batch sizes must be distinct and positive")
    if args.context_length < QUERY_TOKENS or args.topk <= 0:
        parser.error("context must cover six query tokens and topk must be positive")
    if args.index_pattern == "kpool" and args.topk % POOL_SIZE:
        parser.error("topk must be a multiple of four for KPool")
    if min(args.heads, args.warmup, args.forwards, args.samples, args.tuning_samples) < 1:
        parser.error("heads, warmup, forwards, samples and tuning-samples must be positive")
    if any(not math.isfinite(peak) or peak <= 0 for peak in (args.peak_bf16_tflops, args.peak_hbm_gb_s)):
        parser.error("hardware peaks must be finite and positive")
    if args.cpu_fixture and math.ceil(args.context_length / PAGE_SIZE) * max(args.batch) > 1024:
        parser.error("provide a smaller --context-length and --batch for the CPU fixture")
    return args


def initialize_npu(args):
    import torch_npu

    if getattr(args, "baseline_candidate", None) is None:
        if args.native_impl == "vllm-ascend":
            from vllm_ascend.utils import enable_custom_op

            if not enable_custom_op():
                raise RuntimeError("vLLM Ascend custom operators could not be loaded")
        elif not hasattr(torch_npu, "npu_kv_quant_sparse_flash_attention"):
            raise RuntimeError("This torch_npu build does not expose npu_kv_quant_sparse_flash_attention")
    torch.npu.set_device(args.device)
    return {"device": torch.npu.get_device_name(), "torch": torch.__version__, "torch_npu": torch_npu.__version__}


@torch.inference_mode()
def run(args):
    device = "cpu" if args.cpu_fixture else args.device
    metadata = {"torch": torch.__version__} if args.cpu_fixture else initialize_npu(args)
    schedules = []
    if not (args.cpu_fixture or args.native_only):
        schedules = [
            (
                candidate_label(args.candidate, block_i, args.packed_gather),
                load_candidate(args.candidate, block_i, args.packed_gather),
            )
            for block_i in args.block_i or [None]
        ]
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {key: value for key, value in vars(args).items() if key not in {"output", "device"}}
    settings.update(metadata)
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    max_pages = math.ceil(args.context_length / PAGE_SIZE) * max(args.batch)
    storage = prepare_storage(max_pages, device, args.seed, args.rope_dim)
    native_call = load_baseline(args) if not args.cpu_fixture else None
    columns = (
        "batch",
        "tokens",
        "selected",
        *WORKLOAD_COLUMNS,
        "native_us",
        "candidate_us",
        "paired_speedup",
        "speedup_ci95_low",
        "speedup_ci95_high",
        *UTILIZATION_COLUMNS,
        "native_golden_max_abs",
        "candidate_golden_max_abs",
        "candidate_native_max_abs",
    )
    with (
        (args.output / "graph.csv").open("w", encoding="utf-8", newline="") as csv_file,
        (args.output / "samples.jsonl").open("w", encoding="utf-8") as samples_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for batch in args.batch:
            inputs, cpu_metadata = make_inputs(
                batch, args.context_length, args.heads, args.topk, args.seed, storage, args.index_pattern
            )
            rows, golden = selected_golden(inputs, cpu_metadata, batch, args.scale)
            record = {"batch": batch, "tokens": batch * QUERY_TOKENS}
            record.update(workload_counts(inputs, cpu_metadata, args.topk, args.index_pattern))
            details = {"batch": batch, "golden_rows": [rows.start, rows.stop]}
            if args.cpu_fixture:
                check(golden, golden, 0.0, 0.0)
                details["fixture"] = "valid"
            else:
                benchmark_batch(args, inputs, rows, golden, native_call, schedules, record, details)
                add_utilization(record, args.peak_bf16_tflops, args.peak_hbm_gb_s)
            writer.writerow(record)
            samples_file.write(json.dumps(details) + "\n")
            csv_file.flush()
            samples_file.flush()
            print(json.dumps(record), flush=True)


def benchmark_batch(args, inputs, rows, golden, native_call, schedules, record, details):
    base = output_tensor(native_call(inputs, args.scale), inputs).cpu()
    base_accuracy = check(base[rows], golden, args.atol, args.rtol)
    record["native_golden_max_abs"] = base_accuracy["max_abs"]
    details["native_golden"] = base_accuracy
    baseline = None
    if not args.accuracy_only:
        baseline = Graph(native_call, inputs, args.scale, args.warmup, args.forwards)
        details["native_graph_eager"] = check(baseline.output.cpu(), base, args.atol, args.rtol)
    if not schedules:
        if baseline is not None:
            measurements = [baseline.time_us() for _ in range(args.samples)]
            details["native_samples_us"] = measurements
            record["native_us"] = statistics.median(measurements)
        return

    candidates = []
    details["candidates"] = {}
    for name, function in schedules:
        eager = output_tensor(function(inputs, args.scale), inputs).cpu()
        accuracy = {
            "golden": check(eager[rows], golden, args.atol, args.rtol),
            "native": check(eager, base, args.atol, args.rtol),
        }
        results = {"accuracy": accuracy}
        details["candidates"][name] = results
        if baseline is not None:
            runner = Graph(function, inputs, args.scale, args.warmup, args.forwards)
            accuracy["graph_eager"] = check(runner.output.cpu(), eager, args.atol, args.rtol)
            if len(schedules) > 1:
                tuning = paired_timings(baseline, runner, args.tuning_samples)
                results["tuning"] = tuning
                tune_speedup = statistics.median(item["speedup"] for item in tuning)
            else:
                tune_speedup = 1.0
            candidates.append((tune_speedup, name, runner))
    if candidates:
        _, selected, winner = max(candidates, key=lambda item: item[0])
        holdout = paired_timings(baseline, winner, args.samples)
        details["holdout"] = holdout
        record.update(summarize(holdout, args.seed + record["batch"]))
    else:
        selected = schedules[0][0]
    selected_accuracy = details["candidates"][selected]["accuracy"]
    record.update(
        selected=selected,
        candidate_golden_max_abs=selected_accuracy["golden"]["max_abs"],
        candidate_native_max_abs=selected_accuracy["native"]["max_abs"],
    )


if __name__ == "__main__":
    run(arguments())
