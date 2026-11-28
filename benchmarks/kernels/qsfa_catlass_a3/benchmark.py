# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare A3 CATLASS QSFA with the stock CANN operator for GLM-5.4 MTP5 B10."""

import argparse
import json
import math
import random
import statistics
from functools import partial
from pathlib import Path
from typing import NamedTuple

import torch

BATCH = 10
QUERY_TOKENS = 6
ROWS = BATCH * QUERY_TOKENS
CONTEXT = 131072
HEADS = 64
LATENT_DIM = 512
ROPE_DIM = 64
TOPK = 2048
PAGE_SIZE = 128
PAGES = CONTEXT // PAGE_SIZE
QUANT_TILE = 128
SCALE_OFFSET = LATENT_DIM + 2 * ROPE_DIM
PACKED_DIM = SCALE_OFFSET + 4 * (LATENT_DIM // QUANT_TILE)
SOFTMAX_SCALE = 0.0625
PEAK_BF16_TFLOPS = 280.0
PEAK_HBM_GB_S = 1600.0


class Inputs(NamedTuple):
    query: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor
    page_table: torch.Tensor
    query_ends: torch.Tensor
    kv_lengths: torch.Tensor


def packed_pages(pages, generator):
    prefix = (pages, PAGE_SIZE, 1)
    data = torch.randint(-127, 128, (*prefix, LATENT_DIM), dtype=torch.int8, generator=generator)
    scales = torch.rand((*prefix, LATENT_DIM // QUANT_TILE), generator=generator) * 0.02 + 0.005
    rope = torch.randn((*prefix, ROPE_DIM), generator=generator, dtype=torch.bfloat16)
    return torch.cat((data, rope.view(torch.int8), scales.contiguous().view(torch.int8)), dim=-1)


def make_storage(device, seed):
    generator = torch.Generator().manual_seed(seed)
    storage = torch.empty((BATCH * PAGES, PAGE_SIZE, 1, PACKED_DIM), dtype=torch.int8, device=device)
    # Preserve the CPU generator's draw order by packing 128 physical pages at a time.
    for start in range(0, BATCH * PAGES, 128):
        storage[start : start + 128].copy_(packed_pages(128, generator))
    return storage


def make_inputs(storage, seed):
    generator = torch.Generator().manual_seed(seed + BATCH)
    rng = random.Random(seed + BATCH)
    latent = (torch.randn((ROWS, HEADS, LATENT_DIM), generator=generator) * 0.5).to(torch.bfloat16)
    page_table = torch.randperm(BATCH * PAGES, dtype=torch.int32, generator=generator).reshape(BATCH, PAGES)
    rope = (torch.randn((ROWS, HEADS, ROPE_DIM), generator=generator) * 0.5).to(torch.bfloat16)
    query = torch.cat((latent, rope), dim=-1)
    query_ends = torch.arange(1, BATCH + 1, dtype=torch.int32) * QUERY_TOKENS
    kv_lengths = CONTEXT - torch.arange(BATCH, dtype=torch.int32)
    indices = torch.empty((ROWS, 1, TOPK), dtype=torch.int32)
    for request, length in enumerate(kv_lengths.tolist()):
        first_visible = length - QUERY_TOKENS + 1
        shared = rng.sample(range(first_visible), TOPK // 2)
        shared_set = set(shared)
        for token in range(QUERY_TOKENS):
            visible = first_visible + token
            sampled = [index for index in rng.sample(range(visible), TOPK) if index not in shared_set]
            row_indices = shared + sampled[: TOPK - len(shared)]
            rng.shuffle(row_indices)
            indices[request * QUERY_TOKENS + token, 0] = torch.tensor(row_indices, dtype=torch.int32)
    return Inputs(query, storage, indices, page_table, query_ends, kv_lengths)


def causal_limits(kv_lengths):
    first_visible = kv_lengths - QUERY_TOKENS + 1
    return (first_visible[:, None] + torch.arange(QUERY_TOKENS)).flatten()


def check_fixture(inputs):
    page_ids = inputs.page_table.flatten().sort().values
    if not torch.equal(page_ids, torch.arange(BATCH * PAGES, dtype=torch.int32)):
        raise ValueError("The page table must assign each physical page exactly once")
    indices = inputs.indices[:, 0]
    limits = causal_limits(inputs.kv_lengths)[:, None]
    if not torch.all((indices >= 0) & (indices < limits)):
        raise ValueError("Sparse indices must be within the causal window")
    sorted_indices = indices.sort(dim=-1).values
    if not torch.all(sorted_indices[:, 1:] > sorted_indices[:, :-1]):
        raise ValueError("Each query must have distinct sparse indices")


def boundary_inputs(inputs):
    indices = inputs.indices.clone()
    limits = causal_limits(inputs.kv_lengths)[:, None, None]
    # Drop one index per 256-token AIV lane. Both kernels take a valid prefix with -1 padding.
    indices[..., ::256] = limits
    valid = (indices >= 0) & (indices < limits)
    removed = TOPK // 256
    indices[..., : TOPK - removed] = indices[valid].reshape(ROWS, 1, TOPK - removed)
    indices[..., TOPK - removed :] = -1
    return inputs._replace(indices=indices), removed


def selected_golden(inputs):
    """Decode packed KV independently and compute FP32 attention for one request."""
    request = BATCH // 2
    start = request * QUERY_TOKENS
    stop = int(inputs.query_ends[request])
    rows = slice(start, stop)
    query = inputs.query[rows].float()
    output = torch.zeros_like(query[..., :LATENT_DIM])
    packed_rows = inputs.kv.view(-1, PACKED_DIM)
    for local_row, row in enumerate(range(start, stop)):
        limit = int(inputs.kv_lengths[request]) - (stop - row - 1)
        indices = inputs.indices[row, 0]
        tokens = indices[(indices >= 0) & (indices < limit)].long()
        physical_rows = inputs.page_table[request, tokens // PAGE_SIZE].long() * PAGE_SIZE + tokens % PAGE_SIZE
        packed = packed_rows.index_select(0, physical_rows.to(inputs.kv.device)).cpu()
        scales = packed[:, SCALE_OFFSET:].contiguous().view(torch.float32).repeat_interleave(QUANT_TILE, dim=-1)
        latent = (packed[:, :LATENT_DIM].float() * scales).to(torch.bfloat16).float()
        rope = packed[:, LATENT_DIM:SCALE_OFFSET].contiguous().view(torch.bfloat16).float()
        logits = query[local_row, :, :LATENT_DIM] @ latent.T
        logits += query[local_row, :, LATENT_DIM:] @ rope.T
        output[local_row] = (logits * SOFTMAX_SCALE).softmax(dim=-1) @ latent
    return rows, output


def stock_qsfa(operation, inputs):
    return operation(
        query=inputs.query,
        key=inputs.kv,
        value=inputs.kv,
        sparse_indices=inputs.indices,
        scale_value=SOFTMAX_SCALE,
        sparse_block_size=1,
        block_table=inputs.page_table,
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
        rope_head_dim=ROPE_DIM,
    )


def output_tensor(result, device):
    output = result[0] if isinstance(result, (tuple, list)) else result
    if output.shape != (ROWS, HEADS, LATENT_DIM) or output.dtype != torch.bfloat16 or output.device != device:
        raise ValueError(f"Unexpected QSFA output: {output.shape}, {output.dtype}, {output.device}")
    return output


def check(actual, expected, *, exact=False):
    actual, expected = actual.float(), expected.float()
    if not actual.isfinite().all() or not expected.isfinite().all():
        raise ValueError("Non-finite output in accuracy check")
    atol, rtol = (0.0, 0.0) if exact else (0.015, 0.02)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return float((actual - expected).abs().max())


class Graph:
    def __init__(self, call, warmup, forwards):
        self.forwards = forwards
        self.graph = torch.npu.NPUGraph()
        self.start = torch.npu.Event(enable_timing=True)
        self.end = torch.npu.Event(enable_timing=True)
        for _ in range(warmup):
            call()
        torch.npu.synchronize()
        with torch.npu.graph(self.graph):
            for _ in range(forwards):
                self.output = call()
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


def paired_timings(stock, catlass, number):
    pairs = []
    for sample in range(number):
        switched = sample % 2 == 1
        outer, inner = (catlass, stock) if switched else (stock, catlass)
        times = [graph.time_us() for graph in (outer, inner, inner, outer)]
        outer_us, inner_us = (times[0] + times[3]) / 2, (times[1] + times[2]) / 2
        stock_us, catlass_us = (inner_us, outer_us) if switched else (outer_us, inner_us)
        pairs.append(
            {
                "order": "BAAB" if switched else "ABBA",
                "replays_us": times,
                "stock_us": stock_us,
                "catlass_us": catlass_us,
                "speedup": stock_us / catlass_us,
            }
        )
    return pairs


def summarize(pairs, seed, inputs):
    selected_pairs = ROWS * TOPK
    useful_flops = 2 * selected_pairs * HEADS * ((LATENT_DIM + ROPE_DIM) + LATENT_DIM)
    logical_bytes = selected_pairs * PACKED_DIM + ROWS * HEADS * LATENT_DIM * inputs.query.element_size()
    logical_bytes += sum(tensor.numel() * tensor.element_size() for tensor in (inputs.query, *inputs[2:]))
    speedups = [pair["speedup"] for pair in pairs]
    rng = random.Random(seed + BATCH)
    boot = sorted(statistics.median(rng.choices(speedups, k=len(pairs))) for _ in range(2000))
    metrics = {
        "useful_flops_per_forward": useful_flops,
        "logical_io_bytes_per_forward": logical_bytes,
        "paired_speedup": statistics.median(speedups),
        "paired_speedup_ci95_low": boot[50],
        "paired_speedup_ci95_high": boot[1949],
    }
    for name in ("stock", "catlass"):
        times = [pair[f"{name}_us"] for pair in pairs]
        p50 = statistics.median(times)
        metrics[f"{name}_p50_us"] = p50
        metrics[f"{name}_p95_us"] = statistics.quantiles(times, n=100, method="inclusive")[94]
        metrics[f"{name}_useful_mfu_pct"] = 100 * useful_flops / (p50 * PEAK_BF16_TFLOPS * 1e6)
        metrics[f"{name}_logical_mbu_pct"] = 100 * logical_bytes / (p50 * PEAK_HBM_GB_S * 1e3)
    return metrics


@torch.inference_mode()
def run(args):
    import torch_npu
    from catlass_qsfa import qsfa_a3

    torch.npu.set_device(args.device)
    cpu_inputs = make_inputs(make_storage(args.device, args.seed), args.seed)
    check_fixture(cpu_inputs)
    metrics = {
        "batch": BATCH,
        "tokens": ROWS,
        "seed": args.seed,
        "stock": "torch_npu.npu_kv_quant_sparse_flash_attention",
        "candidate": "catlass_cpp_resident_kv_wide_384",
    }
    if args.causal_boundary:
        cpu_inputs, removed = boundary_inputs(cpu_inputs)
        metrics["boundary_filtered_indices_per_row"] = removed
    golden_rows, golden = selected_golden(cpu_inputs)
    inputs = Inputs(*(tensor.to(args.device) for tensor in cpu_inputs))
    stock = partial(stock_qsfa, torch_npu.npu_kv_quant_sparse_flash_attention, inputs)
    candidate = qsfa_a3 if args.catlass_library is None else partial(qsfa_a3, library_path=args.catlass_library)
    device = inputs.query.device

    def stock_call():
        return output_tensor(stock(), device)

    def candidate_call():
        return output_tensor(candidate(*inputs, SOFTMAX_SCALE), device)

    stock_eager = stock_call().cpu()
    metrics["stock_golden_max_abs"] = check(stock_eager[golden_rows], golden)
    if not args.causal_boundary:
        stock_graph = Graph(stock_call, args.warmup, args.forwards)
        metrics["stock_graph_eager_max_abs"] = check(stock_graph.output.cpu(), stock_eager, exact=True)

    candidate_eager = candidate_call().cpu()
    metrics.update(
        catlass_golden_max_abs=check(candidate_eager[golden_rows], golden),
        catlass_stock_max_abs=check(candidate_eager, stock_eager, exact=True),
    )
    if args.causal_boundary:
        return metrics

    candidate_graph = Graph(candidate_call, args.warmup, args.forwards)
    metrics.update(
        catlass_graph_eager_max_abs=check(candidate_graph.output.cpu(), candidate_eager, exact=True),
        graph_forwards_per_replay=args.forwards,
        graph_warmups=args.warmup,
        graph_pairs=args.samples,
    )
    pairs = paired_timings(stock_graph, candidate_graph, args.samples)
    metrics.update(summarize(pairs, args.seed, inputs))
    metrics["pairs"] = pairs
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(93211, 93212), default=93211)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--catlass-library", type=Path, help="override the default CATLASS shared library")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--forwards", type=int, default=9)
    parser.add_argument("--samples", type=int, default=23)
    parser.add_argument("--causal-boundary", action="store_true", help="run the causal boundary accuracy check only")
    parser.add_argument("--output", type=Path, help="write JSON to this file instead of stdout")
    args = parser.parse_args()
    if args.warmup < 1 or args.forwards < 1 or args.samples < 2:
        parser.error("use at least one warmup, one forward and two samples")
    result = json.dumps(run(args), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(result, end="")
    else:
        args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
