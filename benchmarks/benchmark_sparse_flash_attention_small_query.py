# SPDX-License-Identifier: Apache-2.0
"""Replay frozen Indexer inputs for NoPE SFA correctness, latency and memory.

Run capture once with the full custom-op package, then replay the same files
with each build. Random features are synthetic; this is not a model benchmark.
"""

import argparse
import gc
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

HEAD_DIM = 512
QUERY_HEADS = 64
INDEXER_HEADS = 32
INDEXER_DIM = 128
KV_LENGTH = 4096
PAGE_SIZE = 128
SPARSE_COUNT = 2048
GRAPH_CALLS = 8
WARMUP_CALLS = 5
WARMUP_REPLAYS = 10


def capture(dtype, query_count, quantized):
    torch.manual_seed(1001 + query_count)
    pages = KV_LENGTH // PAGE_SIZE
    inputs = {
        "query": torch.randn(query_count, QUERY_HEADS, HEAD_DIM, dtype=dtype),
        "key": torch.randn(pages, PAGE_SIZE, 1, HEAD_DIM, dtype=dtype),
        "block_table": torch.randperm(pages).int().unsqueeze(0),
        "actual_seq_lengths_query": torch.tensor([query_count], dtype=torch.int32),
        "actual_seq_lengths_kv": torch.tensor([KV_LENGTH], dtype=torch.int32),
    }
    metadata = {name: inputs[name].npu() for name in inputs if name not in ("query", "key")}
    options = dict(
        actual_seq_lengths_query=metadata["actual_seq_lengths_query"],
        actual_seq_lengths_key=metadata["actual_seq_lengths_kv"],
        block_table=metadata["block_table"],
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=SPARSE_COUNT,
        sparse_mode=3,
    )
    if quantized:
        indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
            query=torch.randint(-32, 32, (query_count, INDEXER_HEADS, INDEXER_DIM), dtype=torch.int8).npu(),
            key=torch.randint(-32, 32, (pages, PAGE_SIZE, 1, INDEXER_DIM), dtype=torch.int8).npu(),
            weights=torch.rand(query_count, INDEXER_HEADS, dtype=torch.float16).npu(),
            query_dequant_scale=(torch.rand(query_count, INDEXER_HEADS, dtype=torch.float16) / 32).npu(),
            key_dequant_scale=(torch.rand(pages, PAGE_SIZE, 1, dtype=torch.float16) / 32).npu(),
            query_quant_mode=0,
            key_quant_mode=0,
            **options,
        )
    else:
        indices, _ = torch_npu.npu_lightning_indexer(
            query=torch.randn(query_count, INDEXER_HEADS, INDEXER_DIM, dtype=dtype).npu(),
            key=torch.randn(pages, PAGE_SIZE, 1, INDEXER_DIM, dtype=dtype).npu(),
            weights=torch.randn(query_count, INDEXER_HEADS, dtype=dtype).npu(),
            **options,
        )
    inputs["sparse_indices"] = indices.cpu().int().reshape(query_count, 1, SPARSE_COUNT)
    for row, selected in enumerate(inputs["sparse_indices"][:, 0]):
        assert selected.min() >= 0 and selected.max() < KV_LENGTH - query_count + row + 1
        assert selected.unique().numel() == SPARSE_COUNT
    return inputs


def reference(inputs):
    key = inputs["key"][inputs["block_table"][0].long(), :, 0].reshape(-1, HEAD_DIM).double()
    expected = torch.empty_like(inputs["query"], dtype=torch.float64)
    expected_lse = torch.empty(inputs["query"].shape[:2], dtype=torch.float64)
    for row, indices in enumerate(inputs["sparse_indices"][:, 0]):
        selected = indices[(indices >= 0) & (indices < KV_LENGTH - len(expected) + row + 1)].long()
        scores = inputs["query"][row].double() @ key[selected].T / math.sqrt(HEAD_DIM)
        expected[row] = scores.softmax(-1) @ key[selected]
        expected_lse[row] = scores.logsumexp(-1)
    return expected, expected_lse


def run(inputs, return_lse):
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        **inputs,
        value=inputs["key"],
        query_rope=None,
        key_rope=None,
        scale_value=1 / math.sqrt(HEAD_DIM),
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=return_lse,
    )


def check_outputs(outputs, expected, expected_lse, return_lse):
    for output in outputs:
        torch.testing.assert_close(output[0].cpu().double(), expected, atol=0.03, rtol=0.01)
        if return_lse:
            lse = (output[1].cpu().double() + output[2].cpu().double().log()).squeeze(0)
            torch.testing.assert_close(lse, expected_lse, atol=0.005, rtol=0.001)


def measure(inputs, return_lse, samples, replays, expected, expected_lse):
    for _ in range(WARMUP_CALLS):
        run(inputs, return_lse)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [run(inputs, return_lse) for _ in range(GRAPH_CALLS)]
    for _ in range(WARMUP_REPLAYS):
        graph.replay()
    torch.npu.synchronize()
    values = []
    for _ in range(samples):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (replays * GRAPH_CALLS))
    check_outputs(outputs, expected, expected_lse, return_lse)
    del graph, outputs
    return dict(graph_device_us=values, median_device_us=statistics.median(values))


def memory(inputs, return_lse, expected, expected_lse):
    observations = {}
    for label, graph_mode in [("eager", False), ("graph", True)]:
        torch.npu.synchronize()
        before = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        if graph_mode:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = [run(inputs, return_lse) for _ in range(GRAPH_CALLS)]
            graph.replay()
        else:
            outputs = [run(inputs, return_lse)]
        torch.npu.synchronize()
        observations[label] = dict(
            live_before_bytes=before,
            live_after_bytes=torch.npu.memory_allocated(),
            peak_allocated_bytes=torch.npu.max_memory_allocated(),
            peak_reserved_bytes=torch.npu.max_memory_reserved(),
            peak_increase_bytes=torch.npu.max_memory_allocated() - before,
        )
        check_outputs(outputs, expected, expected_lse, return_lse)
        if graph_mode:
            del graph
        del outputs
        gc.collect()
        torch.npu.synchronize()
    return dict(allocator_observations=observations)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["capture", "measure", "memory"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 20, 21, 24, 25, 32])
    parser.add_argument("--return-lse", action="store_true", help="Diagnostic; production NoPE uses LSE disabled.")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path to preserve previous observations.")
    if args.samples < 1 or args.replays < 1 or len(set(args.query_counts)) != len(args.query_counts):
        parser.error("Samples/replays must be positive and query counts unique.")
    if any(count < 1 or count > KV_LENGTH - SPARSE_COUNT + 1 for count in args.query_counts):
        parser.error("Query counts must leave at least 2048 causally visible KV tokens.")
    if args.phase == "capture":
        if args.inputs.exists():
            parser.error("Capture requires a new input directory; replay it unchanged for both builds.")
        args.inputs.mkdir(parents=True)
    elif not args.inputs.is_dir():
        parser.error("Replay requires the frozen input directory from capture.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert enable_custom_op()
    rows = []
    report = dict(
        phase=args.phase,
        device=torch.npu.get_device_name(0),
        torch_version=torch.__version__,
        torch_npu_version=torch_npu.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope=(
            "Unquantized NoPE SFA on synthetic features and actual native/quantized Indexer outputs. "
            "Memory is Torch allocator usage, not device-wide HBM. No model speedup claim."
        ),
        samples=args.samples,
        replays=args.replays,
        graph_calls=GRAPH_CALLS,
        return_lse=args.return_lse,
        rows=rows,
    )
    for kind in ["native", "quant"]:
        for dtype in [torch.float16, torch.bfloat16]:
            for count in args.query_counts:
                label = f"{kind}-{dtype}-{count}"
                path = args.inputs / f"{label}-inputs.pt"
                row = dict(case=label)
                try:
                    if args.phase == "capture":
                        torch.save(capture(dtype, count, kind == "quant"), path)
                    cpu = torch.load(path, weights_only=True)
                    row["capture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    expected, expected_lse = reference(cpu)
                    inputs = {name: tensor.npu() for name, tensor in cpu.items()}
                    actual = run(inputs, args.return_lse)
                    check_outputs([actual], expected, expected_lse, args.return_lse)
                    del actual
                    torch.npu.synchronize()
                    if args.phase == "measure":
                        row.update(measure(inputs, args.return_lse, args.samples, args.replays, expected, expected_lse))
                    elif args.phase == "memory":
                        row.update(memory(inputs, args.return_lse, expected, expected_lse))
                    del inputs
                    gc.collect()
                    torch.npu.synchronize()
                    row["status"] = "passed"
                except Exception as exc:
                    row.update(status="failed", error=repr(exc))
                rows.append(row)
                args.output.write_text(json.dumps(dict(report, complete=False), indent=2) + "\n")
                print(json.dumps(row), flush=True)
    report.update(complete=True, all_passed=all(row["status"] == "passed" for row in rows))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
