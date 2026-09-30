# SPDX-License-Identifier: Apache-2.0
"""Compare separately installed native SFA builds; does not measure model TPOT.

Run each build in a fresh process, with identical cases and an idle device.
Device samples use an NPU graph; wall samples include Python/operator dispatch
and synchronization amortized over 25 calls. Both preserve every observation.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

INPUT_SEED = 930
HEAD_DIM = 512
PAGE_SIZE = 128
SPARSE_CAPACITY = 2048
SCALE_VALUE = 1 / 24
EAGER_WARMUP_CALLS = 5
GRAPH_CALLS = 8
GRAPH_WARMUP_REPLAYS = 10
GRAPH_TIMED_REPLAYS = 100
EAGER_BATCH_CALLS = 25
SAMPLES_PER_PROCESS = 5


def make_inputs(case):
    torch.manual_seed(INPUT_SEED)
    values = {x["name"]: x.get("value", x.get("shape")) for x in case["inputs"]}
    dtype = getattr(torch, case["inputs"][0]["dtype"])
    tokens, heads, dim = values["query"]
    capacity, rope = values["kv_capacity"], values["rope_dim"]
    assert values["query_lengths"] == [tokens] and values["kv_lengths"] == [capacity]
    assert dim == HEAD_DIM and capacity % PAGE_SIZE == 0
    query = torch.randn(tokens, heads, dim, dtype=dtype)
    key = torch.randn(1, capacity, dim, dtype=dtype)
    query_rope = torch.randn(tokens, heads, rope, dtype=dtype) if rope else None
    key_rope = torch.randn(1, capacity, rope, dtype=dtype) if rope else None
    order = torch.randperm(capacity // PAGE_SIZE)
    pages = torch.empty(capacity // PAGE_SIZE, PAGE_SIZE, 1, dim, dtype=dtype)
    pages[order] = key.reshape(-1, PAGE_SIZE, 1, dim)
    rope_pages = None
    if rope:
        rope_pages = torch.empty(*pages.shape[:-1], rope, dtype=dtype)
        rope_pages[order] = key_rope.reshape(-1, PAGE_SIZE, 1, rope)
    indices = torch.full((tokens, 1, SPARSE_CAPACITY), -1, dtype=torch.int32)
    for token, count in enumerate(values["selected_counts"]):
        assert 0 <= count <= min(capacity, SPARSE_CAPACITY)
        indices[token, 0, :count] = torch.randperm(capacity)[:count].sort().values.int()
    inputs = dict(
        query=query.npu(),
        key=pages.npu(),
        sparse_indices=indices.npu(),
        block_table=order.reshape(1, -1).int().npu(),
        actual_seq_lengths_query=torch.tensor([tokens], dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor([capacity], dtype=torch.int32).npu(),
        query_rope=query_rope.npu() if rope else None,
        key_rope=rope_pages.npu() if rope else None,
    )
    inputs["value"] = inputs["key"]
    return inputs


def run(inputs):
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        **inputs,
        scale_value=SCALE_VALUE,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )


def input_hash(inputs):
    digest = hashlib.sha256()
    for name, value in sorted(inputs.items()):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            cpu = value.cpu().contiguous()
            digest.update(str((cpu.dtype, tuple(cpu.shape))).encode())
            digest.update(cpu.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(str(value).encode())
    return digest.hexdigest()


def measure(inputs):
    for _ in range(EAGER_WARMUP_CALLS):
        run(inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [run(inputs) for _ in range(GRAPH_CALLS)]
    for _ in range(GRAPH_WARMUP_REPLAYS):
        graph.replay()
    torch.npu.synchronize()
    device_samples, wall_samples = [], []
    for _ in range(SAMPLES_PER_PROCESS):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(GRAPH_TIMED_REPLAYS):
            graph.replay()
        end.record()
        end.synchronize()
        device_samples.append(start.elapsed_time(end) * 1000 / (GRAPH_CALLS * GRAPH_TIMED_REPLAYS))
        begin = time.perf_counter_ns()
        for _ in range(EAGER_BATCH_CALLS):
            run(inputs)
        torch.npu.synchronize()
        wall_samples.append((time.perf_counter_ns() - begin) / 1000 / EAGER_BATCH_CALLS)
    del graph, outputs
    return device_samples, wall_samples


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).with_name("sparse_flash_attention_kv_padding.jsonl")
    )
    parser.add_argument("--label", required=True, help="Source/build revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Output .pt from an independent baseline build")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        parser.error("Use a new output path to preserve previous attempts")
    assert enable_custom_op()
    cases_hash = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    baseline = torch.load(args.reference, weights_only=True) if args.reference else None
    if baseline is not None and baseline["cases_sha256"] != cases_hash:
        parser.error("Baseline case definitions differ")
    results = dict(
        label=args.label,
        cases_sha256=cases_hash,
        device=torch.npu.get_device_name(0),
        torch_version=torch.__version__,
        torch_npu_version=torch_npu.__version__,
        complete=False,
        rows=[],
    )
    saved = dict(cases_sha256=cases_hash, rows=[])
    for index, line in enumerate(args.cases.read_text().splitlines()):
        case = json.loads(line)
        inputs = make_inputs(case)
        digest = input_hash(inputs)
        actual = [x.cpu() for x in run(inputs)]
        if baseline is not None:
            old = baseline["rows"][index]
            assert old["input_sha256"] == digest, "Inputs differ"
            for x, y in zip(actual, old["outputs"], strict=True):
                assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)), index
        device, wall = measure(inputs)
        row = dict(index=index, case=case, input_sha256=digest, graph_device_us=device, eager_batch_wall_us=wall)
        results["rows"].append(row)
        saved["rows"].append(dict(input_sha256=digest, outputs=actual))
        print(json.dumps(row), flush=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    results["complete"] = True
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    torch.save(saved, args.output.with_suffix(".pt"))


if __name__ == "__main__":
    main()
