# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-device DCP preprocessing/postprocessing versus equivalent torch ops.

The default matrix covers 4..512 tokens, DCP8, Kimi H=96/D=512/K=2048.
Collectives are excluded: their received buffers are constructed beforehand.
Every case passes eager and graph correctness before any timing starts. Raw
paired AB/BA samples include allocations and exclude compilation/capture.
"""

import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch_npu
from vllm.triton_utils import triton

from vllm_ascend.attention.context_parallel.sharded_indexer import (
    local_to_global,
    local_visible_lengths,
    merge_topk_candidates,
    query_visible_lengths,
)
from vllm_ascend.ops.triton import sharded_indexer as si
from vllm_ascend.ops.triton.dcp.dcp_a2a import fused_dcp_lse_combine, pack_dcp_output_lse
from vllm_ascend.ops.triton.dcp.dcp_a2a_batched import (
    _encode_dcp_lse_kernel,
)
from vllm_ascend.ops.triton.query_gather_prep import prep_query_head_major
from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


def torch_pack(ids, scores, world=8, rank=7, interleave=768):
    scores = scores.float()
    valid = (ids >= 0) & torch.isfinite(scores)
    keys = torch.where(valid, local_to_global(ids, world, rank, interleave), -1)
    values = torch.where(valid, scores, -torch.inf)
    return torch.stack((keys[:, 0], values[:, 0].view(torch.int32)), -1)


def torch_unpack(packed, rows):
    x = packed.view(8, rows, -1, 2).permute(1, 0, 2, 3).reshape(rows, -1, 2)
    ids = x[..., 0].to(torch.float32)
    return torch.where(ids >= 0, ids, torch.inf), x[..., 1].contiguous().view(torch.float32)


def torch_remap(ids, order, interleave=768):
    x = ids.float()
    block = torch.floor(x / interleave)
    owner = block - torch.floor(block / 8) * 8
    valid = (x >= 0) & (owner == 7)
    local = torch.floor(x / (8 * interleave)) * interleave + x - block * interleave
    local = torch.where(valid, local, -1).to(ids.dtype)
    _, permutation = torch.sort(order + (~valid).float() * ids.shape[-1], dim=-1)
    return torch.gather(local, -1, permutation.to(torch.int32))


def decode_lse(packed):
    code, hi, mid, lo = (packed[..., 512:516] if packed.shape[-1] > 4 else packed).float().unbind(-1)
    value = (hi * 65536 + mid * 256 + lo) * torch.exp2(code.abs() - 151)
    value = torch.where(code < 0, -value, value)
    return torch.where(code != 0, value, -torch.inf)


def torch_encode_lse(maximum, total):
    lse = maximum + total.log()
    finite, nonzero = torch.isfinite(lse), lse != 0
    magnitude = lse.abs()
    safe = torch.where(finite & nonzero, magnitude, 1)
    exponent = torch.floor(torch.log2(safe)).clamp(-126, 127)
    exponent = torch.where(nonzero, exponent, 0)
    significand = torch.where(finite & nonzero, magnitude * torch.exp2(23 - exponent), 0)
    hi = torch.floor(significand / 65536)
    remainder = significand - hi * 65536
    mid = torch.floor(remainder / 256)
    lo = remainder - mid * 256
    code = torch.where(lse < 0, -(exponent + 128), exponent + 128)
    code = torch.where(finite, code, 0)
    return torch.cat((code, hi, mid, lo), -1).bfloat16()


def torch_output_pack(output, maximum, total):
    packed = torch.cat((output, torch_encode_lse(maximum, total)), -1)
    return packed.view(output.shape[0], 8, 12, 516).permute(1, 2, 0, 3).contiguous()


def torch_combine(recv):
    lse = decode_lse(recv)
    weights = torch.nan_to_num(lse.softmax(0), nan=0.0)
    return (recv[..., :512].float() * weights[..., None]).sum(0).transpose(0, 1).to(recv.dtype)


def cases(rows):
    torch.manual_seed(20261027 + rows)
    topk, world, interleave = 2048, 8, 768
    ids = torch.randint(0, 131072, (rows, 1, topk), dtype=torch.int32, device="npu")
    scores = torch.randn(rows, 1, topk, device="npu").bfloat16().float()
    ids[..., -17:] = -1
    scores[..., :3] = torch.tensor([float("nan"), float("inf"), -float("inf")], device="npu")
    yield (
        "candidate_pack",
        lambda: si.pack_local_candidates(ids, scores, world, 7, interleave),
        lambda: torch_pack(ids, scores),
        "exact",
    )
    native_scores = scores.bfloat16()
    yield (
        "candidate_pack_bf16",
        lambda: si.pack_local_candidates(ids, native_scores, world, 7, interleave),
        lambda: torch_pack(ids, native_scores),
        "exact",
    )
    gathered = torch.cat([si.pack_local_candidates(ids, scores, world, rank, interleave) for rank in range(world)], 0)

    def unpack():
        keys = torch.empty((rows, world * topk), device="npu")
        values = torch.empty_like(keys)
        si._unpack_candidates[(rows, triton.cdiv(world * topk, 2048))](gathered, keys, values, rows, topk, world, 2048)
        return keys, values

    yield "candidate_unpack", unpack, lambda: torch_unpack(gathered, rows), "exact"
    keys, values = torch_unpack(gathered, rows)
    selected = keys[:, :topk].contiguous()

    def select():
        result = torch.empty((rows, 1, topk), dtype=torch.int32, device="npu")
        si._select_candidates[(rows, triton.cdiv(topk, 256))](selected, values, result, world * topk, topk, 256)
        return result

    yield (
        "candidate_select",
        select,
        lambda: torch.where(torch.isfinite(values[:, :topk]), selected, -1).int()[:, None],
        "exact",
    )
    global_ids = torch.where(torch.isfinite(keys), keys, -1).int()[:, None]
    yield (
        "candidate_merge",
        lambda: si.merge_packed_candidates(gathered, rows, world, topk),
        lambda: merge_topk_candidates(global_ids, values[:, None], topk),
        "exact",
    )
    yield (
        "candidate_merge_local",
        lambda: si.merge_packed_candidates(gathered, rows, world, topk, local_rank=7, interleave=interleave),
        lambda: remap_sparse_indices_triton(
            si.merge_packed_candidates(gathered, rows, world, topk), world, 7, interleave
        ),
        "exact",
    )
    packed_tiles = [
        gathered.view(world, rows, topk, 2)[:, start : start + 64].reshape(-1, topk, 2).contiguous()
        for start in range(0, rows, 64)
    ]

    def tiled_merge():
        out = torch.empty((rows, 1, topk), dtype=torch.int32, device="npu")
        for start, tile in zip(range(0, rows, 64), packed_tiles):
            si.merge_packed_candidates(
                tile,
                out[start : start + 64].shape[0],
                world,
                topk,
                local_rank=7,
                interleave=interleave,
                out=out[start : start + 64],
            )
        return out

    def native_merge_reference():
        merged = torch.cat(
            [
                merge_topk_candidates(global_ids[start : start + 64], values[start : start + 64, None], topk)
                for start in range(0, rows, 64)
            ],
            0,
        )
        return remap_sparse_indices_triton(merged, world, 7, interleave)

    yield "candidate_merge_tiled_vs_native", tiled_merge, native_merge_reference, "exact"
    query = torch.randn(rows, 12, 512, dtype=torch.bfloat16, device="npu")
    rope = query[..., :0]
    yield (
        "query_gather_prep",
        lambda: prep_query_head_major(query, rope),
        lambda: torch.cat((query, rope), -1).transpose(0, 1).contiguous(),
        "exact",
    )

    for requests in sorted({min(rows, 16), rows}):
        seq = torch.full((requests,), 1048576, dtype=torch.int32, device="npu")
        ends = torch.arange(1, requests + 1, dtype=torch.int32, device="npu") * rows // requests
        key_out = torch.empty_like(seq)
        visible_out = torch.empty((rows, 1), dtype=torch.int32, device="npu")
        row_ids = torch.arange(rows, dtype=torch.int32, device="npu")
        # Request IDs are precomputed metadata in the existing torch path.
        request_ids = torch.searchsorted(ends, row_ids, right=True).long()

        def metadata(seq=seq, ends=ends, key_out=key_out, visible_out=visible_out):
            si.build_local_lengths(seq, ends, key_out, visible_out, world, 7, interleave)
            return key_out, visible_out

        def metadata_reference(seq=seq, ends=ends, row_ids=row_ids, request_ids=request_ids):
            keys = local_visible_lengths(seq, world, 7, interleave)
            visible = local_visible_lengths(
                query_visible_lengths(seq, ends, row_ids, request_ids), world, 7, interleave
            )
            return keys, visible[:, None]

        yield f"metadata_b{requests}", metadata, metadata_reference, "exact"

    for count in (31, 128, 2048, 4096):
        indices = torch.randint(0, 1048576, (rows, 1, count), dtype=torch.int32, device="npu")
        indices[0] = -1
        if rows > 1:
            indices[1, :, : count // 2] = 7 * interleave
        order = torch.arange(count, dtype=torch.float32, device="npu")
        yield (
            f"remap_k{count}",
            lambda indices=indices: remap_sparse_indices_triton(indices, world, 7, interleave),
            lambda indices=indices, order=order: torch_remap(indices, order),
            "exact",
        )

    output = torch.randn(rows, 96, 512, dtype=torch.bfloat16, device="npu")
    maximum = torch.randn(96, rows, 1, device="npu").transpose(0, 1)
    total = torch.rand(96, rows, 1, device="npu").transpose(0, 1) + 0.25
    maximum[0, 0] = -torch.inf
    total[0, 0] = 0
    init_device_properties_triton()
    cores = get_vectorcore_num()

    def encode():
        encoded = torch.empty((rows, 96, 4), dtype=torch.bfloat16, device="npu")
        _encode_dcp_lse_kernel[(min(triton.cdiv(rows * 96, 256), cores),)](
            maximum, total, encoded, *maximum.stride()[:2], *total.stride()[:2], rows * 96, 96, True, 256
        )
        return encoded

    yield "lse_encode", encode, lambda: torch_encode_lse(maximum, total), "lse"
    packed_fn = lambda: pack_dcp_output_lse(output, maximum, world, 1, softmax_sum=total)
    yield "output_pack", packed_fn, lambda: torch_output_pack(output, maximum, total), "pack"
    if rows >= 128:
        yield (
            "output_pack_vs_hybrid",
            packed_fn,
            lambda: torch.cat((output, encode()), -1).view(rows, world, 12, 516).permute(1, 2, 0, 3).contiguous(),
            "pack",
        )
    yield (
        "output_pack_vs_prior",
        packed_fn,
        lambda: pack_dcp_output_lse(output, maximum + total.log(), world, 1),
        "pack",
    )
    recv = packed_fn()
    yield "output_combine", lambda: fused_dcp_lse_combine(recv, 512, 1), lambda: torch_combine(recv), "float"


def check(actual, expected, kind):
    if isinstance(actual, tuple):
        for a, e in zip(actual, expected):
            check(a, e, kind)
        return
    if kind in ("pack", "lse"):
        if kind == "pack":
            check(actual[..., :512], expected[..., :512], "exact")
        torch.testing.assert_close(decode_lse(actual).cpu(), decode_lse(expected).cpu(), atol=2e-6, rtol=2e-6)
    else:
        torch.testing.assert_close(
            actual.cpu(),
            expected.cpu(),
            atol=0.008 if kind == "float" else (2e-6 if kind == "float32" else 0),
            rtol=0.008 if kind == "float" else (2e-6 if kind == "float32" else 0),
            equal_nan=True,
        )


def validate(fn, reference, kind):
    expected = reference()
    for _ in range(3):
        check(fn(), expected, kind)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = fn()
    for _ in range(10):
        graph.replay()
    check(actual, expected, kind)
    graph.reset()


def measure_pair(fn, reference, mode, iterations, repeats):
    functions = [reference, fn]
    graphs, executors = [], []
    for function in functions:
        for _ in range(5):
            function()
        torch.npu.synchronize()
        if mode == "graph":
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(iterations):
                    function()
            graphs.append(graph)
            execute = graph.replay
        else:

            def execute(function=function):
                for _ in range(iterations):
                    function()

        executors.append(execute)
        for _ in range(3):
            execute()
    torch.npu.synchronize()
    samples = [[], []]
    for repeat in range(repeats):
        for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            executors[index]()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) * 1000 / iterations)
    for graph in graphs:
        graph.reset()
    reference_us, triton_us = map(statistics.median, samples)
    return dict(
        reference_us=reference_us,
        triton_us=triton_us,
        speedup=reference_us / triton_us,
        raw_reference_us=samples[0],
        raw_triton_us=samples[1],
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--profile")
    args = parser.parse_args()
    torch.npu.set_device(0)
    root = Path(__file__).resolve().parents[1]
    paths = [
        "sharded_indexer.py",
        "sparse_index_remap.py",
        "query_gather_prep.py",
        "dcp/dcp_a2a.py",
        "dcp/dcp_a2a_batched.py",
    ]
    hashes = {p: hashlib.sha256((root / "vllm_ascend/ops/triton" / p).read_bytes()).hexdigest() for p in paths}
    result = dict(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        device=torch.npu.get_device_name(),
        hashes=hashes,
        args=vars(args),
        validation=[],
        timings=[],
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)

    def save():
        target.write_text(json.dumps(result, indent=2) + "\n")

    for rows in args.tokens:
        for name, fn, reference, kind in cases(rows):
            if args.only and name not in args.only:
                continue
            validate(fn, reference, kind)
            result["validation"].append(dict(tokens=rows, op=name, status="pass"))
            save()
            print(f"PASS tokens={rows} op={name}", flush=True)
        gc.collect()
    if args.validate_only:
        return
    for rows in args.tokens:
        for name, fn, reference, kind in cases(rows):
            if args.only and name not in args.only:
                continue
            for mode in ("eager", "graph"):
                entry = dict(
                    tokens=rows, op=name, mode=mode, **measure_pair(fn, reference, mode, args.iterations, args.repeats)
                )
                result["timings"].append(entry)
                save()
                print(json.dumps({k: v for k, v in entry.items() if not k.startswith("raw_")}), flush=True)
            if args.profile:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(Path(args.profile) / f"{name}-t{rows}")
                    ),
                    record_shapes=True,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                    ),
                ) as prof:
                    for _ in range(5):
                        fn()
                        torch.npu.synchronize()
                        prof.step()
        gc.collect()


if __name__ == "__main__":
    main()
