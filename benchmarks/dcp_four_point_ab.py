# SPDX-License-Identifier: Apache-2.0
"""Run identical DCP workloads against two separately imported revisions.

Select the implementation with PYTHONPATH, and label it with --variant. The
baseline adapter casts native BF16 scores exactly as its C++ binding does;
the candidate leaves that conversion to pack. Inputs/oracles are built on CPU.
Use torchrun with --nproc-per-node equal to --world for real HCCL measurements.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from vllm.triton_utils import triton

import vllm_ascend
from vllm_ascend.attention.context_parallel import sharded_indexer as cp
from vllm_ascend.ops.triton.dcp import dcp_a2a_batched
from vllm_ascend.ops.triton.dcp.dcp_a2a import fused_dcp_lse_combine, pack_dcp_output_lse
from vllm_ascend.ops.triton.sharded_indexer import merge_packed_candidates, pack_local_candidates
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


def candidate_inputs(rows, world=8, topk=2048, interleave=768, score_dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(20261028 + rows)
    ids = torch.randint(0, 16384, (world, rows, 1, topk), dtype=torch.int32, generator=generator)
    scores = torch.randn(ids.shape, generator=generator).to(score_dtype)
    scores[..., 4:8] = 1.0
    ids[:, 0] = -1
    scores[..., :3] = torch.tensor([torch.nan, torch.inf, -torch.inf]).to(score_dtype)
    ranks = torch.arange(world, dtype=torch.int32).view(world, 1, 1, 1)
    global_ids = ids // interleave * (world * interleave) + ranks * interleave + ids % interleave
    valid = (ids >= 0) & torch.isfinite(scores)
    global_ids = torch.where(valid, global_ids, -1)
    values = torch.where(valid, scores.float(), -torch.inf)
    packed = torch.stack((global_ids[:, :, 0], values[:, :, 0].contiguous().view(torch.int32)), -1)
    keys = global_ids[:, :, 0].permute(1, 0, 2).reshape(rows, -1)
    values = values[:, :, 0].permute(1, 0, 2).reshape(rows, -1)
    order = keys.masked_fill(keys < 0, torch.iinfo(torch.int32).max).argsort(dim=-1, stable=True)
    keys, values = keys.gather(-1, order), values.gather(-1, order)
    order = values.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    expected = torch.where(torch.isfinite(values.gather(-1, order)), keys.gather(-1, order), -1)[:, None]
    return ids, scores, packed.reshape(world * rows, topk, 2), expected


def local_reference(expected, rank, world=8, interleave=768):
    result = torch.full_like(expected, -1)
    for source, target in zip(expected[:, 0], result[:, 0]):
        keep = source[(source >= 0) & ((source // interleave) % world == rank)]
        target[: keep.numel()] = keep // (world * interleave) * interleave + keep % interleave
    return result


def check(actual, expected, *, exact=False):
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        atol=0.008 if expected.is_floating_point() and not exact else 0,
        rtol=0.008 if expected.is_floating_point() and not exact else 0,
        equal_nan=True,
    )


def batched_pack(output, lse, world, block_rows):
    """Exercise the generic kernel without changing production dispatch."""
    tokens, heads, dim = output.shape
    send = torch.empty((world, heads // world, tokens, dim + 4), dtype=output.dtype, device=output.device)
    init_device_properties_triton()
    dcp_a2a_batched._pack_dcp_output_lse_batched_kernel[(get_vectorcore_num(),)](
        output,
        lse,
        lse,
        send,
        *output.stride(),
        *lse.stride()[:2],
        *lse.stride()[:2],
        *send.stride(),
        heads // world,
        dim,
        heads,
        tokens * heads,
        SCATTER_TOKENS=False,
        LSE_PACK_DIM=4,
        FROM_STATS=False,
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_ROWS=block_rows,
        OUTPUT_HEAD_MAJOR=True,
    )
    return send


def batched_combine(recv, dim, block_rows):
    world, heads, tokens, _ = recv.shape
    output = torch.empty((tokens, heads, dim), device=recv.device, dtype=recv.dtype)
    init_device_properties_triton()
    grid = min(get_vectorcore_num(), triton.cdiv(tokens * heads, block_rows))
    kernel = dcp_a2a_batched._fused_dcp_lse_combine_batched_kernel
    parameters = dict(
        DCP_SIZE=world,
        HAS_LOCAL=False,
        BLOCK_D=triton.next_power_of_2(dim),
        BLOCK_ROWS=block_rows,
        multibuffer=False,
    )
    if "BLOCK_RANKS" in kernel.arg_names:
        parameters["BLOCK_RANKS"] = triton.next_power_of_2(world)
    kernel[(grid,)](
        recv,
        output,
        recv,
        recv,
        0,
        0,
        0,
        0,
        0,
        *recv.stride(),
        *output.stride(),
        dim,
        heads,
        tokens * heads,
        **parameters,
    )
    return output


def measure(fn, mode, iterations, repeats):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    graph = None
    if mode == "graph":
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(iterations):
                fn()
        execute = graph.replay
    else:

        def execute():
            for _ in range(iterations):
                fn()

    for _ in range(3):
        execute()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        execute()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    if graph is not None:
        graph.reset()
    return dict(median_us=statistics.median(samples), min_us=min(samples), max_us=max(samples), raw_us=samples)


class HcclGroup:
    def __init__(self, world, rank):
        self.world_size, self.rank_in_group, self.calls = world, rank, 0

    def all_gather(self, tensor, dim=0):
        assert dim == 0
        self.calls += 1
        out = torch.empty(
            (self.world_size * tensor.shape[0], *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype
        )
        dist.all_gather_into_tensor(out, tensor)
        return out


class PreloadedGroup:
    """Replay fixed peer candidates to measure production compute without HCCL."""

    def __init__(self, world, rank, buffers):
        self.world_size, self.rank_in_group = world, rank
        self.buffers, self.calls = buffers, 0

    def all_gather(self, tensor, dim=0):
        assert dim == 0
        result = self.buffers[self.calls % len(self.buffers)]
        self.calls += 1
        assert result.shape[0] == tensor.shape[0] * self.world_size
        return result


def cases(args, rank):
    rows, world, topk, interleave = args.rows, args.world, 2048, 768
    score_dtype = torch.float32 if args.score_dtype == "fp32" else torch.bfloat16
    ids, native_scores, packed, expected = candidate_inputs(rows, world, topk, interleave, score_dtype)
    local_ids, local_scores = ids[rank].npu(), native_scores[rank].npu()

    def adapted_scores():
        return local_scores.float() if args.variant == "baseline" else local_scores

    if args.distributed:
        group = HcclGroup(world, rank)
        tile = 64 if args.variant == "baseline" else cp.gathered_candidate_row_tile(rows, world, topk)

        # Keep the input allocation/copy outside the measured collective path.
        tiles = [
            packed.view(world, rows, topk, 2)[rank, start : start + tile].contiguous().npu()
            for start in range(0, rows, tile)
        ]

        def gather_only():
            for tensor in tiles:
                group.all_gather(tensor)
            return torch.empty(0, device="npu")

        yield "allgather_only", gather_only, torch.empty(0), dict(allgather_calls=(rows + tile - 1) // tile)
        yield (
            "distributed_topk_local",
            lambda: cp.distributed_topk(local_ids, adapted_scores(), group, topk, interleave, return_local=True),
            local_reference(expected, rank, world, interleave),
            dict(allgather_calls=(rows + tile - 1) // tile),
        )
        return

    yield (
        "native_cast_and_pack",
        lambda: pack_local_candidates(local_ids, adapted_scores(), world, rank, interleave),
        packed.view(world, rows, topk, 2)[rank],
        {},
    )
    gathered = packed.npu()
    if args.variant == "candidate":
        # Independent CPU ordering, outside every measured NPU region.
        tiles = packed.view(world, rows, topk, 2)
        order = tiles[..., 0].masked_fill(tiles[..., 0] < 0, torch.iinfo(torch.int32).max).argsort(stable=True)
        values = tiles[..., 1].contiguous().view(torch.float32).gather(-1, order)
        order = order.gather(-1, values.argsort(descending=True, stable=True))
        sorted_tiles = tiles.gather(-2, order[..., None].expand(-1, -1, -1, 2)).contiguous()
        yield (
            "sorted_pack",
            lambda: pack_local_candidates(local_ids, local_scores, world, rank, interleave, sort_candidates=True),
            sorted_tiles[rank],
            {},
        )
        sorted_gathered = sorted_tiles.reshape(world * rows, topk, 2).npu()
        for owner in (None, rank):
            yield (
                "presorted_merge_global" if owner is None else "presorted_merge_local",
                lambda owner=owner: merge_packed_candidates(
                    sorted_gathered, rows, world, topk, local_rank=owner, interleave=interleave, presorted=True
                ),
                expected if owner is None else local_reference(expected, owner, world, interleave),
                {},
            )
    for owner in (None, rank):
        yield (
            "merge_global" if owner is None else "merge_local",
            lambda owner=owner: merge_packed_candidates(
                gathered, rows, world, topk, local_rank=owner, interleave=interleave
            ),
            expected if owner is None else local_reference(expected, owner, world, interleave),
            {},
        )
    query_tile = 64 if args.variant == "baseline" else cp.gathered_candidate_row_tile(rows, world, topk)
    peer_tiles = sorted_tiles if args.variant == "candidate" else packed.view(world, rows, topk, 2)
    buffers = [
        peer_tiles[:, start : start + query_tile].contiguous().reshape(-1, topk, 2).npu()
        for start in range(0, rows, query_tile)
    ]
    replay_group = PreloadedGroup(world, rank, buffers)
    yield (
        "candidate_prepare_compute",
        lambda: cp.distributed_topk(local_ids, adapted_scores(), replay_group, topk, interleave, return_local=True),
        local_reference(expected, rank, world, interleave),
        {"communication": "excluded; fixed peer buffers", "query_tile": query_tile},
    )
    generator = torch.Generator().manual_seed(35)
    output = torch.randn(rows, 96, 512, dtype=torch.bfloat16, generator=generator)
    lse = torch.randn(rows, 96, 1, generator=generator) * 4
    lse[0] = -torch.inf
    output[0] = torch.nan
    if args.head_major:
        output = output.transpose(0, 1).contiguous().transpose(0, 1)
        lse = lse.transpose(0, 1).contiguous().transpose(0, 1)
    device_output, device_lse = output.npu(), lse.npu()
    recv = pack_dcp_output_lse(device_output, device_lse, world, 1)
    parts = output.reshape(rows, world, 96 // world, 512).permute(1, 0, 2, 3).float()
    logs = lse.reshape(rows, world, 96 // world).permute(1, 0, 2)
    weights = torch.nan_to_num(logs.softmax(0), nan=0.0)
    reference = (torch.where(torch.isfinite(logs[..., None]), parts, 0) * weights[..., None]).sum(0).bfloat16()
    yield "lse_pack", lambda: pack_dcp_output_lse(device_output, device_lse, world, 1), recv.cpu(), {}
    yield "lse_combine", lambda: fused_dcp_lse_combine(recv, 512, 1), reference, {}
    for block_rows in args.batched_block_rows:
        yield (
            f"lse_pack_batched_r{block_rows}",
            lambda block_rows=block_rows: batched_pack(device_output, device_lse, world, block_rows),
            recv.cpu(),
            {"block_rows": block_rows, "production_dispatch_changed": False},
        )
        yield (
            f"lse_combine_batched_r{block_rows}",
            lambda block_rows=block_rows: batched_combine(recv, 512, block_rows),
            reference,
            {"block_rows": block_rows, "production_dispatch_changed": False},
        )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-root", required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--score-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--head-major", action="store_true")
    parser.add_argument("--batched-block-rows", type=int, nargs="+", default=[])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--profile")
    args = parser.parse_args()
    if args.rows <= 0 or args.world <= 0 or 96 % args.world:
        parser.error("rows must be positive and world must be a positive divisor of 96")
    if args.batched_block_rows and args.distributed:
        parser.error("direct batched kernels require single-device mode")
    if any(rows not in (1, 2, 4, 8, 16, 32) for rows in args.batched_block_rows):
        parser.error("batched block rows must be one of 1, 2, 4, 8, 16, 32")
    rank = int(os.environ.get("LOCAL_RANK", "0")) if args.distributed else args.rank
    rank = args.world - 1 if rank is None else rank
    if not 0 <= rank < args.world:
        parser.error("rank must be in [0, world)")
    torch.npu.set_device(rank if args.distributed else 0)
    if args.distributed:
        dist.init_process_group("hccl", timeout=timedelta(minutes=10))
        if dist.get_world_size() != args.world:
            raise RuntimeError("The HCCL group size must match --world")
    root = Path(vllm_ascend.__file__).resolve().parent
    expected_root = Path(args.expected_source_root).resolve() / "vllm_ascend"
    if root != expected_root:
        raise RuntimeError(f"Imported {root}, expected {expected_root}; refusing invalid A/B measurement")
    paths = [
        "attention/context_parallel/sharded_indexer.py",
        "ops/triton/sharded_indexer.py",
        "ops/triton/dcp/dcp_a2a.py",
        "ops/triton/dcp/dcp_a2a_batched.py",
    ]
    result = dict(
        args=vars(args),
        rank=rank,
        source=str(root),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        device=torch.npu.get_device_name(),
        hashes={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
        cases=[],
    )
    target = Path(args.output.replace("{rank}", str(rank)))
    target.parent.mkdir(parents=True, exist_ok=True)

    def save():
        target.write_text(json.dumps(result, indent=2) + "\n")

    try:
        for name, fn, expected, metadata in cases(args, rank):
            if args.only and name not in args.only:
                continue
            entry = dict(op=name, **metadata)
            result["cases"].append(entry)
            try:
                for _ in range(3):
                    check(fn(), expected, exact=name.startswith("lse_pack"))
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    captured = fn()
                graph.replay()
                torch.npu.synchronize()
                check(captured, expected, exact=name.startswith("lse_pack"))
                graph.reset()
                entry["validation"] = "pass"
                base_memory = torch.npu.memory_allocated()
                torch.npu.reset_peak_memory_stats()
                fn()
                torch.npu.synchronize()
                entry["peak_extra_bytes"] = torch.npu.max_memory_allocated() - base_memory
                for mode in ("eager", "graph"):
                    if args.distributed:
                        dist.barrier()
                    entry[mode] = measure(fn, mode, args.iterations, args.repeats)
                if args.profile and args.distributed and rank != 0:
                    # Rank zero records the trace, but every peer must enter
                    # the same profiling collectives before destroying HCCL.
                    for _ in range(4):
                        fn()
                        torch.npu.synchronize()
                elif args.profile:
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(Path(args.profile) / name)),
                        record_shapes=True,
                        experimental_config=torch_npu.profiler._ExperimentalConfig(
                            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                        ),
                    ) as prof:
                        for _ in range(4):
                            fn()
                            torch.npu.synchronize()
                            prof.step()
            except Exception:
                entry["error"] = traceback.format_exc()
            save()
            print(json.dumps(dict(rank=rank, **entry)), flush=True)
            gc.collect()
    finally:
        save()
        if args.distributed:
            dist.destroy_process_group()
    failures = [case for case in result["cases"] if "error" in case]
    if failures:
        raise RuntimeError(f"{len(failures)} benchmark cases failed; see {target}")


if __name__ == "__main__":
    main()
