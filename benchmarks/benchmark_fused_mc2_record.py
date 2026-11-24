"""Compare MC2 + Record with integrated Record using DeepSeek-V4.1-Flash dimensions.

Run each mode in a separate process invocation. See benchmark_fused_mc2_record.md.
"""

import argparse
import hashlib
import json
import socket
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.ops.fused_moe.eplb import record_expert_tokens
from vllm_ascend.utils import enable_custom_op

SEED = 1234
K = 5120
INTERMEDIATE_SIZE = 2304
N = 2 * INTERMEDIATE_SIZE
TOPK = 6
NUM_EXPERTS = 384
SWIGLU_LIMIT = 10.0


def build_inputs(rank: int, group: str, world_size: int, m: int):
    torch.manual_seed(SEED)
    experts_per_rank = NUM_EXPERTS // world_size
    x = (torch.randn(m, K) * 0.5).to(torch.bfloat16)
    topk_scores, expert_idx = torch.randn(m, NUM_EXPERTS).topk(TOPK, dim=1)
    expert_idx = expert_idx.to(torch.int32)
    probs = torch.softmax(topk_scores, dim=-1)
    digest = hashlib.sha256()

    def hash_tensor(tensor):
        digest.update(memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast("B"))

    for tensor in (x, expert_idx, probs):
        hash_tensor(tensor)

    def make_weights(shape):
        weights = []
        for _ in range(experts_per_rank):
            weight = torch.randint(-2, 3, shape, dtype=torch.int8)
            hash_tensor(weight)
            weights.append(torch_npu.npu_format_cast(weight.npu(), 29))
        return weights

    weight1 = make_weights((K, N))
    weight2 = make_weights((N // 2, K))
    scale1 = torch.ones(experts_per_rank, N).view(torch.int32).to(torch.int64)
    scale2 = torch.ones(experts_per_rank, K).view(torch.int32).to(torch.int64)
    for tensor in (scale1, scale2):
        hash_tensor(tensor)

    op_kwargs = dict(
        x=x.npu(),
        weight1=weight1,
        weight2=weight2,
        expert_idx=expert_idx.npu(),
        bias1=[],
        bias2=[],
        scale1=[s.npu() for s in scale1],
        scale2=[s.npu() for s in scale2],
        probs=probs.npu(),
        group=group,
        max_output_size=max(4096, m * TOPK * world_size),
        swiglu_limit=SWIGLU_LIMIT,
        world_size=world_size,
        x_active_mask=torch.ones(m, dtype=torch.bool).npu(),
        out=torch.zeros(m, K, dtype=torch.bfloat16, device="npu"),
        expert_token_nums=torch.zeros(experts_per_rank, dtype=torch.int32, device="npu"),
    )
    expected = torch.bincount(expert_idx.flatten().long(), minlength=NUM_EXPERTS)
    expected = expected[rank * experts_per_rank : (rank + 1) * experts_per_rank] * world_size
    return op_kwargs, expected.to(torch.int32), digest.hexdigest()


def worker(
    rank: int,
    mode: str,
    port: int,
    warmup: int,
    iterations: int,
    repeats: int,
    world_size: int,
    m: int,
):
    torch_npu.npu.set_device(rank)
    enable_custom_op()
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f"tcp://127.0.0.1:{port}")
    try:
        group = _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        op_kwargs, expected, input_sha256 = build_inputs(rank, group, world_size, m)
        print(f"INPUT mode={mode} rank={rank} sha256={input_sha256}", flush=True)
        experts_per_rank = NUM_EXPERTS // world_size
        local_start = 3 + rank * experts_per_rank
        load = torch.zeros(3 + NUM_EXPERTS + 2, dtype=torch.int32, device="npu")
        enabled = torch.tensor(True, dtype=torch.bool, device="npu")

        if mode == "original":

            def run():
                torch.ops._C_ascend.dispatch_ffn_combine(**op_kwargs)
                record_expert_tokens(op_kwargs["expert_token_nums"], load, enabled, 1, local_start)

        else:

            def run():
                torch.ops._C_ascend.dispatch_ffn_combine(
                    **op_kwargs,
                    expert_load_view=load,
                    record_enabled=enabled,
                    local_expert_start=local_start,
                )

        for _ in range(warmup):
            run()
        torch.npu.synchronize()

        rank_max_event, rank_max_wall = [], []
        for _ in range(repeats):
            load.zero_()
            torch.npu.synchronize()
            dist.barrier()

            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            for _ in range(iterations):
                run()
            end.record()
            end.synchronize()
            local_event = start.elapsed_time(end) * 1000 / iterations
            local_wall = (time.perf_counter() - wall_start) * 1e6 / iterations

            timings = torch.tensor([local_event, local_wall], dtype=torch.float32, device="npu")
            dist.all_reduce(timings, op=dist.ReduceOp.MAX)
            max_event, max_wall = timings.cpu().tolist()
            rank_max_event.append(max_event)
            rank_max_wall.append(max_wall)
            print(
                f"mode={mode} rank={rank} repeat={len(rank_max_event)} "
                f"event_us_per_call={local_event:.2f} wall_us_per_call={local_wall:.2f} "
                f"rank_max_event_us_per_call={max_event:.2f} "
                f"rank_max_wall_us_per_call={max_wall:.2f}",
                flush=True,
            )

        actual_load = load.cpu()
        expected_load = torch.zeros_like(actual_load)
        expected_load[local_start : local_start + experts_per_rank] = expected * iterations
        assert torch.equal(actual_load, expected_load), f"rank {rank}: expert load mismatch"
        assert torch.equal(op_kwargs["expert_token_nums"].cpu(), expected), f"rank {rank}: count mismatch"
        output = op_kwargs["out"].cpu()
        assert torch.count_nonzero(output) > 0, f"rank {rank}: empty MoE output"
        output_sha256 = hashlib.sha256(output.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        print(f"OUTPUT mode={mode} rank={rank} sha256={output_sha256}", flush=True)

        if rank == 0:
            print(
                "RESULT "
                + json.dumps(
                    {
                        "mode": mode,
                        "spec": {
                            "model_geometry": "DeepSeek-V4.1-Flash-1017",
                            "world_size": world_size,
                            "m": m,
                            "k": K,
                            "n": N,
                            "topk": TOPK,
                            "num_experts": NUM_EXPERTS,
                            "experts_per_rank": experts_per_rank,
                            "weight_format": "list",
                            "weight_dtype": "synthetic_int8",
                            "swiglu_limit": SWIGLU_LIMIT,
                            "seed": SEED,
                        },
                        "warmup": warmup,
                        "iterations": iterations,
                        "repeats": repeats,
                        "input_sha256": input_sha256,
                        "median_rank_max_event_us_per_call": statistics.median(rank_max_event),
                        "median_rank_max_wall_us_per_call": statistics.median(rank_max_wall),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("original", "fused"), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--m", type=int, default=128)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("warmup must be nonnegative; iterations and repeats must be positive")
    if args.world_size < 1 or NUM_EXPERTS % args.world_size:
        parser.error(f"world-size must be a positive divisor of {NUM_EXPERTS}")
    if args.m < 1:
        parser.error("m must be positive")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        worker,
        args=(
            args.mode,
            port,
            args.warmup,
            args.iterations,
            args.repeats,
            args.world_size,
            args.m,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
