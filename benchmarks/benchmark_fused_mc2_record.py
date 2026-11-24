"""Compare original MC2 + Record with MC2-integrated Record on two A3 NPUs.

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
WORLD_SIZE = 2
M, K, N, TOPK, EXPERTS_PER_RANK = 128, 1024, 1024, 7, 40


def build_inputs(rank: int, group: str):
    torch.manual_seed(SEED)
    x = (torch.randn(M, K) * 0.5).to(torch.bfloat16)
    weight1 = torch.randint(-2, 3, (EXPERTS_PER_RANK, K, N), dtype=torch.int8)
    weight2 = torch.randint(-2, 3, (EXPERTS_PER_RANK, N // 2, K), dtype=torch.int8)
    expert_idx = torch.randint(0, WORLD_SIZE * EXPERTS_PER_RANK, (M, TOPK), dtype=torch.int32)
    probs = torch.softmax(torch.randn(M, TOPK), dim=-1)
    scale1 = torch.ones(EXPERTS_PER_RANK, N).view(torch.int32).to(torch.int64)
    scale2 = torch.ones(EXPERTS_PER_RANK, K).view(torch.int32).to(torch.int64)
    digest = hashlib.sha256()
    for tensor in (x, weight1, weight2, expert_idx, probs, scale1, scale2):
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())

    def format_weights(weights):
        return [torch_npu.npu_format_cast(w.npu(), 29) for w in weights]

    op_kwargs = dict(
        x=x.npu(),
        weight1=format_weights(weight1),
        weight2=format_weights(weight2),
        expert_idx=expert_idx.npu(),
        bias1=[],
        bias2=[],
        scale1=[s.npu() for s in scale1],
        scale2=[s.npu() for s in scale2],
        probs=probs.npu(),
        group=group,
        max_output_size=4096,
        swiglu_limit=0.0,
        world_size=WORLD_SIZE,
        x_active_mask=torch.ones(M, dtype=torch.bool).npu(),
        out=torch.zeros(M, K, dtype=torch.bfloat16, device="npu"),
        expert_token_nums=torch.zeros(EXPERTS_PER_RANK, dtype=torch.int32, device="npu"),
    )
    expected = torch.bincount(expert_idx.flatten().long(), minlength=WORLD_SIZE * EXPERTS_PER_RANK)
    expected = expected[rank * EXPERTS_PER_RANK : (rank + 1) * EXPERTS_PER_RANK] * WORLD_SIZE
    return op_kwargs, expected.to(torch.int32), digest.hexdigest()


def worker(rank: int, mode: str, port: int, warmup: int, iterations: int, repeats: int):
    torch_npu.npu.set_device(rank)
    enable_custom_op()
    dist.init_process_group(
        backend="hccl", rank=rank, world_size=WORLD_SIZE, init_method=f"tcp://127.0.0.1:{port}"
    )
    try:
        group = _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        op_kwargs, expected, input_sha256 = build_inputs(rank, group)
        print(f"INPUT mode={mode} rank={rank} sha256={input_sha256}", flush=True)
        local_start = 3 + rank * EXPERTS_PER_RANK
        load = torch.zeros(3 + WORLD_SIZE * EXPERTS_PER_RANK + 2, dtype=torch.int32, device="npu")
        enabled = torch.tensor(True, dtype=torch.bool, device="npu")

        if mode == "original":

            def run():
                torch.ops._C_ascend.dispatch_ffn_combine(**op_kwargs)
                record_expert_tokens(
                    op_kwargs["expert_token_nums"], load, enabled, 1, local_start
                )

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
        expected_load[local_start : local_start + EXPERTS_PER_RANK] = expected * iterations
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
                            "world_size": WORLD_SIZE,
                            "m": M,
                            "k": K,
                            "n": N,
                            "topk": TOPK,
                            "experts_per_rank": EXPERTS_PER_RANK,
                            "weight_format": "list",
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
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("warmup must be nonnegative; iterations and repeats must be positive")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        worker,
        args=(args.mode, port, args.warmup, args.iterations, args.repeats),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    main()
