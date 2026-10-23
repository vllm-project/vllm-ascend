# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure only routing-ID preparation, with NPU events on every TP rank.

torchrun --standalone --nproc-per-node=8 benchmarks/dsv41_routing_input_ids.py
"""

import argparse
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record


@record
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=12)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--calls-per-graph", type=int, default=32)
    parser.add_argument("--dtype", choices=("int32", "int64"), default="int32")
    args = parser.parse_args()
    if min(args.num_tokens, args.layers, args.samples, args.replays, args.calls_per_graph) < 1:
        parser.error("token, layer, sample and replay counts must be positive")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    def progress(phase):
        print(f"[rank {rank}] {phase}", file=sys.stderr, flush=True)

    # Keep worker-specific imports under @record so import failures also
    # reach torchrun's per-rank error.json and first-failure traceback.
    progress("import NPU/vLLM modules")
    import torch_npu  # noqa: F401 - registers torch.npu in each worker
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    progress("set NPU device")
    torch.npu.set_device(local_rank)
    progress("create vLLM config")
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    # initialize_model_parallel reads the current config, and the worker's
    # imports/custom-op execution must retain that same context through replay.
    with set_current_vllm_config(config):
        progress("initialize HCCL world group")
        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank, backend="hccl")
        progress("initialize TP groups")
        initialize_model_parallel(tensor_model_parallel_size=world)
        run_benchmark(args, rank, world, progress)


def run_benchmark(args, rank, world, progress):
    # This local tensor transform only reads the TP group. It has no attention
    # or MoE metadata and does not require a model forward context.
    from vllm_ascend.models.common.ops.sequence_parallel import prepare_sp_routing_input_ids

    dtype = torch.int32 if args.dtype == "int32" else torch.int64
    input_ids = torch.arange(args.num_tokens, dtype=dtype, device="npu") + 10
    input_ids[0] = -1
    padded_rows = (args.num_tokens + world - 1) // world * world

    def baseline():
        # DP1/PCP1: all ranks have equally sized SP shards, including the
        # TP padding. Reproduce the current cast/gather/unpad/normalize path.
        padded = F.pad(input_ids, (0, padded_rows - args.num_tokens))
        local = padded.chunk(world)[rank]
        output = None
        for _ in range(args.layers):
            gathered = torch.empty(padded_rows, dtype=torch.int64, device="npu")
            dist.all_gather_into_tensor(gathered, local.to(torch.int64))
            gathered = torch.cat(gathered.chunk(world))
            output = torch.where(gathered == -1, 0, gathered)
        return output

    def candidate():
        return prepare_sp_routing_input_ids(input_ids)

    graphs = {}
    outputs = {}
    for name, fn in (("baseline", baseline), ("candidate", candidate)):
        progress(f"warmup {name}")
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        dist.barrier()
        graph = torch.npu.NPUGraph()
        progress(f"capture {name}")
        with torch.npu.graph(graph):
            # Amortize replay submission over enough device work to avoid
            # measuring a host-fed stream of tiny candidate graphs.
            for _ in range(args.calls_per_graph):
                outputs[name] = fn()
        graphs[name] = graph
        progress(f"captured {name}")

    # Capture must retain computations, not the first batch's token values.
    progress("verify graph input refresh")
    for offset in (0, 100):
        updated = torch.arange(args.num_tokens, dtype=dtype) + 10 + offset
        updated[0] = -1
        input_ids.copy_(updated)
        for graph in graphs.values():
            graph.replay()
        torch.npu.synchronize()
        expected = F.pad(updated.to(torch.int64), (0, padded_rows - args.num_tokens))
        expected = torch.where(expected == -1, 0, expected)
        for output in outputs.values():
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    timings = {name: [] for name in graphs}
    progress("measure NPU events")
    for sample in range(args.samples):
        order = ("baseline", "candidate") if sample % 2 == 0 else ("candidate", "baseline")
        for name in order:
            dist.barrier()
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) * 1000 / (args.replays * args.calls_per_graph))

    result = {
        "rank": rank,
        "tp_size": world,
        "num_tokens": args.num_tokens,
        "padded_rows": padded_rows,
        "layers": args.layers,
        "calls_per_graph": args.calls_per_graph,
        "dtype": args.dtype,
        "median_us": {name: statistics.median(values) for name, values in timings.items()},
        "samples_us": timings,
        "scope": "routing-ID preparation only; not model latency or throughput",
    }
    # Emit each rank's NPU event times; a fast rank alone is not the model result.
    print(json.dumps(result), flush=True)
    progress("complete")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
