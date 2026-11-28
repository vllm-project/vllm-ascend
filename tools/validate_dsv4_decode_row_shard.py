# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run with torchrun on Ascend: validate fixed-buffer HCCL capture and replay.

This exercises real NPU row selection, HCCL and restoration. The synthetic
TopK payload deliberately bypasses QLI; model accuracy requires a separate
comparison with sharding disabled.
"""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v4.decode_row_shard import make_decode_indexer_row_plan
from vllm_ascend.models.deepseek_v4.decode_row_shard_graph import (
    allocate_graph_decode_indexer_state,
    gather_graph_decode_indexer_topk,
    refresh_graph_decode_indexer_inputs,
    select_graph_decode_indexer_inputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=13)
    parser.add_argument("--rows-per-request", type=int, default=8)
    parser.add_argument("--topk", type=int, default=32)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        plan = make_decode_indexer_row_plan(args.requests, args.rows_per_request, world, rank)
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        table = torch.arange(args.requests * 2, dtype=torch.int32, device=device).reshape(args.requests, 2)
        state = allocate_graph_decode_indexer_state(
            plan,
            table,
            num_heads=2,
            head_dim=4,
            topk=args.topk,
            max_model_len=65536,
            metadata_size=1024,
        )
        rows = args.requests * args.rows_per_request
        query = torch.zeros((rows, 2, 4), dtype=torch.int8, device=device)
        weights = torch.zeros((rows, 2), dtype=torch.float16, device=device)
        scales = torch.ones_like(weights)
        payload = torch.arange(rows * args.topk, dtype=torch.int32, device=device).reshape(rows, 1, args.topk)
        local = torch.empty((len(plan.query_rows), 1, args.topk), dtype=torch.int32, device=device)
        group = SimpleNamespace(world_size=world, rank_in_group=rank, device_group=dist.group.WORLD)
        lens = torch.full((args.requests,), 100, dtype=torch.int32, device=device)
        refresh_graph_decode_indexer_inputs(state, lens, table, args.requests)

        def run():
            select_graph_decode_indexer_inputs(state, query, weights, scales)
            torch.index_select(payload, 0, state.query_rows, out=local)
            return gather_graph_decode_indexer_topk(state, local, group)

        # Materialize the HCCL communicator and all kernels before capture.
        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            stream.wait_stream(torch.npu.default_stream())
            with torch.npu.graph(graph, stream=stream):
                output = run()
        torch.npu.synchronize()

        signatures = {name: value.data_ptr() for name, value in vars(state).items() if isinstance(value, torch.Tensor)}
        live_counts = [args.requests, 1, 0, min(world - 1, args.requests), args.requests - 1, args.requests]
        for step, live in enumerate(live_counts):
            query.fill_(step + 1)
            weights.fill_(step + 2)
            lens.fill_(101 + step)
            refresh_graph_decode_indexer_inputs(state, lens, table, live)
            graph.replay()
            expected = payload.clone()
            expected[live * args.rows_per_request :] = -1
            torch.npu.synchronize()
            # All ranks complete each collective before the host assertions.
            assert torch.equal(output.cpu(), expected.cpu()), (rank, step, live)
            assert torch.equal(state.query_buffer.cpu(), query.index_select(0, state.query_rows).cpu())
            assert torch.equal(state.weights_buffer.cpu(), weights.index_select(0, state.query_rows).cpu())
            assert signatures == {
                name: value.data_ptr() for name, value in vars(state).items() if isinstance(value, torch.Tensor)
            }
        dist.barrier()
        if rank == 0:
            print(
                f"PASS: TP={world}, requests={args.requests}, rows/request={args.rows_per_request}; HCCL graph replay"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
