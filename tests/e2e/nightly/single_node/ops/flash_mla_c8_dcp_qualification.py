#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real DCP8 C8 attention and production HCCL output exchange/combine gate.

torchrun --standalone --nproc_per_node=8 flash_mla_c8_dcp_qualification.py --json result.json
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401
from flash_mla_c8_qualification import C8Case, bench_graph, error
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

from vllm_ascend.attention.context_parallel.common_cp import merge_flash_attention_output
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def local_case(full, rank):
    local_k, local_rope, lengths = [], [], []
    for request, length in enumerate(full.lengths_cpu):
        ids = full.blocks_cpu[request].long()
        keys = full.k_cpu[ids].reshape(-1, 1, 512)[:length]
        rope = full.kr_cpu[ids].reshape(-1, 1, 64)[:length]
        selected = (torch.arange(length) // 128) % 8 == rank
        local_k.append(keys[selected])
        local_rope.append(rope[selected])
        lengths.append(int(selected.sum()))
    case = C8Case(lengths, 96, seed=16468)
    case.q.copy_(full.q)
    case.qr.copy_(full.qr)
    case.sq.copy_(full.sq)
    for request, length in enumerate(lengths):
        for page_index in range(math.ceil(length / 128)):
            start = page_index * 128
            count = min(128, length - start)
            page = int(case.blocks_cpu[request, page_index])
            case.k[page, :count].copy_(local_k[request][start : start + count])
            case.kr[page, :count].copy_(local_rope[request][start : start + count])
    return case


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8, "This gate requires TP8/DCP8."
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    init_device_properties_triton()
    init_distributed_environment(
        world_size=8, rank=rank, local_rank=rank, distributed_init_method="env://", backend="hccl"
    )
    group = init_model_parallel_group(
        [list(range(8))],
        local_rank=rank,
        backend="hccl",
        group_name="flash_mla_c8_dcp_gate",
        use_device_communicator=False,
    )
    results = []
    try:
        full = C8Case([127, 2179, 0], 96)
        expected, _, _ = full.reference()
        expected = expected[:, rank * 12 : (rank + 1) * 12]
        case = local_case(full, rank)
        metadata = case.metadata()

        def run():
            output, lse = case.run(metadata)
            return merge_flash_attention_output(output, lse, group)

        for mode in ("eager", "aclgraph"):
            if mode == "eager":
                output = run()
            else:
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = run()
                for _ in range(3):
                    graph.replay()
            actual = output.cpu().float()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.03)
            assert torch.count_nonzero(actual[-4:]) == 0
            results.append(
                {
                    "gate": "real_dcp8_attention_exchange_combine",
                    "mode": mode,
                    "local_lengths": case.lengths_cpu,
                    **error(actual, expected),
                }
            )
        if args.benchmark:
            # 128K global history / DCP8; B16, DSpark3 => 64 logical Q rows.
            # Timing includes attention, output packing, HCCL, and combine.
            case = C8Case([16384] * 16, 96, shared=True, seed=16468 + rank)
            metadata = case.metadata()
            results.append(
                {
                    "gate": "attention_and_dcp_merge_graph_latency",
                    "tokens": 64,
                    "local_kv_length": 16384,
                    "shared_prefix_fraction": 0.99,
                    **bench_graph(run),
                }
            )
        torch.distributed.barrier(group=group.device_group)
        rank_path = args.json.with_name(f"{args.json.stem}.rank{rank}{args.json.suffix}")
        rank_path.write_text(json.dumps(results, indent=2))
        print(json.dumps({"rank": rank, "results": results}), flush=True)
    finally:
        group.destroy()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
