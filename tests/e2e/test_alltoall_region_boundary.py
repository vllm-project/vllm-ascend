# SPDX-License-Identifier: Apache-2.0
"""Four Gloo ranks: validate the production op's dynamic graph boundary.

TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchrun --standalone --nproc-per-node=4 this_file.py
The CPU test kernel uses real unequal async collectives and a simple expert
transform. It tests schema/fake/dynamic-shape propagation, not NPU W8A8 kernels;
the latter require the separate four-NPU model precision comparison.
"""

import importlib.util
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/fused_moe/alltoall_region.py"
    spec = importlib.util.spec_from_file_location("region_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    op = module.alltoall_routed_experts
    dist.init_process_group("gloo")
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size == 4

    @op.register_kernel("cpu")
    def cpu_region(x, weights, ids, *args):
        targets = ids[:, 0]
        order = torch.argsort(targets, stable=True)
        send_counts = torch.bincount(targets, minlength=size).to(torch.int64)
        gathered = [torch.empty_like(send_counts) for _ in range(size)]
        dist.all_gather(gathered, send_counts)
        recv_counts = torch.stack(gathered)[:, rank]
        sends, recvs = send_counts.tolist(), recv_counts.tolist()
        received = x.new_empty((sum(recvs), x.shape[1]))
        dist.all_to_all_single(received, x[order].contiguous(), recvs, sends, async_op=True).wait()
        # The ragged buffer and its expert output never leave this kernel.
        transformed = received * (rank + 1)
        returned = torch.empty_like(x)
        dist.all_to_all_single(returned, transformed, sends, recvs, async_op=True).wait()
        result = torch.empty_like(x)
        result[order] = returned
        return result, recv_counts.sum().reshape(1)

    graphs = []

    def backend(gm, inputs):
        nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        assert sum(n.target == torch.ops.vllm_ascend.fxrt_alltoall_routed_experts.default for n in nodes) == 1
        assert not any("all_to_all_single" in str(n.target) for n in nodes)
        graphs.append(gm)
        return gm.forward

    def run(x, ids):
        output, counts = op(
            x,
            x.new_ones((x.shape[0], 1)),
            ids,
            [x],
            [x],
            [x],
            [x],
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            False,
            "silu",
            False,
            None,
            False,
            10.0,
            1,
        )
        return output.reshape(x.shape[0], -1) + 1, counts

    compiled = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
    try:
        for rows, shift in ((7, 0), (19, 1), (33, 2), (7, 3), (19, 0)):
            x = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8) + rank
            # Same shape, different routing values includes ranks receiving zero
            # tokens. Each later shape must reuse the same symbolic graph.
            ids = torch.full((rows, 1), (rank + shift) % 2, dtype=torch.int64)
            actual, counts = compiled(x, ids)
            expected = x * (ids + 1) + 1
            assert torch.equal(actual, expected)
            assert counts.shape == (1,) and counts.dtype == torch.int64
        assert len(graphs) == 1, len(graphs)
        print(f"PASS rank={rank} graphs=1 dynamic_tokens=7,19,33 routing_values_changed=True", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
