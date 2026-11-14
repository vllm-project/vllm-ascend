# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with torchrun --nproc-per-node=2/4/8 -m pytest --noconftest (A3/A5)."""

import gc
import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")

from vllm.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.test_glm5_next_dcp_indexer import (  # noqa: E402
    MAX_LEN,
    POOL,
    make_inputs,
    reference,
    require_dtype,
    to_device,
    update_inputs,
)
from vllm_ascend.ops.triton.glm5_next_dcp_indexer import glm5_next_dcp_indexer  # noqa: E402

RANKS = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(RANKS not in (2, 4, 8), reason="requires isolated DCP2/4/8")


@pytest.fixture(scope="module")
def context():
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    init_distributed_environment(
        world_size=RANKS, rank=rank, local_rank=local_rank, distributed_init_method="env://", backend="hccl"
    )
    group = init_model_parallel_group(
        [list(range(RANKS))],
        local_rank=local_rank,
        backend="hccl",
        group_name="glm_kpool_dcp",
        use_device_communicator=False,
    )
    retained = []
    yield rank, group, retained
    torch.npu.synchronize()
    dist.barrier(group=group.cpu_group)
    for graph, *_ in retained:
        graph.reset()
    retained.clear()
    gc.collect()
    group.destroy()
    destroy_distributed_environment()


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("interleave", [128, 512])
def test_real_candidate_collectives_and_graph(context, quantized, interleave):
    rank, group, retained = context
    require_dtype(quantized)
    device = to_device(make_inputs(RANKS, rank, interleave, quantized))
    query, cache, weights, scales, metadata, _ = device

    def run():
        return glm5_next_dcp_indexer(
            query,
            cache,
            weights,
            metadata,
            group,
            topk=2048,
            pool_size=POOL,
            interleave=interleave,
            max_model_len=MAX_LEN,
            scale_cache=scales,
        )

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run()
    for step in range(3):
        host = make_inputs(RANKS, rank, interleave, quantized, step)
        update_inputs(device, host)
        graph.replay()
        expected = reference(host[0], host[2], host[4], 2048, step)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    retained.append((graph, device, result))
