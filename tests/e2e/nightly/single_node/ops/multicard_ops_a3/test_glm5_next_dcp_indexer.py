# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run with torchrun --nproc-per-node=2/4/8 -m pytest --noconftest (A3/A5)."""

import gc
import os
from types import SimpleNamespace

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
from vllm_ascend.attention.context_parallel.kpool_pcp import (  # noqa: E402
    build_kpool_pcp_query_layout,
    gather_kpool_pcp_queries,
    restore_kpool_pcp_queries,
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


@pytest.mark.parametrize("full_dcp", [False, True], ids=["dcp_equals_pcp", "dcp_equals_tp_pcp"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("empty_rank", [False, True])
def test_joint_pcp_dcp_candidate_order(context, full_dcp, quantized, empty_rank):
    rank, world_group, _ = context
    require_dtype(quantized)
    pcp, tp = 2, RANKS // 2
    pcp_ranks = [[t + p * tp for p in range(pcp)] for t in range(tp)]
    # Logical DCP rank order is PCP-first; HCCL gathers in global-rank order.
    # Global candidate IDs must make the merge independent of that difference.
    dcp_ranks = [[r for ranks in pcp_ranks for r in ranks]] if full_dcp else pcp_ranks
    groups = []
    try:
        for name, ranks in (("glm_kpool_joint_pcp", pcp_ranks), ("glm_kpool_joint_dcp", dcp_ranks)):
            groups.append(
                init_model_parallel_group(
                    ranks,
                    local_rank=int(os.environ["LOCAL_RANK"]),
                    backend="hccl",
                    group_name=name,
                    use_device_communicator=False,
                )
            )
        pcp_group, dcp_group = groups
        local_rows = [list(range(8)), []] if empty_rank else [[0, 1, 4, 5, 7], [0, 2, 3, 6]]
        width = max(map(len, local_rows)) + 1
        padded = torch.zeros(pcp, width, dtype=torch.long)
        restore = torch.full((8,), -1, dtype=torch.long)
        for p, rows in enumerate(local_rows):
            padded[p, : len(rows)] = torch.tensor(rows, dtype=torch.long)
            for offset, row in enumerate(rows):
                if restore[row] < 0:
                    restore[row] = p * width + offset
        assert (restore >= 0).all()
        actual = len(local_rows[pcp_group.rank_in_group])
        layout = build_kpool_pcp_query_layout(
            SimpleNamespace(num_input_tokens=width, num_actual_tokens=actual),
            SimpleNamespace(
                padded_gather_idx=padded.flatten().npu(),
                hidden_restore_idx=restore.npu(),
                global_batch=SimpleNamespace(num_tokens=8),
            ),
            pcp_group,
        )
        for step in range(3):
            host = make_inputs(dcp_group.world_size, dcp_group.rank_in_group, 128, quantized, step)
            query, cache, weights, scales, metadata, _ = to_device(host)
            # All ranks, including a rank without local queries, score the same
            # canonical request rows against their own compressed cache shard.
            global_query = gather_kpool_pcp_queries(query[layout.local_indices], layout, pcp_group)
            global_weights = gather_kpool_pcp_queries(weights[layout.local_indices], layout, pcp_group)
            selected = glm5_next_dcp_indexer(
                global_query,
                cache,
                global_weights,
                metadata,
                dcp_group,
                topk=2048,
                pool_size=POOL,
                interleave=128,
                max_model_len=MAX_LEN,
                scale_cache=scales,
            )
            result = restore_kpool_pcp_queries(selected, layout, width, fill_value=-1).cpu()
            reference_rows = reference(host[0], host[2], host[4], 2048, step)
            expected = torch.full((width, 1, 2048 + POOL - 1), -1, dtype=torch.int32)
            expected[:actual] = reference_rows[local_rows[pcp_group.rank_in_group]]
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        dist.barrier(group=world_group.cpu_group)
        for group in reversed(groups):
            group.destroy()
