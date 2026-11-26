# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check routing-ID reuse against real DP/PCP collectives on four CPUs."""

import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import prepare_finalize as prepare_module
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.router import fused_topk_router as router_module
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter

DP_SIZE = 2
PCP_SIZE = 2
WORLD_SIZE = DP_SIZE * PCP_SIZE
NUM_LAYERS = 3
TOKEN_COUNTS = ((2, 4, 1, 3), (2, 4, 1, 3), (5, 1, 3, 2), (1, 2, 1, 2))


class GatherGroup:
    def __init__(self, group, world_size):
        self.group = group
        self.world_size = world_size
        self.id_gathers = 0

    def all_gather(self, tensor, dim=0):
        if tensor.ndim == 1:
            self.id_gathers += 1
        outputs = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(outputs, tensor, group=self.group)
        return torch.cat(outputs, dim=dim)


def _input_ids(rank, count, iteration):
    values = torch.arange(count, dtype=torch.int32) + 100 * rank + 1000 * iteration + 11
    values[0] = -1
    return values


def _expected_ids(counts, iteration):
    """Build the DP-then-PCP row order independently of the collectives."""
    pcp_rows = []
    for pcp_rank in range(PCP_SIZE):
        ranks = [dp_rank * PCP_SIZE + pcp_rank for dp_rank in range(DP_SIZE)]
        dp_width = max(counts[rank] for rank in ranks)
        rows: list[int] = []
        for rank in ranks:
            ids = _input_ids(rank, counts[rank], iteration).tolist()
            rows.extend(ids)
            rows.extend([0] * (dp_width - counts[rank]))
        pcp_rows.append(rows)
    pcp_width = max(map(len, pcp_rows))
    return torch.tensor(
        [value for rows in pcp_rows for value in rows + [0] * (pcp_width - len(rows))],
        dtype=torch.int32,
    )


def _distributed_routing_ids_worker(rank, init_method):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=30),
    )
    try:
        dp_rank, pcp_rank = divmod(rank, PCP_SIZE)
        pcp_groups = [dist.new_group([d * PCP_SIZE + p for p in range(PCP_SIZE)]) for d in range(DP_SIZE)]
        dp_groups = [dist.new_group([d * PCP_SIZE + p for d in range(DP_SIZE)]) for p in range(PCP_SIZE)]
        pcp = GatherGroup(pcp_groups[dp_rank], PCP_SIZE)
        dp = GatherGroup(dp_groups[pcp_rank], DP_SIZE)
        config = SimpleNamespace(is_sequence_parallel=False, dp_size=DP_SIZE, pcp_size=PCP_SIZE, dp_group=dp)
        context = SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)
        max_local_tokens = max(map(max, TOKEN_COUNTS))
        input_storage = torch.empty(max_local_tokens, dtype=torch.int32)
        shared_ids = torch.full((WORLD_SIZE * max_local_tokens,), -999, dtype=torch.int64)
        baseline_routers = [AscendFusedTopKRouter(top_k=2, global_num_experts=4) for _ in range(NUM_LAYERS)]
        cached_routers = [AscendFusedTopKRouter(top_k=2, global_num_experts=4) for _ in range(NUM_LAYERS)]
        for layer_index, router in enumerate(cached_routers):
            router.routing_input_ids_buffer = shared_ids
            router.is_routing_input_ids_producer = layer_index == 0

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(prepare_module, "get_dynamic_mx_quant_scale_alg", lambda: None)
            patch.setattr(prepare_module, "get_pcp_group", lambda: pcp)
            patch.setattr(prepare_module, "_EXTRA_CTX", context)
            patch.setattr(router_module, "_EXTRA_CTX", context)
            prepares = [PrepareAndFinalizeWithAllGather(config) for _ in range(NUM_LAYERS)]
            for iteration, counts in enumerate(TOKEN_COUNTS):
                input_ids = input_storage[: counts[rank]]
                input_ids.copy_(_input_ids(rank, counts[rank], iteration))
                expected = _expected_ids(counts, iteration)
                context.max_tokens_across_dp = max(counts[d * PCP_SIZE + pcp_rank] for d in range(DP_SIZE))
                context.max_tokens_across_pcp = DP_SIZE * max(counts)
                baseline_outputs: list[torch.Tensor] = []
                for reuse, routers in ((False, baseline_routers), (True, cached_routers)):
                    dp.id_gathers = pcp.id_gathers = 0
                    for layer_index, (router, prepare) in enumerate(zip(routers, prepares)):
                        context.moe_comm_method = SimpleNamespace(prepare_finalize=prepare)
                        # Route IDs alongside token-labelled rows so the test
                        # also detects a mismatch with actual router padding.
                        rows = input_ids.to(torch.float32).reshape(-1, 1)
                        prepared = prepare.prepare(rows, rows)
                        num_tokens = prepared.router_logits.shape[0]
                        actual = router._prepare_routing_input_ids(input_ids, num_tokens)
                        router_rows = prepared.router_logits[:, 0].to(input_ids.dtype)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(actual, router_rows, rtol=0, atol=0)
                        if reuse:
                            torch.testing.assert_close(actual, baseline_outputs[layer_index], rtol=0, atol=0)
                            assert actual.data_ptr() == shared_ids.data_ptr()
                        else:
                            baseline_outputs.append(actual.clone())
                        expected_gathers = 1 if reuse else layer_index + 1
                        assert dp.id_gathers == pcp.id_gathers == expected_gathers
                dist.monitored_barrier(timeout=timedelta(seconds=30))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason="fork inherits the CPU test dependency mocks")
def test_routing_ids_reuse_real_dp_pcp_collectives(tmp_path):
    mp.start_processes(
        _distributed_routing_ids_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=WORLD_SIZE,
        start_method="fork",
    )
