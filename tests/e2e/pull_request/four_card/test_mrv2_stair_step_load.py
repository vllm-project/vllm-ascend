# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from vllm.distributed.eplb import eplb_communicator
from vllm.distributed.eplb.rebalance_execute import move_from_buffer
from vllm.utils.network_utils import get_open_port

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb import eplb_state
from vllm_ascend.distributed.eplb.explicit_transfer import stage_explicit_layer_transfer
from vllm_ascend.distributed.eplb.policy import stair
from vllm_ascend.patch.platform.patch_eplb import _wrap_async_rebalance
from vllm_ascend.utils import is_950

_RANKS = 4
_EXPERTS_PER_RANK = 2
_EXPERTS = _RANKS * _EXPERTS_PER_RANK
_WINDOW = 128
_BINS = 64
_WIDTH = 128
_TIMEOUT = 120


def _run_step_validation(rank: int, port: int) -> None:
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    dist.init_process_group(
        "hccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=_RANKS, timeout=timedelta(seconds=_TIMEOUT)
    )
    try:
        groups = SimpleNamespace(device_group=dist.group.WORLD, cpu_group=dist.new_group(backend="gloo"))
        with (
            patch.object(eplb_state, "get_eplb_group", return_value=groups),
            patch.object(stair, "get_eplb_group", return_value=groups),
        ):
            _check_stepwise_plan(rank, device, groups)
    finally:
        dist.destroy_process_group()


def _check_stepwise_plan(rank, device, groups):
    def unexpected_fallback(model_state, eplb_state, physical_to_logical_map_cpu, stream):
        raise AssertionError("The planner must consume the prepared STAIR load")

    policy = stair.StairEplbPolicy(StairConfig(load_window_bins=_BINS))
    state = eplb_state.AscendEplbState.__new__(eplb_state.AscendEplbState)
    state.policy = policy
    state.expert_load_window_size = _WINDOW
    state._rank_node_ids = np.zeros(_RANKS, dtype=np.int64)
    state.rearrange_event = torch.npu.Event()
    # The first layer's alternating hotspots disappear after binning. The
    # second layer is a control whose balanced plan should still be applied.
    alternating = torch.tensor([[10, 1, 8, 1, 5, 5, 5, 5], [1, 10, 2, 7, 5, 5, 5, 5]]).repeat(_BINS, 1)
    constant = torch.tensor([[11, 11, 10, 8, 10, 10, 10, 10]]).repeat(_WINDOW, 1)
    load = torch.stack((alternating, constant), dim=1).int()
    old = torch.arange(_EXPERTS).expand(2, -1).contiguous()
    slots = slice(rank * _EXPERTS_PER_RANK, (rank + 1) * _EXPERTS_PER_RANK)
    local_load = torch.zeros_like(load)
    local_load[:, :, slots] = load[:, :, slots]
    model = SimpleNamespace(
        num_logical_experts=_EXPERTS, num_physical_experts=_EXPERTS, num_moe_layers=2, num_expert_groups=1
    )
    communicator = eplb_communicator.TorchDistNcclEplbCommunicator(groups.device_group)
    model_state = SimpleNamespace(
        model=model,
        expert_load_window=local_load.to(device),
        physical_to_logical_map=old.to(device),
        communicator=communicator,
    )
    state.model_states = {"toy": model_state}
    state._initialize_load_stats_state(model_state)
    state._num_recorded_load_steps = _WINDOW
    state._local_load_collection_mask.fill_(1)
    state._physical_load_sample_slots.copy_(torch.arange(_WINDOW))
    try:
        stats = state.collect_global_load_stats()
        state.publish_async_load_stats(stats)
        plan = _wrap_async_rebalance(unexpected_fallback)(model_state, state, old, None)
        torch.testing.assert_close(plan[0], old[0])
        torch.testing.assert_close(plan[1], torch.tensor([0, 7, 1, 3, 2, 5, 6, 4]))
        torch.testing.assert_close(stats["toy"].values.cpu(), load)
        assert stats["toy"].sample_counts is None
        assert plan.changed_layer_count == 1

        def weight_for(layout):
            return (layout[slots, None] * _WIDTH + torch.arange(_WIDTH)).to(torch.bfloat16)

        weights = [weight_for(old[1]).to(device)]
        buffers = [torch.empty_like(weights[0])]
        metadata = stage_explicit_layer_transfer(
            old[1],
            plan[1],
            plan.source_rank_ids[1],
            plan.source_slot_ids[1],
            weights,
            buffers,
            groups.device_group,
            communicator,
            layer_idx=1,
        )
        move_from_buffer(weights, buffers, metadata, plan[1].numpy(), rank)
        torch.testing.assert_close(weights[0].cpu(), weight_for(plan[1]), rtol=0, atol=0)
    finally:
        policy._stop_planner_process()


@pytest.mark.skipif(not is_950(), reason="A5 STAIR stepwise planner regression")
def test_a5_stair_rejects_stepwise_regression():
    workers = mp.spawn(_run_step_validation, args=(get_open_port(),), nprocs=_RANKS, join=False)
    deadline = time.monotonic() + _TIMEOUT
    try:
        while not workers.join(timeout=5):
            assert time.monotonic() < deadline, "STAIR regression workers timed out"
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
