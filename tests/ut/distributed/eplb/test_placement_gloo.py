# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.ut.distributed.eplb.test_placement import make_metadata
from vllm_ascend.distributed.eplb.placement import EplbPlacement, load_placement
from vllm_ascend.distributed.eplb.placement_runtime import export_committed_placement, install_placement


class _CpuPeerCommunicator:
    """Test transport: real Gloo P2P, without NPU staging or HIXL."""

    def __init__(self):
        self.operations = []

    def set_stream(self, stream):
        pass

    def set_transfer_context(self, old, layer_idx):
        pass

    def add_send(self, tensors, peer, expert):
        for index, tensor in enumerate(tensors):
            self.operations.append(dist.P2POp(dist.isend, tensor, peer, tag=expert * len(tensors) + index))

    def add_recv(self, tensors, peer, expert):
        for index, tensor in enumerate(tensors):
            self.operations.append(dist.P2POp(dist.irecv, tensor, peer, tag=expert * len(tensors) + index))

    def execute(self):
        if self.operations:
            for work in dist.batch_isend_irecv(self.operations):
                work.wait()
        self.operations.clear()


def _install_and_export(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        group = dist.group.WORLD
        old = torch.tensor([[0, 1, 2, 3, 0, 1]])
        placement = EplbPlacement(make_metadata(), ((1, 0, 3, 2, 0, 1),), {"workload": "cpu-gloo"})
        local = old[0, rank * 3 : (rank + 1) * 3]
        weights = [local.float().reshape(3, 1) + 10, local.float().reshape(3, 1) * 2 + 100]
        model_state = SimpleNamespace(
            model=SimpleNamespace(num_logical_experts=4, expert_weights=[weights]),
            physical_to_logical_map=old,
            logical_to_physical_map=torch.full((1, 4, 3), -1, dtype=torch.long),
            logical_replica_count=torch.zeros((1, 4), dtype=torch.long),
            communicator=_CpuPeerCommunicator(),
            expert_buffer=[torch.zeros_like(weight) for weight in weights],
        )
        state = SimpleNamespace(
            async_worker=None,
            model_states={"target": model_state},
            prepare_movement_resources=MagicMock(),
            release_movement_resources=MagicMock(),
            placement_device_group=lambda: group,
            refresh_placement_layer=MagicMock(),
            finish_pending_commits=MagicMock(),
            parallel_config=SimpleNamespace(
                eplb_config=SimpleNamespace(policy="stair", window_size=2, step_interval=2)
            ),
        )
        install_placement(state, placement, group, MagicMock())
        target = torch.tensor(placement.physical_to_logical[0][rank * 3 : (rank + 1) * 3]).float().reshape(3, 1)
        torch.testing.assert_close(weights[0], target + 10)
        torch.testing.assert_close(weights[1], target * 2 + 100)
        result = export_committed_placement(state, make_metadata(), output, "real CPU Gloo test", group)
        assert load_placement(output).digest() == result["sha256"]
        assert model_state.physical_to_logical_map.tolist() == [list(placement.physical_to_logical[0])]
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires CPU Gloo")
class TestPlacementGloo(unittest.TestCase):
    def test_install_and_export_on_two_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            output = str(Path(directory) / "deployment.json")
            mp.spawn(_install_and_export, args=(rendezvous, output), nprocs=2, join=True)
            self.assertEqual(load_placement(output).provenance["workload"], "real CPU Gloo test")
