# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Inactive ranks relinquish native buffers but retain CPU rendezvous."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import PytestBase
from vllm_ascend.distributed.fasteep import communication
from vllm_ascend.distributed.fasteep.communication import ReclaimableCommunicators
from vllm_ascend.distributed.fasteep.hccl import RetainedGroup


class TestCommunicationReclamation(PytestBase):
    def test_constructor_defers_migration_bootstrap(self):
        parent = MagicMock(rank_in_group=0)
        with (
            patch.object(communication.torch.npu, "current_device", return_value=0),
            patch.object(communication.torch, "device", return_value="npu:0"),
            patch("vllm_ascend.distributed.fasteep.hccl.HCCLLibrary") as library,
            patch("vllm_ascend.distributed.fasteep.hccl.PyHcclCommunicator") as create,
        ):
            owner = ReclaimableCommunicators(parent)
        create.assert_not_called()
        library.assert_called_once_with()
        assert owner.root is None
        assert owner.library is library.return_value

    def test_v2_cpu_metadata_uses_group_local_collective(self):
        group = object.__new__(RetainedGroup)
        group.world_size = 3
        group.cpu_group = MagicMock()
        value = torch.tensor([1, 2, 3])
        assert group.cpu_all_reduce_(value) is value
        tensors, options = group.cpu_group.allreduce.call_args.args
        assert tensors[0] is value
        assert options.reduceOp == torch.distributed.ReduceOp.SUM
        group.cpu_group.allreduce.return_value.wait.assert_called_once()

    def owner(self):
        owner = object.__new__(ReclaimableCommunicators)
        owner.rank = 3
        owner.device = "npu:0"
        owner.root = MagicMock()
        owner.active_native = MagicMock()
        owner.active_group = MagicMock()
        owner.cpu_groups = {}
        owner.barrier = MagicMock()
        owner._cpu_group = MagicMock(return_value=None)
        return owner

    def test_inactive_rank_releases_both_native_communicators(self):
        owner = self.owner()
        migration, serving = owner.root, owner.active_native
        with (
            patch.object(communication.torch.npu, "synchronize"),
            patch.object(communication.torch.npu, "mem_get_info", side_effect=[(100, 1000), (400, 1000)]),
        ):
            assert owner.activate(2) is None
        migration.close.assert_called_once()
        serving.close.assert_called_once()
        assert owner.root is None and owner.active_native is None
        assert owner.last_memory_delta == 300
        owner._cpu_group.assert_called_once_with(2)

    def test_close_is_idempotent(self):
        owner = self.owner()
        migration, serving = owner.root, owner.active_native
        cpu = MagicMock()
        owner.cpu_groups = {2: None, 4: cpu}
        with (
            patch.object(communication.torch.npu, "synchronize"),
            patch(
                "vllm_ascend.distributed.fasteep.communication.stateless_destroy_torch_distributed_process_group"
            ) as destroy,
        ):
            owner.close()
            owner.close()
        migration.close.assert_called_once()
        serving.close.assert_called_once()
        destroy.assert_called_once_with(cpu)

    def test_inactive_reclaim_runs_before_readiness_barrier_without_polluting_comm_bytes(self):
        owner = self.owner()
        events = []
        free = [100]
        owner.active_native.close.side_effect = lambda: events.append("close-serving")
        owner.root.close.side_effect = lambda: events.append("close-migration")
        owner.barrier.side_effect = lambda: events.append("barrier")

        def cpu_group(size):
            events.append("cpu-group")
            free[0] = 400

        def reclaim():
            assert owner.root is None and owner.active_native is None
            assert owner.last_memory_delta == 300
            events.append("reclaim")
            free[0] = 9000

        owner._cpu_group.side_effect = cpu_group
        with (
            patch.object(communication.torch.npu, "synchronize"),
            patch.object(communication.torch.npu, "mem_get_info", side_effect=lambda: (free[0], 10000)),
        ):
            assert owner.activate(2, inactive_callback=reclaim) is None
        assert events == ["close-serving", "close-migration", "barrier", "cpu-group", "reclaim", "barrier"]
        assert owner.last_memory_delta == 300

    def test_active_rank_does_not_reclaim_idle_pools(self):
        owner = self.owner()
        owner.rank = 0
        reclaim = MagicMock()
        with (
            patch.object(communication.torch.npu, "synchronize"),
            patch.object(communication.torch.npu, "mem_get_info", return_value=(100, 1000)),
            patch.object(communication, "PyHcclCommunicator") as create,
            patch.object(communication, "RetainedGroup") as group,
        ):
            create.return_value.available = True
            assert owner.activate(2, inactive_callback=reclaim) is group.return_value
        reclaim.assert_not_called()
        assert owner.root is None

    def test_failed_inactive_reclaim_cannot_reach_readiness_barrier(self):
        owner = self.owner()
        reclaim = MagicMock(side_effect=RuntimeError("reclaim failed"))
        with (
            patch.object(communication.torch.npu, "synchronize"),
            patch.object(communication.torch.npu, "mem_get_info", return_value=(100, 1000)),
            pytest.raises(RuntimeError, match="reclaim failed"),
        ):
            owner.activate(2, inactive_callback=reclaim)
        assert owner.root is None and owner.active_group is None
        owner.barrier.assert_called_once()

    def test_migration_communicator_is_restored_only_when_needed(self):
        owner = self.owner()
        owner.root = None
        owner.control = MagicMock()
        owner.parent = MagicMock()
        with patch("vllm_ascend.distributed.fasteep.communication.PyHcclCommunicator") as create:
            create.return_value.available = True
            owner.begin_transition()
            owner.begin_transition()
        create.assert_called_once_with(owner.control, owner.device)

    def test_transfers_and_children_require_a_live_root(self):
        owner = self.owner()
        owner.root = None
        owner.children = {}
        with pytest.raises(RuntimeError, match="communicator is inactive"):
            owner.transfer(0, 0, 1, send=True)
        with pytest.raises(RuntimeError, match="communicator is inactive"):
            owner.child(2)
