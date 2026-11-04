# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Inactive ranks relinquish native buffers but retain CPU rendezvous."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import PytestBase
from vllm_ascend.distributed.fasteep.communication import ReclaimableCommunicators
from vllm_ascend.distributed.fasteep.hccl import RetainedGroup


class TestCommunicationReclamation(PytestBase):
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
            patch("torch.npu.synchronize"),
            patch("torch.npu.mem_get_info", side_effect=[(100, 1000), (400, 1000)]),
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
            patch("torch.npu.synchronize"),
            patch(
                "vllm_ascend.distributed.fasteep.communication.stateless_destroy_torch_distributed_process_group"
            ) as destroy,
        ):
            owner.close()
            owner.close()
        migration.close.assert_called_once()
        serving.close.assert_called_once()
        destroy.assert_called_once_with(cpu)

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
