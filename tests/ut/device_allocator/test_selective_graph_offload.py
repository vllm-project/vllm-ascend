# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Selective graph offload preserves live pools and fails before data loss."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.base import PytestBase
from vllm_ascend.device_allocator import camem


class TestSelectiveGraphOffload(PytestBase):
    def allocator(self):
        allocator = object.__new__(camem.CaMemAllocator)
        allocator.pointer_to_data = {
            100: SimpleNamespace(handle=(0, 16, 100, 0), tag="graphs", is_asleep=False, cpu_backup_tensor=None),
            200: SimpleNamespace(handle=(0, 32, 200, 0), tag="weights", is_asleep=False, cpu_backup_tensor=None),
        }
        return allocator

    def test_selective_offload_and_repeat(self):
        allocator = self.allocator()
        with (
            patch.object(camem.torch.npu, "synchronize"),
            patch.object(camem.torch, "empty", return_value=MagicMock()),
            patch.object(camem, "memcpy", return_value=0) as copy,
            patch.object(camem, "unmap_and_release") as unmap,
        ):
            assert allocator.offload("graphs") == 16
            assert allocator.offload("graphs") == 0
            unmap.assert_called_once_with((0, 16, 100, 0))
            assert copy.call_args.args[1:] == (16, 100, 16, 2)
        assert allocator.pointer_to_data[100].is_asleep
        assert not allocator.pointer_to_data[200].is_asleep

    def test_failed_backup_does_not_unmap(self):
        allocator = self.allocator()
        with (
            patch.object(camem.torch.npu, "synchronize"),
            patch.object(camem.torch, "empty", return_value=MagicMock()),
            patch.object(camem, "memcpy", return_value=1),
            patch.object(camem, "unmap_and_release") as unmap,
        ):
            with pytest.raises(RuntimeError, match="backup failed"):
                allocator.offload("graphs")
            unmap.assert_not_called()
        assert not allocator.pointer_to_data[100].is_asleep

    def test_discarded_allocation_cannot_be_backed_up(self):
        allocator = self.allocator()
        allocator.pointer_to_data[100].is_asleep = True
        with pytest.raises(RuntimeError, match="contents were discarded"):
            allocator.offload("graphs")

    def test_failed_restore_keeps_backup_for_retry(self):
        allocator = self.allocator()
        data = allocator.pointer_to_data[100]
        data.is_asleep = True
        backup = MagicMock()
        backup.numel.return_value = 16
        backup.element_size.return_value = 1
        data.cpu_backup_tensor = backup
        with (
            patch.object(camem.gc, "collect"),
            patch.object(camem.torch.npu, "empty_cache"),
            patch.object(camem, "create_and_map"),
            patch.object(camem, "memcpy", return_value=2),
            patch.object(camem, "unmap_and_release") as unmap,
        ):
            with pytest.raises(RuntimeError, match="restore failed"):
                allocator.wake_up(["graphs"])
            unmap.assert_called_once_with(data.handle)
        assert data.is_asleep
        assert data.cpu_backup_tensor is backup
