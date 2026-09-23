# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.distributed.device_communicators.npu_communicator import _NpuAll2AllManager


class FakeSymmBuffer:
    def __init__(self, world_size=4):
        self.ep_world_size = world_size
        self.mask_buffer = None
        self.ccl = torch.ones(32, dtype=torch.uint8)

    def clean_mask_buffer(self):
        if self.mask_buffer is None:
            self.mask_buffer = torch.zeros(self.ep_world_size, dtype=torch.int32)
        else:
            self.mask_buffer.zero_()

    def update_mask_buffer(self, rank, masked):
        self.mask_buffer[rank] = int(masked)

    def get_local_buffer_tensor(self, dtype):
        assert dtype == torch.uint8
        return self.ccl


@pytest.mark.parametrize("inference_mode", [False, True])
def test_mega_moe_cumulative_mask_and_retry_preserve_captured_storage(inference_mode):
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    with torch.inference_mode(inference_mode):
        buffer = FakeSymmBuffer()
        manager.bind_mega_moe_buffer(buffer, [0, 2, 1, 3])
    address = buffer.mask_buffer.data_ptr()
    manager.update_mask(1)
    manager.update_mask(3)
    assert manager.query_active_mask().tolist() == [0, 1, 0, 1]
    assert buffer.mask_buffer.tolist() == [0, 0, 1, 1]
    manager.clean_buffers()
    assert not buffer.ccl.any()
    assert buffer.mask_buffer.tolist() == [0, 0, 1, 1]
    manager.update_mask(1, False)
    assert manager.query_active_mask().tolist() == [0, 0, 0, 1]
    assert buffer.mask_buffer.tolist() == [0, 0, 0, 1]
    assert buffer.mask_buffer.data_ptr() == address
    # Re-binding the same object must never resurrect an already dead peer.
    manager.bind_mega_moe_buffer(buffer, [0, 2, 1, 3])
    assert buffer.mask_buffer.tolist() == [0, 0, 0, 1]


def test_unmask_clears_stale_communication_flags_before_enabling_peer():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    buffer = FakeSymmBuffer()
    manager.bind_mega_moe_buffer(buffer, list(range(4)))
    manager.update_mask(1)
    update = buffer.update_mask_buffer

    def unmask(rank, masked):
        assert not buffer.ccl.any()
        update(rank, masked)

    buffer.update_mask_buffer = unmask
    manager.update_mask(1, False)
    assert not manager.query_active_mask().any()


def test_fault_queries_never_touch_device_buffer():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    manager.bind_mega_moe_buffer(FakeSymmBuffer(), list(range(4)))
    manager.update_mask(1)
    manager._mega_moe_buffer = Mock(spec=[])
    assert manager.query_active_mask().tolist() == [0, 1, 0, 0]
    assert not manager.query_fault()


def test_mask_allocation_failure_is_not_silently_ignored():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    buffer = FakeSymmBuffer()
    buffer.clean_mask_buffer = Mock(side_effect=RuntimeError("unsupported device"))
    with pytest.raises(RuntimeError, match="unsupported device"):
        manager.bind_mega_moe_buffer(buffer, list(range(4)))
    assert not manager.uses_mega_moe


@pytest.mark.parametrize("order", [[0, 1, 1, 3], [0, 1, 2], [0, 1, 2, 4]])
def test_invalid_mc2_rank_mapping_is_rejected(order):
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    with pytest.raises(ValueError, match="rank sets"):
        manager.bind_mega_moe_buffer(FakeSymmBuffer(), order)


def test_mc2_manager_keeps_dense_rank_translation_and_retry_behavior():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    manager.set_num_local_physical_experts(8)
    info = manager.get_elastic_info()
    address = info.data_ptr()
    manager.update_mask(1)
    manager.clean_buffers()
    assert not manager.uses_mega_moe
    assert info.tolist() == [1, 3, 0, 24, 0, -1, 1, 2, 0, 2, 3, -1]
    assert info.data_ptr() == address


@pytest.mark.parametrize("rank", [-1, 4, True])
def test_invalid_fault_rank_cannot_corrupt_cpu_mask(rank):
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    with pytest.raises(ValueError, match="EP rank"):
        manager.update_mask(rank)
    assert not manager.query_active_mask().any()


def test_binding_buffer_replays_existing_fault_mask():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    manager.update_mask(2)
    buffer = FakeSymmBuffer()
    manager.bind_mega_moe_buffer(buffer, list(range(4)))
    assert buffer.mask_buffer.tolist() == [0, 0, 1, 0]
