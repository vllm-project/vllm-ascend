# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.distributed.device_communicators.npu_communicator import _NpuAll2AllManager
from vllm_ascend.worker.sentinel import npu_worker_sentinel as sentinel_module
from vllm_ascend.worker.sentinel.npu_worker_sentinel import WorkerSentinel


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


def test_mega_moe_cleanup_and_mask_replay_preserve_captured_storage():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    with torch.inference_mode():
        buffer = FakeSymmBuffer()
        manager.bind_mega_moe_buffer(buffer)
    address = buffer.mask_buffer.data_ptr()
    # MegaMoe recovery must never read or write MC2-only tensors.
    manager._mc2_elastic_info = Mock(spec=[])
    manager._mc2_elastic_info_host = Mock(spec=[])
    manager.update_mask(1)
    manager.update_mask(3)
    assert manager.query_active_mask().tolist() == [0, 1, 0, 1]
    assert buffer.mask_buffer.tolist() == [0, 1, 0, 1]
    manager.clean_buffers()
    assert not buffer.ccl.any()
    assert not manager.query_active_mask().any()
    assert not buffer.mask_buffer.any()
    assert buffer.mask_buffer.data_ptr() == address
    # retry replays the cumulative dead set after cleanup.
    manager.update_mask(1)
    manager.update_mask(3)
    assert manager.query_active_mask().tolist() == [0, 1, 0, 1]
    assert buffer.mask_buffer.tolist() == [0, 1, 0, 1]
    assert buffer.mask_buffer.data_ptr() == address
    manager.update_mask(1, False)
    assert manager.query_active_mask().tolist() == [0, 0, 0, 1]
    assert buffer.mask_buffer.tolist() == [0, 0, 0, 1]


def test_mc2_manager_keeps_dense_rank_translation_and_retry_behavior():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    manager.set_num_local_physical_experts(8)
    info = manager.get_mc2_elastic_info()
    address = info.data_ptr()
    manager.update_mask(1)
    assert not manager.has_mega_moe_buffer
    assert info.tolist() == [1, 3, 0, 24, 0, -1, 1, 2, 0, 2, 3, -1]
    manager.clean_buffers()
    assert not manager.query_active_mask().any()
    assert not info.any()
    assert info.data_ptr() == address
    manager.update_mask(1)
    assert info.tolist() == [1, 3, 0, 24, 0, -1, 1, 2, 0, 2, 3, -1]
    assert info.data_ptr() == address


def test_retry_publishes_device_writes_before_leaving_quarantine():
    sentinel = object.__new__(WorkerSentinel)
    sentinel.worker_faulted = True
    sentinel.reset_device = Mock()
    request = object()

    def synchronize_while_quarantined():
        retry.assert_called_once_with(request)
        assert sentinel.worker_faulted

    with (
        patch.object(sentinel_module.GPUWorkerSentinel, "retry") as retry,
        patch.object(torch.npu, "synchronize", side_effect=synchronize_while_quarantined) as synchronize,
    ):
        sentinel.retry(request)
    sentinel.reset_device.assert_called_once()
    retry.assert_called_once_with(request)
    synchronize.assert_called_once_with()
    assert not sentinel.worker_faulted


@pytest.mark.parametrize("mega_moe", [False, True])
def test_scale_down_uses_backend_specific_physical_expert_ids(mega_moe):
    # Rank 1 is dead: the expert on rank 2 must keep id 8 for MegaMoe,
    # but become id 4 for MC2. A tail-rank-only test would miss this bug.
    table = torch.tensor([[0], [8], [12]], dtype=torch.int32)
    layer = SimpleNamespace(eplb_state=SimpleNamespace(expert_replica_routing_table=table))
    state = SimpleNamespace(
        physical_to_logical_map=torch.empty(1, 16),
        logical_replica_count=torch.empty(1, 3),
        model=SimpleNamespace(moe_layers=[layer]),
    )
    sentinel = object.__new__(WorkerSentinel)
    sentinel._eplb_model_state = lambda: state
    sentinel.worker = SimpleNamespace(model_runner=SimpleNamespace(model=state.model), vllm_config=object())
    manager = SimpleNamespace(
        has_mega_moe_buffer=mega_moe,
        query_active_mask=lambda: torch.tensor([0, 1, 0, 0]),
        get_mc2_elastic_info=Mock(return_value=torch.zeros(12)),
    )
    with (
        patch.object(sentinel_module, "mark_dead_expert_slots_inplace"),
        patch.object(sentinel_module, "redistribute_expert_placement", return_value={(0, 1)}),
        patch.object(sentinel_module, "_commit_eplb_maps"),
        patch.object(sentinel_module, "rebuild_model_expert_maps"),
        patch.object(sentinel_module, "reload_experts_from_disk") as reload_experts,
        patch.object(sentinel_module, "refresh_model_routing_tables"),
        patch.object(sentinel_module, "get_ep_group", return_value=SimpleNamespace(world_size=4)),
        patch.object(sentinel_module, "get_ep_all2all_manager", return_value=manager),
    ):
        sentinel._redistribute_experts({1})
    reload_experts.assert_called_once_with(state.model, sentinel.worker.vllm_config, {(0, 1)})
    assert table.flatten().tolist() == ([0, 8, 12] if mega_moe else [0, 4, 8])
