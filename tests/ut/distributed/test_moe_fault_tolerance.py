# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.worker.sentinel import gpu_worker_sentinel as upstream_sentinel

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
def test_mega_moe_cleanup_and_mask_replay_preserve_captured_storage(inference_mode):
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    with torch.inference_mode(inference_mode):
        buffer = FakeSymmBuffer()
        manager.bind_mega_moe_buffer(buffer)
    address = buffer.mask_buffer.data_ptr()
    manager.update_mask(1)
    manager.update_mask(3)
    assert manager.query_active_mask().tolist() == [0, 1, 0, 1]
    assert buffer.mask_buffer.tolist() == [0, 1, 0, 1]
    manager.clean_buffers()
    assert not buffer.ccl.any()
    assert not manager.query_active_mask().any()
    assert not buffer.mask_buffer.any()
    assert not manager.get_elastic_info().any()
    assert buffer.mask_buffer.data_ptr() == address
    # retry replays the cumulative dead set after cleanup.
    manager.update_mask(1)
    manager.update_mask(3)
    assert manager.query_active_mask().tolist() == [0, 1, 0, 1]
    assert buffer.mask_buffer.tolist() == [0, 1, 0, 1]
    assert buffer.mask_buffer.data_ptr() == address


def test_unmask_only_updates_the_requested_rank():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    buffer = FakeSymmBuffer()
    manager.bind_mega_moe_buffer(buffer)
    manager.update_mask(1)
    manager.update_mask(3)
    manager.update_mask(1, False)
    assert manager.query_active_mask().tolist() == [0, 0, 0, 1]
    assert buffer.mask_buffer.tolist() == [0, 0, 0, 1]
    assert buffer.ccl.eq(1).all()


def test_fault_queries_never_touch_device_buffer():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    manager.bind_mega_moe_buffer(FakeSymmBuffer())
    manager.update_mask(1)
    manager._mega_moe_buffer = Mock(spec=[])
    assert manager.query_active_mask().tolist() == [0, 1, 0, 0]
    assert not manager.query_fault()


def test_mask_allocation_failure_is_not_silently_ignored():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    buffer = FakeSymmBuffer()
    buffer.clean_mask_buffer = Mock(side_effect=RuntimeError("unsupported device"))
    with pytest.raises(RuntimeError, match="unsupported device"):
        manager.bind_mega_moe_buffer(buffer)
    assert not manager.has_mega_moe_buffer


def test_mc2_manager_keeps_dense_rank_translation_and_retry_behavior():
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    manager.set_num_local_physical_experts(8)
    info = manager.get_elastic_info()
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
    manager.bind_mega_moe_buffer(buffer)
    assert buffer.mask_buffer.tolist() == [0, 0, 1, 0]


@pytest.mark.parametrize("mega_moe", [False, True])
def test_upstream_retry_replays_dead_ranks_across_consecutive_recoveries(mega_moe):
    manager = _NpuAll2AllManager(4, torch.device("cpu"))
    manager.set_num_local_physical_experts(8)
    buffer = FakeSymmBuffer()
    if mega_moe:
        manager.bind_mega_moe_buffer(buffer)
    sentinel = SimpleNamespace(
        _clean_worker_state=Mock(),
        worker=SimpleNamespace(
            rank=0,
            model_runner=object(),
            parallel_config=SimpleNamespace(
                data_parallel_size=4,
                world_size=1,
                tensor_parallel_size=1,
                prefill_context_parallel_size=1,
                enable_eplb=False,
            ),
        ),
    )

    def check_cleanup(*args):
        assert not manager.query_active_mask().any()
        if mega_moe:
            assert not buffer.mask_buffer.any()
            assert not buffer.ccl.any()

    with (
        patch.object(torch.accelerator, "synchronize"),
        patch.object(upstream_sentinel, "get_cached_tcp_store_client"),
        patch.object(upstream_sentinel, "reset_eplb_async_state"),
        patch.object(upstream_sentinel, "_reinit_cpu_group", side_effect=check_cleanup),
        patch.object(upstream_sentinel, "get_dp_group", return_value=SimpleNamespace()),
        patch.object(upstream_sentinel, "get_ep_all2all_manager", return_value=manager),
    ):
        for dead_ranks in ([1], [1, 3]):
            request = SimpleNamespace(
                params={
                    "dp_master_ip": "127.0.0.1",
                    "dp_group_rank": 0,
                    "dp_group_size": 4 - len(dead_ranks),
                    "recovery_round": len(dead_ranks),
                    "recovery_store_port": 12345,
                    "dead_dp_ranks": dead_ranks,
                }
            )
            upstream_sentinel.WorkerSentinel.retry(sentinel, request)
            expected = [int(rank in dead_ranks) for rank in range(4)]
            assert manager.query_active_mask().tolist() == expected
            if mega_moe:
                assert buffer.mask_buffer.tolist() == expected
