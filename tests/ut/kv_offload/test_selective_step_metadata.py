# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for steady Decode metadata and precomputed tail slots."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import CopySfaTopkSlotAllocator
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    update_sparse_kv_offload_metadata,
)


class Buffer:
    def __init__(self, size, dtype):
        self.np = np.zeros(size, dtype=dtype)
        self.copy_to_gpu = Mock()


def test_steady_identity_and_mapping_refresh_on_layout_changes():
    update = update_sparse_kv_offload_metadata
    ids, owners = Buffer(4, np.int64), Buffer(8, np.int32)

    def step(reqs=("a", "b"), instances=(101, 102), bounds=(0, 1, 2), reuse=True, padded=4):
        return update(
            bounds[-1],
            len(reqs),
            padded,
            padded,
            list(reqs),
            np.array(bounds),
            ids,
            owners,
            request_instance_ids=list(instances),
            reuse=reuse,
        )

    step()
    step()
    assert ids.copy_to_gpu.call_count == owners.copy_to_gpu.call_count == 1
    np.testing.assert_array_equal(ids.np, [101, 102, 0, 0])
    np.testing.assert_array_equal(owners.np[:4], [0, 1, 0, 0])
    step(("b", "a"), (102, 101))
    assert ids.copy_to_gpu.call_count == 2
    assert owners.copy_to_gpu.call_count == 1
    step(("b", "a"), (202, 101))  # Same text ID after recovery, new instance.
    assert ids.copy_to_gpu.call_count == 3
    step(bounds=(0, 2, 3))
    assert owners.copy_to_gpu.call_count == 2
    np.testing.assert_array_equal(owners.np[:4], [0, 0, 1, 0])
    step(reqs=("a",), instances=(101,), bounds=(0, 1), padded=2)
    np.testing.assert_array_equal(ids.np[:2], [101, 0])
    np.testing.assert_array_equal(owners.np[:2], [0, 0])
    step(reuse=False)
    before = (ids.copy_to_gpu.call_count, owners.copy_to_gpu.call_count)
    step()
    assert ids.copy_to_gpu.call_count == before[0] + 1
    assert owners.copy_to_gpu.call_count == before[1] + 1
    with pytest.raises(ValueError, match="positive"):
        step(instances=(0, 102))


def test_tail_slots_padding_reorder_boundary_and_recovery():
    allocator = CopySfaTopkSlotAllocator(2)
    packed = np.empty(12, dtype=np.int64)

    def step(reqs, positions, dummy=False):
        allocator.prepare_tail_inputs(reqs, positions, 128, 2304, 2048, packed, dummy=dummy)
        return packed[:8].reshape(4, 2).copy(), packed[8:].copy()

    rows, slots = step(["a", "b"], [127, 129])
    np.testing.assert_array_equal(slots, [2175, 4353, -1, -1])
    rows, slots = step(["b", "a"], [130, 128])
    np.testing.assert_array_equal(rows, [[1, 1], [0, 0], [-1, -1], [-1, -1]])
    np.testing.assert_array_equal(slots, [4354, 2048, -1, -1])
    allocator.release("a")
    rows, slots = step(["a", "b"], [123, 131])
    assert rows[0, 1] == 123 and slots[0] == 2171
    step(["a"], [124], dummy=True)
    assert np.all(packed == -1)
    with pytest.raises(ValueError):
        step(["a"], [-1])


def test_shared_tail_write_uses_precomputed_slots(monkeypatch):
    scatter = Mock()
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_kv_cache", scatter)
    manager = SimpleNamespace(
        _get_offload_layer_id=lambda _: 0,
        topk_buffers_k=[torch.zeros(2, 256, 4)],
        topk_buffers_v=[torch.zeros(2, 256, 2)],
        block_size=128,
        tp_rank=1,
        queue_selective_tail_writeback=Mock(),
    )
    slots = torch.tensor([128, -1])
    device_slots = torch.tensor([256, -1])
    # The fast path must not reconstruct masks or addresses on the device.
    monkeypatch.setattr(torch, "where", Mock(side_effect=AssertionError("recomputed tail slots")))
    SparseKVOffloadManager.write_selective_tail(
        manager, "layer", slots, object(), torch.zeros(2, 4), torch.zeros(2, 2), device_slots=device_slots
    )
    assert scatter.call_args.kwargs["slot_mapping"] is device_slots
    manager.queue_selective_tail_writeback.assert_called_once()


def test_fused_tail_writeback_queues_only_full_block(monkeypatch):
    context = SimpleNamespace()
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager.get_forward_context",
        lambda: context,
    )
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager._get_offload_layer_id = lambda _: 0
    manager.block_size = 128
    manager.selective_row_stride = 256
    manager.topk_buffer_size = 128
    manager.topk_buffers_k = [torch.zeros(2, 256, 1, 4)]
    manager.topk_buffers_v = [torch.zeros(2, 256, 1, 2)]
    manager.gvas_k_bases = [65536]
    manager.gvas_v_bases = [131072]
    manager.token_size_bytes_k = 16
    manager.token_size_bytes_v = 8
    manager.d2h_src_ptrs_npu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_dst_ptrs_npu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_lengths_npu = torch.zeros(4, dtype=torch.int32)
    manager._submit_current_kv_writeback = Mock()

    manager.queue_selective_tail_writeback("layer", torch.tensor([255, 256]), torch.tensor([[0, 0], [1, 0]]))
    manager._submit_current_kv_writeback.assert_not_called()
    (submit,) = context._selective_tail_writebacks
    submit()
    assert manager.d2h_lengths_npu.tolist() == [2048, 0, 1024, 0]
    assert manager.d2h_src_ptrs_npu[0] == manager.topk_buffers_k[0].data_ptr() + 128 * 16
    assert manager.d2h_dst_ptrs_npu[0] == 65536 + 128 * 16
    manager._submit_current_kv_writeback.assert_called_once_with(torch.device("cpu"), 2, tail=True)
