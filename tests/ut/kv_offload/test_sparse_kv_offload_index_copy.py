from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("memfabric_hybrid")

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (  # noqa: E402
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (  # noqa: E402
    MooncakeHostPool,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    SparseKVOffloadManager,
)


def _set_mooncake_allocator(manager):
    allocator = object.__new__(MooncakeHostPool)
    allocator.topology = SimpleNamespace()
    manager._host_kv_allocator = allocator


def test_eager_current_kv_index_copy_filters_invalid_slots():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = 1 * torch.bfloat16.itemsize
    manager.max_d2h_index_copy_tokens = 4

    host_k = torch.zeros((8, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((8, 1), dtype=torch.bfloat16)
    current_k = torch.tensor(
        [[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16
    )
    current_v = torch.tensor([[7], [8], [9]], dtype=torch.bfloat16)

    manager._offload_new_kv_via_index_copy(
        slot_mapping=torch.tensor([3, -1, 5], dtype=torch.int64),
        k_cache_cpu=host_k,
        v_cache_cpu=host_v,
        k=current_k,
        v=current_v,
        capturing=False,
    )

    assert torch.equal(host_k[3], current_k[0])
    assert torch.equal(host_v[3], current_v[0])
    assert torch.equal(host_k[5], current_k[2])
    assert torch.equal(host_v[5], current_v[2])
    assert torch.count_nonzero(host_k[0]).item() == 0
    assert torch.count_nonzero(host_v[0]).item() == 0


def test_graph_mooncake_writeback_waits_for_save_stream():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.use_fused_overlap = True
    manager.layer_name_to_offload_id = {"layer.0": 0}
    manager.current_kv_by_layer = {}
    manager._offload_new_kv_on_current_stream = MagicMock()
    _set_mooncake_allocator(manager)
    manager.current_kv_save_stream = MagicMock()
    current_stream = MagicMock()
    current_kv_ready = object()
    current_stream.record_event.return_value = current_kv_ready

    slot_mapping = torch.tensor([2], dtype=torch.int64)
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    current_k = torch.ones((1, 2), dtype=torch.bfloat16)
    current_v = torch.ones((1, 1), dtype=torch.bfloat16)

    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=current_stream),
        patch.object(
            manager_module.torch_npu.npu,
            "stream",
            side_effect=lambda _: nullcontext(),
        ),
    ):
        manager.offload_new_kv(
            "layer.0",
            slot_mapping,
            host_k,
            host_v,
            None,
            None,
            current_k,
            current_v,
            capturing=True,
        )
        manager.wait_for_current_kv_writeback(capturing=True)

    manager._offload_new_kv_on_current_stream.assert_called_once_with(
        slot_mapping,
        host_k,
        host_v,
        None,
        None,
        current_k,
        current_v,
        False,
        True,
        True,
    )
    assert manager.current_kv_by_layer[0] == (current_k, current_v)
    current_stream.record_event.assert_not_called()
    manager.current_kv_save_stream.wait_event.assert_not_called()
    current_stream.wait_stream.assert_called_once_with(manager.current_kv_save_stream)


def test_graph_mooncake_index_copy_runs_on_save_stream():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = torch.bfloat16.itemsize
    manager.max_d2h_index_copy_tokens = 4
    manager.d2h_slot_mapping_cpu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_src_idx_cpu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_dst_idx_cpu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_index_count_cpu = torch.zeros(1, dtype=torch.int32)
    manager.d2h_src_idx_npu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_dst_idx_npu = torch.zeros(4, dtype=torch.int64)
    manager.d2h_index_count_npu = torch.zeros(1, dtype=torch.int32)
    manager.current_kv_save_stream = MagicMock()
    current_stream = MagicMock()
    descriptors_ready = object()
    current_stream.record_event.return_value = descriptors_ready

    def enqueue_descriptors(
        slot_mapping,
        num_actual_tokens,
        max_num_tokens,
        num_host_slots,
        src_idx,
        dst_idx,
        count,
    ):
        assert slot_mapping[0].item() == 2
        assert num_actual_tokens == 1
        assert max_num_tokens == 4
        assert num_host_slots == 4
        src_idx.zero_()
        dst_idx.fill_(2)
        count.fill_(1)

    sparse_kv_ops = SimpleNamespace(
        sparse_kv_enqueue_current_kv_index_copy_descriptors=MagicMock(
            side_effect=enqueue_descriptors,
        ),
    )
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    current_k = torch.tensor([[1, 2]], dtype=torch.bfloat16)
    current_v = torch.tensor([[3]], dtype=torch.bfloat16)

    with (
        patch.object(manager_module, "_sparse_kv_ops", return_value=sparse_kv_ops),
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=current_stream),
        patch.object(
            manager_module.torch_npu.npu,
            "stream",
            side_effect=lambda _: nullcontext(),
        ),
    ):
        manager._offload_new_kv_via_index_copy(
            slot_mapping=torch.tensor([2], dtype=torch.int64),
            k_cache_cpu=host_k,
            v_cache_cpu=host_v,
            k=current_k,
            v=current_v,
            capturing=True,
            prepare_descriptors=True,
        )

    assert torch.equal(host_k[2], current_k[0])
    assert torch.equal(host_v[2], current_v[0])
    current_stream.record_event.assert_called_once_with()
    manager.current_kv_save_stream.wait_event.assert_called_once_with(
        descriptors_ready
    )
    sparse_kv_ops.sparse_kv_enqueue_current_kv_index_copy_descriptors.assert_called_once()


def test_graph_mooncake_prepares_descriptors_on_first_layer_only():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.tp_rank = 0
    manager.use_fused_overlap = True
    manager.layer_name_to_offload_id = {
        "layer.0": 0,
        "layer.1": 1,
    }
    manager.current_kv_by_layer = {}
    manager._offload_new_kv_on_current_stream = MagicMock()
    _set_mooncake_allocator(manager)

    slot_mapping = torch.tensor([2], dtype=torch.int64)
    host_k = torch.zeros((4, 2), dtype=torch.bfloat16)
    host_v = torch.zeros((4, 1), dtype=torch.bfloat16)
    current_k = torch.ones((1, 2), dtype=torch.bfloat16)
    current_v = torch.ones((1, 1), dtype=torch.bfloat16)

    for layer_name in ("layer.0", "layer.1"):
        manager.offload_new_kv(
            layer_name,
            slot_mapping,
            host_k,
            host_v,
            None,
            None,
            current_k,
            current_v,
            capturing=True,
        )

    assert manager._offload_new_kv_on_current_stream.call_count == 2
    first_call, second_call = (
        manager._offload_new_kv_on_current_stream.call_args_list
    )
    assert first_call.args[-3:] == (
        False,
        True,
        True,
    )
    assert second_call.args[-3:] == (
        False,
        True,
        False,
    )
