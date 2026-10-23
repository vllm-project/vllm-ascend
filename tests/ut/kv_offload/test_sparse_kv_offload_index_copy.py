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


def _make_host_cache_with_private_sink(manager, k_shape, v_shape):
    physical_k = torch.zeros(
        (k_shape[0] + 1, *k_shape[1:]),
        dtype=torch.bfloat16,
    )
    physical_v = torch.zeros(
        (v_shape[0] + 1, *v_shape[1:]),
        dtype=torch.bfloat16,
    )
    logical_k = physical_k[:-1]
    logical_v = physical_v[:-1]
    manager._mooncake_writeback_raw_views = {
        logical_k.data_ptr(): physical_k.view(torch.int8),
        logical_v.data_ptr(): physical_v.view(torch.int8),
    }
    return logical_k, logical_v, physical_k, physical_v


def test_eager_current_kv_index_copy_filters_invalid_slots():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = 1 * torch.bfloat16.itemsize
    manager.max_d2h_index_copy_tokens = 4

    host_k, host_v, physical_k, physical_v = (
        _make_host_cache_with_private_sink(manager, (8, 2), (8, 1))
    )
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
    assert torch.equal(physical_k[-1], current_k[1])
    assert torch.equal(physical_v[-1], current_v[1])


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

    manager._offload_new_kv_on_current_stream.assert_called_once()
    call_args = manager._offload_new_kv_on_current_stream.call_args
    assert call_args.args == (
        slot_mapping,
        host_k,
        host_v,
        None,
        None,
        current_k,
        current_v,
    )
    assert call_args.kwargs == {
        "has_prefill": False,
        "capturing": True,
        "prepare_index_copy_descriptors": True,
    }
    assert manager.current_kv_by_layer[0] == (current_k, current_v)
    current_stream.record_event.assert_not_called()
    manager.current_kv_save_stream.wait_event.assert_not_called()
    current_stream.wait_stream.assert_called_once_with(manager.current_kv_save_stream)


def test_graph_mooncake_index_copy_runs_on_save_stream():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.token_size_bytes_k = 2 * torch.bfloat16.itemsize
    manager.token_size_bytes_v = torch.bfloat16.itemsize
    manager.max_d2h_index_copy_tokens = 4
    manager.d2h_dst_idx_npu = torch.zeros(4, dtype=torch.int64)
    manager.current_kv_save_stream = MagicMock()
    current_stream = MagicMock()
    current_kv_ready = object()
    current_stream.record_event.return_value = current_kv_ready

    def build_descriptors(
        slots,
        destinations,
        host_capacity,
    ):
        assert slots.tolist() == [2, -1]
        assert host_capacity == 4
        destinations[:2].copy_(torch.tensor([2, 4], dtype=torch.int64))

    descriptor_builder = MagicMock(side_effect=build_descriptors)
    manager.fused_index_copy_descriptor_builder = descriptor_builder
    host_k, host_v, physical_k, physical_v = _make_host_cache_with_private_sink(
        manager,
        (4, 2),
        (4, 1),
    )
    current_k = torch.tensor([[1, 2], [4, 5]], dtype=torch.bfloat16)
    current_v = torch.tensor([[3], [6]], dtype=torch.bfloat16)

    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=current_stream),
        patch.object(
            manager_module.torch_npu.npu,
            "stream",
            side_effect=lambda _: nullcontext(),
        ),
    ):
        manager._offload_new_kv_via_index_copy(
            slot_mapping=torch.tensor([2, -1], dtype=torch.int64),
            k_cache_cpu=host_k,
            v_cache_cpu=host_v,
            k=current_k,
            v=current_v,
            capturing=True,
            prepare_descriptors=True,
        )

    assert torch.equal(host_k[2], current_k[0])
    assert torch.equal(host_v[2], current_v[0])
    assert torch.equal(physical_k[-1], current_k[1])
    assert torch.equal(physical_v[-1], current_v[1])
    current_stream.record_event.assert_called_once_with()
    manager.current_kv_save_stream.wait_event.assert_called_once_with(
        current_kv_ready
    )
    descriptor_builder.assert_called_once()


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
    assert first_call.kwargs == {
        "has_prefill": False,
        "capturing": True,
        "prepare_index_copy_descriptors": True,
    }
    assert second_call.kwargs == {
        "has_prefill": False,
        "capturing": True,
        "prepare_index_copy_descriptors": False,
    }
