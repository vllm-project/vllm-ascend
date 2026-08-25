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
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    FSA_EXTERNAL_PLAN_READY_MARKER,
    FSA_PAIRED_SELECTION_COPY_MARKER,
    FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT,
    FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT,
    SparseKVOffloadManager,
)


def _make_plan_manager():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_fused_overlap = True
    manager.tp_rank = 0
    manager.topk = 4
    manager.topk_buffer_size = 8
    manager.max_model_len = 64
    manager.max_num_topk_rows = 4
    manager.lru_workspace_threads = 2
    manager.layer_name_to_offload_id = {
        "layer.0": 0,
        "layer.1": 1,
        "layer.2": 2,
        "layer.3": 3,
        "layer.4": 4,
    }
    manager.lru_req_ids_cpu = torch.empty(4, dtype=torch.int64)
    manager.lru_topk_indices_cpu = torch.empty((4, 4), dtype=torch.int32)
    manager.lru_stable_prefix_lens_cpu = torch.empty(4, dtype=torch.int32)
    manager.lru_visible_seq_lens_cpu = torch.empty(4, dtype=torch.int32)
    manager.lru_physical_row_workspace = torch.arange(12, dtype=torch.int32)
    manager.fused_plan_metadata_npu = torch.zeros(5, dtype=torch.int32)
    manager.fused_plan_status_npu = manager.fused_plan_metadata_npu[:1]
    manager.fused_plan_current_linear_slots_npu = manager.fused_plan_metadata_npu[1:]
    manager.current_kv_by_layer = {}
    manager.fused_overlap_membership_map = None
    manager.fused_overlap_membership_map_rows = 0
    manager.fused_overlap_plan_owner_layer_id = None
    manager.fused_overlap_plan_topk = None
    manager.fused_overlap_plan_num_tokens = 0
    manager.fused_overlap_plan_membership_map = None
    pointer_names = (
        "lru_req_ids_ptr",
        "lru_topk_indices_ptr",
        "lru_stable_prefix_lens_ptr",
        "lru_visible_seq_lens_ptr",
        "lru_current_slots_ptr",
        "lru_token_mark_workspace_ptr",
        "lru_token_pos_workspace_ptr",
        "lru_slot_workspace_ptr",
        "lru_miss_position_workspace_ptr",
        "lru_epochs_ptr",
        "lru_physical_row_workspace_ptr",
    )
    for pointer, name in enumerate(pointer_names, start=101):
        setattr(manager, name, pointer)
    manager.lru_last_req_ids_ptrs = [201 + layer for layer in range(5)]
    manager.lru_slot_to_token_ptrs = [211 + layer for layer in range(5)]
    manager.lru_slots_ptrs = [221 + layer for layer in range(5)]
    manager.lru_miss_count_ptrs = [231 + layer for layer in range(5)]
    manager.lru_miss_tokens_ptrs = [241 + layer for layer in range(5)]
    manager.lru_miss_slots_ptrs = [251 + layer for layer in range(5)]
    manager.tp_group = MagicMock()
    manager.sparse_kv_offload_cpp = SimpleNamespace(
        lru_resident_compact_with_plan_stable_rows=MagicMock(),
        enqueue_lru_resident_compact_with_plan_stable_rows=MagicMock(),
    )
    return manager


def test_external_lru_planner_thread_warmup_runs_on_tp0():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_fused_overlap = True
    manager.tp_rank = 0
    manager.lru_workspace_threads = 8
    manager.sparse_kv_offload_cpp = SimpleNamespace(
        warmup_lru_resident_threads=MagicMock(return_value=8),
    )

    assert manager._warmup_external_lru_planner_threads() == 8

    manager.sparse_kv_offload_cpp.warmup_lru_resident_threads.assert_called_once_with(8)


@pytest.mark.parametrize(
    ("use_fused_overlap", "tp_rank"),
    [(False, 0), (True, 1)],
)
def test_external_lru_planner_thread_warmup_is_gated(
    use_fused_overlap,
    tp_rank,
):
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_fused_overlap = use_fused_overlap
    manager.tp_rank = tp_rank
    manager.lru_workspace_threads = 8
    manager.sparse_kv_offload_cpp = SimpleNamespace(
        warmup_lru_resident_threads=MagicMock(),
    )

    assert manager._warmup_external_lru_planner_threads() == 0

    manager.sparse_kv_offload_cpp.warmup_lru_resident_threads.assert_not_called()


def test_mapped_membership_allocation_initializes_external_plan_control():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_fused_overlap = True
    manager.topk = 2048
    manager.tp_rank = 0
    manager.tp_group = MagicMock()
    real_zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return real_zeros(*args, **kwargs)

    with (
        patch.object(manager_module.torch, "zeros", side_effect=cpu_zeros),
        patch.object(
            manager_module.offload,
            "empty",
            side_effect=lambda shape, dtype, pin_memory: torch.empty(shape, dtype=dtype),
        ),
    ):
        membership = manager.allocate_fused_overlap_membership_map(3)
        membership_again = manager.allocate_fused_overlap_membership_map(3)

    assert membership.shape == (3, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT)
    assert membership_again.data_ptr() == membership.data_ptr()
    assert membership.dtype == torch.int16
    control = membership[
        :,
        FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT
        + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
    ]
    assert control[:, 0].tolist() == [-1, -1, -1]
    assert control[:, 1].tolist() == [FSA_EXTERNAL_PLAN_READY_MARKER] * 3
    assert control[:, 2].tolist() == [2048] * 3
    assert control[:, 3].tolist() == [14336] * 3
    assert control[:, 7].tolist() == [FSA_PAIRED_SELECTION_COPY_MARKER] * 3
    manager.tp_group.broadcast.assert_called_once()
    manager.tp_group.barrier.assert_called_once_with()


def test_external_lru_plan_is_reused_by_three_skip_layers_and_replanned_at_owner():
    manager = _make_plan_manager()
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    topk = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    req_ids = torch.tensor([101], dtype=torch.int64)
    stable_prefix_lens = torch.tensor([10], dtype=torch.int32)
    visible_seq_lens = torch.tensor([11], dtype=torch.int32)

    common_args = dict(
        num_tokens=1,
        topk_indices_npu=topk,
        req_ids_npu=req_ids,
        stable_prefix_lens_npu=stable_prefix_lens,
        visible_seq_lens_npu=visible_seq_lens,
        selection_membership_map=membership,
        capturing=False,
    )
    assert manager.prepare_fused_overlap_external_plan(layer_name="layer.0", skip_topk=False, **common_args)
    for layer_id in range(1, 4):
        assert manager.prepare_fused_overlap_external_plan(
            layer_name=f"layer.{layer_id}", skip_topk=True, **common_args
        )

    planner = manager.sparse_kv_offload_cpp.lru_resident_compact_with_plan_stable_rows
    assert planner.call_count == 1
    manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_metadata_npu, src=0)
    assert manager.fused_overlap_plan_owner_layer_id == 0

    assert manager.prepare_fused_overlap_external_plan(layer_name="layer.4", skip_topk=False, **common_args)
    assert planner.call_count == 2
    assert planner.call_args_list[-1].args[4] == manager.lru_slot_to_token_ptrs[4]
    assert manager.fused_overlap_plan_owner_layer_id == 4


def test_eager_external_plan_copies_inputs_and_preserves_cpp_argument_order():
    manager = _make_plan_manager()
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    topk = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.int32)
    req_ids = torch.tensor([101, 202], dtype=torch.int64)
    stable_prefix_lens = torch.tensor([10, 20], dtype=torch.int32)
    visible_seq_lens = torch.tensor([11, 21], dtype=torch.int32)

    assert manager.prepare_fused_overlap_external_plan(
        layer_name="layer.0",
        num_tokens=2,
        topk_indices_npu=topk,
        req_ids_npu=req_ids,
        stable_prefix_lens_npu=stable_prefix_lens,
        visible_seq_lens_npu=visible_seq_lens,
        selection_membership_map=membership,
        capturing=False,
    )

    torch.testing.assert_close(manager.lru_topk_indices_cpu[:2], topk)
    torch.testing.assert_close(manager.lru_req_ids_cpu[:2], req_ids)
    torch.testing.assert_close(manager.lru_stable_prefix_lens_cpu[:2], stable_prefix_lens)
    torch.testing.assert_close(manager.lru_visible_seq_lens_cpu[:2], visible_seq_lens)
    plan_start = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - manager.topk
    planner = manager.sparse_kv_offload_cpp.lru_resident_compact_with_plan_stable_rows
    planner.assert_called_once_with(
        manager.lru_req_ids_ptr,
        manager.lru_last_req_ids_ptrs[0],
        manager.lru_topk_indices_ptr,
        manager.lru_stable_prefix_lens_ptr,
        manager.lru_slot_to_token_ptrs[0],
        manager.lru_slots_ptrs[0],
        manager.lru_current_slots_ptr,
        manager.lru_miss_count_ptrs[0],
        manager.lru_miss_tokens_ptrs[0],
        manager.lru_miss_slots_ptrs[0],
        manager.lru_token_mark_workspace_ptr,
        manager.lru_token_pos_workspace_ptr,
        manager.lru_slot_workspace_ptr,
        manager.lru_miss_position_workspace_ptr,
        manager.lru_epochs_ptr,
        manager.lru_physical_row_workspace_ptr,
        manager.max_num_topk_rows,
        membership[:, plan_start:].data_ptr(),
        membership.stride(0),
        2,
        manager.topk,
        manager.topk_buffer_size,
        manager.max_model_len,
        manager.lru_workspace_threads,
        manager.lru_workspace_threads,
        manager.lru_visible_seq_lens_ptr,
    )
    manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_metadata_npu, src=0)


def test_capture_external_plan_and_current_kv_use_separate_side_streams():
    manager = _make_plan_manager()
    manager.current_kv_save_stream = MagicMock()
    manager.fused_plan_stream = MagicMock()
    current_stream = MagicMock()
    input_event = object()
    current_stream.record_event.return_value = input_event
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    topk = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    req_ids = torch.tensor([101], dtype=torch.int64)
    stable_prefix_lens = torch.tensor([10], dtype=torch.int32)
    visible_seq_lens = torch.tensor([11], dtype=torch.int32)
    manager._offload_new_kv_on_current_stream = MagicMock()

    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=current_stream),
        patch.object(
            manager_module.torch_npu.npu,
            "stream",
            side_effect=lambda _: nullcontext(),
        ),
    ):
        manager.offload_new_kv(
            layer_name="layer.0",
            slot_mapping=torch.tensor([0]),
            k_cache_cpu=torch.empty(1),
            v_cache_cpu=torch.empty(1),
            k_cache_npu=None,
            v_cache_npu=None,
            k=torch.empty(1),
            v=torch.empty(1),
            capturing=True,
        )
        assert manager.prepare_fused_overlap_external_plan(
            layer_name="layer.0",
            num_tokens=1,
            topk_indices_npu=topk,
            req_ids_npu=req_ids,
            stable_prefix_lens_npu=stable_prefix_lens,
            visible_seq_lens_npu=visible_seq_lens,
            selection_membership_map=membership,
            capturing=True,
        )
        manager.inject_current_kv_into_selection = MagicMock()
        manager.wait_for_current_kv_writeback(capturing=True)

    assert current_stream.record_event.call_count == 2
    manager.current_kv_save_stream.wait_event.assert_called_once_with(input_event)
    manager.fused_plan_stream.wait_event.assert_called_once_with(input_event)
    manager.sparse_kv_offload_cpp.enqueue_lru_resident_compact_with_plan_stable_rows.assert_called_once()
    manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_metadata_npu, src=0)
    current_stream.wait_stream.assert_called_once_with(manager.current_kv_save_stream)
    assert manager.current_kv_by_layer[0][0].numel() == 1
    assert manager.current_kv_by_layer[0][1].numel() == 1


def test_capture_without_fused_overlap_stays_on_current_stream():
    manager = _make_plan_manager()
    manager.use_fused_overlap = False
    manager.current_kv_save_stream = MagicMock()
    manager.fused_plan_stream = MagicMock()
    current_stream = MagicMock()
    manager._offload_new_kv_on_current_stream = MagicMock()

    with patch.object(
        manager_module.torch_npu.npu,
        "current_stream",
        return_value=current_stream,
    ):
        manager.offload_new_kv(
            layer_name="layer.0",
            slot_mapping=torch.tensor([0]),
            k_cache_cpu=torch.empty(1),
            v_cache_cpu=torch.empty(1),
            k_cache_npu=None,
            v_cache_npu=None,
            k=torch.empty(1),
            v=torch.empty(1),
            capturing=True,
        )
        manager.wait_for_current_kv_writeback(capturing=True)

    manager._offload_new_kv_on_current_stream.assert_called_once()
    current_stream.record_event.assert_not_called()
    current_stream.wait_stream.assert_not_called()
    manager.current_kv_save_stream.wait_event.assert_not_called()
    manager.fused_plan_stream.wait_event.assert_not_called()


def test_current_kv_injection_uses_planner_linear_slots_and_sentinel():
    manager = _make_plan_manager()
    manager.topk_buffer_size = 4
    manager.current_kv_by_layer[0] = (
        torch.tensor([[10.0, 11.0], [20.0, 21.0]]),
        torch.tensor([[30.0], [40.0]]),
    )
    manager.fused_plan_current_linear_slots_npu[:2].copy_(torch.tensor([6, 3]))
    selection_kv = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    selection_rope = torch.arange(8, dtype=torch.float32).reshape(8, 1)

    def scatter(destination, indices, updates):
        destination.index_copy_(0, indices.reshape(-1).to(torch.int64), updates)

    with patch.object(
        manager_module.torch_npu, "npu_scatter_nd_update_", side_effect=scatter, create=True
    ) as scatter_op:
        manager.inject_current_kv_into_selection(
            layer_name="layer.0",
            num_tokens=2,
            selection_kv_cache=selection_kv,
            selection_k_rope=selection_rope,
            capturing=False,
        )

    torch.testing.assert_close(selection_kv[6], torch.tensor([10.0, 11.0]))
    torch.testing.assert_close(selection_rope[6], torch.tensor([30.0]))
    torch.testing.assert_close(selection_kv[3], torch.tensor([20.0, 21.0]))
    torch.testing.assert_close(selection_rope[3], torch.tensor([40.0]))
    torch.testing.assert_close(selection_kv[0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(selection_rope[0], torch.tensor([0.0]))
    assert scatter_op.call_count == 2
