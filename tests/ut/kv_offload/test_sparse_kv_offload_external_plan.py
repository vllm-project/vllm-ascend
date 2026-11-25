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
    HostMemoryRegion,
    MooncakeHostPool,
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
    manager.fused_async_plan = False
    manager.tp_size = 1
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
    manager.fused_overlap_planner_membership_map = None
    manager.fused_overlap_membership_plan_device_staging = None
    manager.fused_plan_broadcast_buffer = None
    manager._host_kv_allocator = None
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


def _enable_mooncake_membership_staging(manager):
    allocator = object.__new__(MooncakeHostPool)
    allocator.topology = SimpleNamespace()
    manager._host_kv_allocator = allocator
    return allocator


@pytest.mark.parametrize("topk", [3, 4])
def test_combined_plan_buffer_preserves_metadata_and_int16_bits(topk):
    manager = _make_plan_manager()
    manager.topk = topk
    metadata = torch.tensor([0, 65537, 131079, -1, 2147483647], dtype=torch.int32)
    manager.fused_plan_metadata_npu.copy_(metadata)
    manager._allocate_fused_plan_staging(3, torch.device("cpu"))
    plan = manager.fused_overlap_membership_plan_device_staging
    plan.copy_(torch.arange(plan.numel(), dtype=torch.int16).view_as(plan) - 16000)
    payload = manager.fused_plan_broadcast_buffer
    assert payload.dtype == torch.int32
    assert payload.is_contiguous()
    assert plan.is_contiguous()
    assert plan.data_ptr() == payload.data_ptr() + metadata.nbytes
    assert payload.nbytes >= metadata.nbytes + plan.nbytes
    assert payload.nbytes - metadata.nbytes - plan.nbytes < torch.int32.itemsize
    torch.testing.assert_close(manager.fused_plan_metadata_npu, metadata)

    peer = _make_plan_manager()
    peer.topk = topk
    peer._allocate_fused_plan_staging(3, torch.device("cpu"))
    peer.fused_plan_broadcast_buffer.copy_(payload)
    torch.testing.assert_close(peer.fused_plan_metadata_npu, metadata)
    torch.testing.assert_close(peer.fused_overlap_membership_plan_device_staging, plan)
    peer.fused_plan_status_npu.fill_(1)
    assert peer.fused_plan_broadcast_buffer[0] == 1
    torch.testing.assert_close(peer.fused_plan_current_linear_slots_npu, metadata[1:])


@pytest.mark.parametrize("mode", ["eager", "async", "capture", "failure"])
def test_combined_plan_broadcast_publishes_to_each_rank(mode):
    # Simulate the same collective on two ranks, including failed TP0 planning.
    payloads = []
    for rank in (0, 1):
        manager = _make_plan_manager()
        manager.tp_rank = rank
        manager.tp_size = 2
        manager.fused_async_plan = mode == "async"
        manager.fused_plan_stream = MagicMock()
        manager.current_kv_save_stream = MagicMock()
        _enable_mooncake_membership_staging(manager)
        manager._allocate_fused_plan_staging(4, torch.device("cpu"))
        plan = manager.fused_overlap_membership_plan_device_staging
        planner_storage = torch.full_like(plan, -12345)
        manager.fused_overlap_planner_membership_map = planner_storage if rank == 0 else None
        manager.lru_physical_row_workspace[8:10] = torch.tensor([65537, 131079])
        membership = torch.full((4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT), -1, dtype=torch.int16)
        manager.fused_overlap_membership_map = membership
        if mode == "failure":
            manager.sparse_kv_offload_cpp.lru_resident_compact_with_plan_stable_rows.side_effect = ValueError(
                "bad plan"
            )

        def broadcast(buffer, src, *, rank=rank, manager=manager):
            assert src == 0
            assert buffer is manager.fused_plan_broadcast_buffer
            if rank == 0:
                if mode == "capture":
                    manager.fused_plan_stream.wait_stream.assert_called_once_with(manager.current_kv_save_stream)
                payloads.append(buffer.clone())
            else:
                buffer.copy_(payloads[0])

        manager.tp_group.broadcast.side_effect = broadcast
        stream = MagicMock()
        expected_error = (
            pytest.raises(RuntimeError, match="external planner failed") if mode == "failure" else nullcontext()
        )
        with (
            patch.object(manager_module.torch_npu.npu, "current_stream", return_value=stream),
            patch.object(manager_module.torch_npu.npu, "stream", side_effect=lambda _: nullcontext()),
            patch.object(torch.Tensor, "record_stream"),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected host value read"))
            if mode in ("async", "capture")
            else nullcontext(),
            expected_error,
        ):
            manager.prepare_fused_overlap_external_plan(
                "layer.0",
                2,
                torch.zeros((2, 4), dtype=torch.int32),
                torch.tensor([101, 202]),
                torch.tensor([10, 20], dtype=torch.int32),
                torch.tensor([11, 21], dtype=torch.int32),
                membership,
                capturing=mode == "capture",
            )
        manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_broadcast_buffer, src=0)
        if mode == "failure":
            assert manager.fused_plan_status_npu.item() == 1
            assert manager.fused_overlap_plan_owner_layer_id is None
            assert torch.all(membership == -1)
        else:
            start = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - manager.topk
            torch.testing.assert_close(membership[:2, start : start + plan.shape[1]], planner_storage[:2])
            torch.testing.assert_close(
                manager.fused_plan_current_linear_slots_npu[:2], torch.tensor([65537, 131079], dtype=torch.int32)
            )
            assert torch.all(membership[2:] == -1)
        if mode in ("async", "capture"):
            stream.wait_stream.assert_called_once_with(manager.fused_plan_stream)
        if rank == 1:
            manager.sparse_kv_offload_cpp.lru_resident_compact_with_plan_stable_rows.assert_not_called()
            manager.sparse_kv_offload_cpp.enqueue_lru_resident_compact_with_plan_stable_rows.assert_not_called()


def test_empty_aligned_swapped_tensor_returns_aligned_view():
    raw_tensors = []

    def empty_swapped(shape, *, dtype, device):
        assert device == torch.device("npu:0")
        raw = torch.empty(shape, dtype=dtype)
        raw_tensors.append(raw)
        return raw

    with (
        patch.object(
            manager_module.torch_npu,
            "empty_with_swapped_memory",
            side_effect=empty_swapped,
            create=True,
        ),
        patch.object(
            manager_module.torch_npu.npu,
            "current_device",
            return_value=0,
        ),
    ):
        aligned = manager_module.empty_aligned_swapped_tensor(
            [3, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT],
            torch.int16,
        )

    assert raw_tensors
    assert aligned.shape == (3, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT)
    assert aligned.data_ptr() % manager_module._CPU_CACHE_ALIGNMENT == 0


def test_mooncake_membership_uses_per_rank_swapped_storage_and_staging(
    monkeypatch,
):
    # CPU UT validates staging geometry/ordering, not accelerator pinning.
    allocate_tensor = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return allocate_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.use_fused_overlap = True
    manager.topk = 2048
    manager.tp_rank = 0
    manager.tp_group = MagicMock()
    manager.fused_overlap_membership_map = None
    manager.fused_overlap_membership_map_rows = 0
    manager.fused_overlap_planner_membership_map = None
    manager.fused_overlap_membership_plan_device_staging = None
    manager.fused_plan_broadcast_buffer = None
    _enable_mooncake_membership_staging(manager)
    manager.fused_plan_metadata_npu = torch.zeros(4, dtype=torch.int32)
    swapped_membership = torch.empty(
        (3, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        dtype=torch.int16,
    )
    synchronization_order = []
    manager.tp_group.barrier.side_effect = lambda: synchronization_order.append("barrier")
    current_stream = MagicMock()
    current_stream.synchronize.side_effect = lambda: synchronization_order.append("stream_sync")

    with (
        patch.object(
            manager_module,
            "empty_aligned_swapped_tensor",
            return_value=swapped_membership,
        ) as allocate,
        patch.object(
            manager_module.torch_npu.npu,
            "current_stream",
            return_value=current_stream,
        ),
    ):
        membership = manager.allocate_fused_overlap_membership_map(3)

    planner = manager.fused_overlap_planner_membership_map
    device_staging = manager.fused_overlap_membership_plan_device_staging
    plan_shape = (
        3,
        manager.topk + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
    )
    assert membership.shape == (
        3,
        FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT,
    )
    assert planner is not None
    assert planner.shape == plan_shape
    assert planner.is_contiguous()
    assert planner.stride() == (plan_shape[1], 1)
    assert device_staging is not None
    assert device_staging.shape == plan_shape
    assert device_staging.is_contiguous()
    assert membership.data_ptr() != planner.data_ptr()
    assert device_staging.data_ptr() != planner.data_ptr()
    assert device_staging.data_ptr() != membership.data_ptr()
    allocate.assert_called_once_with(
        [3, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT],
        torch.int16,
    )
    current_stream.synchronize.assert_called_once_with()
    manager.tp_group.barrier.assert_called_once_with()
    assert synchronization_order == ["stream_sync", "barrier"]


def test_mooncake_eager_external_plan_publishes_through_npu_staging():
    manager = _make_plan_manager()
    _enable_mooncake_membership_staging(manager)
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    plan_width = manager.topk + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
    planner_membership = torch.full(
        (4, plan_width),
        7,
        dtype=torch.int16,
    )
    manager.fused_overlap_membership_map = membership
    manager.fused_overlap_planner_membership_map = planner_membership
    manager._allocate_fused_plan_staging(4, torch.device("cpu"))
    sparse_kv_ops = SimpleNamespace(
        sparse_kv_lru_resident_compact_with_plan_stable_rows=(
            manager.sparse_kv_offload_cpp.lru_resident_compact_with_plan_stable_rows
        ),
        sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows=(
            manager.sparse_kv_offload_cpp.enqueue_lru_resident_compact_with_plan_stable_rows
        ),
    )

    with nullcontext():
        assert manager.prepare_fused_overlap_external_plan(
            layer_name="layer.0",
            num_tokens=2,
            topk_indices_npu=torch.tensor(
                [[1, 2, 3, 4], [4, 3, 2, 1]],
                dtype=torch.int32,
            ),
            req_ids_npu=torch.tensor([101, 202], dtype=torch.int64),
            stable_prefix_lens_npu=torch.tensor([10, 20], dtype=torch.int32),
            visible_seq_lens_npu=torch.tensor([11, 21], dtype=torch.int32),
            selection_membership_map=membership,
            capturing=False,
        )

    plan_start = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - manager.topk
    plan_end = plan_start + plan_width
    torch.testing.assert_close(
        membership[:2, plan_start:plan_end],
        planner_membership[:2],
    )
    torch.testing.assert_close(
        manager.fused_overlap_membership_plan_device_staging[:2],
        planner_membership[:2],
    )
    planner = sparse_kv_ops.sparse_kv_lru_resident_compact_with_plan_stable_rows
    assert planner.call_args.args[17] == planner_membership[:2].data_ptr()
    assert planner.call_args.args[18] == plan_width
    manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_broadcast_buffer, src=0)


def test_non_tp0_mooncake_plan_does_not_require_cpu_planner_storage():
    manager = _make_plan_manager()
    manager.tp_rank = 1
    _enable_mooncake_membership_staging(manager)
    membership = torch.full(
        (4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT),
        -1,
        dtype=torch.int16,
    )
    manager.fused_overlap_membership_map = membership
    manager.fused_overlap_planner_membership_map = None
    plan_width = manager.topk + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
    manager._allocate_fused_plan_staging(4, torch.device("cpu"))
    manager.fused_overlap_membership_plan_device_staging.fill_(9)
    sparse_kv_ops = SimpleNamespace(
        sparse_kv_lru_resident_compact_with_plan_stable_rows=(
            manager.sparse_kv_offload_cpp.lru_resident_compact_with_plan_stable_rows
        ),
        sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows=(
            manager.sparse_kv_offload_cpp.enqueue_lru_resident_compact_with_plan_stable_rows
        ),
    )

    with nullcontext():
        assert manager.prepare_fused_overlap_external_plan(
            layer_name="layer.0",
            num_tokens=1,
            topk_indices_npu=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            req_ids_npu=torch.tensor([101], dtype=torch.int64),
            stable_prefix_lens_npu=torch.tensor([10], dtype=torch.int32),
            visible_seq_lens_npu=torch.tensor([11], dtype=torch.int32),
            selection_membership_map=membership,
            capturing=False,
        )

    (sparse_kv_ops.sparse_kv_lru_resident_compact_with_plan_stable_rows.assert_not_called())
    (sparse_kv_ops.sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows.assert_not_called())
    manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_broadcast_buffer, src=0)
    plan_start = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - manager.topk
    torch.testing.assert_close(
        membership[:1, plan_start : plan_start + plan_width],
        manager.fused_overlap_membership_plan_device_staging[:1],
    )


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


def test_capture_external_plan_side_stream_and_mooncake_writeback_wait():
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
    plan_width = manager.topk + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
    planner_membership = torch.full(
        (4, plan_width),
        7,
        dtype=torch.int16,
    )
    _enable_mooncake_membership_staging(manager)
    manager.fused_overlap_membership_map = membership
    manager.fused_overlap_planner_membership_map = planner_membership
    manager._allocate_fused_plan_staging(4, torch.device("cpu"))
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

    current_stream.record_event.assert_called_once_with()
    manager.current_kv_save_stream.wait_event.assert_not_called()
    manager.fused_plan_stream.wait_event.assert_called_once_with(input_event)
    writeback_kwargs = manager._offload_new_kv_on_current_stream.call_args.kwargs
    assert writeback_kwargs == {
        "has_prefill": False,
        "capturing": True,
        "prepare_index_copy_descriptors": True,
    }
    planner = manager.sparse_kv_offload_cpp.enqueue_lru_resident_compact_with_plan_stable_rows
    planner.assert_called_once()
    assert planner.call_args.args[17] == planner_membership[:1].data_ptr()
    assert planner.call_args.args[18] == plan_width
    manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_broadcast_buffer, src=0)
    current_stream.wait_stream.assert_any_call(manager.fused_plan_stream)
    current_stream.wait_stream.assert_any_call(manager.current_kv_save_stream)
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


@pytest.mark.parametrize("peer_change", [None, "offset", "invalid"])
def test_mooncake_local_views_and_fused_inputs_share_mapping(peer_change):
    manager = _make_plan_manager()
    manager.tp_rank = 1
    manager.offload_layer_names = ["layer.0"]
    manager.kv_cache_config = SimpleNamespace(num_blocks=4)
    pool = MooncakeHostPool(
        HostMemoryRegion(torch.empty(256, dtype=torch.int8)),
        manager_module.HostPoolTopology(tp_rank=1, tp_size=2),
    )
    manager._host_kv_allocator = pool
    k, v = [t.view(torch.bfloat16).reshape(4, 2, 1, 2) for t in pool.allocate_tensors([32, 32], 16)]
    manager.k_caches_cpu, manager.v_caches_cpu = [k], [v]

    def gather(output, value, group):
        descriptor, error = value
        assert error is None
        peer = value
        if peer_change == "offset":
            peer = ((descriptor[0] + 16, *descriptor[1:]), None)
        elif peer_change == "invalid":
            peer = (None, "outside pool")
        output[:] = [peer, value]

    with patch.object(torch.distributed, "all_gather_object", side_effect=gather):
        if peer_change:
            with pytest.raises(ValueError, match="layout mismatch"):
                manager._register_local_mooncake_views()
            return
        manager._register_local_mooncake_views()
    local = manager.get_local_host_kv_views("layer.0")
    fused = manager.get_fused_overlap_cpu_kv_inputs("layer.0")
    assert local[0] is fused[0] is k
    assert local[1] is fused[1] is v
    assert manager.gvas_k_bases == [k.data_ptr()]
    assert manager.gvas_v_bases == [v.data_ptr()]
    assert manager.cpu_block_lens == [(8, 8)]
    manager.tp_group.broadcast.assert_not_called()


def test_async_plan_orders_input_lifetime_callback_and_publication():
    manager = _make_plan_manager()
    manager.fused_async_plan = True
    manager.fused_plan_stream = MagicMock()
    manager.current_kv_save_stream = MagicMock()
    stream = MagicMock()
    operations = []
    manager.fused_plan_stream.wait_event.side_effect = lambda event: operations.append("input_ready")
    manager.sparse_kv_offload_cpp.enqueue_lru_resident_compact_with_plan_stable_rows.side_effect = (
        lambda *args: operations.append("planner")
    )
    manager.tp_group.broadcast.side_effect = lambda *args, **kwargs: operations.append("publish")
    topk = torch.arange(4, dtype=torch.int32).view(1, 4)
    request_ids = torch.tensor([101], dtype=torch.int64)
    stable = torch.tensor([10], dtype=torch.int32)
    visible = torch.tensor([11], dtype=torch.int32)
    membership = torch.full((4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT), -1, dtype=torch.int16)
    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=stream),
        patch.object(manager_module.torch_npu.npu, "stream", return_value=nullcontext()),
        patch.object(torch.Tensor, "record_stream", side_effect=lambda target: operations.append("retain")) as retain,
        patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected host value read")),
    ):
        assert manager.prepare_fused_overlap_external_plan("layer.0", 1, topk, request_ids, stable, visible, membership)
    assert operations == ["input_ready", "retain", "retain", "retain", "retain", "planner", "publish"]
    assert retain.call_count == 4
    manager.fused_plan_stream.wait_event.assert_called_once_with(stream.record_event.return_value)
    manager.sparse_kv_offload_cpp.lru_resident_compact_with_plan_stable_rows.assert_not_called()
    torch.testing.assert_close(manager.lru_topk_indices_cpu[:1], topk)
    torch.testing.assert_close(manager.lru_req_ids_cpu[:1], request_ids)


def test_synchronous_planner_failure_is_published_and_not_reused():
    manager = _make_plan_manager()
    manager.sparse_kv_offload_cpp.lru_resident_compact_with_plan_stable_rows.side_effect = ValueError("bad plan")
    membership = torch.full((4, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT), -1, dtype=torch.int16)
    with pytest.raises(RuntimeError, match="external planner failed.*bad plan"):
        manager.prepare_fused_overlap_external_plan(
            "layer.0",
            1,
            torch.zeros(1, 4, dtype=torch.int32),
            torch.tensor([101]),
            torch.tensor([10], dtype=torch.int32),
            torch.tensor([11], dtype=torch.int32),
            membership,
        )
    assert manager.fused_plan_status_npu.item() == 1
    assert manager.fused_overlap_plan_owner_layer_id is None
    manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_metadata_npu, src=0)
