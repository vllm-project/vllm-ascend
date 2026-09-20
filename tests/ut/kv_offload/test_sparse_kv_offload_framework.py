from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (
    sparse_kv_offload_manager as manager_module,
)


def make_manager():
    manager = manager_module.SparseKVOffloadManager.__new__(manager_module.SparseKVOffloadManager)
    manager.use_fused_overlap = True
    manager.tp_rank = 0
    manager.tp_size = 1
    manager.tp_group = MagicMock()
    manager.fused_plan_status_npu = torch.zeros(1, dtype=torch.int32)
    manager.fused_overlap_plan_owner_layer_id = None
    manager.fused_async_plan = False
    manager.current_kv_by_layer = {}
    manager.layer_name_to_offload_id = {"layer.0": 0, "layer.1": 1}
    manager.current_kv_save_stream = MagicMock()
    manager.fused_plan_stream = MagicMock()
    return manager


def test_graph_mtp_waits_for_writeback_before_selection_read():
    manager = make_manager()
    manager.fused_async_plan = True
    manager.current_kv_by_layer[0] = (torch.ones(4, 8), torch.ones(4, 2))
    manager.fused_plan_current_linear_slots_npu = torch.arange(4, dtype=torch.int32)
    operations = []
    stream = MagicMock()
    stream.wait_stream.side_effect = lambda target: operations.append(target)
    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=stream),
        patch.object(
            manager_module.torch_npu, "npu_scatter_nd_update_", side_effect=lambda *args: operations.append("scatter")
        ),
    ):
        manager.inject_current_kv_into_selection("layer.0", 4, torch.empty(4, 8), torch.empty(4, 2), capturing=True)
    assert operations == [manager.fused_plan_stream, manager.current_kv_save_stream, "scatter", "scatter"]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("owner_layer", [None, 0])
def test_graph_writeback_publishes_readiness_before_peer_selection(rank, owner_layer):
    manager = make_manager()
    manager.tp_rank, manager.tp_size = rank, 2
    manager.fused_async_plan = True
    manager.fused_overlap_plan_owner_layer_id = owner_layer
    manager.current_kv_by_layer[0] = (torch.ones(4, 8), torch.ones(4, 2))
    manager.fused_plan_current_linear_slots_npu = torch.arange(4, dtype=torch.int32)
    operations = []
    stream = MagicMock()
    stream.wait_stream.side_effect = lambda target: operations.append(target)
    manager.tp_group.broadcast.side_effect = lambda *args, **kwargs: operations.append("publish")
    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=stream),
        patch.object(
            manager_module.torch_npu, "npu_scatter_nd_update_", side_effect=lambda *args: operations.append("scatter")
        ),
    ):
        manager.inject_current_kv_into_selection("layer.0", 4, torch.empty(4, 8), torch.empty(4, 2), capturing=True)
    expected = [manager.fused_plan_stream]
    if rank == 0:
        expected.append(manager.current_kv_save_stream)
    if owner_layer is None:
        assert operations == expected + ["publish", "scatter", "scatter"]
        manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_status_npu, src=0)
    else:
        assert operations == expected + ["scatter", "scatter"]
        manager.tp_group.broadcast.assert_not_called()


@pytest.mark.parametrize("shared_storage", [True, False])
def test_cross_layer_owner_plan_ready_before_cpu_copy(shared_storage):
    manager = make_manager()
    manager.fused_async_plan = True
    manager.topk = 4
    manager.max_num_topk_rows = 4
    manager.fused_overlap_plan_owner_layer_id = 0
    manager.fused_overlap_plan_topk = 4
    manager.fused_overlap_plan_num_tokens = 4
    owner_map = torch.zeros(4, manager_module.FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT, dtype=torch.int16)
    manager.fused_overlap_plan_membership_map = owner_map
    target_map = owner_map if shared_storage else torch.zeros_like(owner_map)
    manager.fused_plan_stream.synchronize.side_effect = lambda: owner_map.fill_(7)
    assert manager.prepare_fused_overlap_external_plan(
        "layer.1",
        4,
        torch.zeros(4, 4, dtype=torch.int32),
        torch.arange(4),
        torch.ones(4, dtype=torch.int32),
        torch.ones(4, dtype=torch.int32),
        target_map,
        skip_topk=True,
    )
    assert manager.fused_plan_stream.synchronize.called != shared_storage
    if not shared_storage:
        offset = manager_module.FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT
        assert torch.all(target_map[:, offset - 4 : offset + 8] == 7)


def test_graph_writeback_preserves_final_join():
    manager = make_manager()
    manager.current_kv_by_layer[0] = (torch.ones(4, 8), torch.ones(4, 2))
    manager.fused_plan_current_linear_slots_npu = torch.arange(4, dtype=torch.int32)
    events = []
    stream = MagicMock()
    stream.wait_stream.side_effect = lambda target: events.append(target)
    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=stream),
        patch.object(
            manager_module.torch_npu, "npu_scatter_nd_update_", side_effect=lambda *args: events.append("scatter")
        ),
    ):
        manager.inject_current_kv_into_selection(
            "layer.0",
            4,
            torch.empty(4, 8),
            torch.empty(4, 2),
            capturing=True,
        )
        manager.wait_for_current_kv_writeback(capturing=True)
    assert events == [
        manager.fused_plan_stream,
        manager.current_kv_save_stream,
        "scatter",
        "scatter",
        manager.current_kv_save_stream,
    ]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("layer_id", [0, 1])
def test_graph_current_kv_ready_before_owner_and_reused_attention(rank, layer_id):
    manager = make_manager()
    manager.tp_rank, manager.tp_size = rank, 2
    manager.fused_overlap_plan_owner_layer_id = 0
    manager.current_kv_by_layer[layer_id] = (torch.ones(4, 8), torch.ones(4, 2))
    manager.fused_plan_current_linear_slots_npu = torch.arange(4, dtype=torch.int32)
    operations = []
    stream = MagicMock()
    stream.wait_stream.side_effect = lambda target: operations.append(target)
    manager.tp_group.broadcast.side_effect = lambda *args, **kwargs: operations.append("publish")
    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=stream),
        patch.object(
            manager_module.torch_npu, "npu_scatter_nd_update_", side_effect=lambda *args: operations.append("scatter")
        ),
    ):
        manager.inject_current_kv_into_selection(
            f"layer.{layer_id}", 4, torch.empty(4, 8), torch.empty(4, 2), capturing=True
        )
    expected = [manager.fused_plan_stream]
    if rank == 0:
        expected.append(manager.current_kv_save_stream)
    if layer_id != 0:
        expected.append("publish")
        manager.tp_group.broadcast.assert_called_once_with(manager.fused_plan_status_npu, src=0)
    else:
        manager.tp_group.broadcast.assert_not_called()
    assert operations == expected + ["scatter", "scatter"]


@pytest.mark.parametrize(
    "enabled,rank,fused,rows,expected",
    [
        (False, 0, True, 4, False),
        (True, 0, True, 4, True),
        (True, 1, True, 4, False),
        (True, 0, False, 4, False),
        (True, 0, True, 0, False),
    ],
)
def test_fused_descriptor_initialization_is_opt_in(enabled, rank, fused, rows, expected):
    manager = make_manager()
    manager.tp_rank, manager.use_fused_overlap, manager.max_num_tokens = rank, fused, rows
    manager.fused_reuse_writeback_layout = True
    helper = SimpleNamespace(build_writeback_descriptors=MagicMock(), warmup_writeback_descriptors=MagicMock())
    helper_name = "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.writeback_descriptors"
    settings = SimpleNamespace(VLLM_ASCEND_FSA_FUSED_WRITEBACK_DESCRIPTORS=enabled)
    with (
        patch.dict("sys.modules", {helper_name: helper}),
        patch.dict(manager._initialize_fused_writeback_descriptors.__func__.__globals__, envs=settings),
    ):
        manager._initialize_fused_writeback_descriptors()
    assert helper.warmup_writeback_descriptors.called == expected
    if expected:
        helper.warmup_writeback_descriptors.assert_called_once_with(manager)
        assert manager.fused_writeback_descriptor_builder is helper.build_writeback_descriptors
        assert not manager.fused_reuse_writeback_layout
    else:
        assert manager.fused_writeback_descriptor_builder is None
        assert manager.fused_reuse_writeback_layout


@pytest.mark.parametrize(
    "dtype,rows,expected",
    [
        (torch.bfloat16, 4, True),
        (torch.float16, 4, False),
        (torch.bfloat16, 0, False),
    ],
)
def test_fused_descriptor_dispatch_preserves_memfabric(dtype, rows, expected):
    manager = make_layout_manager()
    manager.fused_reuse_writeback_layout = False
    manager.fused_writeback_descriptor_builder = MagicMock()
    slots = torch.tensor([1, 2, -1, 100], dtype=torch.int64)[:rows]
    keys = torch.ones(rows, 8, dtype=dtype)
    ropes = torch.ones(rows, 2, dtype=dtype)
    with patch.object(manager_module.offload, "sparse_copy", return_value=0) as copy:
        manager._offload_new_kv_on_current_stream(
            slots, torch.empty(8, 8, dtype=dtype), torch.empty(8, 2, dtype=dtype), None, None, keys, ropes
        )
    assert manager.fused_writeback_descriptor_builder.called == expected
    copy.assert_called_once()
    if expected:
        torch.testing.assert_close(manager.fused_writeback_descriptor_builder.call_args.args[0], slots)
    else:
        assert manager.d2h_size_npu.item() == rows * 2


def test_fused_descriptor_copy_errors_and_row_validation():
    manager = make_layout_manager()
    manager.fused_reuse_writeback_layout = False
    manager.fused_writeback_descriptor_builder = MagicMock()
    slots = torch.arange(4, dtype=torch.int64)
    keys, ropes = torch.ones(4, 8, dtype=torch.bfloat16), torch.ones(4, 2, dtype=torch.bfloat16)
    pools = (torch.empty(8, 8, dtype=torch.bfloat16), torch.empty(8, 2, dtype=torch.bfloat16))
    with patch.object(manager_module.offload, "sparse_copy", return_value=-1) as copy:
        with pytest.raises(ValueError, match="row counts"):
            manager._offload_new_kv_on_current_stream(slots, *pools, None, None, keys[:3], ropes)
        copy.assert_not_called()
        manager.fused_writeback_descriptor_builder.assert_not_called()
        with pytest.raises(RuntimeError, match="sparse_copy failed"):
            manager._offload_new_kv_on_current_stream(slots, *pools, None, None, keys, ropes)
        copy.assert_called_once()


@pytest.mark.parametrize(
    "enabled,rank,fused,rows,expected",
    [
        (False, 0, True, 4, False),
        (True, 0, True, 4, True),
        (True, 1, True, 4, True),
        (True, 0, False, 4, False),
        (True, 0, True, 0, False),
    ],
)
def test_paired_scatter_warms_each_rank_only_when_enabled(enabled, rank, fused, rows, expected):
    manager = make_manager()
    manager.tp_rank, manager.use_fused_overlap, manager.max_num_tokens = rank, fused, rows
    helper = SimpleNamespace(try_paired_current_scatter=MagicMock(), warmup_current_scatter=MagicMock())
    helper_name = "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.current_kv_scatter"
    settings = SimpleNamespace(VLLM_ASCEND_FSA_PAIRED_CURRENT_SCATTER=enabled)
    with (
        patch.dict("sys.modules", {helper_name: helper}),
        patch.dict(manager._initialize_paired_current_scatter.__func__.__globals__, envs=settings),
    ):
        manager._initialize_paired_current_scatter()
    assert helper.warmup_current_scatter.called == expected
    assert (manager.fused_current_scatter is helper.try_paired_current_scatter) == expected


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("supported", [True, False])
def test_paired_scatter_keeps_readiness_and_native_fallback(rank, supported):
    manager = make_manager()
    manager.tp_rank, manager.tp_size = rank, 2
    manager.current_kv_by_layer[0] = (torch.ones(4, 8), torch.ones(4, 2))
    manager.fused_plan_current_linear_slots_npu = torch.arange(4, dtype=torch.int32)
    operations = []
    stream = MagicMock()
    stream.wait_stream.side_effect = lambda target: operations.append(target)
    manager.tp_group.broadcast.side_effect = lambda *unused, **kwargs: operations.append("publish")

    def scatter(*unused):
        operations.append("paired")
        return supported

    manager.fused_current_scatter = scatter
    with (
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=stream),
        patch.object(
            manager_module.torch_npu, "npu_scatter_nd_update_", side_effect=lambda *unused: operations.append("native")
        ),
    ):
        manager.inject_current_kv_into_selection("layer.0", 4, torch.empty(4, 8), torch.empty(4, 2), capturing=True)
    expected = [manager.fused_plan_stream]
    if rank == 0:
        expected.append(manager.current_kv_save_stream)
    expected += ["publish", "paired"]
    if not supported:
        expected += ["native", "native"]
    assert operations == expected


def make_layout_manager():
    manager = make_manager()
    manager.fused_reuse_writeback_layout = True
    manager.max_num_tokens = 4
    manager.token_size_bytes_k = 16
    manager.token_size_bytes_v = 4
    manager.d2h_token_indices_npu = torch.arange(4, dtype=torch.int64)
    manager.d2h_src_ptrs_npu = torch.empty(8, dtype=torch.int64)
    manager.d2h_dst_ptrs_npu = torch.empty(8, dtype=torch.int64)
    manager.d2h_lengths_npu = torch.empty(8, dtype=torch.int32)
    manager.d2h_size_npu = torch.empty(1, dtype=torch.int32)
    return manager


def check_layout_writeback(manager, token, slots, key_pool, rope_pool, stream_id=0):
    context = SimpleNamespace(ascend_graph_capture_token=token)
    stream = MagicMock(npu_stream=stream_id)
    keys = torch.ones(4, 8, dtype=torch.bfloat16)
    ropes = torch.ones(4, 2, dtype=torch.bfloat16)
    with (
        patch.dict(manager.offload_new_kv.__func__.__globals__, get_forward_context=lambda: context),
        patch.object(manager_module.torch_npu.npu, "current_stream", return_value=stream),
        patch.object(manager_module.torch_npu.npu, "stream", return_value=nullcontext()),
        patch.object(manager_module.offload, "sparse_copy", return_value=0),
    ):
        manager.offload_new_kv("layer.0", slots, key_pool, rope_pool, None, None, keys, ropes, capturing=True)
    capacity = key_pool.shape[0]
    for row, slot in enumerate(slots.tolist()):
        valid = 0 <= slot < capacity
        assert manager.d2h_lengths_npu[row].item() == (16 if valid else 0)
        assert manager.d2h_lengths_npu[row + 4].item() == (4 if valid else 0)
        if valid:
            assert manager.d2h_dst_ptrs_npu[row].item() == key_pool[slot].data_ptr()
            assert manager.d2h_dst_ptrs_npu[row + 4].item() == rope_pool[slot].data_ptr()
    assert manager.d2h_size_npu.item() == 8


def test_layout_reuse_updates_layer_specific_pool_addresses():
    manager = make_layout_manager()
    token = object()
    slots = torch.tensor([0, 2, -1, 8], dtype=torch.int64)
    pools = [(torch.empty(8, 8, dtype=torch.bfloat16), torch.empty(8, 2, dtype=torch.bfloat16)) for _ in range(2)]
    for key_pool, rope_pool in pools:
        check_layout_writeback(manager, token, slots, key_pool, rope_pool)


@pytest.mark.parametrize("new_token", [True, False])
def test_layout_new_capture_or_missing_scope_refreshes_slot_values(new_token):
    manager = make_layout_manager()
    slots = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
    key_pool, rope_pool = torch.empty(8, 8, dtype=torch.bfloat16), torch.empty(8, 2, dtype=torch.bfloat16)
    check_layout_writeback(manager, object(), slots, key_pool, rope_pool)
    slots.copy_(torch.tensor([1, 3, -1, 8]))
    check_layout_writeback(manager, object() if new_token else None, slots, key_pool, rope_pool)


def test_layout_capacity_change_rechecks_validity():
    manager = make_layout_manager()
    slots = torch.tensor([0, 2, 7, 15], dtype=torch.int64)
    token = object()
    for capacity in (16, 8):
        check_layout_writeback(
            manager,
            token,
            slots,
            torch.empty(capacity, 8, dtype=torch.bfloat16),
            torch.empty(capacity, 2, dtype=torch.bfloat16),
        )


def test_layout_stream_change_builds_its_own_dependencies():
    manager = make_layout_manager()
    slots = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
    key_pool, rope_pool = torch.empty(8, 8, dtype=torch.bfloat16), torch.empty(8, 2, dtype=torch.bfloat16)
    token = object()
    check_layout_writeback(manager, token, slots, key_pool, rope_pool, stream_id=0)
    slots.copy_(torch.tensor([1, 3, 5, 7]))
    check_layout_writeback(manager, token, slots, key_pool, rope_pool, stream_id=1)
