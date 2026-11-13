from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import threading

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    _SELECTIVE_NATIVE_TP_QUARANTINE,
    SelectiveNativeTPCoordinator,
    SparseKVOffloadManager,
)
from vllm_ascend.compilation.updatable_graph import _ACTIVE_GRAPH


def make_manager():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager._selective_native_graphs = {}
    manager._selective_native_graph_layers = {}
    manager._selective_native_graph_keys = {}
    manager._selective_native_failed = False
    manager._selective_native_inflight = False
    manager._selective_native_stopping = False
    manager.tp_size = 1
    manager._selective_native_groups = {}
    manager._selective_native_tp_groups = {}
    manager._selective_native_epoch = 0
    return manager


def test_activation_precedes_requests_and_sets_model_state():
    manager = make_manager()
    manager.tp_size = 1
    manager.topk = manager.topk_buffer_size = 4
    manager.use_fused_copy_sfa = False
    manager.max_model_len = 16
    manager.topk_buffers_k = [torch.empty((2, 8))]
    state = SimpleNamespace(_selective_request_ids={"active": 1}, _offload_live_req_ids={},
                            _selective_native_experiment=False)
    manager._selective_model_state = state
    with pytest.raises(RuntimeError, match="before admitting"):
        manager.enable_selective_native_experiment()
    assert not state._selective_native_experiment
    state._selective_request_ids.clear()
    state._offload_live_req_ids["active"] = 0
    with pytest.raises(RuntimeError, match="before admitting"):
        manager.enable_selective_native_experiment()
    state._offload_live_req_ids.clear()
    manager.enable_selective_native_experiment()
    assert state._selective_native_experiment


def test_tp_coordinator_shares_plan_and_stage_within_one_group():
    barrier = threading.Barrier(2)
    published = threading.Event()
    names = []
    coordinators = [None, None]
    errors = []
    thread_state = threading.local()
    statuses = [None, None]

    class Group:
        cpu_group = object()
        def broadcast_object(self, value, src):
            if value is not None:
                names.append(value)
                published.set()
            assert published.wait(10)
            return names[0]

        def barrier(self):
            barrier.wait(timeout=10)

    def create(rank):
        thread_state.rank = rank
        try:
            coordinators[rank] = SelectiveNativeTPCoordinator(
                Group(), rank, 2, 2, 4, 2, 4, 128, 128, [0, 1, 2]
            )
        except BaseException as error:
            errors.append(error)

    def gather(output, status, group):
        statuses[thread_state.rank] = status
        barrier.wait(timeout=10)
        output[:] = statuses
        barrier.wait(timeout=10)

    with (
        patch.object(torch.ops._C_ascend, "register_pinned_host_mmap", lambda owner: None, create=True),
        patch.object(torch.ops._C_ascend, "unregister_pinned_host_mmap", lambda owner: None, create=True),
        patch.object(torch.distributed, "all_gather_object", side_effect=gather),
    ):
        threads = [threading.Thread(target=create, args=(rank,)) for rank in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        assert not errors
        assert all(not thread.is_alive() for thread in threads)
        assert all(item is not None for item in coordinators)
        coordinators[0].buffers[0][0] = 17
        coordinators[0].stage_k[0, 0, 0] = 31
        assert int(coordinators[1].buffers[0][0]) == 17
        assert int(coordinators[1].stage_k[0, 0, 0]) == 31
        for item in coordinators:
            item.close()


def test_tp_coordinator_one_rank_registration_failure_is_shared():
    barrier = threading.Barrier(2)
    published = threading.Event()
    thread_state = threading.local()
    names, errors, unregistrations = [], [], []
    statuses = [None, None]

    class Group:
        cpu_group = object()

        def broadcast_object(self, value, src):
            if value is not None:
                names.append(value)
                published.set()
            assert published.wait(10)
            return names[0]

        def barrier(self):
            barrier.wait(timeout=10)

    def gather(output, status, group):
        statuses[thread_state.rank] = status
        barrier.wait(timeout=10)
        output[:] = statuses
        barrier.wait(timeout=10)

    def register(_owner):
        if thread_state.rank == 1:
            raise RuntimeError("injected registration failure")

    def unregister(_owner):
        unregistrations.append(thread_state.rank)

    def create(rank):
        thread_state.rank = rank
        try:
            SelectiveNativeTPCoordinator(Group(), rank, 2, 2, 4, 2, 4, 128, 128, [0, 1, 2])
        except BaseException as error:
            errors.append(error)

    initial = len(_SELECTIVE_NATIVE_TP_QUARANTINE)
    with (
        patch.object(torch.ops._C_ascend, "register_pinned_host_mmap", register, create=True),
        patch.object(torch.ops._C_ascend, "unregister_pinned_host_mmap", unregister, create=True),
        patch.object(torch.distributed, "all_gather_object", side_effect=gather),
    ):
        threads = [threading.Thread(target=create, args=(rank,)) for rank in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        assert all(not thread.is_alive() for thread in threads)
        assert len(errors) == 2 and all("registration failed" in str(error) for error in errors)
        assert unregistrations == [0]
        assert len(_SELECTIVE_NATIVE_TP_QUARANTINE) == initial + 1
        # The mock registration never touched a runtime; cleanup the isolated test mapping.
        quarantined = _SELECTIVE_NATIVE_TP_QUARANTINE.pop()
        quarantined.owner = None
        quarantined.mapping.close()


def test_missing_graph_binding_fails_before_start():
    manager = make_manager()
    graph = object()
    manager._selective_native_graphs[graph] = [11]
    manager._selective_native_graph_keys[graph] = [(1,)]
    with patch.object(torch.ops._C_ascend, "selective_native_start", create=True) as start:
        with pytest.raises(RuntimeError, match="binding"):
            manager.start_selective_native_graphs({"a": graph, "b": object()})
    start.assert_not_called()


def test_handle_lifecycle_uses_graph_object_and_orders_completion():
    manager = make_manager()
    graph = MagicMock()
    manager._selective_native_graphs[graph] = [11, 12]
    manager._selective_native_graph_keys[graph] = [(2,), (1,)]
    anchor = object()
    trace = MagicMock()
    with (
        patch.object(torch.ops._C_ascend, "selective_native_start", trace.start, create=True),
        patch.object(torch.ops._C_ascend, "selective_native_begin", trace.begin, create=True),
        patch.object(torch.ops._C_ascend, "selective_native_record_complete", trace.record, create=True),
        patch.object(torch.ops._C_ascend, "selective_native_retire", trace.retire, create=True),
        patch.object(torch.ops._C_ascend, "selective_native_stop", trace.stop, create=True),
        patch.object(torch.ops._C_ascend, "selective_native_release", trace.release, create=True),
    ):
        manager.start_selective_native_graphs({"a": graph})
        epoch = manager.begin_selective_native_graph(graph, anchor)
        manager.finish_selective_native_graph(graph, epoch, anchor)
        manager.stop_selective_native_graphs()
        manager.release_selective_native_graphs()
    assert [call[0] for call in trace.mock_calls] == [
        "start", "start", "begin", "begin", "record", "record", "retire", "retire", "stop", "stop",
        "release", "release",
    ]
    assert [call.args[0] for call in trace.retire.call_args_list] == [12, 11]
    graph.reset.assert_called_once()
    assert manager._selective_native_graphs == {}


def test_tp_missing_group_fails_before_worker_start():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(cpu_group=object())
    graph = object()
    manager._selective_native_graphs[graph] = [11, 12]
    manager._selective_native_graph_keys[graph] = [(2,), (1,)]

    def gather(output, local, group):
        output[:] = [local, (None, [(local[1][0][0], ((1,),))])]

    with (
        patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        patch.object(torch.ops._C_ascend, "selective_native_start", create=True) as start,
    ):
        with pytest.raises(RuntimeError, match="bindings differ"):
            manager.start_selective_native_graphs({"descriptor": graph})
    start.assert_not_called()


def test_tp_replay_descriptor_mismatch_is_rejected():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(cpu_group=object())

    def gather(output, local, group):
        output[:] = [local, ("different",)]

    with patch.object(torch.distributed, "all_gather_object", side_effect=gather):
        with pytest.raises(RuntimeError, match="descriptors differ"):
            manager.check_selective_native_graph_step(SimpleNamespace(num_tokens=1))


def test_tp_capture_failure_is_reported_to_successful_peer():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(cpu_group=object())

    def gather(output, local, group):
        output[:] = [local, "other rank capture failed"]

    with patch.object(torch.distributed, "all_gather_object", side_effect=gather):
        with pytest.raises(RuntimeError, match="other rank capture failed"):
            manager.check_selective_native_capture_status(None)


def test_tp_recapture_keeps_shared_mapping_until_shutdown():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(cpu_group=object())
    coordinator = SimpleNamespace(close=MagicMock())
    manager._selective_native_tp_groups[(0,)] = coordinator
    graph = MagicMock()
    manager._selective_native_graphs[graph] = [11]
    manager._selective_native_graph_keys[graph] = [(1,)]

    def gather(output, value, group):
        output[:] = [value, value]

    with (
        patch.object(torch.ops._C_ascend, "selective_native_stop", create=True),
        patch.object(torch.ops._C_ascend, "selective_native_release", create=True),
        patch.object(torch.distributed, "all_gather_object", side_effect=gather) as all_gather,
    ):
        manager.stop_selective_native_graphs()
        manager.release_selective_native_graphs()
        coordinator.close.assert_not_called()
        assert (0,) in manager._selective_native_tp_groups
        manager.release_selective_native_graphs(close_tp=True)
    coordinator.close.assert_called_once()
    assert all_gather.call_count == 5


def test_tp_stop_failure_is_shared_and_resources_remain_owned():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(cpu_group=object())
    graph = MagicMock()
    manager._selective_native_graphs[graph] = [11]
    manager._selective_native_graph_keys[graph] = [(1,)]
    calls = []

    def gather(output, value, group):
        calls.append(value)
        output[:] = [value, "peer stop failed"] if len(calls) == 2 else [value, value]

    with (patch.object(torch.distributed, "all_gather_object", side_effect=gather),
          patch.object(torch.ops._C_ascend, "selective_native_stop", create=True)):
        with pytest.raises(RuntimeError, match="worker stop failed"):
            manager.stop_selective_native_graphs()
    assert manager._selective_native_failed
    assert graph in manager._selective_native_graphs
    graph.reset.assert_not_called()


def test_completion_failure_does_not_retire_or_release_graph():
    manager = make_manager()
    graph = MagicMock()
    manager._selective_native_graphs[graph] = [11]
    manager._selective_native_graph_keys[graph] = [(1,)]
    with (
        patch.object(torch.ops._C_ascend, "selective_native_record_complete", side_effect=RuntimeError("event"),
                     create=True),
        patch.object(torch.ops._C_ascend, "selective_native_retire", create=True) as retire,
    ):
        with pytest.raises(RuntimeError, match="event"):
            manager.finish_selective_native_graph(graph, 1, object())
    retire.assert_not_called()
    graph.reset.assert_not_called()
    assert graph in manager._selective_native_graphs


def test_modified_pages_select_only_live_resident_identities():
    manager = make_manager()
    resident = torch.tensor([[[11, -1, 12, -1]], [[5, 6, -1, -1]], [[11, -1, -1, -1]]], dtype=torch.int32)
    identity = torch.full((3, 9), -1, dtype=torch.int64)
    identity[:, 0] = torch.tensor([101, 102, 103])
    manager._selective_native_groups[(0,)] = [None, None, resident, identity]
    assert manager.selective_native_cache_ids_for_modified_pages({11}, {101, 102}) == {101}
    assert manager.selective_native_cache_ids_for_modified_pages({99}, {101, 102}) == set()
    manager._selective_native_inflight = True
    with pytest.raises(RuntimeError, match="in flight"):
        manager.selective_native_cache_ids_for_modified_pages({11}, {101, 102})


@pytest.mark.parametrize("num_layers", [1, 2])
def test_two_graphs_share_group_state_but_not_staging(num_layers):
    manager = make_manager()
    manager._selective_native_experiment = True
    manager.tp_size = 1
    manager.tp_rank = 0
    manager.kv_cache_config = SimpleNamespace(num_blocks=8)
    manager.topk = manager.topk_buffer_size = 4
    manager.block_size = 4
    manager.max_model_len = 16
    manager.selective_row_stride = 8
    manager.token_size_bytes_k = manager.token_size_bytes_v = 2
    manager.k_caches_cpu = [torch.empty((8, 4, 1, 1), dtype=torch.bfloat16)]
    manager.v_caches_cpu = [torch.empty((8, 4, 1, 1), dtype=torch.bfloat16)]
    manager.topk_buffers_k = [torch.empty((2, 8, 1, 1), dtype=torch.bfloat16)]
    manager.topk_buffers_v = [torch.empty((2, 8, 1, 1), dtype=torch.bfloat16)]
    if num_layers == 2:
        manager.k_caches_cpu.append(torch.empty_like(manager.k_caches_cpu[0]))
        manager.v_caches_cpu.append(torch.empty_like(manager.v_caches_cpu[0]))
        manager.topk_buffers_k.append(torch.empty_like(manager.topk_buffers_k[0]))
        manager.topk_buffers_v.append(torch.empty_like(manager.topk_buffers_v[0]))
    manager.selective_physical = torch.empty((num_layers, 2, 1, 4), dtype=torch.int32)
    manager._selective_native_rows = torch.arange(2, dtype=torch.int64)
    manager._selective_native_offsets = torch.arange(8, dtype=torch.int64)
    manager.selective_copy_counts = torch.tensor([8, 16], dtype=torch.int32)
    trace = []
    manager.selective_prefetch_ready = [SimpleNamespace(record=lambda: trace.append("ready0")),
                                        SimpleNamespace(record=lambda: trace.append("ready1"))][:num_layers]
    manager.offload_layer_names = ["model.layers.1.attn", "model.layers.2.attn"][:num_layers]
    manager.copy_sfa_kv = MagicMock(side_effect=lambda *args: trace.append("scatter"))
    metadata = SimpleNamespace(
        req_ids_tensor=torch.tensor([1 << 32, 0]),
        selective_tail_rows=torch.tensor([[-1, 0], [-1, 0]]),
        block_table=torch.ones((2, 4), dtype=torch.int32),
    )
    seq = torch.tensor([8, 0], dtype=torch.int32)
    topk = torch.tensor([[0, 1, 2, 3], [-1, -1, -1, -1]])
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    with (
        patch.object(torch, "empty", side_effect=cpu_empty),
        patch.object(torch.ops._C_ascend, "selective_group_native_create", side_effect=[11, 12], create=True) as create,
        patch.object(torch.ops._C_ascend, "selective_native_capture", create=True),
        patch.object(torch.ops._C_ascend, "selective_native_graph_identity", return_value=17, create=True),
        patch.object(torch.ops._C_ascend, "selective_native_scatter_done", create=True,
                     side_effect=lambda *args: trace.append("scatter_done")),
    ):
        for graph in (object(), object()):
            token = _ACTIVE_GRAPH.set(graph)
            try:
                manager._prepare_selective_native_kv([(i, metadata, seq) for i in range(num_layers)], topk)
            finally:
                _ACTIVE_GRAPH.reset(token)
    first, second = [call.args for call in create.call_args_list]
    assert all(first[0][i] is second[0][i] for i in range(8, 16))
    assert first[1][6] is not second[1][6]  # Host staging belongs to each graph.
    assert first[0][2][:, 0].tolist() == [1 << 32, 0]
    assert first[0][2][:, 4].tolist() == [8, 0]
    expected = (["scatter", "scatter_done", "ready0"] if num_layers == 1 else
                ["scatter", "ready0", "scatter", "scatter_done", "ready1"])
    assert trace == expected * 2


def test_tp_stop_collective_failure_keeps_graph_and_quarantines():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(cpu_group=object())
    graph = MagicMock()
    manager._selective_native_graphs[graph] = [11]
    with patch.object(torch.distributed, "all_gather_object", side_effect=RuntimeError("collective failed")):
        with pytest.raises(RuntimeError, match="collective failed"):
            manager.stop_selective_native_graphs()
    assert manager._selective_native_failed
    assert graph in manager._selective_native_graphs
    graph.reset.assert_not_called()


def test_tp_graph_drop_collective_failure_keeps_graph_and_quarantines():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(cpu_group=object())
    graph = MagicMock()
    manager._selective_native_graphs[graph] = [11]
    with patch.object(torch.distributed, "all_gather_object", side_effect=RuntimeError("collective failed")):
        with pytest.raises(RuntimeError, match="collective failed"):
            manager.check_selective_native_phase_status("graph drop", None)
    assert manager._selective_native_failed
    assert graph in manager._selective_native_graphs
    graph.reset.assert_not_called()


def test_tp_coordinator_close_failure_is_shared_and_quarantined():
    manager = make_manager()
    manager.tp_size = 2
    manager.tp_group = SimpleNamespace(cpu_group=object())
    coordinator = SimpleNamespace(close=MagicMock())
    manager._selective_native_tp_groups[(0,)] = coordinator
    calls = []
    def gather(output, value, group):
        calls.append(value)
        output[:] = [value, "peer close failed"] if len(calls) == 2 else [value, value]
    with patch.object(torch.distributed, "all_gather_object", side_effect=gather):
        with pytest.raises(RuntimeError, match="coordinator close failed"):
            manager.release_selective_native_graphs(close_tp=True)
    assert manager._selective_native_failed
    assert manager._selective_native_tp_groups[(0,)] is coordinator


def test_shared_exchange_uses_exact_payload_and_confirms():
    manager = make_manager()
    manager.tp_size, manager.tp_rank = 2, 0
    manager._selective_native_shared_exchange_requested = True
    manager._selective_tp_exchange_epoch = 0
    storage = MagicMock()
    manager._selective_native_tp_groups[(0,)] = SimpleNamespace(exchange=storage)
    def exchange(buf, rank, world, epoch, payload):
        assert buf is storage and (rank, world, epoch) == (0, 2, 1)
        return [payload, payload]
    with (patch.object(torch.ops._C_ascend, "selective_tp_exchange_cpu", side_effect=exchange, create=True),
          patch.object(torch.ops._C_ascend, "selective_tp_exchange_confirm_cpu", create=True) as confirm,
          patch.object(torch.distributed, "all_gather_object") as old):
        assert manager.selective_native_all_gather(((7, 9), ("request",)), "pages") == [
            ((7, 9), ("request",)), ((7, 9), ("request",))
        ]
    confirm.assert_called_once_with(storage, 0, 2, 1)
    old.assert_not_called()


def test_shared_exchange_oversize_uses_existing_collective():
    manager = make_manager()
    manager.tp_size, manager.tp_rank = 2, 0
    manager.tp_group = SimpleNamespace(cpu_group=object())
    manager._selective_native_shared_exchange_requested = True
    manager._selective_tp_exchange_epoch = 0
    storage = MagicMock()
    manager._selective_native_tp_groups[(0,)] = SimpleNamespace(exchange=storage)
    def gather(output, value, group):
        output[:] = [value, value]
    with (patch.object(torch.ops._C_ascend, "selective_tp_exchange_cpu", return_value=[], create=True),
          patch.object(torch.ops._C_ascend, "selective_tp_exchange_confirm_cpu", create=True) as confirm,
          patch.object(torch.distributed, "all_gather_object", side_effect=gather)):
        assert manager.selective_native_all_gather([1, 2], "pages") == [[1, 2], [1, 2]]
    confirm.assert_called_once_with(storage, 0, 2, 1)


def test_shared_failure_marks_exchange_and_every_group_control():
    manager = make_manager()
    manager.tp_size = 2
    manager._selective_native_shared_exchange_requested = True
    exchange = torch.zeros(8, dtype=torch.uint8)
    control1, control2 = torch.zeros(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64)
    manager._selective_native_tp_groups = {
        (0,): SimpleNamespace(exchange=exchange, buffers=[control1]),
        (1,): SimpleNamespace(exchange=None, buffers=[control2]),
    }
    with patch.object(torch.ops._C_ascend, "selective_tp_exchange_fail_cpu", create=True) as mark:
        manager.mark_selective_native_failure()
    assert manager._selective_native_failed
    assert mark.call_count == 3
    pointers = [call.args[0].data_ptr() for call in mark.call_args_list]
    assert pointers == [exchange.data_ptr(), control1.data_ptr(), control2.data_ptr()]
