# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.outputs import KVConnectorOutput

# Clean up stale mock modules installed by other kv offload tests that replace
# real kv_transfer packages with fake modules, breaking imports of this package.
_kv_xfer = "vllm_ascend.distributed.kv_transfer"
_vllm_kv_xfer = "vllm.distributed.kv_transfer"
_saved_modules: dict[str, types.ModuleType] = {}
_to_remove = []
for _module_name in list(sys.modules):
    if _module_name.startswith(_kv_xfer) or _module_name.startswith(_vllm_kv_xfer):
        _to_remove.append(_module_name)
for _module_name in _to_remove:
    _saved_modules[_module_name] = sys.modules.pop(_module_name)

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.manager import (  # noqa: E402
    PreemptedRequestState,
    RecomputeCPUOffloadScheduler,
    TransferMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.metadata import (  # noqa: E402
    INVALID_JOB_ID,
    RecomputeCPUOffloadMetadata,
    RecomputeCPUOffloadWorkerMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.recompute_cpu_offload_connector import (  # noqa: E402
    RecomputeCPUOffloadConnectorV1,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.worker import (  # noqa: E402
    RecomputeCPUOffloadWorker,
)

for _module_name, _module in _saved_modules.items():
    sys.modules[_module_name] = _module


def test_recompute_cpu_offload_worker_metadata_aggregate():
    metadata = RecomputeCPUOffloadWorkerMetadata(completed_store_events={1: 1, 2: 2})
    other = RecomputeCPUOffloadWorkerMetadata(completed_store_events={2: 3, 4: 1})

    merged = metadata.aggregate(other)

    assert isinstance(merged, RecomputeCPUOffloadWorkerMetadata)
    assert merged.completed_store_events == {1: 1, 2: 5, 4: 1}


def test_recompute_cpu_offload_metadata_defaults_are_empty():
    metadata = RecomputeCPUOffloadMetadata()

    assert metadata.need_flush is False
    assert metadata.preempt_store_event == INVALID_JOB_ID
    assert metadata.preempt_store_gpu_blocks == []
    assert metadata.preempt_store_cpu_blocks == []
    assert metadata.preempt_load_event == INVALID_JOB_ID
    assert metadata.preempt_load_gpu_blocks == []
    assert metadata.preempt_load_cpu_blocks == []
    assert metadata.preempt_load_event_to_reqs == {}


def test_recompute_cpu_offload_connector_scheduler_methods_forward():
    connector = RecomputeCPUOffloadConnectorV1.__new__(RecomputeCPUOffloadConnectorV1)
    scheduler_manager = MagicMock()
    scheduler_manager.get_num_new_matched_tokens.return_value = (8, True)
    scheduler_manager.update_state_before_preempt.return_value = True
    scheduler_manager.has_pending_transfers.return_value = True
    scheduler_manager.has_preempted_request.return_value = True
    connector.scheduler_manager = scheduler_manager

    request = SimpleNamespace(request_id="req-1")
    blocks = MagicMock()
    block_ids = ([1, 2],)

    assert connector.get_num_new_matched_tokens(request, 4) == (8, True)
    connector.update_state_after_alloc(request, blocks, 8)
    assert connector.update_state_before_preempt(request, block_ids, 16) is True
    assert connector.has_pending_transfers() is True
    assert connector.has_preempted_request("req-1") is True

    scheduler_manager.get_num_new_matched_tokens.assert_called_once_with(request, 4)
    scheduler_manager.update_state_after_alloc.assert_called_once_with(request, blocks, 8)
    scheduler_manager.update_state_before_preempt.assert_called_once_with(request, block_ids, 16)


def test_recompute_cpu_offload_connector_worker_methods_forward():
    connector = RecomputeCPUOffloadConnectorV1.__new__(RecomputeCPUOffloadConnectorV1)
    worker_handler = MagicMock()
    worker_handler.get_finished.return_value = (None, {"req-1"})
    worker_handler.build_connector_worker_meta.return_value = RecomputeCPUOffloadWorkerMetadata(
        completed_store_events={3: 1}
    )
    connector.worker_handler = worker_handler

    metadata = RecomputeCPUOffloadMetadata(preempt_load_event=3)
    connector.bind_connector_metadata(metadata)
    connector.handle_preemptions(metadata)
    connector.start_load_kv(MagicMock())
    connector.wait_for_layer_load("layer.0")

    assert connector.get_finished(set()) == (None, {"req-1"})
    assert connector.build_connector_worker_meta().completed_store_events == {3: 1}

    worker_handler.bind_connector_metadata.assert_called_once_with(metadata)
    worker_handler.handle_preemptions.assert_called_once_with(metadata)
    worker_handler.start_load_kv.assert_called_once_with()
    worker_handler.wait_for_layer_load.assert_called_once_with()


def test_recompute_cpu_offload_connector_defaults_without_scheduler_manager():
    connector = RecomputeCPUOffloadConnectorV1.__new__(RecomputeCPUOffloadConnectorV1)
    connector.scheduler_manager = None

    assert connector.get_num_new_matched_tokens(MagicMock(), 0) == (0, False)
    assert connector.update_state_before_preempt(MagicMock(), ([],), 1) is False
    assert isinstance(
        connector.build_connector_meta(MagicMock()),
        RecomputeCPUOffloadMetadata,
    )
    assert connector.request_finished(MagicMock(), []) == (False, None)
    assert connector.request_finished_all_groups(MagicMock(), ([],)) == (
        False,
        None,
    )
    assert connector.has_pending_transfers() is False
    assert connector.has_preempted_request("req-1") is False
    assert connector.take_events() == []
    assert connector.reset_cache() is None


def test_recompute_cpu_offload_scheduler_get_num_new_matched_tokens_states():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._preempted_req_states = {}
    scheduler._cleanup_preempt_cache_request = MagicMock()
    request = SimpleNamespace(request_id="req-1", num_tokens=10)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)

    scheduler._preempted_req_states["req-1"] = PreemptedRequestState(
        req_id="req-1",
        cpu_block_ids=([1],),
        num_computed_tokens=8,
        store_transfer_meta=TransferMeta([11], [1]),
        ready=False,
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)

    scheduler._preempted_req_states["req-1"].ready = True
    assert scheduler.get_num_new_matched_tokens(request, 3) == (5, True)
    assert scheduler._preempted_req_states["req-1"].load_start_tokens == 3

    assert scheduler.get_num_new_matched_tokens(request, 8) == (0, False)
    scheduler._cleanup_preempt_cache_request.assert_called_once_with("req-1")


def test_recompute_cpu_offload_scheduler_update_state_after_alloc_errors():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._prepare_preempt_load_after_alloc = MagicMock(return_value=False)
    request = SimpleNamespace(request_id="req-1")
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([1, 2],)

    scheduler.update_state_after_alloc(request, blocks, 0)
    scheduler._prepare_preempt_load_after_alloc.assert_not_called()

    try:
        scheduler.update_state_after_alloc(request, blocks, 2)
    except RuntimeError as exc:
        assert "Failed to prepare recompute H2D load" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when load mapping fails")

    scheduler._prepare_preempt_load_after_alloc.assert_called_once_with(request, ([1, 2],), 2)


def test_recompute_cpu_offload_scheduler_aligns_sliding_window_blocks():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._group_is_sliding_window = [True, False]

    assert scheduler._align_group_block_ids(0, [7, 8], 4) == [0, 0, 7, 8]
    assert scheduler._align_group_block_ids(0, [5, 6, 7, 8, 9], 4) == [
        5,
        6,
        7,
        8,
    ]
    assert scheduler._align_group_block_ids(1, [7, 8], 4) == [7, 8]
    assert scheduler._align_group_block_ids(0, [7, 8], 0) == []


def test_recompute_cpu_offload_scheduler_d2h_keeps_sliding_window_offsets():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._group_is_sliding_window = [True]
    scheduler.cpu_kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]
    )
    scheduler.enable_offload_prefix_caching = False
    scheduler._pending_hash_blocks = {}
    scheduler._gpu_block_pool = SimpleNamespace(
        blocks={
            20: SimpleNamespace(block_id=20, block_hash=None),
            21: SimpleNamespace(block_id=21, block_hash=None),
        },
        _maybe_evict_cached_block=MagicMock(),
    )
    cpu_blocks = [
        SimpleNamespace(block_id=101, _block_hash=None),
        SimpleNamespace(block_id=102, _block_hash=None),
    ]
    scheduler.cpu_block_pool = SimpleNamespace(
        get_num_free_blocks=MagicMock(return_value=8),
        get_new_blocks=MagicMock(return_value=cpu_blocks),
        cached_block_hash_to_block=SimpleNamespace(get_one_block=MagicMock(return_value=None)),
    )
    scheduler._preempted_req_states = {}

    assert scheduler._create_preempt_state("req-1", ([20, 21],), 64) is True

    state = scheduler._preempted_req_states["req-1"]
    assert state.cpu_block_ids == ([0, 0, 101, 102],)
    assert state.store_transfer_meta == TransferMeta([20, 21], [101, 102])
    assert state.ready is False


def test_recompute_cpu_offload_scheduler_h2d_skips_sliding_window_null_blocks():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._group_is_sliding_window = [True]
    scheduler.cpu_kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]
    )
    scheduler._gpu_block_pool = SimpleNamespace(blocks={30: "gpu30", 31: "gpu31"}, touch=MagicMock())
    scheduler._preempted_req_states = {
        "req-1": PreemptedRequestState(
            req_id="req-1",
            cpu_block_ids=([0, 0, 4, 5],),
            num_computed_tokens=64,
            store_transfer_meta=TransferMeta([20, 21], [4, 5]),
            load_start_tokens=0,
            ready=True,
        )
    }

    prepared = scheduler._prepare_preempt_load_after_alloc(
        SimpleNamespace(request_id="req-1"),
        ([30, 31],),
        num_external_tokens=64,
    )

    assert prepared is True
    state = scheduler._preempted_req_states["req-1"]
    assert state.load_transfer_meta == TransferMeta([30, 31], [4, 5])
    touched = list(scheduler._gpu_block_pool.touch.call_args.args[0])
    assert touched == ["gpu30", "gpu31"]


def test_recompute_cpu_offload_scheduler_h2d_clips_mtp_tail_blocks():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._group_is_sliding_window = [False]
    scheduler.cpu_kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))]
    )
    scheduler._gpu_block_pool = SimpleNamespace(
        blocks={10: "gpu10", 11: "gpu11", 12: "gpu12"},
        touch=MagicMock(),
    )
    scheduler._preempted_req_states = {
        "req-1": PreemptedRequestState(
            req_id="req-1",
            cpu_block_ids=([1, 2, 3, 4],),
            num_computed_tokens=64,
            store_transfer_meta=TransferMeta([20, 21, 22, 23], [1, 2, 3, 4]),
            load_start_tokens=0,
            ready=True,
        )
    }

    prepared = scheduler._prepare_preempt_load_after_alloc(
        SimpleNamespace(request_id="req-1"),
        ([10, 11, 12],),
        num_external_tokens=64,
    )

    assert prepared is True
    state = scheduler._preempted_req_states["req-1"]
    assert state.load_transfer_meta == TransferMeta([10, 11, 12], [1, 2, 3])


def test_recompute_cpu_offload_scheduler_build_connector_meta_assigns_events():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._store_event_counter = 4
    scheduler._load_event_counter = 7
    scheduler._preempt_store_event_to_blocks = {}
    scheduler._preempt_store_event_to_reqs = {}
    scheduler._preempt_load_event_to_reqs = {}
    scheduler._pending_hash_blocks = {"hash": MagicMock()}
    scheduler._preempted_req_states = {
        "store-req": PreemptedRequestState(
            req_id="store-req",
            cpu_block_ids=([2],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([10], [2]),
            ready=False,
        ),
        "load-req": PreemptedRequestState(
            req_id="load-req",
            cpu_block_ids=([3],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([], []),
            load_transfer_meta=TransferMeta([11], [3]),
            ready=True,
        ),
    }
    scheduler_output = SimpleNamespace(preempted_req_ids={"store-req"})

    metadata = scheduler.build_connector_meta(scheduler_output)

    assert metadata.need_flush is True
    assert metadata.preempt_store_event == 4
    assert metadata.preempt_store_gpu_blocks == [10]
    assert metadata.preempt_store_cpu_blocks == [2]
    assert metadata.preempt_load_event == 7
    assert metadata.preempt_load_gpu_blocks == [11]
    assert metadata.preempt_load_cpu_blocks == [3]
    assert metadata.preempt_load_event_to_reqs == {7: ["load-req"]}
    assert scheduler._preempted_req_states["store-req"].store_event == 4
    assert scheduler._preempted_req_states["load-req"].load_event == 7
    assert scheduler._pending_hash_blocks == {}


def test_recompute_cpu_offload_scheduler_update_connector_output_marks_store_ready():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._expected_worker_count = 2
    scheduler._store_event_pending_counts = {}
    scheduler._preempted_req_states = {}
    scheduler._process_preempt_store_event = MagicMock()
    output = KVConnectorOutput(
        finished_recving=set(),
        kv_connector_worker_meta=RecomputeCPUOffloadWorkerMetadata(completed_store_events={5: 1}),
    )

    scheduler.update_connector_output(output)

    assert scheduler._store_event_pending_counts == {5: 1}
    scheduler._process_preempt_store_event.assert_not_called()

    scheduler.update_connector_output(output)

    assert scheduler._store_event_pending_counts == {}
    scheduler._process_preempt_store_event.assert_called_once_with(5)


def test_recompute_cpu_offload_scheduler_request_finished_ready_and_pending():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._preempted_req_states = {
        "ready": PreemptedRequestState(
            req_id="ready",
            cpu_block_ids=([1],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([11], [1]),
            ready=True,
        ),
        "pending": PreemptedRequestState(
            req_id="pending",
            cpu_block_ids=([2],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([12], [2]),
            ready=False,
        ),
        "loading": PreemptedRequestState(
            req_id="loading",
            cpu_block_ids=([3],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([13], [3]),
            load_event=5,
            ready=True,
        ),
    }
    scheduler._cleanup_preempt_cache_request = MagicMock()

    assert scheduler.request_finished(SimpleNamespace(request_id="ready"), []) == (
        False,
        None,
    )
    assert scheduler.request_finished(SimpleNamespace(request_id="pending"), []) == (False, None)
    assert scheduler.request_finished(SimpleNamespace(request_id="loading"), []) == (False, None)

    scheduler._cleanup_preempt_cache_request.assert_called_once_with("ready")
    assert scheduler._preempted_req_states["pending"].finished is True
    assert scheduler._preempted_req_states["loading"].finished is False


def test_recompute_cpu_offload_scheduler_process_store_event_finishes_pending_req():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    cpu_block = MagicMock()
    cpu_block.block_hash = None
    scheduler.cpu_block_pool = SimpleNamespace(blocks={4: cpu_block})
    scheduler._preempt_store_event_to_blocks = {7: TransferMeta([1], [4])}
    scheduler._preempt_store_event_to_reqs = {7: ["req-1"]}
    scheduler._preempted_req_states = {
        "req-1": PreemptedRequestState(
            req_id="req-1",
            cpu_block_ids=([4],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([1], [4]),
            ready=False,
            finished=True,
        )
    }
    scheduler._cleanup_preempt_cache_request = MagicMock()

    scheduler._process_preempt_store_event(7)

    assert scheduler._preempted_req_states["req-1"].ready is True
    scheduler._cleanup_preempt_cache_request.assert_called_once_with("req-1")
    assert scheduler._preempt_store_event_to_blocks == {}
    assert scheduler._preempt_store_event_to_reqs == {}


def test_recompute_cpu_offload_scheduler_pending_and_reset_cache_paths():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._store_event_pending_counts = {}
    scheduler._preempt_store_event_to_blocks = {}
    scheduler._preempted_req_states = {}

    assert scheduler.has_pending_transfers() is False

    scheduler._preempted_req_states["not-ready"] = PreemptedRequestState(
        req_id="not-ready",
        cpu_block_ids=([1],),
        num_computed_tokens=8,
        store_transfer_meta=TransferMeta([11], [1]),
        ready=False,
    )
    assert scheduler.has_pending_transfers() is True

    scheduler._preempted_req_states.clear()
    scheduler._preempt_store_event_to_reqs = {"unused": []}
    scheduler._preempt_load_event_to_reqs = {1: ["req-1"]}
    scheduler._pending_hash_blocks = {"hash": MagicMock()}
    scheduler.cpu_block_pool = MagicMock()
    scheduler.cpu_block_pool.reset_prefix_cache.return_value = True
    scheduler._cleanup_preempt_cache_request = MagicMock()

    assert scheduler.reset_cache() is True
    scheduler.cpu_block_pool.reset_prefix_cache.assert_called_once_with()
    assert scheduler._preempt_store_event_to_reqs == {}
    assert scheduler._preempt_load_event_to_reqs == {}
    assert scheduler._pending_hash_blocks == {}


def test_recompute_cpu_offload_scheduler_cleanup_preempt_load_request():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._preempt_load_event_to_reqs = {2: ["req-1"]}
    scheduler._preempted_req_states = {
        "req-1": PreemptedRequestState(
            req_id="req-1",
            cpu_block_ids=([4],),
            num_computed_tokens=8,
            store_transfer_meta=TransferMeta([1], [4]),
            load_event=2,
            load_transfer_meta=TransferMeta([10, 11], [4, 5]),
            ready=True,
        )
    }
    scheduler._gpu_block_pool = SimpleNamespace(
        blocks={10: "gpu10", 11: "gpu11"},
        free_blocks=MagicMock(),
    )
    scheduler._cleanup_preempt_cache_request = MagicMock()

    scheduler._cleanup_preempt_load_request("req-1")

    assert scheduler._preempt_load_event_to_reqs == {}
    freed = list(scheduler._gpu_block_pool.free_blocks.call_args.args[0])
    assert freed == ["gpu10", "gpu11"]
    scheduler._cleanup_preempt_cache_request.assert_called_once_with("req-1")


def test_recompute_cpu_offload_scheduler_cleanup_skips_null_cpu_blocks():
    scheduler = RecomputeCPUOffloadScheduler.__new__(RecomputeCPUOffloadScheduler)
    scheduler._preempted_req_states = {
        "req-1": PreemptedRequestState(
            req_id="req-1",
            cpu_block_ids=([0, 4], [0]),
            num_computed_tokens=32,
            store_transfer_meta=TransferMeta([10], [4]),
            ready=True,
        )
    }
    scheduler.cpu_block_pool = SimpleNamespace(blocks={4: "cpu4"}, free_blocks=MagicMock())

    scheduler._cleanup_preempt_cache_request("req-1")

    freed = list(scheduler.cpu_block_pool.free_blocks.call_args.args[0])
    assert freed == ["cpu4"]


def test_recompute_cpu_offload_worker_metadata_and_empty_transfers():
    worker = RecomputeCPUOffloadWorker.__new__(RecomputeCPUOffloadWorker)
    worker._connector_metadata = None
    worker._pending_load_event_indices = set()
    worker._submitted_load_event_indices = set()
    worker._completed_store_events = {}
    worker._load_events = []
    worker._load_hwm = -1
    worker.load_stream = None
    worker._load_stream_waited = False

    metadata = RecomputeCPUOffloadMetadata(
        preempt_store_event=1,
        preempt_load_event=2,
        preempt_load_event_to_reqs={2: ["req-1"]},
    )
    worker.bind_connector_metadata(metadata)
    assert worker._connector_metadata is metadata
    assert worker._pending_load_event_indices == {2}

    worker._submit_transfer([], [], 1, is_store=True)
    assert worker.build_connector_worker_meta().completed_store_events == {1: 1}
    assert worker.build_connector_worker_meta() is None

    worker._submit_transfer([], [], 2, is_store=False)
    assert worker.get_finished(set()) == (None, {"req-1"})
    assert worker.get_finished(set()) == (None, None)

    worker.clear_connector_metadata()
    assert worker._connector_metadata is None


def test_recompute_cpu_offload_worker_preempt_and_load_entrypoints():
    worker = RecomputeCPUOffloadWorker.__new__(RecomputeCPUOffloadWorker)
    worker._submit_transfer = MagicMock()
    worker._flush_and_sync_all = MagicMock()
    worker._connector_metadata = None
    metadata = RecomputeCPUOffloadMetadata(
        need_flush=True,
        preempt_store_event=3,
        preempt_store_gpu_blocks=[1],
        preempt_store_cpu_blocks=[2],
        preempt_load_event=4,
        preempt_load_gpu_blocks=[5],
        preempt_load_cpu_blocks=[6],
    )

    worker.handle_preemptions(metadata)

    worker._flush_and_sync_all.assert_called_once_with()
    worker._submit_transfer.assert_called_once_with(
        [1],
        [2],
        3,
        is_store=True,
        sync=True,
    )

    worker._submit_transfer.reset_mock()
    worker.start_load_kv()
    worker._submit_transfer.assert_not_called()

    worker._connector_metadata = metadata
    worker.start_load_kv()
    worker._submit_transfer.assert_called_once_with(
        [6],
        [5],
        4,
        is_store=False,
        sync=True,
    )


def test_recompute_cpu_offload_worker_wait_for_layer_load_once():
    worker = RecomputeCPUOffloadWorker.__new__(RecomputeCPUOffloadWorker)
    stream = MagicMock()
    current_stream = MagicMock()
    worker.load_stream = stream
    worker._connector_metadata = RecomputeCPUOffloadMetadata(preempt_load_event=1)
    worker._load_stream_waited = False

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.worker.torch.npu.current_stream",
        return_value=current_stream,
    ):
        worker.wait_for_layer_load()
        worker.wait_for_layer_load()

    current_stream.wait_stream.assert_called_once_with(stream)
    assert worker._load_stream_waited is True


def test_recompute_scheduler_remote_kv_restore_keeps_exact_token_position():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = MagicMock()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = {"req-1"}
    scheduler.kv_cache_manager = MagicMock()

    request = SimpleNamespace(
        request_id="req-1",
        num_computed_tokens=9,
        num_tokens=9,
        num_preemptions=1,
        spec_token_ids=[],
    )

    scheduler._update_waiting_for_remote_kv(request)

    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(request, 8)
    assert request.num_computed_tokens == 8
    assert request.spec_token_ids == []
    assert scheduler.finished_recving_kv_req_ids == set()


def test_recompute_scheduler_remote_kv_restore_frees_failed_empty_load():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = MagicMock()
    scheduler.failed_recving_kv_req_ids = {"req-1"}
    scheduler.finished_recving_kv_req_ids = {"req-1"}
    scheduler.kv_cache_manager = MagicMock()

    request = SimpleNamespace(
        request_id="req-1",
        num_computed_tokens=0,
    )

    scheduler._update_waiting_for_remote_kv(request)

    scheduler.kv_cache_manager.free.assert_called_once_with(request)
    scheduler.kv_cache_manager.cache_blocks.assert_not_called()
    assert scheduler.failed_recving_kv_req_ids == set()
    assert scheduler.finished_recving_kv_req_ids == set()
