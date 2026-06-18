# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.manager import (
    PreemptedRequestState,
    RecomputeCPUOffloadScheduler,
    TransferMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.metadata import (
    INVALID_JOB_ID,
    RecomputeCPUOffloadMetadata,
    RecomputeCPUOffloadWorkerMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.recompute_cpu_offload_connector import (
    RecomputeCPUOffloadConnectorV1,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.recompute_cpu_offload.worker import (
    RecomputeCPUOffloadWorker,
)


def test_recompute_cpu_offload_worker_metadata_aggregate():
    metadata = RecomputeCPUOffloadWorkerMetadata(
        completed_store_events={1: 1, 2: 2}
    )
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
    connector = RecomputeCPUOffloadConnectorV1.__new__(
        RecomputeCPUOffloadConnectorV1
    )
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

    scheduler_manager.get_num_new_matched_tokens.assert_called_once_with(
        request, 4
    )
    scheduler_manager.update_state_after_alloc.assert_called_once_with(
        request, blocks, 8
    )
    scheduler_manager.update_state_before_preempt.assert_called_once_with(
        request, block_ids, 16
    )


def test_recompute_cpu_offload_connector_worker_methods_forward():
    connector = RecomputeCPUOffloadConnectorV1.__new__(
        RecomputeCPUOffloadConnectorV1
    )
    worker_handler = MagicMock()
    worker_handler.get_finished.return_value = (None, {"req-1"})
    worker_handler.build_connector_worker_meta.return_value = (
        RecomputeCPUOffloadWorkerMetadata(completed_store_events={3: 1})
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
    connector = RecomputeCPUOffloadConnectorV1.__new__(
        RecomputeCPUOffloadConnectorV1
    )
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

    scheduler._prepare_preempt_load_after_alloc.assert_called_once_with(
        request, ([1, 2],), 2
    )


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
        kv_connector_worker_meta=RecomputeCPUOffloadWorkerMetadata(
            completed_store_events={5: 1}
        ),
    )

    scheduler.update_connector_output(output)

    assert scheduler._store_event_pending_counts == {5: 1}
    scheduler._process_preempt_store_event.assert_not_called()

    scheduler.update_connector_output(output)

    assert scheduler._store_event_pending_counts == {}
    scheduler._process_preempt_store_event.assert_called_once_with(5)


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


def test_recompute_scheduler_remote_kv_restore_keeps_exact_token_position():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.connector = MagicMock()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = {"req-1"}
    scheduler.kv_cache_manager = MagicMock()
    scheduler.is_mtp_kv_consumer = True
    scheduler.num_spec_tokens = 2
    scheduler.max_model_len = 32

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
    assert request.spec_token_ids == [PLACEHOLDER_TOKEN_ID] * 2
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
