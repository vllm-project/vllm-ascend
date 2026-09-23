# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiKVConnectorMetadata
from vllm.v1.worker import kv_connector_model_runner_mixin as runner_mixin

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.connector import SfaRemoteD2HConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import (
    SfaPDConsumerMetadata,
    SfaPDProducerMetadata,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import (
    SFAPDRD2HConsumerWorker,
    SFAPDRD2HProducerWorker,
)
from vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine import BACKEND_MEMFABRIC


def _connector(worker, *, producer):
    connector = object.__new__(SfaRemoteD2HConnector)
    connector.connector_worker = worker
    connector.is_producer = producer
    connector.is_consumer = not producer
    connector._connector_metadata = None
    connector.get_finished = lambda _: (set(), set())
    connector.get_block_ids_with_load_errors = lambda: set()
    connector.get_kv_connector_stats = lambda: None
    connector.get_kv_connector_kv_cache_events = lambda: None
    connector.build_connector_worker_meta = lambda: None
    return connector


def _runner_connector(connector, multi):
    if not multi:
        return connector
    combined = object.__new__(AscendMultiConnector)
    combined._connectors = [connector]
    combined._extra_async_saves = {}
    combined._connector_metadata = None
    combined._configure_layerwise_reuse_completion()
    return combined


def _scheduler_output(metadata, *, multi, has_sync_loads):
    return SimpleNamespace(
        kv_connector_metadata=MultiKVConnectorMetadata((metadata,)) if multi else metadata,
        has_sync_kv_loads=has_sync_loads,
        finished_req_ids=set(),
    )


@pytest.mark.parametrize("has_sync_loads", [False, True])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("early_dispatch", [False, True])
@pytest.mark.parametrize("tp_size,tp_rank,remote_tp_size", [(8, 3, 8), (16, 7, 4)])
def test_producer_prepares_before_forward_and_preserves_target_state_for_mtp(
    has_sync_loads, multi, early_dispatch, tp_size, tp_rank, remote_tp_size
):
    # Use real worker setup and send-task construction, mocking only transport
    # and device events. Two target layers plus one MTP layer run twice.
    worker = object.__new__(SFAPDRD2HProducerWorker)
    worker._backend = BACKEND_MEMFABRIC
    worker.tp_size = tp_size
    worker.tp_rank = tp_rank
    worker.total_layers = 3
    worker.last_layer_idx = 2
    worker.current_layer = 0
    worker._pd_dispatched_layers = set()
    worker.use_mla = True
    worker.main_group_idx = 0
    worker.indexer_group_idx = 1
    worker.stage_layer_names = [f"model.layers.{i}.self_attn" for i in range(3)]
    worker.layer_metadata = {
        name: SimpleNamespace(tensor_group_idx=[0, 1], has_indexer=True) for name in worker.stage_layer_names
    }
    sender = MagicMock()
    sender.send_queue = queue.Queue()
    worker.kv_send_layer_thread = sender
    child = _connector(worker, producer=True)
    connector = _runner_connector(child, multi)
    attn_metadata = SimpleNamespace(reshape_cache_event=object())
    tp_ratio = tp_size // remote_tp_size
    mapped_port = 24000 + tp_rank // tp_ratio

    with (
        patch.object(runner_mixin, "get_kv_transfer_group", return_value=connector),
        patch.object(runner_mixin, "get_forward_context", return_value=SimpleNamespace(attn_metadata={})),
    ):
        for step in range(2):
            metadata = SfaPDProducerMetadata()
            req_id = f"request-{step}"
            metadata.add_new_req(
                req_id,
                local_block_ids=[[step + 1], [step + 2]],
                kv_transfer_params={
                    "remote_host": "decode-host",
                    "remote_port": 24000,
                    "remote_tp_size": remote_tp_size,
                },
                chunk_finish=True,
            )
            request = metadata.requests[req_id]
            output = _scheduler_output(metadata, multi=multi, has_sync_loads=has_sync_loads)

            def run_layer(layer_id, req_id=req_id):
                # Empty names exercise stage-local counter resolution, used
                # by the offload hooks. Each layer must enqueue only once.
                if early_dispatch:
                    connector.on_kv_cache_written("")
                connector.save_kv_layer("", None, attn_metadata)
                task = sender.send_queue.get_nowait()
                assert task.layer_idx == layer_id
                assert task.send_request[req_id].remote_port == mapped_port
                assert task.send_request[req_id].tp_ratio == tp_ratio
                assert task.send_request[req_id].group_member_idx == tp_rank % tp_ratio
                assert sender.send_queue.empty()

            with runner_mixin.KVConnectorModelRunnerMixin._get_kv_connector_output(output, defer_finalize=True):
                assert request.remote_port == mapped_port
                assert worker.current_layer == 0
                assert worker._pd_dispatched_layers == set()
                run_layer(0)
                run_layer(1)

            # The deferred hook must neither rewind the counter/dedup set nor
            # offset the port again on the metadata retained by queued tasks.
            assert request.remote_port == mapped_port
            assert worker.current_layer == 2
            assert worker._pd_dispatched_layers == {0, 1}
            run_layer(2)
            assert worker.current_layer == 3
            assert worker._pd_dispatched_layers == {0, 1, 2}
            connector.clear_connector_metadata()
            assert not child.has_connector_metadata()
    assert sender.mark_layer_pending.call_count == 6
    assert sender.record_p_save_event.call_count == 6


@pytest.mark.parametrize("has_sync_loads", [False, True])
@pytest.mark.parametrize("multi", [False, True])
def test_consumer_destinations_still_follow_ordered_load_hook(has_sync_loads, multi):
    worker = object.__new__(SFAPDRD2HConsumerWorker)
    worker.request_map = {}
    worker._dest_blocks_by_req = {}
    worker._cpu_blocks_by_req = {}
    worker.nano_slots_by_req = {}
    worker._nano_tail_by_req = {}
    child = _connector(worker, producer=False)
    connector = _runner_connector(child, multi)
    metadata = SfaPDConsumerMetadata()
    req_id = "request-internal"
    metadata.add_request(req_id, [1], [2], pool_slot=3, tail_tokens=7, tail_block_index=4, kv_tokens=519)
    output = _scheduler_output(metadata, multi=multi, has_sync_loads=has_sync_loads)
    with (
        patch.object(worker, "start_load_kv", wraps=worker.start_load_kv) as start_load,
        patch.object(runner_mixin, "get_kv_transfer_group", return_value=connector),
        patch.object(runner_mixin, "get_forward_context", return_value=SimpleNamespace(attn_metadata={})),
    ):
        with runner_mixin.KVConnectorModelRunnerMixin._get_kv_connector_output(output):
            assert start_load.call_count == int(has_sync_loads)
            assert child.get_nano_slot_bindings() == ({req_id: 3} if has_sync_loads else {})
        start_load.assert_called_once_with(metadata)
    external_id = get_external_request_id(req_id)
    assert worker._dest_blocks_by_req[external_id] == ([1], [2])
    assert worker._nano_tail_by_req[external_id].tail_tokens == 7
    assert not child.has_connector_metadata()


def test_empty_producer_step_resets_dispatch_state_at_bind():
    worker = object.__new__(SFAPDRD2HProducerWorker)
    worker._backend = BACKEND_MEMFABRIC
    worker.current_layer = 3
    worker._pd_dispatched_layers = {0, 1, 2}
    connector = _connector(worker, producer=True)

    connector.bind_connector_metadata(SfaPDProducerMetadata())

    assert worker.current_layer == 0
    assert worker._pd_dispatched_layers == set()
    connector.start_load_kv(None)
    connector.clear_connector_metadata()
