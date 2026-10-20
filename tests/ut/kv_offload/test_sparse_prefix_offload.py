# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import SFAPDRD2HConsumerWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.sparse_scheduler import (
    SparsePrefixOffloadingScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.sparse_worker import (
    SparsePrefixOffloadingWorker,
    _select_tensors,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefix_connector import SfaPrefixOffloadingConnector


@pytest.mark.parametrize("blocks_per_chunk", [1, 2, 4])
@pytest.mark.parametrize("start", [0, 1, 3])
@pytest.mark.parametrize("host_first", [False, True])
def test_host_round_trip_preserves_bytes_and_strided_padding(blocks_per_chunk, start, host_first):
    # Two Host components in group 0; group 1 belongs to the indexer. Both
    # flattened transfer specs must advance over that group, even without refs.
    owner = [torch.arange(96, dtype=torch.int8).reshape(8, 12) + i for i in (0, 1)]
    live = [part[:, 2:8] for part in owner]
    originals = [part.clone() for part in owner]
    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(part, 6) for part in live],
        [[CanonicalKVCacheRef(0, 6), CanonicalKVCacheRef(1, 6)], []],
    )
    worker = SparsePrefixOffloadingWorker.__new__(SparsePrefixOffloadingWorker)
    worker._host = caches
    worker._blocks_per_chunk = blocks_per_chunk
    worker._cpu = [torch.full((8, 6 * blocks_per_chunk), -1, dtype=torch.int8) for _ in live]
    count = 3
    chunks = (start % blocks_per_chunk + count + blocks_per_chunk - 1) // blocks_per_chunk
    gpu = GPULoadStoreSpec([2, 4, 5, 7], [count, 1], [start, 0])
    cpu = CPULoadStoreSpec(list(range(chunks)) + [7])
    if not host_first:
        caches.group_data_refs.reverse()
        gpu = GPULoadStoreSpec([7, 2, 4, 5], [1, count], [0, start])
        cpu = CPULoadStoreSpec([7] + list(range(chunks)))
    ready = MagicMock()
    copied, _ = worker._copy_host(gpu, cpu, True, ready)
    ready.synchronize.assert_called_once()
    assert copied == count * 6 * 2
    for part in live:
        part[[2, 4, 5]] = 0
    copied, _ = worker._copy_host(cpu, gpu, False, None)
    assert copied == count * 6 * 2
    for actual, expected in zip(owner, originals):
        assert torch.equal(actual, expected)


def test_host_copy_rejects_inconsistent_group_geometry():
    worker = SparsePrefixOffloadingWorker.__new__(SparsePrefixOffloadingWorker)
    worker._host = CanonicalKVCaches([], [[]])
    worker._cpu = []
    worker._blocks_per_chunk = 2
    with pytest.raises(ValueError, match="block counts"):
        worker._copy_host(GPULoadStoreSpec([1], [1], [0]), CPULoadStoreSpec([1, 2]), True, None)


def test_device_selection_preserves_group_slots_and_remaps_refs():
    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(torch.zeros((2, 4), dtype=torch.int8), 4)],
        [[], [CanonicalKVCacheRef(0, 3)]],
    )
    selected = _select_tensors(caches, "cpu")
    assert selected.tensors[0] is caches.tensors[0]
    assert selected.group_data_refs == caches.group_data_refs
    assert _select_tensors(caches, "npu").group_data_refs == [[], []]


@pytest.mark.parametrize("host_first", [False, True])
def test_worker_completion_requires_both_copies(host_first):
    worker = SparsePrefixOffloadingWorker.__new__(SparsePrefixOffloadingWorker)
    host = Future()
    worker._host_jobs = {10: host}
    worker._device_results = {}
    worker._device = MagicMock()
    device = TransferResult(10, True, 20, 0.25)
    worker._device.get_finished.return_value = [] if host_first else [device]
    if host_first:
        host.set_result((12, 0.5))
    assert worker.get_finished() == []
    worker._device.get_finished.return_value = [device] if host_first else []
    if not host_first:
        host.set_result((12, 0.5))
    assert worker.get_finished() == [replace(device, transfer_size=32, transfer_time=0.5)]
    worker._device.get_finished.return_value = []
    assert worker.get_finished() == []


def test_failed_host_copy_cannot_publish_successful_device_copy():
    worker = SparsePrefixOffloadingWorker.__new__(SparsePrefixOffloadingWorker)
    future = Future()
    future.set_exception(RuntimeError("CPU copy failed"))
    worker._host_jobs = {1: future}
    worker._device_results = {}
    worker._device = MagicMock()
    worker._device.get_finished.return_value = [TransferResult(1, True)]
    with pytest.raises(RuntimeError, match="CPU copy failed"):
        worker.get_finished()


def _connector(paired=True):
    connector = SfaPrefixOffloadingConnector.__new__(SfaPrefixOffloadingConnector)
    connector._connectors = [MagicMock(), MagicMock()]
    for child in connector._connectors:
        child.get_transfer_results.return_value = KVConnectorTransferResults()
        child.get_block_ids_with_load_errors.return_value = set()
    worker = connector._connectors[0].connector_worker
    worker._dest_blocks_by_req = {"req": ([1, 2], [3, 4])}
    worker._gather_tp_read_status.side_effect = lambda done, failed: [(set(done), set(failed))]
    connector._receives = {"req": set()}
    connector._expected = {"req": {0, 1} if paired else {0}}
    connector._cancelled = set()
    connector._failed = set()
    connector._blocked_errors = {}
    connector._ready_errors = set()
    connector._extra_async_saves = {}
    return connector


@pytest.mark.parametrize("first", [0, 1])
def test_connector_joins_loads_across_steps(first):
    connector = _connector()
    connector._connectors[first].get_transfer_results.return_value = KVConnectorTransferResults(
        finished_recving={"req"}
    )
    assert not connector.get_transfer_results(set()).finished_recving
    connector._connectors[first].get_transfer_results.return_value = KVConnectorTransferResults()
    connector._connectors[1 - first].get_transfer_results.return_value = KVConnectorTransferResults(
        finished_recving={"req"}
    )
    assert connector.get_transfer_results(set()).finished_recving == {"req"}
    connector._connectors[0].connector_worker.finish_prefix_cache_loads.assert_called_with({"req"})
    assert not connector._receives


def test_main_host_completion_on_tp0_gates_other_ranks():
    connector = _connector()
    for child in connector._connectors:
        child.get_transfer_results.return_value = KVConnectorTransferResults(finished_recving={"req"})
    worker = connector._connectors[0].connector_worker
    worker._gather_tp_read_status.side_effect = lambda done, failed: [(done, failed), (set(), set())]
    assert not connector.get_transfer_results(set()).finished_recving
    worker.finish_prefix_cache_loads.assert_called_with(set())
    for child in connector._connectors:
        child.get_transfer_results.return_value = KVConnectorTransferResults()
    worker._gather_tp_read_status.side_effect = lambda done, failed: [(done, failed), (done, failed)]
    assert connector.get_transfer_results(set()).finished_recving == {"req"}


@pytest.mark.parametrize("paired", [False, True])
def test_cancelled_request_keeps_dma_destinations_until_loads_drain(paired):
    connector = _connector(paired)
    assert not connector.get_transfer_results({"req"}).finished_sending
    for child in connector._connectors:
        child.get_transfer_results.assert_called_once_with(set())
        child.get_transfer_results.return_value = KVConnectorTransferResults(finished_recving={"req"})
    result = connector.get_transfer_results(set())
    assert result.finished_recving == result.finished_sending == {"req"}
    connector._connectors[0].connector_worker.finish_prefix_cache_loads.assert_called_with(set())
    connector._connectors[0].connector_worker._cleanup_request_state.assert_called_once_with({"req"})
    for child in connector._connectors:
        assert child.get_transfer_results.call_count == 2


def test_failed_pd_load_does_not_release_blocks_while_indexer_restore_is_pending():
    connector = _connector()
    pd, native = connector._connectors
    pd.get_transfer_results.return_value = KVConnectorTransferResults(finished_recving={"req"})
    pd.get_block_ids_with_load_errors.return_value = {1, 3}
    assert not connector.get_transfer_results(set()).finished_recving
    assert not connector.get_block_ids_with_load_errors()
    pd.get_transfer_results.return_value = KVConnectorTransferResults()
    pd.get_block_ids_with_load_errors.return_value = set()
    native.get_transfer_results.return_value = KVConnectorTransferResults(finished_recving={"req"})
    result = connector.get_transfer_results(set())
    assert result.failed_recving == result.finished_recving == {"req"}
    assert connector.get_block_ids_with_load_errors() == {1, 3}
    assert not connector.get_block_ids_with_load_errors()
    connector._connectors[0].connector_worker.finish_prefix_cache_loads.assert_called_with(set())


def _config(prefix=None):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_consumer=True,
            is_kv_producer=False,
            kv_connector_extra_config={"prefix_cache": prefix or {"cpu_bytes_to_use": 4096}},
            engine_id="decode",
            kv_port=1234,
            kv_load_failure_policy="fail",
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        additional_config={"sparse_kv_offload_config": {"enabled": True}},
        speculative_config=None,
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"cpu_bytes_to_use": 0},
        {"cpu_bytes_to_use": 4096, "canonical_layout": True},
        {"cpu_bytes_to_use": 4096, "secondary_tiers": [{"type": "disk"}]},
        {"cpu_bytes_to_use": 4096, "spec_name": "TieringOffloadingSpec"},
    ],
)
def test_unsupported_prefix_pool_configs_fail_early(extra):
    with pytest.raises(ValueError):
        SfaPrefixOffloadingConnector._get_connector_classes_and_configs(_config(extra))


def test_child_config_isolation_and_existing_pd_options():
    config = _config()
    config.kv_transfer_config.kv_connector_extra_config["memfabric_port"] = 4321
    children = SfaPrefixOffloadingConnector._get_connector_classes_and_configs(config)
    assert len(children) == 2
    pd, native = (child.kv_transfer_config for _, child in children)
    assert pd.kv_role == "kv_consumer"
    assert pd.kv_connector_extra_config == {"memfabric_port": 4321}
    assert native.kv_role == "kv_both"
    assert native.kv_connector_extra_config == {"cpu_bytes_to_use": 4096, "sparse_prefix_cache": True}
    assert "prefix_cache" in config.kv_transfer_config.kv_connector_extra_config


@pytest.mark.parametrize("cached", [0, 256])
def test_lookup_and_allocation_coordinate_the_absolute_prefix_boundary(cached):
    connector = _connector()
    connector._prefix_matches = {}
    connector._prefix_hits = {}
    connector._requests_to_connector = {}
    connector._new_pairs = set()
    connector._new_receives = set()
    connector._scheduler_pending = set()
    pd, native = connector._connectors
    pd.connector_scheduler.prefix_cache_tokens = {}
    pd.get_num_new_matched_tokens.return_value = (896, True)
    native.get_num_new_matched_tokens.return_value = (cached, bool(cached))
    # A 128-token local APC hit is not yet committed to request at this hook.
    request = SimpleNamespace(request_id="req", num_computed_tokens=0)
    assert connector.get_num_new_matched_tokens(request, 128) == (896, True)
    blocks = object()
    connector.update_state_after_alloc(request, blocks, 896)
    native.update_state_after_alloc.assert_called_once_with(request, blocks, cached)
    pd.update_state_after_alloc.assert_called_once_with(request, blocks, 896)
    assert pd.connector_scheduler.prefix_cache_tokens == {"req": 128 + cached}
    assert connector._new_receives == connector._scheduler_pending == {"req"}
    assert connector._new_pairs == ({"req"} if cached else set())


def test_no_local_prefill_is_advertised_without_pd():
    connector = _connector()
    connector._connectors[0].get_num_new_matched_tokens.return_value = (0, False)
    connector._connectors[1].get_num_new_matched_tokens.return_value = (256, True)
    assert connector.get_num_new_matched_tokens(SimpleNamespace(request_id="req"), 0) == (0, False)


def test_dense_hot_restore_uses_the_completed_requests_absolute_prefix_once():
    worker = SFAPDRD2HConsumerWorker.__new__(SFAPDRD2HConsumerWorker)
    worker._prefix_hot_restores = {"req": (2, [7, 4], 192), "pending": (3, [8], 128)}
    worker._copy_sfa_block_size = 128
    worker.offload_manager = MagicMock()
    worker.finish_prefix_cache_loads({"req"})
    call = worker.offload_manager.dense_fill_copy_sfa_rows.call_args
    assert call.args == ({2: (0, 192)},)
    assert call.kwargs["block_size"] == 128
    assert call.kwargs["block_table"].tolist() == [[7, 4]]
    assert set(worker._prefix_hot_restores) == {"pending"}
    worker.finish_prefix_cache_loads({"req"})
    assert worker.offload_manager.dense_fill_copy_sfa_rows.call_count == 1


@pytest.mark.parametrize("status", [RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR])
def test_cancelled_or_failed_pd_prefix_is_not_a_store_candidate(status):
    scheduler = SparsePrefixOffloadingScheduler.__new__(SparsePrefixOffloadingScheduler)
    scheduler._req_status = {
        "bad": SimpleNamespace(req=SimpleNamespace(status=status)),
        "valid": SimpleNamespace(req=SimpleNamespace(status=RequestStatus.FINISHED_STOPPED)),
    }
    output = SimpleNamespace(finished_req_ids={"bad", "valid"}, num_scheduled_tokens={"bad": 1, "running": 1})
    filtered = scheduler._store_ready_output(output)
    assert filtered.finished_req_ids == {"valid"}
    assert filtered.num_scheduled_tokens == {"running": 1}
    # The outer upstream build still sees the original IDs and performs its
    # normal load/fence finalization. Only store admission uses the filtered copy.
    assert output.finished_req_ids == {"bad", "valid"}
