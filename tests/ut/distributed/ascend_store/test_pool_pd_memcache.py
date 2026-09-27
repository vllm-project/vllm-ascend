# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_backend import TestMemcacheBackendMethods as _BackendFixture
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import BatchResultShapeError
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pd_transfer import PoolPDTransfer


@pytest.fixture
def worker():
    patches = unittest.TestCase()
    result = make_worker(patches, extra_config={"backend": "memcache", "pool_pd": True}, use_layerwise=True)
    result.layerwise_offload = False
    result.num_kv_cache_groups = 1
    result.grouped_block_size = [16]
    result.kv_cache_group_families = ["default"]
    result.group_block_len = {0: [64]}
    result.group_num_layers = {0: 2}
    result.hash_block_size = 16
    result.page_size_bytes = 64
    result.head_or_tp_rank = 0
    result._allocated_gvas = {}
    result.m_store = MagicMock()
    result.m_store.batch_is_committed.side_effect = lambda keys: [False] * len(keys)
    result.m_store.batch_is_exist.side_effect = lambda keys: [1] * len(keys)
    try:
        yield result
    finally:
        patches.doCleanups()


def make_pd_request(length=20, decode=False):
    transfer = PoolPDTransfer.create(list(range(length)), 16, "layout", [f"{i + 1:04x}" for i in range(length // 16)])
    blocks = list(range(10, 10 + (length + 15) // 16))
    return ReqMeta(
        req_id="decode-local-id" if decode else "prefill-local-id",
        block_ids=blocks,
        block_ids_np=np.asarray(blocks, dtype=np.int64),
        block_ids_by_group_np=[np.asarray(blocks, dtype=np.int64)],
        block_hashes=[b"different-local-hash"] * (length // 16),
        token_len_chunk=length,
        save_start_token=0,
        save_end_token=length // 16 * 16,
        target_token_len=length,
        can_save=not decode,
        partial_block_index=length // 16 if length % 16 else None,
        pd_transfer=transfer.to_wire(),
        load_spec=LoadSpec(0, length - 1, can_load=True, kvpool_store_skip_tokens=length) if decode else None,
    )


def key_info(gva):
    return MagicMock(size=lambda: 128 if gva else 0, gva_list=lambda: [gva] if gva else [])


@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33])
def test_memcache_pd_loads_full_snapshot_from_producer_keys(worker, length):
    request = make_pd_request(length, decode=True)
    infos = [key_info(201 + index) for index in range((length + 15) // 16)]
    worker.m_store.batch_get_key_info.return_value = infos
    worker.m_store.batch_add_lease.side_effect = lambda keys, ttl: [0] * len(keys)
    worker._prepare_load_gvas([request])
    worker._process_load_for_layer_batch([request], 0)
    worker._process_load_for_layer_batch([request], 1)
    expected = [worker._make_layerwise_full_key(0, value) for value in request.pd_transfer["block_hashes"]]
    if length % 16:
        expected.append(worker._make_layerwise_full_key(0, request.pd_transfer["tail_id"]))
        assert request.partial_load_gva_per_group == [201 + length // 16]
    assert request.load_keys == expected
    for layer_tasks in worker.layer_load_tasks:
        ranges = layer_tasks[0].block_ranges
        assert ranges[0].partial_block_index == (length // 16 if length % 16 else None)
    assert request.load_block_gvas_by_group_np[0][: length // 16].tolist() == list(range(201, 201 + length // 16))


def test_memcache_pd_last_chunk_saves_tail_with_no_new_full_blocks(worker):
    request = make_pd_request()
    request.save_start_token = 16
    worker.m_store.batch_alloc.return_value = [101]
    worker._alloc_gvas_for_save([request])
    worker._process_save_for_layer_batch([request], 1)
    key = worker._make_layerwise_full_key(0, request.pd_transfer["tail_id"])
    assert request.save_keys == [key]
    assert request.partial_save_gva_per_group == [101]
    assert worker.layer_save_tasks[1][0].block_ranges[0].partial_block_index == 1
    assert key not in worker._allocated_gvas


def test_producer_prefix_load_does_not_fetch_uncomputed_prompt_tail(worker):
    request = make_pd_request(33)
    request.load_spec = LoadSpec(0, 16, can_load=True, kvpool_store_skip_tokens=16)
    worker.m_store.batch_get_key_info.return_value = [key_info(201)]
    worker.m_store.batch_add_lease.return_value = [0]
    worker._prepare_load_gvas([request])
    worker._process_load_for_layer_batch([request], 0)
    assert request.load_keys == [worker._make_layerwise_full_key(0, "0001")]
    assert worker.layer_load_tasks[0][0].block_ranges[0].partial_block_index is None


def test_memcache_pd_preserves_interior_committed_blocks(worker):
    request = make_pd_request(48)
    worker.m_store.batch_is_committed.side_effect = lambda keys: [False, True, False]
    worker.m_store.batch_alloc.return_value = [101, 103]
    worker._mask_readable_pd_blocks(request)
    worker._alloc_gvas_for_save([request])
    assert request.store_masks == ([True, False, True],)
    assert request.block_gvas_np.tolist() == [101, 0, 103]
    allocated = worker.m_store.batch_alloc.call_args.args[0]
    assert allocated == [worker._make_layerwise_full_key(0, value) for value in ("0001", "0003")]


def test_memcache_pd_deduplicates_interior_writes_in_same_batch(worker):
    request = make_pd_request(48)
    worker._allocated_gvas[worker._make_layerwise_full_key(0, "0002")] = 102
    worker.m_store.batch_alloc.return_value = [101, 103]
    worker._alloc_gvas_for_save([request])
    assert request.block_gvas_np.tolist() == [101, 0, 103]


@pytest.mark.parametrize("length", [1, 16])
@pytest.mark.parametrize("result", [[], [0]])
def test_memcache_pd_allocation_failure_stops_save(worker, length, result):
    worker.m_store.batch_alloc.return_value = result
    with pytest.raises(RuntimeError, match="allocation failed"):
        worker._alloc_gvas_for_save([make_pd_request(length)])


def test_memcache_pd_failed_lease_releases_successes_and_stops_load(worker):
    request = make_pd_request(32, decode=True)
    worker.m_store.batch_get_key_info.return_value = [key_info(201), key_info(202)]
    worker.m_store.batch_add_lease.return_value = [0, -3101]
    with pytest.raises(RuntimeError, match="pool PD load failed"):
        worker._prepare_load_gvas([request])
    worker.m_store.batch_remove_lease.assert_called_once_with([worker._make_layerwise_full_key(0, "0001")])
    assert worker.get_block_ids_with_load_errors() == set()


def test_memcache_pd_metadata_mismatch_stops_before_acquiring_leases(worker):
    worker.m_store.batch_get_key_info.return_value = []
    with pytest.raises(RuntimeError, match="metadata result count mismatch"):
        worker._prepare_load_gvas([make_pd_request(decode=True)])
    worker.m_store.batch_add_lease.assert_not_called()


def test_memcache_allocated_metadata_is_not_committed_until_read_lease_succeeds():
    backend = _BackendFixture()._make_backend()
    backend.store.batch_get_key_info.return_value = [key_info(101), key_info(102), key_info(0)]
    backend.store.batch_add_lease.return_value = [-3101, 0]
    backend.store.batch_remove_lease.return_value = 0
    assert backend.batch_is_committed(["writing", "committed", "missing"]) == [False, True, False]
    backend.store.batch_remove_lease.assert_called_once_with(["committed"])


def test_memcache_bad_publication_probe_count_releases_known_leases():
    backend = _BackendFixture()._make_backend()
    backend.store.batch_get_key_info.return_value = [key_info(101), key_info(102)]
    backend.store.batch_add_lease.return_value = [0]
    backend.store.batch_remove_lease.return_value = 0
    with pytest.raises(BatchResultShapeError):
        backend.batch_is_committed(["a", "b"])
    backend.store.batch_remove_lease.assert_called_once_with(["a"])


def test_memcache_pd_requires_explicit_publication_sdk():
    with (
        patch.object(MemcacheBackend, "_setup_store", return_value=object()),
        pytest.raises(RuntimeError, match="requires MemCache batch_write_finish"),
    ):
        MemcacheBackend(MagicMock(), device_id=0, extra_config={"pool_pd": True})
