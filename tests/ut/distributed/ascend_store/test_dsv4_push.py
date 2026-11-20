# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vllm.config import KVTransferConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_scheduler import LayerwiseKVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import BaseKVPoolScheduler
from vllm_ascend.models.deepseek_v4.cache_reuse import uses_dspark_push
from vllm_ascend.models.deepseek_v4.cache_transfer import select_window_pages


@pytest.mark.parametrize("rotating,expected", [(True, 4096), (False, 8192)])
def test_dsv4_prefix_lookup_probes_resumable_boundary_without_speculation(rotating, expected):
    scheduler = object.__new__(LayerwiseKVPoolScheduler)
    scheduler.layerwise_offload = rotating
    scheduler.kv_cache_config = SimpleNamespace(dsv4_cache_format="dsv4")
    scheduler.use_eagle = False
    scheduler.cache_transfer_granularity = 4096
    scheduler.cacheable_group_ids = {0}
    scheduler.original_block_size = [128, 32]
    scheduler.cache_coordinator = object()
    scheduler._get_or_create_request_tracker = Mock()
    scheduler._lookup_layerwise_with_coordinator = Mock(return_value=expected)
    request = SimpleNamespace(request_id="request", num_tokens=8192)
    assert scheduler._get_layerwise_hit_tokens(request, 8192, 0) == expected
    scheduler._lookup_layerwise_with_coordinator.assert_called_once_with(request, expected, 0)


@pytest.mark.parametrize("rotating,expected", [(True, 65), (False, 66)])
def test_dsv4_save_metadata_stops_before_verifier_only_rows(monkeypatch, rotating, expected):
    scheduler = object.__new__(LayerwiseKVPoolScheduler)
    scheduler.layerwise_offload = rotating
    scheduler.kv_cache_config = SimpleNamespace(dsv4_cache_format="dsv4")
    scheduler._unfinished_requests = {"request": (SimpleNamespace(all_token_ids=list(range(65))),)}
    monkeypatch.setattr(BaseKVPoolScheduler, "_build_req_meta", lambda self, tracker: tracker.token_len)
    assert scheduler._build_req_meta(SimpleNamespace(req_id="request", token_len=66)) == expected


@pytest.mark.parametrize(
    "name,extra,role,expected",
    [
        ("MooncakeConnectorV2", {"use_layerwise": True}, "kv_producer", True),
        ("MooncakeConnectorV2", {}, "kv_producer", False),
        ("MooncakePushConnector", {}, "kv_producer", True),
        ("MooncakePushConnector", {}, "kv_consumer", False),
        ("AscendStoreConnector", {"use_layerwise": True}, "kv_producer", False),
    ],
)
def test_dspark_uses_actual_push_producer(name, extra, role, expected):
    child = dict(kv_connector=name, kv_role=role, kv_connector_extra_config=extra)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MultiConnector", kv_role="kv_both", kv_connector_extra_config={"connectors": [child]}
        ),
    )
    assert uses_dspark_push(config) is expected
    config.kv_transfer_config = KVTransferConfig(**child)
    assert uses_dspark_push(config) is expected
    config.speculative_config.method = "mtp"
    assert not uses_dspark_push(config)


def test_previous_chunk_uses_just_its_window_and_separate_public_checkpoints():
    # W=8, two tokens per page, 64 prompt tokens already processed.
    import torch
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator

    coordinator = object.__new__(AscendStoreCoordinator)
    coordinator.group_effective_specs = [
        SlidingWindowSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.bfloat16, sliding_window=8),
    ]
    public = np.zeros(32, dtype=bool)
    public[[3, 8]] = True
    ids = np.arange(1, 33)
    ids[8] = 0
    window = coordinator.continuation_block_range(0, 64)
    live, shareable = select_window_pages(ids, window, public, public_end=16)
    assert np.flatnonzero(live).tolist() == [3, 28, 29, 30, 31]
    assert np.flatnonzero(shareable).tolist() == [3]
    restored, checkpoint = select_window_pages(ids, window)
    assert np.flatnonzero(restored).tolist() == [28, 29, 30, 31]
    assert checkpoint is None
    assert np.flatnonzero(public).tolist() == [3, 8]


@pytest.mark.parametrize(
    "rotating,model,dcp,target,selected",
    [
        (True, "deepseek_v4", 1, 9, True),
        (False, "deepseek_v4", 1, 9, False),
        (True, "glm5_next", 1, 9, False),
        (True, "deepseek_v4", 2, 9, False),
        (True, "deepseek_v4", 1, 11, False),
    ],
)
def test_gva_applies_model_window_only_to_rotating_prompt_groups(rotating, model, dcp, target, selected):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes.gva import GvaDataPlane

    layout = {"_layerwise_reuse_layout": SimpleNamespace(has_layer_reuse=rotating)}
    coordinator = Mock()
    coordinator.continuation_block_range.side_effect = lambda gid, end: (3, 5) if gid == 0 else None
    worker = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model),
        kv_cache_config=SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config=layout)),
        _reused_layers_by_group={0: {2}, 1: {3}},
        grouped_block_size=[2, 2, 2],
        cache_coordinator=coordinator,
        dcp_size=dcp,
    )
    checkpoint = np.array([True, False, False, False, False])
    masks = (checkpoint, None, checkpoint)
    meta = SimpleNamespace(
        block_ids_by_group_np=None,
        block_ids_by_group=[[3, 4, 5, 0, 7]] * 3,
        store_masks=masks,
        load_masks=masks,
        save_end_token=8,
        target_token_len=target,
        num_prompt_tokens=10,
    )
    plane = GvaDataPlane(worker)
    public = plane._apply_window_masks(meta, target, saving=True)
    plane._apply_window_masks(meta, target - 1, saving=False)
    if selected:
        assert set(public) == {0}
        assert np.flatnonzero(public[0]).tolist() == [0]
        assert np.flatnonzero(meta.store_masks[0]).tolist() == [0, 4]
        assert np.flatnonzero(meta.load_masks[0]).tolist() == [4]
        assert meta.store_masks[1] is None and meta.store_masks[2] is checkpoint
    else:
        assert not public and meta.store_masks is masks and meta.load_masks is masks
        coordinator.continuation_block_range.assert_not_called()


def test_shared_read_keeps_one_lease_through_final_draft_write():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.planes.gva import GvaDataPlane
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker

    worker = object.__new__(LayerwiseKVPoolWorker)
    worker._step_load_keys = set()
    worker.m_store = Mock()
    info = SimpleNamespace(size=lambda: 16, gva_list=lambda: [123])
    worker.m_store.batch_get_key_info.side_effect = lambda keys, **kw: [info] * len(keys)
    worker.m_store.batch_add_lease.side_effect = lambda keys, ttl: [0] * len(keys)
    worker.m_store.batch_remove_lease.return_value = 0
    worker.kv_recv_thread = None
    plane = GvaDataPlane(worker)
    for _ in range(2):
        assert plane._lease_load_gvas(["draft"]) == [123]
    assert worker.m_store.batch_add_lease.call_count == 1
    worker.release_step_load_leases()
    worker.m_store.batch_remove_lease.assert_called_once_with(["draft"])


@pytest.mark.parametrize("peer_pcp,peer_dcp,mismatch", [(2, 1, False), (1, 1, True), (2, 2, True)])
def test_producer_uses_native_pcp_topology_check(peer_pcp, peer_dcp, mismatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import infer_dcp_mismatch_info

    assert (
        infer_dcp_mismatch_info("kv_producer", {"decode_pcp_size": peer_pcp, "decode_dcp_size": peer_dcp}, 1, 2)
        is mismatch
    )


@pytest.mark.parametrize("lose_draft_page", [False, True])
def test_cold_request_requires_both_registered_groups_from_ordinary_store(request, lose_draft_page):
    from tests.ut.distributed.ascend_store.test_pool_worker import TestKVPoolWorkerProcessLayerData
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec

    case = TestKVPoolWorkerProcessLayerData()
    request.addfinalizer(case.doCleanups)
    writer = case._make_gva_worker(2)
    reader = case._make_gva_worker(2)
    assert writer._snapshot_keys is not reader._snapshot_keys
    # Two ordinary groups, one for target history and one for a persistent draft.
    # The real GVA plane produces the keys here. An in-memory store emulates publication.
    allocated = {}

    def alloc(keys, sizes, ttl):
        for key in keys:
            allocated.setdefault(key, 1024 + 128 * len(allocated))
        return [allocated[key] for key in keys]

    store = writer.m_store
    reader.m_store = store
    store.batch_alloc.side_effect = alloc
    a = case._make_gva_request(2, can_save=True)
    a.req_id = "A"
    writer._data_plane._alloc_gvas_for_save([a])
    assert len(a.save_keys) == 2
    visible = {key: allocated[key] for key in a.save_keys}
    if lose_draft_page:
        visible.pop(a.save_keys[1])

    def query(keys, **kwargs):
        return [
            SimpleNamespace(size=lambda: 128, gva_list=lambda k=k: [visible[k]]) if k in visible else None for k in keys
        ]

    store.batch_get_key_info.side_effect = query
    store.batch_add_lease.side_effect = lambda keys, ttl: [0] * len(keys)
    b = case._make_gva_request(2, load_spec=LoadSpec(0, 16, True))
    b.req_id = "B"
    b.block_ids_by_group_np = [np.array([17]), np.array([29])]
    result = reader._data_plane._prepare_load_gvas([b])
    assert not reader._snapshot_keys
    if lose_draft_page:
        assert result is False and not b.load_spec.can_load
        assert reader.get_failed_requests() == {"B"}
    else:
        assert result is None and b.load_spec.can_load
        assert b.load_keys == a.save_keys
        assert [int(group[0]) for group in b.load_block_gvas_by_group_np] == [allocated[k] for k in a.save_keys]


def test_partial_keys_share_the_full_key_format_namespace():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend

    parallel = SimpleNamespace(pipeline_parallel_size=1, prefill_context_parallel_size=1)

    def keys(fmt):
        config = SimpleNamespace(kv_cache_groups=[], dsv4_cache_format=fmt)
        bound = MemcacheBackend.bind_layerwise_keys(
            vllm_config=SimpleNamespace(parallel_config=parallel),
            kv_cache_config=config,
            model_name="flash",
            use_hybrid=False,
            grouped_block_size=[32],
        )
        return bound.make_full_key(0, "hash", 0, 0), bound.make_partial_key("flash", "B", 0, 3, 8, 0)

    assert keys(None) == ("flash@hash@0", "flash@partial@B@0@3@8@0")
    for fmt in ("A", "B"):
        assert keys(fmt) == (f"flash@dsv4:{fmt}@hash@0", f"flash@dsv4:{fmt}@partial@B@0@3@8@0")


@pytest.mark.parametrize("rotating", [False, True])
def test_leases_end_after_the_last_reader_on_cancellation(monkeypatch, rotating):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import BaseKVPoolWorker

    worker = object.__new__(LayerwiseKVPoolWorker)
    worker.kv_cache_config = SimpleNamespace(dsv4_dspark_push=True)
    worker.layerwise_offload = rotating
    worker._step_load_keys = {"shared"}
    worker._snapshot_keys = {"cancelled": {}, "next": {}}
    worker.kv_recv_thread = None
    worker.use_block_key_layerwise = False
    worker.m_store = Mock()
    worker.m_store.batch_remove_lease.return_value = 0
    monkeypatch.setattr(BaseKVPoolWorker, "get_finished", lambda *a: (set(), set()))
    meta = SimpleNamespace(preempted_req_ids={"cancelled"})
    worker.get_finished(set(), meta)
    assert worker._snapshot_keys == {"next": {}}
    assert ("shared" in worker._step_load_keys) == rotating
    # V2 runs the late draft reader here before clearing metadata.
    store = object.__new__(AscendStoreConnector)
    store.connector_worker = worker
    store.clear_connector_metadata()
    store.clear_connector_metadata()
    worker.m_store.batch_remove_lease.assert_called_once_with(["shared"])


def test_late_reader_setup_failure_releases_only_confirmed_leases(request):
    from tests.ut.distributed.ascend_store.test_pool_worker import TestKVPoolWorkerProcessLayerData
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec

    case = TestKVPoolWorkerProcessLayerData()
    request.addfinalizer(case.doCleanups)
    worker = case._make_gva_worker(2)
    worker.kv_cache_config = SimpleNamespace(dsv4_dspark_push=True)
    worker.layerwise_max_transfer_blocks = 1
    worker.kv_recv_thread = None
    store = worker.m_store
    info = SimpleNamespace(size=lambda: 16, gva_list=lambda: [123])
    store.batch_get_key_info.side_effect = [[info], [info], RuntimeError("second batch failed")]
    store.batch_add_lease.return_value = [0]
    store.batch_remove_lease.return_value = 0
    req = case._make_gva_request(2, load_spec=LoadSpec(0, 16, True), can_save=True)
    with pytest.raises(RuntimeError, match="second batch"):
        worker._data_plane.prepare_sessions([req])
    store.batch_remove_lease.assert_called_once_with(store.batch_add_lease.call_args.args[0])
    assert not worker._step_load_keys
    store.batch_alloc.assert_not_called()
    store.batch_write_finish.assert_not_called()
