# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from vllm.config import KVTransferConfig

from vllm_ascend.models.deepseek_v4.cache_reuse import uses_dspark_push
from vllm_ascend.models.deepseek_v4.cache_transfer import prompt_window_masks


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


@pytest.mark.parametrize("method", [None, "mtp", "dspark"])
def test_previous_chunk_uses_just_its_window_and_separate_public_checkpoints(method):
    # W=8, two tokens per page, 64 prompt tokens already processed.
    import torch
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator

    coordinator = object.__new__(AscendStoreCoordinator)
    coordinator.group_effective_specs = [
        SlidingWindowSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.bfloat16, sliding_window=8),
    ]
    worker = SimpleNamespace(
        layerwise_offload=True,
        kv_cache_config=SimpleNamespace(dsv4_cache_format="test"),
        cache_coordinator=coordinator,
        dcp_size=1,
        _reused_layers_by_group={0: [0, 1]},
        num_kv_cache_groups=2,
        grouped_block_size=[2, 2],
        speculative_config=method,
    )
    public = np.zeros(32, dtype=bool)
    public[[3, 8]] = True
    ids = np.arange(1, 33)
    ids[8] = 0
    resident = np.ones(32, dtype=bool)
    req = SimpleNamespace(
        store_masks=(public, resident),
        load_masks=(public, resident),
        target_token_len=64,
        num_prompt_tokens=128,
        save_end_token=32,
        block_ids_by_group_np=[ids, ids],
    )
    public_pages = prompt_window_masks(worker, req, 64, saving=True)
    assert np.flatnonzero(req.store_masks[0]).tolist() == [3, 28, 29, 30, 31]
    assert not public_pages[0][28:].any()
    prompt_window_masks(worker, req, 64, saving=False)
    assert np.flatnonzero(req.load_masks[0]).tolist() == [28, 29, 30, 31]
    assert req.load_masks[1] is resident
    req.target_token_len = 129
    assert prompt_window_masks(worker, req, 128, saving=True) == {}


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


@pytest.mark.parametrize(
    "local_dcp,peer_dcp,replicated,mismatch",
    [(1, 1, True, False), (1, 1, False, True), (2, 2, True, True), (1, 2, True, True)],
)
def test_only_replicated_pcp_can_differ_between_peers(local_dcp, peer_dcp, replicated, mismatch):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import infer_dcp_mismatch_info

    assert (
        infer_dcp_mismatch_info(
            "kv_producer", {"decode_pcp_size": 1, "decode_dcp_size": peer_dcp}, local_dcp, 2, replicated_pcp=replicated
        )
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
