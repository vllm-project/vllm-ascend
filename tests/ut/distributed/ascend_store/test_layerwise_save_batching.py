# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real group planning/scatter with a key-addressable store."""

from unittest.mock import Mock

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.test_dsv4_layerwise_runtime import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta


def make_case(req_id="r", groups=6):
    worker = make_worker()
    worker._failed_requests = set()
    worker.num_kv_cache_groups = groups
    worker.grouped_block_size = [128] * groups
    worker.group_uses_align_state = [group % 2 == 1 for group in range(groups)]
    worker.cacheable_group_ids = set(range(groups))
    worker._global_group_alloc_size = lambda group: (group + 1) * 4096
    worker._make_layerwise_full_key = lambda group, key: f"g{group}:public:{key}"
    worker._make_layerwise_partial_key = lambda r, group, block, end: f"g{group}:{r.req_id}:{block}:{end}"
    worker.m_store.batch_alloc = Mock(wraps=worker.m_store.batch_alloc)
    worker.m_store.batch_is_exist = Mock(wraps=worker.m_store.batch_is_exist)
    request = ReqMeta(
        req_id,
        target_token_len=257,
        save_end_token=256,
        block_ids_by_group_np=[np.array([0, 20 + group, 30 + group]) for group in range(groups)],
        block_hashes=["h0", "h1"],
        can_save=True,
    )
    request.store_masks = [[False, True, True] for _ in range(groups)]
    return worker, request


@pytest.mark.parametrize("limit", [0, 1, 4, 1024])
def test_batching_preserves_sizes_masks_private_public_and_partial_addresses(limit):
    worker, request = make_case()
    worker.layerwise_max_transfer_blocks = limit
    worker._alloc_gvas_for_save([request])
    assert request.can_save
    addresses = worker.m_store.addresses
    for group in range(6):
        public = f"g{group}:public:h1"
        full = public if group % 2 == 0 else f"g{group}:r:1:256@full@h1"
        partial = f"g{group}:r:2:257"
        assert request.block_gvas_by_group_np[group].tolist() == [0, addresses[full], 0]
        assert request.partial_save_gva_per_group[group] == addresses[partial]
        if group % 2:
            ids, gvas = request.public_save_blocks[group]
            assert ids.tolist() == [20 + group]
            assert gvas.tolist() == [addresses[public]]
    assert set(request.save_keys) == set(addresses)
    assert not any("h0" in key for key in addresses)
    for call in worker.m_store.batch_alloc.call_args_list:
        keys, sizes, _ = call.args
        assert sizes == [(int(key[1]) + 1) * 4096 for key in keys]
        assert limit == 0 or len(keys) <= limit
    if limit in (0, 1024):
        # Six groups: two existence policies and one mixed-size allocation.
        assert worker.m_store.batch_is_exist.call_count == 2
        assert worker.m_store.batch_alloc.call_count == 1


def test_public_target_has_one_writer_but_private_snapshots_stay_per_request():
    worker, first = make_case("first")
    _, second = make_case("second")
    worker._alloc_gvas_for_save([first, second])
    assert not set(first.save_keys).intersection(second.save_keys)
    assert all(":public:" not in key for key in second.save_keys)
    assert set(first.public_save_blocks) == {1, 3, 5}
    assert not second.public_save_blocks
    assert all(second.partial_save_gva_per_group)
    assert all(first.partial_save_gva_per_group[g] != second.partial_save_gva_per_group[g] for g in range(6))


@pytest.mark.parametrize("failure", ["exception", "malformed", "unknown"])
def test_exists_failure_never_overwrites_public_pages(failure):
    worker, request = make_case()
    if failure == "exception":
        worker.m_store.batch_is_exist.side_effect = RuntimeError("unreachable")
    else:
        worker.m_store.batch_is_exist.side_effect = lambda keys: [] if failure == "malformed" else [-1] * len(keys)
    worker._alloc_gvas_for_save([request])
    assert request.can_save
    assert not request.public_save_blocks
    assert all(":public:" not in key for key in request.save_keys)
    # Required execution snapshots still get allocated with fail-open checks.
    assert all(request.partial_save_gva_per_group)
    assert all(request.block_gvas_by_group_np[g][1] > 0 for g in (1, 3, 5))


@pytest.mark.parametrize("failure", ["zero", "negative", "short", "none", "public_ssd"])
def test_failed_mixed_batch_stops_saves_without_deleting_shared_keys(failure):
    worker, request = make_case()
    original = worker.m_store.batch_alloc

    def allocate(keys, sizes, ttl):
        values = original(keys, sizes, ttl)
        if failure == "none":
            return None
        if failure == "short":
            return values[:-1]
        if "g1:public:h1" not in keys:
            return values
        index = keys.index("g1:public:h1")
        values[index] = {"zero": 0, "negative": -1, "public_ssd": (1 << 64) - 1}[failure]
        return values

    worker.m_store.batch_alloc = allocate
    worker.m_store.batch_delete = Mock()
    worker._alloc_gvas_for_save([request])
    assert not request.can_save
    assert not request.save_keys
    assert worker._save_disabled.is_set()
    assert "r" not in worker._snapshot_keys
    worker.m_store.batch_delete.assert_not_called()


def test_committed_public_pages_are_skipped_and_addresses_reacquired_next_step():
    worker, first = make_case()
    worker._alloc_gvas_for_save([first])
    worker.m_store.committed.update(key for key in first.save_keys if ":public:" in key)
    before = dict(worker.m_store.addresses)
    worker.m_store.addresses = {key: address + 1000 for key, address in before.items()}
    _, second = make_case()
    worker._alloc_gvas_for_save([second])
    assert all(":public:" not in key for key in second.save_keys)
    assert second.partial_save_gva_per_group == [before[f"g{g}:r:2:257"] + 1000 for g in range(6)]
    assert not second.public_save_blocks


def test_non_owner_records_snapshot_but_allocates_nothing():
    worker, request = make_case()
    worker._is_layerwise_save_owner = lambda group: False
    worker._alloc_gvas_for_save([request])
    assert len(worker._snapshot_keys["r"]) == 6
    assert request.can_save
    assert not request.save_keys
    worker.m_store.batch_alloc.assert_not_called()
    worker.m_store.batch_is_exist.assert_not_called()


def test_different_token_block_sizes_scatter_to_their_own_group():
    worker, request = make_case(groups=2)
    worker.grouped_block_size = [128, 256]
    request.target_token_len = 513
    request.save_end_token = 512
    request.block_hashes = ["h0", "h1", "h2", "h3"]
    request.store_masks = None
    request.block_ids_by_group_np = [np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8])]
    worker._alloc_gvas_for_save([request])
    addresses = worker.m_store.addresses
    assert request.block_gvas_by_group_np[0].tolist() == [addresses[f"g0:public:h{i}"] for i in range(4)] + [0]
    assert request.block_gvas_by_group_np[1].tolist() == [
        addresses["g1:r:0:256@full@h1"],
        addresses["g1:r:1:512@full@h3"],
        0,
    ]
    assert request.partial_save_gva_per_group == [addresses["g0:r:4:513"], addresses["g1:r:2:513"]]


def test_failure_after_successful_batch_does_not_publish_partial_request():
    worker, request = make_case()
    worker.layerwise_max_transfer_blocks = 2
    original = worker.m_store.batch_alloc
    calls = 0

    def allocate(keys, sizes, ttl):
        nonlocal calls
        calls += 1
        return original(keys, sizes, ttl) if calls == 1 else [0] * len(keys)

    worker.m_store.batch_alloc = allocate
    worker._alloc_gvas_for_save([request])
    assert calls == 2
    assert not request.can_save
    assert not request.save_keys
    assert not request.public_save_blocks
    assert worker._failed_requests == {"r"}


def test_no_rotation_keeps_fail_open_policy_and_does_not_save_partial_pages():
    worker, request = make_case()
    worker.layerwise_offload = False
    worker.m_store.batch_is_exist.side_effect = RuntimeError("unavailable")
    worker._alloc_gvas_for_save([request])
    assert request.can_save
    assert len(request.save_keys) == 6
    assert all(":public:" in key for key in request.save_keys)
    assert request.partial_save_gva_per_group == [0] * 6
    assert not worker._snapshot_keys


@pytest.mark.parametrize("reason", ["disabled", "consumer", "cannot_put", "cannot_save", "uncacheable"])
def test_no_save_paths_do_not_query_pool(reason):
    worker, request = make_case()
    if reason == "disabled":
        worker.use_layerwise_transfer = False
    elif reason == "consumer":
        worker.kv_role = "kv_consumer"
    elif reason == "cannot_put":
        worker.can_put = False
    elif reason == "cannot_save":
        request.can_save = False
    else:
        worker.cacheable_group_ids = set()
    worker._alloc_gvas_for_save([request])
    worker.m_store.batch_alloc.assert_not_called()
    worker.m_store.batch_is_exist.assert_not_called()
