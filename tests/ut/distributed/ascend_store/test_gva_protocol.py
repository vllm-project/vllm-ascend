#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import unittest
from unittest.mock import MagicMock

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.gva_protocol import (
    GVAHitChecker,
    GVAKeyFactory,
    GVASession,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LoadSpec,
    ReqMeta,
    get_partial_block_index,
)

_BLOCK_HASH_HEX = "ab" * 32


class _FakeKeyInfo:
    def __init__(self, size, gva=777):
        self._size = size
        self._gva = gva

    def size(self):
        return self._size

    def gva_list(self):
        return [self._gva]


class _FakeStore:
    """Minimal stand-in for the memcache backend surface used by GVASession."""

    def __init__(self):
        self.calls: list[str] = []
        self.ensure_initialized = MagicMock(side_effect=lambda: self.calls.append("ensure_initialized"))
        self.batch_is_exist = MagicMock(side_effect=lambda keys: [1] * len(keys))
        self.batch_get_key_info = MagicMock(return_value=[])
        self.batch_alloc = MagicMock(side_effect=lambda keys, sizes: [100 + i for i in range(len(keys))])
        self.batch_add_lease = MagicMock(side_effect=lambda keys, ttl=0: [0] * len(keys))
        self.batch_remove_lease = MagicMock(return_value=0)


def _make_session(
    store=None,
    num_kv_cache_groups=1,
    grouped_block_size=None,
    hash_block_size=16,
    layerwise_offload=False,
    use_eagle=False,
    kv_role="kv_producer",
    consumer_is_to_put=False,
    tp_rank=0,
    put_step=1,
    head_or_tp_rank=0,
    on_invalid_blocks=None,
):
    invalid_calls: list[list[int]] = []
    if on_invalid_blocks is None:
        on_invalid_blocks = invalid_calls.append
    session = GVASession(
        store=store or _FakeStore(),
        model_name="test-model",
        head_or_tp_rank=head_or_tp_rank,
        tp_rank=tp_rank,
        put_step=put_step,
        num_kv_cache_groups=num_kv_cache_groups,
        grouped_block_size=grouped_block_size or [16],
        hash_block_size=hash_block_size,
        layerwise_offload=layerwise_offload,
        use_eagle=use_eagle,
        kv_role=kv_role,
        consumer_is_to_put=consumer_is_to_put,
        on_invalid_blocks=on_invalid_blocks,
    )
    session.bind_layout({0: [4096]}, 65536)
    return session


def _make_req(
    req_id="req-1",
    target_token_len=32,
    save_start_token=0,
    num_hashes=2,
    can_save=True,
    load_spec=None,
    block_ids_np=None,
    block_hashes=None,
):
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=target_token_len,
        block_ids_by_group=[[i + 1 for i in range(num_hashes)]],
        block_hashes=block_hashes if block_hashes is not None else [bytes([i]) * 32 for i in range(num_hashes)],
        can_save=can_save,
        load_spec=load_spec,
        save_start_token=save_start_token,
        target_token_len=target_token_len,
        block_ids_np=block_ids_np if block_ids_np is not None else np.asarray([i + 1 for i in range(num_hashes)]),
    )


# =========================================================================
# GVAKeyFactory — byte-level key format snapshots
# =========================================================================
class TestGVAKeyFactory(unittest.TestCase):
    def test_full_key_single_group_snapshot(self):
        # PR #11585 compat format: model@hash@rank
        key = GVAKeyFactory.full_key("m", 0, _BLOCK_HASH_HEX, 3, num_groups=1)
        self.assertEqual(key, f"m@{_BLOCK_HASH_HEX}@3")

    def test_full_key_multi_group_snapshot(self):
        key = GVAKeyFactory.full_key("m", 2, _BLOCK_HASH_HEX, 3, num_groups=2)
        self.assertEqual(key, f"m@2@{_BLOCK_HASH_HEX}@3")

    def test_partial_key_snapshot(self):
        key = GVAKeyFactory.partial_key("m", "req-9", 1, 4, 128, 3)
        self.assertEqual(key, "m@partial@req-9@1@4@128@3")

    def test_hit_check_keys_single_group_snapshot(self):
        keys = GVAKeyFactory.hit_check_keys("m", 0, _BLOCK_HASH_HEX, num_ranks=2, num_groups=1)
        self.assertEqual(keys, [f"m@{_BLOCK_HASH_HEX}@0", f"m@{_BLOCK_HASH_HEX}@1"])

    def test_hit_check_keys_multi_group_snapshot(self):
        keys = GVAKeyFactory.hit_check_keys("m", 1, _BLOCK_HASH_HEX, num_ranks=2, num_groups=2)
        self.assertEqual(keys, [f"m@1@{_BLOCK_HASH_HEX}@0", f"m@1@{_BLOCK_HASH_HEX}@1"])


# =========================================================================
# get_partial_block_index
# =========================================================================
class TestGetPartialBlockIndex(unittest.TestCase):
    def test_disabled_or_nonpositive(self):
        self.assertIsNone(get_partial_block_index(10, 16, 2, enabled=False))
        self.assertIsNone(get_partial_block_index(0, 16, 2, enabled=True))
        self.assertIsNone(get_partial_block_index(-1, 16, 2, enabled=True))

    def test_exact_alignment(self):
        # 32 tokens / block 16 = 2 full blocks, no remainder
        self.assertIsNone(get_partial_block_index(32, 16, 4, enabled=True))

    def test_exact_alignment_beyond_hash_count(self):
        # full_blocks (4) > hash_count (2) -> last full block is partial
        self.assertEqual(get_partial_block_index(64, 16, 2, enabled=True), 3)

    def test_trailing_partial_block(self):
        self.assertEqual(get_partial_block_index(33, 16, 4, enabled=True), 2)


# =========================================================================
# GVASession construction (UT 2 — LIFE self-ensure)
# =========================================================================
class TestGVASessionConstruction(unittest.TestCase):
    def test_construction_triggers_ensure_initialized(self):
        store = _FakeStore()
        _make_session(store=store)
        store.ensure_initialized.assert_called_once_with()

    def test_construction_calls_ensure_before_any_store_use(self):
        store = _FakeStore()
        _make_session(store=store)
        # ensure_initialized must be the first store interaction
        self.assertEqual(store.calls[0], "ensure_initialized")


# =========================================================================
# GVASession.alloc_gvas_for_save
# =========================================================================
class TestAllocGVAsForSave(unittest.TestCase):
    def test_role_gate_consumer_not_to_put(self):
        session = _make_session(kv_role="kv_consumer", consumer_is_to_put=False)
        req = _make_req()
        session.alloc_gvas_for_save([req])
        session._store.batch_alloc.assert_not_called()

    def test_tp_rank_gate(self):
        session = _make_session(tp_rank=1, put_step=2)
        req = _make_req()
        session.alloc_gvas_for_save([req])
        session._store.batch_alloc.assert_not_called()

    def test_can_save_gate(self):
        session = _make_session()
        req = _make_req(can_save=False)
        session.alloc_gvas_for_save([req])
        session._store.batch_alloc.assert_not_called()

    def test_allocates_new_keys_and_writes_reqmeta(self):
        session = _make_session()
        req = _make_req(target_token_len=32, num_hashes=2)
        session.alloc_gvas_for_save([req])

        self.assertEqual(len(session._store.batch_alloc.call_args_list), 1)
        keys, sizes = session._store.batch_alloc.call_args_list[0].args
        # alloc_size falls back to sum(group_block_len) = sum({0: [4096]})
        self.assertEqual(sizes, [4096, 4096])
        self.assertEqual(len(keys), 2)
        # ReqMeta field writes are part of the protocol contract
        self.assertEqual(req.block_gvas_np.tolist(), [100, 101])
        self.assertEqual(req.block_ids_by_group_np[0].tolist(), [1, 2])
        self.assertEqual(req.save_keys, keys)
        self.assertEqual(req.gva_block_offset, 0)

    def test_skips_blocks_still_in_memcache(self):
        # Block already allocated in a previous step -> save range starts
        # after it, so the cached block is neither re-allocated nor
        # re-published in save_keys
        session = _make_session()
        req = _make_req(target_token_len=32, num_hashes=2)
        first_key = GVAKeyFactory.full_key("test-model", 0, req.block_hashes[0].hex(), 0, 1)
        session._allocated_gvas[first_key] = 555
        session._store.batch_is_exist.side_effect = lambda keys: [1] * len(keys)

        session.alloc_gvas_for_save([req])
        keys, _ = session._store.batch_alloc.call_args_list[0].args
        # Only the second block needs a fresh allocation
        self.assertEqual(len(keys), 1)
        second_key = GVAKeyFactory.full_key("test-model", 0, req.block_hashes[1].hex(), 0, 1)
        self.assertEqual(keys, [second_key])
        # Blocks before save_start_block stay 0 (already saved previously)
        self.assertEqual(req.block_gvas_np.tolist(), [0, 100])
        self.assertEqual(req.save_keys, [second_key])

    def test_evicted_allocated_gva_is_reallocated(self):
        # batch_is_exist reports 0 -> cached entry dropped and re-allocated
        session = _make_session()
        req = _make_req(target_token_len=32, num_hashes=2)
        first_key = GVAKeyFactory.full_key("test-model", 0, req.block_hashes[0].hex(), 0, 1)
        session._allocated_gvas[first_key] = 555
        session._store.batch_is_exist.side_effect = lambda keys: [0] + [1] * (len(keys) - 1)

        session.alloc_gvas_for_save([req])
        keys, _ = session._store.batch_alloc.call_args_list[0].args
        self.assertEqual(len(keys), 2)
        self.assertEqual(req.block_gvas_np.tolist(), [100, 101])

    def test_partial_allocation_failure_logs_and_zero(self):
        session = _make_session(layerwise_offload=True)
        # 33 tokens / block 16 -> partial block index 2, beyond num_hashes=2
        # so use num_hashes=3 to make the partial block indexable
        req = _make_req(target_token_len=33, num_hashes=3)
        # batch_alloc: first call allocates 3 full keys, second call (partial)
        # returns 0 -> failure branch
        session._store.batch_alloc.side_effect = [
            [100, 101, 102],
            [0],
        ]
        session.alloc_gvas_for_save([req])
        self.assertEqual(req.partial_save_gva_per_group, [0])
        # Partial keys must not be retained
        self.assertFalse(any("partial" in k for k in session._allocated_gvas))

    def test_partial_allocation_success(self):
        session = _make_session(layerwise_offload=True)
        req = _make_req(target_token_len=33, num_hashes=3)
        session._store.batch_alloc.side_effect = [
            [100, 101, 102],
            [200],
        ]
        session.alloc_gvas_for_save([req])
        self.assertEqual(req.partial_save_gva_per_group, [200])
        # Request-scoped partial key is popped after publishing
        self.assertFalse(any("partial" in k for k in session._allocated_gvas))
        self.assertIn("partial", req.save_keys[-1])


# =========================================================================
# GVASession.prepare_load_gvas
# =========================================================================
class TestPrepareLoadGVAs(unittest.TestCase):
    def _make_load_req(self, cached_tokens=32, vllm_cached_tokens=0, num_hashes=2, use_eagle=False):
        load_spec = LoadSpec(
            vllm_cached_tokens=vllm_cached_tokens,
            kvpool_cached_tokens=cached_tokens,
            can_load=True,
        )
        return _make_req(target_token_len=cached_tokens, num_hashes=num_hashes, load_spec=load_spec)

    def test_load_gvas_written_to_reqmeta(self):
        session = _make_session()
        req = self._make_load_req()
        session._store.batch_get_key_info.return_value = [
            _FakeKeyInfo(1, gva=100),
            _FakeKeyInfo(1, gva=101),
        ]
        session.prepare_load_gvas([req])

        keys = session._store.batch_get_key_info.call_args_list[0].args[0]
        self.assertEqual(len(keys), 2)
        self.assertEqual(req.load_block_gvas_np.tolist(), [100, 101])
        self.assertEqual(req.load_keys, keys)
        self.assertEqual(req.load_gva_block_offset, 0)

    def test_invalid_gva_reports_block(self):
        # size-0 key infos mark the block invalid (single-group path
        # reports via callback)
        reported: list[list[int]] = []
        session = _make_session(on_invalid_blocks=reported.append)
        req = self._make_load_req()
        session._store.batch_get_key_info.return_value = [
            _FakeKeyInfo(1, gva=100),
            _FakeKeyInfo(0),
        ]
        session.prepare_load_gvas([req])
        self.assertEqual(reported, [[2]])
        # Invalid block's GVA stays 0 in the padded array
        self.assertEqual(req.load_block_gvas_np.tolist(), [100, 0])

    def test_empty_key_info_return_writes_zero_gvas(self):
        # UT 4 golden: an uninitialized (lazy) store returns [] from
        # batch_get_key_info; the protocol must not raise and silently
        # writes all-zero GVAs with no lease and no failure report.
        # This locks the CURRENT contract (documented silent-degradation
        # behavior of the lazy-init path) so any future change to it is
        # deliberate and visible.
        reported: list[list[int]] = []
        session = _make_session(on_invalid_blocks=reported.append)
        req = self._make_load_req()
        session._store.batch_get_key_info.return_value = []

        session.prepare_load_gvas([req])

        self.assertEqual(req.load_block_gvas_np.tolist(), [0, 0])
        self.assertEqual(req.load_keys, [])
        session._store.batch_add_lease.assert_not_called()
        session._store.batch_remove_lease.assert_not_called()
        self.assertEqual(reported, [])

    def test_multi_group_failure_raises_and_releases_leases(self):
        session = _make_session(num_kv_cache_groups=2, grouped_block_size=[16, 16])
        session.bind_layout({0: [4096], 1: [4096]}, 65536)
        req = self._make_load_req()
        req.block_ids_by_group_np = None
        # First key valid+leased, second invalid -> multi-group path raises
        # and releases the lease acquired for the valid key
        session._store.batch_get_key_info.return_value = [
            _FakeKeyInfo(1, gva=100),
            _FakeKeyInfo(0),
        ]
        with self.assertRaises(RuntimeError):
            session.prepare_load_gvas([req])
        session._store.batch_remove_lease.assert_called_once()

    def test_lease_failure_zeroes_gva_and_reports(self):
        reported: list[list[int]] = []
        session = _make_session(on_invalid_blocks=reported.append)
        req = self._make_load_req()
        session._store.batch_get_key_info.return_value = [
            _FakeKeyInfo(1, gva=100),
            _FakeKeyInfo(1, gva=101),
        ]
        session._store.batch_add_lease.side_effect = lambda keys, ttl=0: [0, 1]  # second lease fails
        session.prepare_load_gvas([req])
        self.assertEqual(reported, [[2]])
        self.assertEqual(req.load_block_gvas_np.tolist(), [100, 0])

    def test_no_requests_with_load_spec(self):
        session = _make_session()
        req = _make_req(load_spec=None)
        session.prepare_load_gvas([req])
        session._store.batch_get_key_info.assert_not_called()


# =========================================================================
# GVAHitChecker
# =========================================================================
class TestGVAHitChecker(unittest.TestCase):
    def _make_checker(self, store=None, grouped_block_size=None, hash_block_size=16, use_layerwise=True):
        return GVAHitChecker(
            store=store or _FakeStore(),
            model_name="test-model",
            head_or_tp_ranks=2,
            grouped_block_size=grouped_block_size or [16],
            hash_block_size=hash_block_size,
            num_groups=1,
            use_layerwise=use_layerwise,
        )

    def _make_request(self, num_hashes, request_id="req-1"):
        request = MagicMock()
        request.request_id = request_id
        request.block_hashes = [bytes([i]) * 32 for i in range(num_hashes)]
        return request

    def test_all_ranks_hit_blocks(self):
        store = _FakeStore()
        checker = self._make_checker(store=store)
        request = self._make_request(2)
        # 2 blocks x 2 ranks = 4 key infos, all valid
        store.batch_get_key_info.return_value = [_FakeKeyInfo(1)] * 4
        hit = checker.hit_tokens(request, token_len=32, num_computed_tokens=0)
        self.assertEqual(hit, 32)
        # Key layout: all-rank keys, block-major
        keys = store.batch_get_key_info.call_args_list[0].args[0]
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], f"test-model@{request.block_hashes[0].hex()}@0")
        self.assertEqual(keys[1], f"test-model@{request.block_hashes[0].hex()}@1")
        self.assertEqual(keys[2], f"test-model@{request.block_hashes[1].hex()}@0")
        self.assertEqual(keys[3], f"test-model@{request.block_hashes[1].hex()}@1")

    def test_partial_hit_stops_at_first_miss(self):
        store = _FakeStore()
        checker = self._make_checker(store=store)
        request = self._make_request(2)
        # Block 0: both ranks valid; block 1: rank 0 invalid -> stop
        store.batch_get_key_info.return_value = [
            _FakeKeyInfo(1),
            _FakeKeyInfo(1),
            _FakeKeyInfo(0),
            _FakeKeyInfo(1),
        ]
        hit = checker.hit_tokens(request, token_len=32, num_computed_tokens=0)
        self.assertEqual(hit, 16)

    def test_all_groups_hit_takes_min(self):
        store = _FakeStore()
        checker = self._make_checker(store=store, grouped_block_size=[16, 32])
        request = self._make_request(4)

        def key_info(keys):
            if len(keys) == 8:
                # Group 0 (bs 16): 4 blocks, all ranks valid -> 64 tokens
                return [_FakeKeyInfo(1)] * 8
            # Group 1 (bs 32): block 0 fully valid, block 1 rank 0 invalid
            # -> 1 hit block -> 32 tokens
            return [
                _FakeKeyInfo(1),
                _FakeKeyInfo(1),
                _FakeKeyInfo(0),
                _FakeKeyInfo(1),
            ]

        store.batch_get_key_info.side_effect = key_info
        hit = checker.hit_tokens(request, token_len=64, num_computed_tokens=0)
        self.assertEqual(hit, 32)

    def test_mismatched_key_info_count_returns_zero_for_group(self):
        store = _FakeStore()
        checker = self._make_checker(store=store)
        request = self._make_request(2)
        store.batch_get_key_info.return_value = [_FakeKeyInfo(1)]  # expected 4
        hit = checker.hit_tokens(request, token_len=32, num_computed_tokens=0)
        self.assertEqual(hit, 0)

    def test_query_start_block_respects_use_layerwise(self):
        # use_layerwise=True always queries from block 0; False starts at
        # num_computed_tokens // block_size
        store = _FakeStore()
        checker = self._make_checker(store=store, use_layerwise=False)
        request = self._make_request(4)
        store.batch_get_key_info.return_value = [_FakeKeyInfo(1)] * 4
        hit = checker.hit_tokens(request, token_len=64, num_computed_tokens=32)
        # Queries blocks [2, 4): 2 blocks hit, plus query_start 2 -> 4 blocks
        keys = store.batch_get_key_info.call_args_list[0].args[0]
        self.assertEqual(len(keys), 4)
        self.assertEqual(hit, 64)


if __name__ == "__main__":
    unittest.main()
