# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Unit tests for the Mooncake D2RH connector.

Covers block-map translation, CPU staging, handshake metadata, content-based
host caching, offset-qualified block maps, and scheduler helpers.
"""

import sys
import threading
import types
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# The connector eagerly imports ``mooncake.engine``, which is not a UT
# dependency. Sibling test files install the same stub at import time, but
# depending on their collection order made this module fail when run alone.
if "mooncake.engine" not in sys.modules:
    try:
        import_module("mooncake.engine")
    except ImportError:
        _fake_engine = types.ModuleType("mooncake.engine")
        _fake_engine.__dict__["TransferEngine"] = MagicMock()
        sys.modules["mooncake.engine"] = _fake_engine

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_d2rh_connector as d2rh  # noqa: E402


class TestD2RHPorts:
    @staticmethod
    def _config(extra_config: dict[str, int] | None = None):
        extra_config = extra_config or {}
        return SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                get_from_extra_config=lambda name, default: extra_config.get(name, default)
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                prefill_context_parallel_size=1,
                data_parallel_rank=1,
            ),
        )

    def test_uses_default_port_bases(self):
        config = self._config()

        assert d2rh.get_d2rh_zmq_port(config, tp_rank=1, pp_rank=1) == 38107
        assert d2rh.get_scheduler_ready_zmq_port(config) == 38204

    def test_uses_configured_port_bases(self):
        config = self._config(
            {
                "d2rh_zmq_port": 39100,
                "d2rh_scheduler_ready_port": 39200,
            }
        )

        assert d2rh.get_d2rh_zmq_port(config, tp_rank=1, pp_rank=1) == 39107
        assert d2rh.get_scheduler_ready_zmq_port(config) == 39204


class TestMooncakeAgentMetadata:
    def test_metadata_carries_handshake_port(self):
        metadata = d2rh.MooncakeAgentMetadata(
            engine_id="engine",
            handshake_port=30007,
            te_rpc_port=1234,
            kv_group2layeridx={0: ({"kv_cache_spec_type": "FullAttentionSpec"}, [0])},
            block_size=16,
            kv_caches_base_addr=[[1000, 2000]],
            block_size_scale=[[1, 1]],
            num_blocks=8,
            block_lens=[[128, 128]],
            block_strides=[[256, 256]],
            local_ip="127.0.0.1",
        )

        assert metadata.block_strides == [[256, 256]]
        assert metadata.kv_group2layeridx[0][1] == [0]
        # The handshake port must survive metadata registration so tuple-key
        # lookup can validate the remote worker endpoint.
        assert metadata.handshake_port == 30007


class TestBuildStartPullParams:
    def test_carries_group_pulls_per_port(self):
        scheduler = object.__new__(d2rh.MooncakeConnectorScheduler)
        scheduler._prefill_tp_size = 4
        scheduler._decode_tp_size = 2
        scheduler._prefill_pp_size = 1
        scheduler.num_key_value_heads = 4
        scheduler.is_deepseek_mla = False
        scheduler.use_sparse = False
        scheduler.tp_size = 2
        scheduler.kv_cache_groups = [
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=16, num_kv_heads=4),
                layer_names=["layer.0"],
            )
        ]

        params = scheduler._build_start_pull_params(
            "req",
            {
                "remote_request_id": "remote-req",
                "remote_port": 30000,
                "remote_host": "p-host",
                "remote_engine_id": "p-engine",
                "remote_block_ids": ([1, 2],),
            },
            decode_tp_rank=0,
        )

        assert params["remote_handshake_ports"] == [30000, 30001]
        assert len(params["group_pulls_by_port"]) == 2
        assert params["group_pulls_by_port"][0][0].group_id == 0
        assert params["group_pulls_by_port"][1][0].is_group_transfer_end is True


class TestSchedulerSocketReuse:
    @staticmethod
    def _scheduler():
        scheduler = object.__new__(d2rh.MooncakeConnectorScheduler)
        scheduler.local_host = "d-host"
        scheduler.timeout = 1.0
        scheduler.encoder = MagicMock()
        scheduler.encoder.encode.return_value = b"request"
        scheduler.remote_poller = MagicMock()
        scheduler._get_remote_socket = MagicMock(return_value=MagicMock())
        scheduler._return_remote_socket = MagicMock()
        scheduler._discard_remote_socket = MagicMock()
        return scheduler

    def test_matching_reply_returns_socket_to_pool(self):
        scheduler = self._scheduler()

        with (
            patch.object(d2rh, "ensure_zmq_send"),
            patch.object(d2rh, "ensure_zmq_recv", return_value=b"ACK"),
        ):
            assert scheduler._send_start_pull("req", {}, 38100) == b"ACK"

        socket = scheduler._get_remote_socket.return_value
        scheduler._return_remote_socket.assert_called_once_with(socket, "d-host", 38100)
        scheduler._discard_remote_socket.assert_not_called()

    def test_failed_reply_discards_req_socket(self):
        scheduler = self._scheduler()

        with (
            patch.object(d2rh, "ensure_zmq_send"),
            patch.object(d2rh, "ensure_zmq_recv", side_effect=RuntimeError("timeout")),
            pytest.raises(RuntimeError, match="timeout"),
        ):
            scheduler._send_start_pull("req", {}, 38100)

        socket = scheduler._get_remote_socket.return_value
        scheduler._discard_remote_socket.assert_called_once_with(socket)
        scheduler._return_remote_socket.assert_not_called()


class TestBlockMapHelpers:
    def test_roundtrip_keeps_manager_api_as_block_ids(self):
        block_map = d2rh._build_block_map(([10, 11], [20]), ([0, 1], [0]))
        assert block_map == {(0, 10): 0, (0, 11): 1, (1, 20): 0}
        assert d2rh._group_block_map_values(block_map) == ([0, 1], [0])

    def test_empty_groups_produce_empty_map(self):
        assert d2rh._build_block_map((), ()) == {}
        assert d2rh._group_block_map_values({}) == ()


class TestD2RHCPUCacheManager:
    def test_reuses_freed_blocks(self):
        manager = d2rh.D2RHCPUCacheManager(2)
        first = manager.alloc_block_map(([10, 11],))
        assert first == {(0, 10): 0, (0, 11): 1}
        assert manager.alloc_block_map(([12],)) is None
        manager.free_block_map({(0, 10): 0})
        assert manager.alloc_block_map(([12],)) == {(0, 12): 0}

    def test_allocates_blocks_per_group(self):
        manager = d2rh.D2RHCPUCacheManager(4)

        block_map = manager.alloc_block_map(([10, 11], [20, 21]))
        assert block_map == {(0, 10): 0, (0, 11): 1, (1, 20): 2, (1, 21): 3}
        assert manager.alloc_block_map(([], [22])) is None

        manager.free_block_map({(1, 20): 2})
        assert manager.alloc_block_map(([], [22])) == {(1, 22): 2}

    def test_group_allocation_is_atomic(self):
        manager = d2rh.D2RHCPUCacheManager(1)

        assert manager.alloc_block_map(([10], [20, 21])) is None
        assert manager.alloc_block_map(([10], [])) == {(0, 10): 0}

    def test_alloc_deduplicates_repeated_remote_ids(self):
        manager = d2rh.D2RHCPUCacheManager(2)
        # ([10, 10],) is one logical block after dedup, not two.
        assert manager.alloc_block_map(([10, 10],)) == {(0, 10): 0}
        assert manager.alloc_block_map(([11, 12],)) is None

    def test_free_with_unknown_keys_is_noop(self):
        manager = d2rh.D2RHCPUCacheManager(2)
        manager.alloc_block_map(([10],))
        manager.free_block_map({(0, 99): 7})
        assert manager.alloc_block_map(([11],)) == {(0, 11): 1}

    def test_double_free_keeps_block_sets_consistent(self):
        manager = d2rh.D2RHCPUCacheManager(2)
        block_map = manager.alloc_block_map(([10],))
        manager.free_block_map(block_map)
        free_after_first = list(manager.free_queue)
        manager.free_block_map(block_map)
        assert list(manager.free_queue) == free_after_first
        assert manager.used_set == set()


class TestHostContentCache:
    """Exercise host-cache allocation, commit, and release semantics."""

    @staticmethod
    def _manager(capacity: int = 4) -> d2rh.D2RHCPUCacheManager:
        return d2rh.D2RHCPUCacheManager(capacity)

    @staticmethod
    def _pulls(group_id: int = 0, remote_tp_offset: int = 0) -> list[list[dict]]:
        return [[{"group_id": group_id, "remote_tp_offset": remote_tp_offset}]]

    def _alloc_commit(self, manager, block_ids, hashes):
        block_map, _, misses = manager.alloc_sharded_block_map(block_ids, self._pulls(), hashes)
        manager.commit_block_map(block_map, misses)
        return block_map

    def test_miss_registers_pending_and_commit_promotes_to_hit(self):
        manager = self._manager()
        result = manager.alloc_sharded_block_map(([10, 11],), self._pulls(), ([b"\x01", b"\x02"],))
        assert result is not None
        block_map, cache_hits, cacheable_misses = result
        assert cache_hits == set()
        assert cacheable_misses == {
            (0, 10, 0): (0, 0, b"\x01"),
            (0, 11, 0): (0, 0, b"\x02"),
        }
        # Pending entries are not exposed as hits until committed.
        assert set(manager.pending_by_key) == {(0, 0, b"\x01"), (0, 0, b"\x02")}
        assert manager.cache_stats() == (0, 2, 2)

        manager.commit_block_map(block_map, cacheable_misses)
        assert manager.pending_by_key == {}
        assert manager.cache_stats() == (2, 0, 2)

        result = manager.alloc_sharded_block_map(([10, 11],), self._pulls(), ([b"\x01", b"\x02"],))
        block_map2, cache_hits2, misses2 = result
        assert cache_hits2 == {(0, 10, 0), (0, 11, 0)}
        assert misses2 == {}
        assert block_map2 == block_map

    def test_hit_pins_block_and_free_keeps_it_cached(self):
        manager = self._manager()
        block_map = self._alloc_commit(manager, ([10],), ([b"\x01"],))

        result = manager.alloc_sharded_block_map(([10],), self._pulls(), ([b"\x01"],))
        hit_block_map, cache_hits, _ = result
        assert cache_hits == {(0, 10, 0)}
        assert manager.pin_count[hit_block_map[(0, 10, 0)]] == 2

        manager.free_block_map(hit_block_map)
        manager.free_block_map(block_map)
        block_id = block_map[(0, 10, 0)]
        assert manager.pin_count[block_id] == 0
        # Cached blocks stay resident (evictable) instead of returning to the
        # free queue, so the next identical request still hits.
        assert manager.cache_by_key[(0, 0, b"\x01")] == block_id
        assert block_id not in manager.free_queue
        assert manager.cache_stats() == (1, 0, 3)

    def test_pending_entry_is_not_exposed_as_hit(self):
        manager = self._manager()
        first_block_map, first_hits, first_misses = manager.alloc_sharded_block_map(
            ([10],), self._pulls(), ([b"\x01"],)
        )
        # A concurrent loader for the same content must not see a hit while
        # the first load is still pending, and must not re-register pending.
        second_block_map, second_hits, second_misses = manager.alloc_sharded_block_map(
            ([10],), self._pulls(), ([b"\x01"],)
        )
        assert first_hits == set()
        assert second_hits == set()
        assert second_block_map[(0, 10, 0)] != first_block_map[(0, 10, 0)]
        assert second_misses == {}

        manager.commit_block_map(first_block_map, first_misses)
        assert manager.cache_by_key[(0, 0, b"\x01")] == first_block_map[(0, 10, 0)]

        manager.free_block_map(second_block_map)
        # The transient block returns to the free queue; the committed one stays cached.
        assert second_block_map[(0, 10, 0)] in manager.free_queue
        assert manager.cache_stats() == (1, 0, 3)

    def test_lru_eviction_reclaims_unpinned_cached_blocks(self):
        manager = self._manager(capacity=2)
        first_map = self._alloc_commit(manager, ([10],), ([b"\x01"],))
        second_map = self._alloc_commit(manager, ([11],), ([b"\x02"],))
        manager.free_block_map(first_map)
        manager.free_block_map(second_map)
        assert manager.cache_stats() == (2, 0, 0)
        assert list(manager.evictable_blocks) == [
            first_map[(0, 10, 0)],
            second_map[(0, 11, 0)],
        ]

        # Two new hashes must evict the two unpinned cached blocks (LRU order).
        result = manager.alloc_sharded_block_map(([20, 21],), self._pulls(), ([b"\x03", b"\x04"],))
        assert result is not None
        block_map, _, misses = result
        assert sorted(block_map.values()) == [0, 1]
        assert set(misses) == {(0, 20, 0), (0, 21, 0)}
        # The old entries were evicted; the new ones are pending until committed.
        assert manager.cache_by_key == {}
        manager.commit_block_map(block_map, misses)
        assert manager.cache_by_key == {(0, 0, b"\x03"): 0, (0, 0, b"\x04"): 1}
        assert manager.cache_stats() == (2, 0, 0)

    def test_pinned_cached_block_is_never_evicted_and_alloc_rolls_back(self):
        manager = self._manager(capacity=2)
        first_map = self._alloc_commit(manager, ([10],), ([b"\x01"],))
        second_map = self._alloc_commit(manager, ([11],), ([b"\x02"],))
        manager.free_block_map(first_map)
        manager.free_block_map(second_map)

        # Re-touch the first entry so it is pinned while still cached.
        result = manager.alloc_sharded_block_map(([10],), self._pulls(), ([b"\x01"],))
        hit_block_map, cache_hits, _ = result
        assert cache_hits == {(0, 10, 0)}
        assert manager.pin_count[hit_block_map[(0, 10, 0)]] == 1
        assert list(manager.evictable_blocks) == [second_map[(0, 11, 0)]]

        # Two new hashes cannot be served: only one evictable block remains.
        result = manager.alloc_sharded_block_map(([20, 21],), self._pulls(), ([b"\x03", b"\x04"],))
        assert result is None
        # The rollback released the transiently acquired block and its
        # pending entry; the pinned hit stays cached.
        cached, pending, free = manager.cache_stats()
        assert (cached, free) == (1, 1)
        assert manager.pending_by_key == {}
        assert manager.cache_by_key == {(0, 0, b"\x01"): hit_block_map[(0, 10, 0)]}

    def test_alloc_without_hashes_bypasses_cache_entirely(self):
        manager = self._manager()
        result = manager.alloc_sharded_block_map(([10],), self._pulls(), None)
        block_map, cache_hits, misses = result
        assert cache_hits == set()
        assert misses == {}
        manager.commit_block_map(block_map, misses)
        assert manager.cache_by_key == {}
        assert manager.cache_stats() == (0, 0, 3)

    def test_commit_replaces_stale_block_for_same_content(self):
        manager = self._manager()
        first_block_map, _, first_misses = manager.alloc_sharded_block_map(([10],), self._pulls(), ([b"\x01"],))
        manager.commit_block_map(first_block_map, first_misses)
        stale_block_id = first_block_map[(0, 10, 0)]
        manager.free_block_map(first_block_map)

        second_block_map, _, _ = manager.alloc_sharded_block_map(([11],), self._pulls(), ([b"\x02"],))
        fresh_block_id = second_block_map[(0, 11, 0)]
        # Simulate a re-load of the *same* content into the fresh block and
        # commit: the stale block must be replaced and released.
        manager.pending_by_key[(0, 0, b"\x01")] = fresh_block_id
        manager.pending_key_by_block[fresh_block_id] = (0, 0, b"\x01")
        manager.commit_block_map(second_block_map, {(0, 11, 0): (0, 0, b"\x01")})

        assert manager.cache_by_key[(0, 0, b"\x01")] == fresh_block_id
        assert stale_block_id in manager.free_queue


class TestKVCacheRecvingThreadHop2:
    @staticmethod
    def _thread(block_map_by_req):
        thread = object.__new__(d2rh.KVCacheRecvingThread)
        thread.remote_local_block_map = block_map_by_req
        thread.cpu_host = "127.0.0.1"
        # Match the worker state used by the full-block-map diagnostic path.
        thread.log_full_block_map = False
        # _handle_request records the H2D request mapping before delegating.
        thread._h2d_remote_request_ids = {}
        return thread

    @staticmethod
    def _req_meta(remote_block_ids=([10, 11], [7]), group_pulls=None):
        meta = {
            "request_id": "req",
            "remote_request_id": "remote-req",
            "remote_block_ids": remote_block_ids,
            "remote_engine_id": "p-engine",
            "remote_host": "p-host",
            "remote_handshake_port": 30000,
        }
        if group_pulls is not None:
            meta["group_pulls"] = group_pulls
        return meta

    def test_translates_group_aware_block_map_for_hop2(self):
        thread = self._thread({"remote-req": {(0, 10): 3, (0, 11): 4, (1, 7): 5}})
        req_meta = self._req_meta()

        with patch.object(d2rh.KVCacheRecvingThread, "_transfer_staged_kv_cache_all_groups") as mock_transfer:
            d2rh.KVCacheRecvingThread._transfer_kv_cache_all_groups(thread, req_meta)

        mock_transfer.assert_called_once()
        transferred_meta = mock_transfer.call_args.args[0]
        assert transferred_meta["remote_block_ids"] == ([3, 4], [5])
        assert transferred_meta["remote_engine_id"] == d2rh.CPU_STAGING_ENGINE_ID
        assert transferred_meta["remote_handshake_port"] == d2rh.CPU_STAGING_HANDSHAKE_PORT
        assert transferred_meta["remote_host"] == "127.0.0.1"

    def test_prefers_offset_qualified_keys_over_legacy(self):
        thread = self._thread({"remote-req": {(0, 10, 2): 7, (0, 10): 3}})
        req_meta = self._req_meta(
            remote_block_ids=([10],),
            group_pulls=[SimpleNamespace(group_id=0, remote_tp_offset=2)],
        )

        with patch.object(d2rh.KVCacheRecvingThread, "_transfer_staged_kv_cache_all_groups") as mock_transfer:
            d2rh.KVCacheRecvingThread._transfer_kv_cache_all_groups(thread, req_meta)

        assert mock_transfer.call_args.args[0]["remote_block_ids"] == ([7],)

    def test_missing_block_map_key_raises(self):
        thread = self._thread({"remote-req": {(0, 10): 3}})
        req_meta = self._req_meta(remote_block_ids=([10, 11],))

        with pytest.raises(RuntimeError, match="missing key"):
            d2rh.KVCacheRecvingThread._transfer_kv_cache_all_groups(thread, req_meta)

    def test_missing_request_map_raises(self):
        thread = self._thread({})
        req_meta = self._req_meta(remote_block_ids=([10],))

        with pytest.raises(RuntimeError, match="block map missing for request"):
            d2rh.KVCacheRecvingThread._transfer_kv_cache_all_groups(thread, req_meta)

    def test_handle_request_records_mapping_and_defers_cleanup(self):
        manager = d2rh.D2RHCPUCacheManager(4)
        block_map = manager.alloc_block_map(([10],))
        thread = self._thread({"remote-req": dict(block_map)})
        thread.cpu_kvcache_manager = manager
        req_meta = {"request_id": "req", "remote_request_id": "remote-req"}

        with patch.object(d2rh.BaseKVCacheRecvingThread, "_handle_request") as mock_handle:
            d2rh.KVCacheRecvingThread._handle_request(thread, req_meta)

        mock_handle.assert_called_once_with(req_meta)
        assert thread._h2d_remote_request_ids == {"req": "remote-req"}
        # Staging blocks are released only after the base completion counter
        # reports every H2D shard done, not when the request is submitted.
        assert thread.remote_local_block_map == {"remote-req": dict(block_map)}
        assert list(manager.free_queue) == [1, 2, 3]

    def test_mark_request_task_done_frees_block_map_on_completion(self):
        manager = d2rh.D2RHCPUCacheManager(4)
        block_map = manager.alloc_block_map(([10],))
        thread = self._thread({"remote-req": dict(block_map), "req": dict(block_map)})
        thread.cpu_kvcache_manager = manager
        thread._h2d_remote_request_ids = {"req": "remote-req"}

        with patch.object(d2rh.BaseKVCacheRecvingThread, "_mark_request_task_done", return_value=True):
            assert d2rh.KVCacheRecvingThread._mark_request_task_done(thread, "req", True) is True

        assert thread.remote_local_block_map == {}
        assert thread._h2d_remote_request_ids == {}
        assert list(manager.free_queue) == [1, 2, 3, 0]

    def test_mark_request_task_done_keeps_block_map_until_completed(self):
        manager = d2rh.D2RHCPUCacheManager(4)
        block_map = manager.alloc_block_map(([10],))
        thread = self._thread({"remote-req": dict(block_map), "req": dict(block_map)})
        thread.cpu_kvcache_manager = manager
        thread._h2d_remote_request_ids = {"req": "remote-req"}

        with patch.object(d2rh.BaseKVCacheRecvingThread, "_mark_request_task_done", return_value=False):
            assert d2rh.KVCacheRecvingThread._mark_request_task_done(thread, "req", False) is False

        assert thread.remote_local_block_map == {"remote-req": dict(block_map), "req": dict(block_map)}
        assert thread._h2d_remote_request_ids == {"req": "remote-req"}
        assert list(manager.free_queue) == [1, 2, 3]


class TestPrefixFingerprint:
    @staticmethod
    def _scheduler(hash_block_size: int | None = 2):
        scheduler = object.__new__(d2rh.MooncakeConnectorScheduler)
        cache_config = (
            SimpleNamespace() if hash_block_size is None else SimpleNamespace(hash_block_size=hash_block_size)
        )
        scheduler.vllm_config = SimpleNamespace(cache_config=cache_config)
        scheduler.block_size = 16
        return scheduler

    @staticmethod
    def _request(block_hashes, token_ids):
        return SimpleNamespace(block_hashes=block_hashes, prompt_token_ids=token_ids)

    def test_is_deterministic_and_token_sensitive(self):
        scheduler = self._scheduler()
        request = self._request([b"\xaa" * 32], [1, 2, 3])
        first = scheduler._d2rh_prefix_fingerprint(request, 3)
        assert first == scheduler._d2rh_prefix_fingerprint(request, 3)
        other_tokens = self._request([b"\xaa" * 32], [1, 2, 4])
        assert first != scheduler._d2rh_prefix_fingerprint(other_tokens, 3)

    def test_covers_partial_block_tail_and_block_boundary(self):
        scheduler = self._scheduler()
        request = self._request([b"\xaa" * 32], [1, 2, 3])
        # end_token=2 stops at the block boundary: chained hash only, no tail.
        boundary = scheduler._d2rh_prefix_fingerprint(request, 2)
        # end_token=3 additionally folds the tail token into the digest.
        with_tail = scheduler._d2rh_prefix_fingerprint(request, 3)
        assert boundary != with_tail
        # Tokens beyond end_token do not affect the fingerprint.
        assert boundary == scheduler._d2rh_prefix_fingerprint(self._request([b"\xaa" * 32], [1, 2]), 2)

    def test_falls_back_to_block_size_when_hash_block_size_missing(self):
        scheduler = self._scheduler(hash_block_size=None)
        request = self._request([b"\xaa" * 32], [1, 2, 3])
        digest = scheduler._d2rh_prefix_fingerprint(request, 3)
        assert digest == scheduler._d2rh_prefix_fingerprint(request, 3)
        # No complete block at block_size=16, so every token up to end_token
        # is folded in and token changes are visible.
        assert digest != scheduler._d2rh_prefix_fingerprint(self._request([b"\xaa" * 32], [1, 2, 4]), 3)


class TestGetNumNewMatchedTokens:
    @staticmethod
    def _scheduler(ready: bool = True):
        scheduler = object.__new__(d2rh.MooncakeConnectorScheduler)
        scheduler.host_cache_hash_source = "prefill"
        scheduler.all_requests = set()
        scheduler.listeningthread = SimpleNamespace(
            ready_lock=threading.Lock(),
            ready_request={"req1"} if ready else set(),
            ready_count={"req1": 1},
        )
        scheduler._state_prefill_token_count = lambda num_prompt_tokens: num_prompt_tokens
        return scheduler

    @staticmethod
    def _request(params=None):
        return SimpleNamespace(
            request_id="req1",
            kv_transfer_params={"do_remote_prefill": True} if params is None else params,
            block_hashes=[b"\x01"],
            prompt_token_ids=list(range(100)),
        )

    def test_ready_request_returns_remaining_tokens_and_publishes_count(self):
        scheduler = self._scheduler()
        scheduler.all_requests.add("req1")
        request = self._request()
        params = request.kv_transfer_params

        assert scheduler.get_num_new_matched_tokens(request, 40) == (60, True)
        # The D-local prefix hit is published into kv_transfer_params.
        assert params["num_computed_tokens"] == 40

    def test_not_ready_request_returns_none(self):
        scheduler = self._scheduler(ready=False)
        scheduler.all_requests.add("req1")
        request = self._request()
        params = request.kv_transfer_params

        assert scheduler.get_num_new_matched_tokens(request, 40) == (None, False)
        assert params["num_computed_tokens"] == 40

    def test_hbm_query_is_logged_once_per_request(self):
        scheduler = self._scheduler()
        scheduler._decode_tp_size = 1
        scheduler._prefill_tp_size = 2
        scheduler._prefill_pp_size = 1
        scheduler.num_key_value_heads = 4
        scheduler.is_deepseek_mla = False
        scheduler.use_sparse = False
        scheduler.tp_size = 2
        scheduler.kv_cache_groups = []
        scheduler.vllm_config = SimpleNamespace()
        scheduler._send_start_pull = lambda request_id, params, port: b"ACK"
        request = self._request(
            {
                "do_remote_prefill": True,
                "remote_request_id": "remote-req",
                "remote_port": 30000,
                "remote_host": "p-host",
                "remote_engine_id": "p-engine",
                "remote_block_ids": ([1, 2],),
            }
        )

        with (
            patch.object(d2rh, "get_d2rh_zmq_port", return_value=38100),
            patch.object(d2rh, "get_remote_ranks_for_req", return_value=[[0]]),
            patch.object(
                d2rh,
                "resolve_remote_host_for_handshake_port",
                side_effect=lambda *args: ("p-host", "p-engine"),
            ),
            patch.object(d2rh.logger, "info") as info_log,
        ):
            assert scheduler.get_num_new_matched_tokens(request, 0) == (100, True)
            assert scheduler.get_num_new_matched_tokens(request, 0) == (100, True)

        hbm_logs = [call for call in info_log.call_args_list if call.args[0].startswith("D2RH_HBM_QUERY")]
        assert len(hbm_logs) == 1

    def test_staging_full_discards_request_and_returns_none(self):
        scheduler = self._scheduler()
        scheduler._decode_tp_size = 1
        scheduler._prefill_tp_size = 2
        scheduler._prefill_pp_size = 1
        scheduler.num_key_value_heads = 4
        scheduler.is_deepseek_mla = False
        scheduler.use_sparse = False
        scheduler.tp_size = 2
        scheduler.kv_cache_groups = []
        scheduler.vllm_config = SimpleNamespace()
        scheduler._send_start_pull = lambda request_id, params, port: d2rh.STAGING_FULL
        request = self._request(
            {
                "do_remote_prefill": True,
                "remote_request_id": "remote-req",
                "remote_port": 30000,
                "remote_host": "p-host",
                "remote_engine_id": "p-engine",
                "remote_block_ids": ([1, 2],),
            }
        )

        with (
            patch.object(d2rh, "get_d2rh_zmq_port", return_value=38100),
            patch.object(d2rh, "get_remote_ranks_for_req", return_value=[[0]]),
            patch.object(
                d2rh,
                "resolve_remote_host_for_handshake_port",
                side_effect=lambda *args: ("p-host", "p-engine"),
            ),
        ):
            assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)

        assert request.kv_transfer_params["num_computed_tokens"] == 0
        assert "req1" not in scheduler.all_requests
        assert "req1" not in scheduler.listeningthread.ready_count


class TestRequestFinished:
    @staticmethod
    def _scheduler():
        scheduler = object.__new__(d2rh.MooncakeConnectorScheduler)
        scheduler.host_cache_hash_source = "prefill"
        return scheduler

    def test_publishes_block_hashes(self):
        scheduler = self._scheduler()
        request = SimpleNamespace(request_id="req1", kv_transfer_params={}, prompt_token_ids=[1] * 64)
        base_params = {"remote_block_ids": ([1],)}
        sentinel_hashes = (["ab" * 32],)
        scheduler._d2rh_get_transfer_block_hashes = MagicMock(return_value=sentinel_hashes)

        with patch.object(
            d2rh.BaseMooncakeConnectorScheduler,
            "request_finished",
            return_value=(False, base_params),
        ) as mock_base:
            delay_free, params = scheduler.request_finished(request, ([1],))

        assert mock_base.called
        assert delay_free is False
        scheduler._d2rh_get_transfer_block_hashes.assert_called_once_with(request, ([1],), ([1],))
        # The content hashes ride the transfer params so D workers can hit
        # the host cache before the first pull completes.
        assert params["d2rh_block_hashes"] is sentinel_hashes

    def test_passes_through_when_base_returns_no_params(self):
        scheduler = self._scheduler()
        request = SimpleNamespace(request_id="req1", kv_transfer_params=None, prompt_token_ids=[])

        with patch.object(
            d2rh.BaseMooncakeConnectorScheduler,
            "request_finished",
            return_value=(False, None),
        ):
            delay_free, params = scheduler.request_finished(request, ([],))

        assert (delay_free, params) == (False, None)


class TestUpdateStateAfterAlloc:
    def test_consumer_discards_request_state(self):
        scheduler = object.__new__(d2rh.MooncakeConnectorScheduler)
        scheduler.kv_role = "kv_consumer"
        scheduler.all_requests = {"req1"}
        scheduler.listeningthread = SimpleNamespace(
            ready_lock=threading.Lock(),
            ready_request={"req1"},
            ready_count={"req1": 3},
        )
        request = SimpleNamespace(request_id="req1")

        with patch.object(d2rh.BaseMooncakeConnectorScheduler, "update_state_after_alloc") as mock_base:
            scheduler.update_state_after_alloc(request, [], 0)

        mock_base.assert_called_once_with(request, [], 0)
        assert "req1" not in scheduler.all_requests
        assert "req1" not in scheduler.listeningthread.ready_request
        assert "req1" not in scheduler.listeningthread.ready_count


class TestSchedulerBlockSize:
    """D2RH-local DeepSeek-V4 scheduler block-size compatibility.

    Hybrid KV cache initialization stores the smallest physical group block
    size in cache_config; the resolved DSV4 SWA spec retains the logical
    block size needed by Mooncake transfer metadata.
    """

    @staticmethod
    def _swa_spec(block_size: int, model_version: str | None) -> AscendSlidingWindowMLASpec:
        return AscendSlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.float16,
            sliding_window=128,
            model_version=model_version,
        )

    def test_dsv4_swa_spec_overrides_hybrid_cache_quantum(self):
        scheduler = object.__new__(d2rh.MooncakeConnectorScheduler)
        scheduler.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=2))
        scheduler.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4096), layer_names=["state"]),
                SimpleNamespace(
                    kv_cache_spec=self._swa_spec(32, "deepseek_v4"),
                    layer_names=["swa"],
                ),
            ]
        )
        assert scheduler._get_scheduler_block_size() == 32

    def test_non_dsv4_swa_spec_falls_back_to_cache_config(self):
        scheduler = object.__new__(d2rh.MooncakeConnectorScheduler)
        scheduler.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))
        scheduler.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=self._swa_spec(32, None),
                    layer_names=["swa"],
                ),
            ]
        )
        # isinstance matches but model_version does not: the scheduler must
        # keep cache_config.block_size instead of the SWA spec size.
        assert scheduler._get_scheduler_block_size() == 128

    def test_refreshes_block_size_before_base_scheduler_snapshot(self):
        calls: list[tuple[str, int]] = []
        extras = {
            "prefill": {"tp_size": 1, "pp_size": 1},
            "decode": {"tp_size": 1},
        }
        config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=2),
            kv_transfer_config=SimpleNamespace(
                kv_role="kv_producer",
                get_from_extra_config=lambda name, default: extras.get(name, default),
            ),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_key_value_heads=1),
                is_deepseek_mla=True,
            ),
        )
        cache_config = SimpleNamespace(kv_cache_groups=[])

        def refresh(vllm_config):
            calls.append(("refresh", vllm_config.cache_config.block_size))
            vllm_config.cache_config.block_size = 32

        def base_init(scheduler, vllm_config, engine_id, kv_cache_config):
            calls.append(("base", vllm_config.cache_config.block_size))
            scheduler.vllm_config = vllm_config
            scheduler.kv_cache_config = kv_cache_config

        with (
            patch.object(d2rh, "refresh_block_size", side_effect=refresh) as refresh_mock,
            patch.object(d2rh.BaseMooncakeConnectorScheduler, "__init__", new=base_init),
            patch.object(d2rh, "get_ip", return_value="127.0.0.1"),
        ):
            scheduler = d2rh.MooncakeConnectorScheduler(config, "engine", cache_config)

        refresh_mock.assert_called_once_with(config)
        assert calls == [("refresh", 2), ("base", 32)]
        assert scheduler.block_size == 32


class TestRegisterKvCaches:
    def test_propagates_handshake_port_to_agent_metadata(self):
        worker = object.__new__(d2rh.MooncakeConnectorWorker)
        worker.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=False, hf_text_config=SimpleNamespace()),
        )
        worker.kv_cache_config = SimpleNamespace(num_blocks=8)
        worker.kv_role = "kv_producer"
        worker._is_hma_required = False
        worker.handshake_port = 30007
        worker.te_rpc_port = 12345
        worker.block_size = 16
        worker.engine_id = "engine"
        worker.tp_rank = 0
        worker._prefill_tp_size = 2
        worker.side_channel_host = "127.0.0.1"
        worker.side_channel_port = 30000
        worker.pcp_rank = 0
        # Empty layer names keep the D2RH register-region bookkeeping happy
        # while leaving the metadata layer count at zero.
        worker._build_kv_group2layeridx = lambda: {0: ({"layer_names": []}, [])}
        worker._requires_group_aware_attention_transfer = lambda: False

        def _capture_thread(*args, **kwargs):
            # register_kv_caches blocks on the ready event; mark it set so the
            # wait loop exits immediately.
            for arg in args:
                if isinstance(arg, threading.Event):
                    arg.set()
            return MagicMock()

        # The lifecycle is D2RH-local, so all collaborators resolve here too.
        with (
            patch.object(d2rh, "enable_sfa_dcp_replicated_indexer", return_value=False),
            patch.object(d2rh, "validate_register_region_count"),
            patch.object(d2rh, "global_te"),
            patch.object(d2rh, "get_ip", return_value="127.0.0.1"),
            patch.object(d2rh, "MooncakeAgentMetadata") as mock_metadata,
            patch.object(d2rh, "KVCacheSendingThread", side_effect=_capture_thread),
        ):
            worker.register_kv_caches({})

        mock_metadata.assert_called_once()
        kwargs = mock_metadata.call_args.kwargs
        # The worker must forward its handshake port into registered metadata.
        assert kwargs["handshake_port"] == 30007
        assert kwargs["engine_id"] == "engine"
        assert worker.xfer_handshake_metadata is mock_metadata.return_value
