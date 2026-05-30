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

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES,
    MooncakeStoreConfig,
    _classify_replica_tier,
    _convert_to_bytes,
    _estimate_disk_offload_staging_bytes,
    _get_replica_tiers_by_key,
    _get_usable_disk_offload_buffer_budget_bytes,
    _log_mooncake_load_tier_summary,
    _parse_global_segment_size,
    _split_disk_offload_load_batches,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import (
    YuanrongConfig,
    YuanrongHelper,
)


# =========================================================================
# Backend ABC
# =========================================================================
class TestBackendABC(unittest.TestCase):
    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            Backend(MagicMock())  # type: ignore[abstract]


# =========================================================================
# MooncakeStoreConfig
# =========================================================================
class TestMooncakeStoreConfig(unittest.TestCase):
    def test_from_file(self):
        config = {
            "metadata_server": "127.0.0.1:2379",
            "global_segment_size": "2GB",
            "local_buffer_size": "1GB",
            "protocol": "ascend",
            "device_name": "npu0",
            "master_server_address": "127.0.0.1:8080",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            path = f.name

        try:
            cfg = MooncakeStoreConfig.from_file(path)
            self.assertEqual(cfg.metadata_server, "127.0.0.1:2379")
            self.assertEqual(cfg.global_segment_size, 2 * 1024**3)
            self.assertEqual(cfg.local_buffer_size, 1 * 1024**3)
            self.assertEqual(cfg.protocol, "ascend")
            self.assertEqual(cfg.device_name, "npu0")
        finally:
            os.unlink(path)

    def test_from_file_defaults(self):
        config = {
            "metadata_server": "localhost:2379",
            "master_server_address": "localhost:8080",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            path = f.name

        try:
            cfg = MooncakeStoreConfig.from_file(path)
            self.assertEqual(cfg.protocol, "ascend")
            self.assertEqual(cfg.device_name, "")
        finally:
            os.unlink(path)

    def test_load_from_env_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MOONCAKE_CONFIG_PATH", None)
            with self.assertRaises(ValueError):
                MooncakeStoreConfig.load_from_env()

    def test_load_from_env(self):
        config = {
            "metadata_server": "host:1234",
            "master_server_address": "host:5678",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            path = f.name

        try:
            with patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": path}):
                cfg = MooncakeStoreConfig.load_from_env()
                self.assertEqual(cfg.metadata_server, "host:1234")
        finally:
            os.unlink(path)


class TestParseGlobalSegmentSize(unittest.TestCase):
    def test_int(self):
        self.assertEqual(_parse_global_segment_size(1024), 1024)

    def test_gb(self):
        self.assertEqual(_parse_global_segment_size("2GB"), 2 * 1024**3)

    def test_mb(self):
        self.assertEqual(_parse_global_segment_size("512MB"), 512 * 1024**2)

    def test_kb(self):
        self.assertEqual(_parse_global_segment_size("256KB"), 256 * 1024)

    def test_b(self):
        self.assertEqual(_parse_global_segment_size("4096B"), 4096)

    def test_no_unit(self):
        self.assertEqual(_parse_global_segment_size("2048"), 2048)

    def test_float_input(self):
        self.assertEqual(_parse_global_segment_size(2048.0), 2048)

    def test_empty_string(self):
        with self.assertRaises(ValueError):
            _parse_global_segment_size("")

    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            _parse_global_segment_size("abcGB")

    def test_unsupported_type(self):
        with self.assertRaises(TypeError):
            _parse_global_segment_size(None)  # type: ignore[arg-type]


class TestConvertToBytes(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(_convert_to_bytes("10", 1, "10"), 10)
        self.assertEqual(_convert_to_bytes("1.5", 1024, "1.5KB"), int(1.5 * 1024))

    def test_invalid_number(self):
        with self.assertRaises(ValueError):
            _convert_to_bytes("abc", 1, "abc")


# =========================================================================
# YuanrongConfig
# =========================================================================
class TestYuanrongConfig(unittest.TestCase):
    def test_load_from_env(self):
        with patch.dict(
            os.environ,
            {
                "DS_WORKER_ADDR": "host:1234",
                "DS_ENABLE_EXCLUSIVE_CONNECTION": "1",
                "DS_ENABLE_REMOTE_H2D": "0",
            },
        ):
            cfg = YuanrongConfig.load_from_env()
            self.assertEqual(cfg.worker_addr, "host:1234")
            self.assertTrue(cfg.enable_exclusive_connection)
            self.assertFalse(cfg.enable_remote_h2d)

    def test_load_from_env_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DS_WORKER_ADDR", None)
            with self.assertRaises(ValueError):
                YuanrongConfig.load_from_env()

    def test_load_from_env_defaults(self):
        with patch.dict(os.environ, {"DS_WORKER_ADDR": "h:1"}):
            cfg = YuanrongConfig.load_from_env()
            self.assertFalse(cfg.enable_exclusive_connection)
            self.assertFalse(cfg.enable_remote_h2d)


# =========================================================================
# YuanrongHelper
# =========================================================================
class TestYuanrongHelper(unittest.TestCase):
    def setUp(self):
        self.blob_cls = MagicMock()
        self.blob_list_cls = MagicMock()
        self.helper = YuanrongHelper(self.blob_cls, self.blob_list_cls)

    def test_normalize_keys_short_valid(self):
        keys = ["abc-123", "key_2"]
        result = self.helper.normalize_keys(keys)
        self.assertEqual(result, keys)

    def test_normalize_keys_with_invalid_chars(self):
        keys = ["key with spaces/and.dots"]
        result = self.helper.normalize_keys(keys)
        self.assertEqual(len(result), 1)
        # Should not contain the original invalid chars
        self.assertNotIn(" ", result[0])
        self.assertNotIn("/", result[0])
        # Should have hash suffix
        self.assertIn("__", result[0])

    def test_normalize_keys_long_key(self):
        long_key = "a" * 300
        result = self.helper.normalize_keys([long_key])
        self.assertEqual(len(result), 1)
        self.assertLessEqual(len(result[0]), 255)

    def test_make_blob_lists(self):
        self.helper._device_id = 0
        addrs = [[100, 200], [300, 400]]
        sizes = [[10, 20], [30, 40]]
        result = self.helper.make_blob_lists(addrs, sizes)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.blob_cls.call_count, 4)

    def test_make_blob_lists_length_mismatch(self):
        self.helper._device_id = 0
        with self.assertRaises(ValueError):
            self.helper.make_blob_lists([[1]], [[1, 2], [3, 4]])

    def test_make_blob_lists_inner_length_mismatch(self):
        self.helper._device_id = 0
        with self.assertRaises(ValueError):
            self.helper.make_blob_lists([[1, 2]], [[1]])

    def test_make_blob_lists_no_device(self):
        self.helper._device_id = None
        with self.assertRaises(RuntimeError):
            self.helper.make_blob_lists([[1]], [[1]])


# =========================================================================
# MooncakeBackend (mocked store)
# =========================================================================
class TestMooncakeBackendMethods(unittest.TestCase):
    def _make_backend(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

        with (
            patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": "/dev/null"}),
            patch.object(MooncakeBackend, "__init__", lambda self, pc: None),
        ):
            backend = MooncakeBackend.__new__(MooncakeBackend)
            backend.store = MagicMock()
            backend.config = MagicMock()
            # Defaults expected by the post-PR get()/put() paths. Tests can
            # override individually when they want disk-offload semantics.
            backend.disk_offload_buffer_budget_bytes = None
            backend.usable_disk_offload_buffer_budget_bytes = None
            backend.replicate_config = None
            return backend

    def test_exists(self):
        b = self._make_backend()
        b.store.batch_is_exist.return_value = [1, 0]
        result = b.exists(["k1", "k2"])
        self.assertEqual(result, [1, 0])

    def test_put(self):
        b = self._make_backend()
        b.store.batch_put_from_multi_buffers.return_value = [0, 0]
        b.put(["k1"], [[100]], [[10]])
        b.store.batch_put_from_multi_buffers.assert_called_once()

    def test_put_error(self):
        b = self._make_backend()
        b.store.batch_put_from_multi_buffers.return_value = [-1]
        b.put(["k1"], [[100]], [[10]])  # Should log error but not raise

    def test_put_exception(self):
        b = self._make_backend()
        b.store.batch_put_from_multi_buffers.side_effect = Exception("fail")
        b.put(["k1"], [[100]], [[10]])  # Should log error but not raise

    def test_get(self):
        b = self._make_backend()
        b.store.batch_get_into_multi_buffers.return_value = [0]
        b.get(["k1"], [[100]], [[10]])
        b.store.batch_get_into_multi_buffers.assert_called_once()

    def test_get_error(self):
        b = self._make_backend()
        b.store.batch_get_into_multi_buffers.return_value = [-1]
        b.get(["k1"], [[100]], [[10]])

    def test_get_exception(self):
        b = self._make_backend()
        b.store.batch_get_into_multi_buffers.side_effect = Exception("fail")
        b.get(["k1"], [[100]], [[10]])

    def test_register_buffer(self):
        b = self._make_backend()
        with (
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.os") as mock_os,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te"
            ) as mock_te,
        ):
            mock_os.getenv.return_value = "0"
            b.register_buffer([100], [200])
            mock_te.register_buffer.assert_called_once()


# =========================================================================
# YuanrongBackend (mocked store)
# =========================================================================
class TestYuanrongBackendMethods(unittest.TestCase):
    def _make_backend(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import YuanrongBackend

        with patch.object(YuanrongBackend, "__init__", lambda self, pc: None):
            backend = YuanrongBackend.__new__(YuanrongBackend)
            backend._helper = MagicMock()
            backend._helper._device_id = 0
            backend._helper.normalize_keys = lambda keys: keys
            backend._helper.make_blob_lists = lambda a, s: [MagicMock() for _ in a]
            backend._hetero_client = MagicMock()
            backend._ds_set_param = MagicMock()
            backend.rank = 0
            return backend

    def test_exists_empty(self):
        b = self._make_backend()
        result = b.exists([])
        self.assertEqual(result, [])

    def test_exists(self):
        b = self._make_backend()
        b._hetero_client.exist.return_value = [True, False]
        result = b.exists(["k1", "k2"])
        self.assertEqual(result, [1, 0])

    def test_exists_exception(self):
        b = self._make_backend()
        b._hetero_client.exist.side_effect = Exception("fail")
        result = b.exists(["k1"])
        self.assertEqual(result, [0])

    def test_get_empty(self):
        b = self._make_backend()
        b.get([], [], [])
        b._hetero_client.mget_h2d.assert_not_called()

    def test_get(self):
        b = self._make_backend()
        b._hetero_client.mget_h2d.return_value = []
        b.get(["k1"], [[100]], [[10]])
        b._hetero_client.mget_h2d.assert_called_once()

    def test_get_failed_keys(self):
        b = self._make_backend()
        b._hetero_client.mget_h2d.return_value = ["k1"]
        b.get(["k1"], [[100]], [[10]])  # Should log error

    def test_get_exception(self):
        b = self._make_backend()
        b._hetero_client.mget_h2d.side_effect = Exception("fail")
        b.get(["k1"], [[100]], [[10]])

    def test_put_empty(self):
        b = self._make_backend()
        b.put([], [], [])
        b._hetero_client.mset_d2h.assert_not_called()

    def test_put(self):
        b = self._make_backend()
        b.put(["k1"], [[100]], [[10]])
        b._hetero_client.mset_d2h.assert_called_once()

    def test_put_exception(self):
        b = self._make_backend()
        b._hetero_client.mset_d2h.side_effect = Exception("fail")
        b.put(["k1"], [[100]], [[10]])

    def test_register_buffer(self):
        b = self._make_backend()
        b._helper._device_id = None
        b._ensure_device_ready = MagicMock()
        b.register_buffer([100], [200])
        b._ensure_device_ready.assert_called_once()

    def test_ensure_device_ready(self):
        b = self._make_backend()
        b._helper._device_id = None
        b.set_device = MagicMock()
        b._ensure_device_ready()
        b.set_device.assert_called_once()

    def test_ensure_device_ready_already_set(self):
        b = self._make_backend()
        b._helper._device_id = 0
        b.set_device = MagicMock()
        b._ensure_device_ready()
        b.set_device.assert_not_called()


# =========================================================================
# MemcacheBackend (mocked store)
# =========================================================================
class TestMemcacheBackendMethods(unittest.TestCase):
    def _make_backend(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend

        with patch.object(MemcacheBackend, "__init__", lambda self, pc: None):
            backend = MemcacheBackend.__new__(MemcacheBackend)
            backend.store = MagicMock()
            backend.local_rank = 0
            return backend

    def test_exists(self):
        b = self._make_backend()
        b.store.batch_is_exist.return_value = [1]
        self.assertEqual(b.exists(["k1"]), [1])

    def test_register_buffer(self):
        b = self._make_backend()
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.get_ascend_device_type"
        ) as mock_dt:
            from vllm_ascend.utils import AscendDeviceType

            mock_dt.return_value = AscendDeviceType.A2
            b.register_buffer([100], [200])

    def test_get(self):
        b = self._make_backend()
        b.store.batch_get_into_layers.return_value = [0]
        b.get(["k1"], [[100]], [[10]])
        b.store.batch_get_into_layers.assert_called_once()

    def test_get_error(self):
        b = self._make_backend()
        b.store.batch_get_into_layers.return_value = [1]  # non-zero = error
        b.get(["k1"], [[100]], [[10]])

    def test_get_exception(self):
        b = self._make_backend()
        b.store.batch_get_into_layers.side_effect = Exception("fail")
        b.get(["k1"], [[100]], [[10]])

    def test_put(self):
        b = self._make_backend()
        b.store.batch_put_from_layers.return_value = [0]
        b.put(["k1"], [[100]], [[10]])
        b.store.batch_put_from_layers.assert_called_once()

    def test_put_error(self):
        b = self._make_backend()
        b.store.batch_put_from_layers.return_value = [1]
        b.put(["k1"], [[100]], [[10]])

    def test_put_exception(self):
        b = self._make_backend()
        b.store.batch_put_from_layers.side_effect = Exception("fail")
        b.put(["k1"], [[100]], [[10]])


# =========================================================================
# MooncakeStoreConfig validation (__post_init__)
# =========================================================================
class TestMooncakeStoreConfigValidation(unittest.TestCase):
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            metadata_server="m:1",
            protocol="ascend",
            device_name="npu0",
            master_server_address="m:2",
        )
        kwargs.update(overrides)
        return kwargs

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError) as ctx:
            MooncakeStoreConfig(**self._base_kwargs(mode="unknown"))  # type: ignore[arg-type]
        self.assertIn("unknown Mooncake mode", str(ctx.exception))

    def test_zero_local_buffer_raises(self):
        with self.assertRaises(ValueError):
            MooncakeStoreConfig(**self._base_kwargs(local_buffer_size=0))

    def test_embedded_zero_segment_raises(self):
        with self.assertRaises(ValueError) as ctx:
            MooncakeStoreConfig(**self._base_kwargs(global_segment_size=0))
        self.assertIn("embedded", str(ctx.exception))

    def test_standalone_nonzero_segment_raises(self):
        with self.assertRaises(ValueError) as ctx:
            MooncakeStoreConfig(
                **self._base_kwargs(mode="standalone-store", global_segment_size=1024)
            )
        self.assertIn("standalone-store", str(ctx.exception))

    def test_standalone_zero_segment_ok(self):
        cfg = MooncakeStoreConfig(
            **self._base_kwargs(mode="standalone-store", global_segment_size=0)
        )
        self.assertEqual(cfg.mode, "standalone-store")
        self.assertFalse(cfg.enable_offload)

    def test_from_file_parses_enable_offload(self):
        config = {
            "metadata_server": "m:1",
            "master_server_address": "m:2",
            "mode": "standalone-store",
            "global_segment_size": 0,
            "enable_offload": True,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            path = f.name
        try:
            cfg = MooncakeStoreConfig.from_file(path)
            self.assertEqual(cfg.mode, "standalone-store")
            self.assertEqual(cfg.global_segment_size, 0)
            self.assertTrue(cfg.enable_offload)
        finally:
            os.unlink(path)


# =========================================================================
# Disk-offload staging helpers
# =========================================================================
class TestEstimateDiskOffloadStagingBytes(unittest.TestCase):
    def test_aligns_up_and_pads(self):
        # 1 byte payload rounds up to one 4 KiB block + two padding blocks.
        self.assertEqual(_estimate_disk_offload_staging_bytes([1]), 4096 + 2 * 4096)

    def test_scatter_gather_sums(self):
        self.assertEqual(
            _estimate_disk_offload_staging_bytes([4096, 4096]),
            8192 + 2 * 4096,
        )

    def test_empty_list(self):
        # Empty payload still pays the two-block padding (rare but documented).
        self.assertEqual(_estimate_disk_offload_staging_bytes([]), 2 * 4096)


class TestUsableDiskOffloadBudget(unittest.TestCase):
    def test_applies_ratio(self):
        # Default ratio is 0.9; mocking is overkill since the function reads
        # the live env wrapper. We assert ``int(budget * ratio) <= budget``.
        budget = 10_000_000
        usable = _get_usable_disk_offload_buffer_budget_bytes(budget)
        self.assertGreater(usable, 0)
        self.assertLessEqual(usable, budget)

    def test_min_one(self):
        # Round-down on a tiny budget still produces at least 1.
        self.assertGreaterEqual(_get_usable_disk_offload_buffer_budget_bytes(1), 1)


class TestSplitDiskOffloadLoadBatches(unittest.TestCase):
    def test_under_budget_single_batch(self):
        keys = ["a", "b"]
        sizes = [[100], [100]]
        addrs = [[0], [0]]
        # Soft budget large enough for both keys.
        batches, oversize = _split_disk_offload_load_batches(
            keys, addrs, sizes, usable_budget_bytes=10**9, raw_budget_bytes=10**9
        )
        self.assertIsNone(oversize)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0], keys)

    def test_splits_when_soft_budget_exceeded(self):
        keys = ["a", "b", "c"]
        sizes = [[4096], [4096], [4096]]
        addrs = [[0], [0], [0]]
        # Each key estimate = 4096 + 8192 = 12288. Soft cap of 16384
        # forces one key per batch (12288 + 12288 > 16384).
        batches, oversize = _split_disk_offload_load_batches(
            keys, addrs, sizes, usable_budget_bytes=16384, raw_budget_bytes=10**9
        )
        self.assertIsNone(oversize)
        self.assertEqual([b[0] for b in batches], [["a"], ["b"], ["c"]])

    def test_oversized_key_returns_skip_signal(self):
        keys = ["a", "huge", "c"]
        sizes = [[100], [10**9], [100]]
        addrs = [[0], [0], [0]]
        batches, oversize = _split_disk_offload_load_batches(
            keys, addrs, sizes, usable_budget_bytes=10_000, raw_budget_bytes=20_000
        )
        # "huge" exceeds the *hard* cap → propagate skip-signal upstream.
        self.assertEqual(oversize, "huge")
        self.assertEqual(batches, [])

    def test_single_overflowing_key_solo_batch(self):
        # Key bigger than soft cap but smaller than hard cap is sent solo.
        keys = ["a", "fat"]
        sizes = [[100], [12_000]]
        addrs = [[0], [0]]
        batches, oversize = _split_disk_offload_load_batches(
            keys,
            addrs,
            sizes,
            usable_budget_bytes=10_000,
            raw_budget_bytes=50_000,
        )
        self.assertIsNone(oversize)
        # ``a`` fits the soft cap → its own batch; ``fat`` exceeds the
        # soft cap → solo batch (never co-batched).
        batch_keys = [b[0] for b in batches]
        self.assertEqual(batch_keys, [["a"], ["fat"]])


# =========================================================================
# Replica-tier classification + logging
# =========================================================================
class _FakeReplicaDesc:
    def __init__(self, memory=False, disk=False, local_disk=False):
        self._memory = memory
        self._disk = disk
        self._local_disk = local_disk

    def is_memory_replica(self):
        return self._memory

    def is_disk_replica(self):
        return self._disk

    def is_local_disk_replica(self):
        return self._local_disk


class TestClassifyReplicaTier(unittest.TestCase):
    def test_empty_unknown(self):
        self.assertEqual(_classify_replica_tier(None), "unknown")
        self.assertEqual(_classify_replica_tier([]), "unknown")
        self.assertEqual(_classify_replica_tier({}), "unknown")

    def test_memory(self):
        self.assertEqual(_classify_replica_tier([_FakeReplicaDesc(memory=True)]), "memory")

    def test_disk(self):
        self.assertEqual(_classify_replica_tier([_FakeReplicaDesc(disk=True)]), "disk")

    def test_local_disk_is_disk(self):
        # Older mooncake exposes ``is_local_disk_replica``; classify as disk.
        self.assertEqual(_classify_replica_tier([_FakeReplicaDesc(local_disk=True)]), "disk")

    def test_unknown_when_no_predicate_matches(self):
        # Replica with neither predicate truthy → "unknown".
        self.assertEqual(_classify_replica_tier([_FakeReplicaDesc()]), "unknown")

    def test_unindexable_returns_unknown(self):
        # Some types raise TypeError on indexing — the helper must swallow that.
        class NotIndexable:
            def __bool__(self):
                return True

        self.assertEqual(_classify_replica_tier(NotIndexable()), "unknown")


class TestGetReplicaTiersByKey(unittest.TestCase):
    def test_mapping_return(self):
        store = MagicMock()
        store.batch_get_replica_desc.return_value = {
            "a": [_FakeReplicaDesc(memory=True)],
            "b": [_FakeReplicaDesc(disk=True)],
        }
        tiers = _get_replica_tiers_by_key(store, ["a", "b"])
        self.assertEqual(tiers, {"a": "memory", "b": "disk"})

    def test_list_return_positional(self):
        # The fallback path: store returns a list parallel to keys.
        store = MagicMock()
        store.batch_get_replica_desc.return_value = [
            [_FakeReplicaDesc(memory=True)],
            [_FakeReplicaDesc(disk=True)],
        ]
        tiers = _get_replica_tiers_by_key(store, ["k0", "k1"])
        self.assertEqual(tiers, {"k0": "memory", "k1": "disk"})

    def test_api_exception_returns_all_unknown(self):
        store = MagicMock()
        store.batch_get_replica_desc.side_effect = Exception("api gone")
        tiers = _get_replica_tiers_by_key(store, ["a", "b"])
        self.assertEqual(tiers, {"a": "unknown", "b": "unknown"})


class TestLogMooncakeLoadTierSummary(unittest.TestCase):
    def test_summary_counts(self):
        # Should not raise; we only assert the function tolerates partial
        # input (load_results shorter than batch_keys).
        _log_mooncake_load_tier_summary(
            batch_keys=["a", "b", "c"],
            load_results=[100, -1],  # third key has no result → treated as failed
            tiers_by_key={"a": "memory", "b": "disk", "c": "unknown"},
        )

    def test_unexpected_tier_falls_back_to_unknown(self):
        # Tier not in {memory,disk,unknown} should not crash the summarizer.
        _log_mooncake_load_tier_summary(
            batch_keys=["k"],
            load_results=[42],
            tiers_by_key={"k": "garbage"},
        )


# =========================================================================
# MooncakeBackend.get() — disk-offload batching path
# =========================================================================
class TestMooncakeBackendGetWithOffload(unittest.TestCase):
    def _make_backend(self, *, disk_offload: bool, hard_cap: int | None = None):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
            MooncakeBackend,
        )

        with patch.object(MooncakeBackend, "__init__", lambda self, pc: None):
            backend = MooncakeBackend.__new__(MooncakeBackend)
        backend.store = MagicMock()
        backend.config = MagicMock()
        if disk_offload:
            hard = hard_cap if hard_cap is not None else DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES
            backend.disk_offload_buffer_budget_bytes = hard
            backend.usable_disk_offload_buffer_budget_bytes = (
                _get_usable_disk_offload_buffer_budget_bytes(hard)
            )
        else:
            backend.disk_offload_buffer_budget_bytes = None
            backend.usable_disk_offload_buffer_budget_bytes = None
        return backend

    def test_empty_keys_skips_store_call(self):
        b = self._make_backend(disk_offload=False)
        b.get([], [], [])
        b.store.batch_get_into_multi_buffers.assert_not_called()

    def test_no_offload_uses_single_batch(self):
        b = self._make_backend(disk_offload=False)
        b.store.batch_get_into_multi_buffers.return_value = [0, 0]
        b.get(["k1", "k2"], [[10], [10]], [[100], [100]])
        # Single batched call, all keys at once.
        b.store.batch_get_into_multi_buffers.assert_called_once_with(
            ["k1", "k2"], [[10], [10]], [[100], [100]]
        )

    def test_offload_splits_when_total_exceeds_soft_budget(self):
        # Tiny hard cap → soft cap = 0.9 * 100_000 ≈ 90_000 bytes.
        # Each key staging = 4096 + 8192 = 12288 bytes.
        # Eight keys ≈ 98_304 bytes > soft cap → splits into 2 sub-batches.
        b = self._make_backend(disk_offload=True, hard_cap=100_000)

        # Return a success result whose length matches the batch — the
        # post-PR ``get()`` uses ``zip(..., strict=True)`` for failure
        # detection, so a fixed-length mock crashes on partial batches.
        def _ok_per_batch(batch_keys, _addrs, _sizes):
            return [0] * len(batch_keys)

        b.store.batch_get_into_multi_buffers.side_effect = _ok_per_batch
        keys = [f"k{i}" for i in range(8)]
        sizes = [[4096] for _ in range(8)]
        addrs = [[i] for i in range(8)]
        b.get(keys, addrs, sizes)
        # Should have split into >1 sub-batch (no single batch covers all).
        self.assertGreater(b.store.batch_get_into_multi_buffers.call_count, 1)
        # And every batch should respect the soft cap.
        for call_args in b.store.batch_get_into_multi_buffers.call_args_list:
            batch_keys = call_args.args[0]
            self.assertLessEqual(len(batch_keys), 8)

    def test_offload_skips_oversized_key(self):
        b = self._make_backend(disk_offload=True, hard_cap=20_000)
        keys = ["normal", "monster"]
        # Monster key staging > hard cap → entire request is skipped.
        sizes = [[100], [100_000]]
        addrs = [[0], [0]]
        b.get(keys, addrs, sizes)
        b.store.batch_get_into_multi_buffers.assert_not_called()

    def test_get_swallows_backend_exception(self):
        b = self._make_backend(disk_offload=False)
        b.store.batch_get_into_multi_buffers.side_effect = Exception("backend down")
        # Must not raise — transfer threads expect get() to log and return.
        b.get(["k1"], [[0]], [[10]])

    def test_get_stops_on_first_failed_sub_batch(self):
        b = self._make_backend(disk_offload=True, hard_cap=100_000)

        # Match each sub-batch's length and return all-failures (-1).
        def _fail_per_batch(batch_keys, _addrs, _sizes):
            return [-1] * len(batch_keys)

        b.store.batch_get_into_multi_buffers.side_effect = _fail_per_batch
        keys = [f"k{i}" for i in range(8)]
        sizes = [[4096] for _ in range(8)]
        addrs = [[i] for i in range(8)]
        b.get(keys, addrs, sizes)
        # Stopped after first failing sub-batch (didn't dispatch all batches).
        # split: 7-then-1 → call_count is 1 since we break on first failure.
        self.assertEqual(b.store.batch_get_into_multi_buffers.call_count, 1)

    def test_tier_log_enabled_calls_batch_get_replica_desc(self):
        b = self._make_backend(disk_offload=False)
        b.store.batch_get_into_multi_buffers.return_value = [0]
        b.store.batch_get_replica_desc.return_value = {
            "k1": [_FakeReplicaDesc(memory=True)],
        }
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend"
            ".mooncake_backend.ascend_envs"
        ) as mock_envs:
            mock_envs.VLLM_MOONCAKE_STORE_TIER_LOG = True
            b.get(["k1"], [[0]], [[10]])
        b.store.batch_get_replica_desc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
