#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    MooncakeStoreConfig,
    _convert_to_bytes,
    _parse_global_segment_size,
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
        with patch.dict(os.environ, {
            "DS_WORKER_ADDR": "host:1234",
            "DS_ENABLE_EXCLUSIVE_CONNECTION": "1",
            "DS_ENABLE_REMOTE_H2D": "0",
        }):
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
        with patch.dict(os.environ, {"MOONCAKE_CONFIG_PATH": "/dev/null"}):
            with patch.object(MooncakeBackend, "__init__", lambda self, pc: None):
                backend = MooncakeBackend.__new__(MooncakeBackend)
                backend.store = MagicMock()
                backend.config = MagicMock()
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

    def test_set_device(self):
        b = self._make_backend()
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_world_group") as mock_wg:
            mock_wg.return_value.local_rank = 0
            b.set_device()

    def test_register_buffer(self):
        b = self._make_backend()
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.os") as mock_os:
            mock_os.getenv.return_value = "0"
            with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te") as mock_te:
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

    def test_set_device(self):
        b = self._make_backend()
        import torch
        b.set_device()

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

    def test_set_device(self):
        b = self._make_backend()
        import torch
        b.set_device()

    def test_register_buffer(self):
        b = self._make_backend()
        with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.get_ascend_device_type") as mock_dt:
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


if __name__ == "__main__":
    unittest.main()
