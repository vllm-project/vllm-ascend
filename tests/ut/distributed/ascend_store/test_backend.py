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
    MooncakeStoreConfig,
    _convert_to_bytes,
    _parse_global_segment_size,
    _ssd_setup_kwargs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import (
    YuanrongConfig,
    YuanrongHelper,
)


def _format_log_call(call):
    args = call.args
    return args[0] % args[1:]


# =========================================================================
# Backend ABC
# =========================================================================
class TestBackendABC(unittest.TestCase):
    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            Backend(MagicMock())  # type: ignore[abstract]


def _make_mooncake_store_config(**overrides) -> MooncakeStoreConfig:
    """Build MooncakeStoreConfig via from_file(); inherits from_file() defaults."""
    config = dict(overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        f.flush()
        path = f.name
    try:
        return MooncakeStoreConfig.from_file(path)
    finally:
        os.unlink(path)


# =========================================================================
# MooncakeStoreConfig
# =========================================================================
class TestMooncakeStoreConfig(unittest.TestCase):
    def test_from_file(self):
        cfg = _make_mooncake_store_config(
            metadata_server="127.0.0.1:2379",
            global_segment_size="2GB",
            local_buffer_size="1GB",
            protocol="ascend",
            device_name="npu0",
            master_server_address="127.0.0.1:8080",
        )
        self.assertEqual(cfg.global_segment_size, 2 * 1024**3)
        self.assertEqual(cfg.local_buffer_size, 1024**3)
        self.assertEqual(cfg.device_name, "npu0")

        defaults = _make_mooncake_store_config()
        self.assertEqual(defaults.protocol, "ascend")
        self.assertEqual(defaults.device_name, "")
        self.assertFalse(defaults.enable_ssd_offload)

        ssd_path = TestMooncakeStoreConfig._writable_ssd_path()
        self.addCleanup(lambda: os.rmdir(ssd_path))
        ssd = _make_mooncake_store_config(enable_ssd_offload=True, ssd_offload_path=ssd_path)
        self.assertEqual(ssd.ssd_offload_path, ssd_path)

    def test_ssd_offload_validation(self):
        for path in ("relative/path", None):
            with self.subTest(path=path), self.assertRaises(ValueError):
                kwargs = {"ssd_offload_path": path} if path else {}
                _make_mooncake_store_config(enable_ssd_offload=True, **kwargs)

    @staticmethod
    def _writable_ssd_path() -> str:
        return tempfile.mkdtemp(prefix="mooncake_ssd_ut_")

    def test_ssd_setup_kwargs(self):
        target = (
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend."
            "mooncake_backend._mooncake_setup_supports_ssd_offload"
        )
        with patch(target, return_value=False):
            self.assertEqual(_ssd_setup_kwargs(_make_mooncake_store_config()), {})

        ssd_path = TestMooncakeStoreConfig._writable_ssd_path()
        self.addCleanup(lambda: os.rmdir(ssd_path))
        cfg = _make_mooncake_store_config(enable_ssd_offload=True, ssd_offload_path=ssd_path)
        with patch(target, return_value=False), self.assertRaises(RuntimeError):
            _ssd_setup_kwargs(cfg)
        with patch(target, return_value=True):
            self.assertEqual(
                _ssd_setup_kwargs(cfg),
                {"enable_ssd_offload": True, "ssd_offload_path": ssd_path},
            )

    def test_load_from_env(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            MooncakeStoreConfig.load_from_env()

        config = {"metadata_server": "host:1234", "master_server_address": "host:5678"}
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
    def test_valid_values(self):
        cases = [
            (1024, 1024),
            ("2GB", 2 * 1024**3),
            ("512MB", 512 * 1024**2),
            ("256KB", 256 * 1024),
            ("4096B", 4096),
            ("2048", 2048),
            (2048.0, 2048),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(_parse_global_segment_size(value), expected)

    def test_invalid_values(self):
        for value, error in [("", ValueError), ("abcGB", ValueError), (None, TypeError)]:
            with self.subTest(value=value), self.assertRaises(error):
                _parse_global_segment_size(value)  # type: ignore[arg-type]


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
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            YuanrongConfig.load_from_env()

        with patch.dict(os.environ, {"DS_WORKER_ADDR": "h:1"}):
            cfg = YuanrongConfig.load_from_env()
            self.assertFalse(cfg.enable_exclusive_connection)
            self.assertFalse(cfg.enable_remote_h2d)

        env = {
            "DS_WORKER_ADDR": "host:1234",
            "DS_ENABLE_EXCLUSIVE_CONNECTION": "1",
            "DS_ENABLE_REMOTE_H2D": "0",
        }
        with patch.dict(os.environ, env):
            cfg = YuanrongConfig.load_from_env()
            self.assertEqual(cfg.worker_addr, "host:1234")
            self.assertTrue(cfg.enable_exclusive_connection)
            self.assertFalse(cfg.enable_remote_h2d)


# =========================================================================
# YuanrongHelper
# =========================================================================
class TestYuanrongHelper(unittest.TestCase):
    def setUp(self):
        self.blob_cls = MagicMock()
        self.blob_list_cls = MagicMock()
        self.helper = YuanrongHelper(self.blob_cls, self.blob_list_cls)

    def test_normalize_keys(self):
        valid = ["abc-123", "key_2", "a" * 1024]
        self.assertEqual(self.helper.normalize_keys(valid), valid)
        for key in ("key with spaces/and.dots", "a" * 1025):
            with self.subTest(key_length=len(key)):
                normalized = self.helper.normalize_keys([key])[0]
                self.assertLessEqual(len(normalized), 1024)
                self.assertIn("__", normalized)

    def test_make_blob_lists(self):
        self.helper._device_id = 0
        result = self.helper.make_blob_lists([[100, 200], [300, 400]], [[10, 20], [30, 40]])
        self.assertEqual(len(result), 2)
        self.assertEqual(self.blob_cls.call_count, 4)

        cases = [
            (0, [[1]], [[1, 2], [3, 4]], ValueError),
            (0, [[1, 2]], [[1]], ValueError),
            (None, [[1]], [[1]], RuntimeError),
        ]
        for device_id, addrs, sizes, error in cases:
            with self.subTest(error=error):
                self.helper._device_id = device_id
                with self.assertRaises(error):
                    self.helper.make_blob_lists(addrs, sizes)


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
            backend.local_seg = "127.0.0.1:1234"
            backend._lazy_init = False
            backend._store_initialized = True
            backend._use_fabric_mem = False
            backend._store_init_lock = MagicMock()
            backend.local_seg = None
            return backend

    def test_exists(self):
        b = self._make_backend()
        b.store.batch_is_exist.return_value = [1, 0]
        result = b.exists(["k1", "k2"])
        self.assertEqual(result, [1, 0])

    def test_transfers(self):
        module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.logger"
        for operation, store_method in [
            ("put", "batch_put_from_multi_buffers"),
            ("get", "batch_get_into_multi_buffers"),
        ]:
            for result in ([0], [-1], RuntimeError("backend fail")):
                with self.subTest(operation=operation, result=result):
                    backend = self._make_backend()
                    method = getattr(backend.store, store_method)
                    if isinstance(result, Exception):
                        method.side_effect = result
                    else:
                        method.return_value = result
                    with patch(module) as logger:
                        getattr(backend, operation)(["k1"], [[100]], [[10]])
                    method.assert_called_once()
                    if result != [0]:
                        logger.error.assert_called()

    def test_register_buffer(self):
        b = self._make_backend()
        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te"
            ) as mock_te,
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_ip"),
        ):
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
            backend._is_a2 = False
            backend._registered_buffers = None
            backend._buffers_registered = False
            backend.config = YuanrongConfig(
                worker_addr="127.0.0.1:0",
                enable_exclusive_connection=False,
                enable_remote_h2d=False,
            )
            backend.rank = 0
            return backend

    def test_exists(self):
        cases = [([], None, []), (["k1", "k2"], [True, False], [1, 0]), (["k1"], Exception("fail"), [0])]
        for keys, result, expected in cases:
            with self.subTest(keys=keys, result=result):
                backend = self._make_backend()
                if isinstance(result, Exception):
                    backend._hetero_client.exist.side_effect = result
                else:
                    backend._hetero_client.exist.return_value = result
                self.assertEqual(backend.exists(keys), expected)

    def test_get(self):
        cases = [
            ([], [], []),
            (["k1"], [], [0]),
            (["k1", "k2", "k3"], ["k2"], [0, 1, 0]),
            (["k1"], ["k1"], [1]),
            (["k1"], RuntimeError("backend fail"), None),
        ]
        for keys, result, expected in cases:
            with self.subTest(keys=keys, result=result):
                backend = self._make_backend()
                if isinstance(result, Exception):
                    backend._hetero_client.mget_h2d.side_effect = result
                else:
                    backend._hetero_client.mget_h2d.return_value = result
                count = len(keys)
                actual = backend.get(keys, [[100]] * count, [[10]] * count)
                self.assertEqual(actual, expected)

    def test_put(self):
        for keys, error in [([], None), (["k1"], None), (["k1"], RuntimeError("backend fail"))]:
            with self.subTest(keys=keys, error=error):
                backend = self._make_backend()
                backend._hetero_client.mset_d2h.side_effect = error
                backend.put(keys, [[100]] * len(keys), [[10]] * len(keys))
                if keys:
                    backend._hetero_client.mset_d2h.assert_called_once()
                else:
                    backend._hetero_client.mset_d2h.assert_not_called()

    def test_register_buffer(self):
        cases = [(False, False, 0), (True, False, 1), (True, True, 0)]
        for enabled, is_a2, expected_calls in cases:
            with self.subTest(enabled=enabled, is_a2=is_a2):
                backend = self._make_backend()
                backend.config.enable_remote_h2d = enabled
                backend._is_a2 = is_a2
                backend.register_buffer([100], [200])
                self.assertEqual(backend._hetero_client.pre_register_device_memory.call_count, expected_calls)
                if expected_calls:
                    backend.register_buffer([300], [400])
                    backend._hetero_client.pre_register_device_memory.assert_called_once_with([100], [200])

    def test_register_buffers_if_needed(self):
        cases = [(True, None, False), (True, ([100], [200]), True), (False, ([100], [200]), False)]
        for enabled, buffers, registered in cases:
            with self.subTest(enabled=enabled, buffers=buffers, registered=registered):
                backend = self._make_backend()
                backend.config.enable_remote_h2d = enabled
                backend._registered_buffers = buffers
                backend._buffers_registered = registered
                backend._register_buffers_if_needed()
                backend._hetero_client.pre_register_device_memory.assert_not_called()

    def test_ensure_device_ready(self):
        for device_id, expected_calls in [(None, 1), (0, 0)]:
            with self.subTest(device_id=device_id):
                backend = self._make_backend()
                backend._helper._device_id = device_id
                backend.set_device = MagicMock()
                backend._ensure_device_ready()
                self.assertEqual(backend.set_device.call_count, expected_calls)


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
            # Set internal state to avoid lazy init logic during tests
            backend._lazy_init = False
            backend._store_initialized = True
            backend._pending_buffers = None
            return backend

    def test_exists(self):
        b = self._make_backend()
        b.store.batch_is_exist.return_value = [1]
        self.assertEqual(b.exists(["k1"]), [1])

    def test_register_buffer(self):
        b = self._make_backend()
        b.register_buffer([100], [200])
        b.store.register_buffer.assert_called_once()

    def test_transfers(self):
        logger_path = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.logger"
        for operation, store_method in [("get", "batch_get_into_layers"), ("put", "batch_put_from_layers")]:
            for result in ([0], [1], RuntimeError("backend fail")):
                with self.subTest(operation=operation, result=result):
                    backend = self._make_backend()
                    method = getattr(backend.store, store_method)
                    if isinstance(result, Exception):
                        method.side_effect = result
                    else:
                        method.return_value = result
                    with patch(logger_path) as logger:
                        getattr(backend, operation)(["k1"], [[100]], [[10]])
                    method.assert_called_once()
                    if result != [0]:
                        logger.error.assert_called()


if __name__ == "__main__":
    unittest.main()
