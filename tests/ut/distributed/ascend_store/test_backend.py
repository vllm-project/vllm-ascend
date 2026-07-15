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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MmcDirect
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


def _make_mooncake_runtime_config(**overrides) -> MooncakeStoreConfig:
    config = {
        "metadata_server": "127.0.0.1:2379",
        "global_segment_size": 4096,
        "local_buffer_size": 2048,
        "protocol": "ascend",
        "device_name": "npu0",
        "master_server_address": "127.0.0.1:8080",
        "preferred_segment": False,
        "prefer_alloc_in_same_node": True,
    }
    config.update(overrides)
    return MooncakeStoreConfig(**config)


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
            self.assertFalse(cfg.enable_ssd_offload)
            self.assertEqual(cfg.ssd_offload_path, "")
        finally:
            os.unlink(path)

    def test_from_file_ssd_offload(self):
        ssd_path = TestMooncakeStoreConfig._writable_ssd_path()
        self.addCleanup(lambda: os.rmdir(ssd_path))
        cfg = _make_mooncake_store_config(
            enable_ssd_offload=True,
            ssd_offload_path=ssd_path,
        )
        self.assertTrue(cfg.enable_ssd_offload)
        self.assertEqual(cfg.ssd_offload_path, ssd_path)

    def test_ssd_offload_requires_absolute_path(self):
        with self.assertRaises(ValueError):
            _make_mooncake_store_config(
                enable_ssd_offload=True,
                ssd_offload_path="relative/path",
            )

    def test_ssd_offload_requires_path_in_json(self):
        with self.assertRaises(ValueError):
            _make_mooncake_store_config(enable_ssd_offload=True)

    @staticmethod
    def _writable_ssd_path() -> str:
        return tempfile.mkdtemp(prefix="mooncake_ssd_ut_")

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend."
        "mooncake_backend._mooncake_setup_supports_ssd_offload",
        return_value=False,
    )
    def test_ssd_setup_kwargs_off_when_disabled(self, _mock_supports):
        cfg = _make_mooncake_store_config()
        self.assertEqual(_ssd_setup_kwargs(cfg), {})

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend."
        "mooncake_backend._mooncake_setup_supports_ssd_offload",
        return_value=False,
    )
    def test_ssd_setup_kwargs_raises_on_old_mooncake(self, _mock_supports):
        ssd_path = TestMooncakeStoreConfig._writable_ssd_path()
        self.addCleanup(lambda: os.rmdir(ssd_path))
        cfg = _make_mooncake_store_config(
            enable_ssd_offload=True,
            ssd_offload_path=ssd_path,
        )
        with self.assertRaises(RuntimeError):
            _ssd_setup_kwargs(cfg)

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend."
        "mooncake_backend._mooncake_setup_supports_ssd_offload",
        return_value=True,
    )
    def test_ssd_setup_kwargs_when_supported(self, _mock_supports):
        ssd_path = TestMooncakeStoreConfig._writable_ssd_path()
        self.addCleanup(lambda: os.rmdir(ssd_path))
        cfg = _make_mooncake_store_config(
            enable_ssd_offload=True,
            ssd_offload_path=ssd_path,
        )
        self.assertEqual(
            _ssd_setup_kwargs(cfg),
            {
                "enable_ssd_offload": cfg.enable_ssd_offload,
                "ssd_offload_path": cfg.ssd_offload_path,
            },
        )

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

    def test_normalize_keys_at_max_length(self):
        max_length_key = "a" * 1024
        result = self.helper.normalize_keys([max_length_key])
        self.assertEqual(result, [max_length_key])

    def test_normalize_keys_over_max_length(self):
        long_key = "a" * 1025
        result = self.helper.normalize_keys([long_key])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1024)
        self.assertIn("__", result[0])

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
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te"
            ) as mock_te,
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_ip"),
        ):
            b.register_buffer([100], [200])
            mock_te.register_buffer.assert_called_once()

    def test_init_fabric_memory_sets_local_buffer_zero(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

        cfg = _make_mooncake_runtime_config()
        with (
            patch.dict(os.environ, {"ASCEND_ENABLE_USE_FABRIC_MEM": "1"}),
            patch.object(MooncakeStoreConfig, "load_from_env", return_value=cfg),
            patch("mooncake.store.MooncakeDistributedStore", create=True) as mock_store_cls,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_ip",
                return_value="10.0.0.1",
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te"
            ) as mock_te,
        ):
            mock_store_cls.return_value.setup.return_value = 0
            backend = MooncakeBackend(MagicMock())

        setup_kwargs = mock_store_cls.return_value.setup.call_args.kwargs
        self.assertTrue(backend._use_fabric_mem)
        self.assertEqual(setup_kwargs["local_hostname"], "10.0.0.1")
        self.assertEqual(setup_kwargs["local_buffer_size"], 0)
        self.assertNotIn("engine", setup_kwargs)
        mock_te.get_transfer_engine.assert_not_called()

    def test_register_buffer_skips_transfer_engine_for_fabric_memory(self):
        b = self._make_backend()
        b._use_fabric_mem = True

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te"
        ) as mock_te:
            b.register_buffer([100], [200])

        mock_te.get_transfer_engine.assert_not_called()
        mock_te.register_buffer.assert_not_called()

    def test_lazy_init_fabric_memory_put_initializes_store(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

        cfg = _make_mooncake_runtime_config()
        store = MagicMock()
        store.batch_put_from_multi_buffers.return_value = [0]
        with (
            patch.dict(os.environ, {"ASCEND_ENABLE_USE_FABRIC_MEM": "1"}),
            patch.object(MooncakeStoreConfig, "load_from_env", return_value=cfg),
            patch.object(MooncakeBackend, "_setup_store", return_value=store) as mock_setup,
        ):
            backend = MooncakeBackend(MagicMock(), lazy_init=True)
            self.assertFalse(backend._store_initialized)
            mock_setup.assert_not_called()

            backend.put(["k1"], [[100]], [[10]])

        mock_setup.assert_called_once_with()
        store.batch_put_from_multi_buffers.assert_called_once()
        self.assertTrue(backend._store_initialized)

    def test_init_non_fabric_uses_transfer_engine(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

        cfg = _make_mooncake_runtime_config()
        transfer_engine = MagicMock()
        transfer_engine.get_rpc_port.return_value = 12345
        transfer_engine.get_engine.return_value = "engine"
        with (
            patch.dict(os.environ, {"ASCEND_ENABLE_USE_FABRIC_MEM": "0"}),
            patch.object(MooncakeStoreConfig, "load_from_env", return_value=cfg),
            patch("mooncake.store.MooncakeDistributedStore", create=True) as mock_store_cls,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_ip",
                return_value="10.0.0.2",
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te"
            ) as mock_te,
        ):
            mock_store_cls.return_value.setup.return_value = 0
            mock_te.get_transfer_engine.return_value = transfer_engine
            backend = MooncakeBackend(MagicMock())

        setup_kwargs = mock_store_cls.return_value.setup.call_args.kwargs
        mock_te.get_transfer_engine.assert_called_once_with("10.0.0.2", device_name=None)
        self.assertEqual(backend.local_seg, "10.0.0.2:12345")
        self.assertEqual(setup_kwargs["local_hostname"], "10.0.0.2:12345")
        self.assertEqual(setup_kwargs["local_buffer_size"], cfg.local_buffer_size)
        self.assertEqual(setup_kwargs["engine"], "engine")

    def test_exists_lazy_uninitialized_returns_missing(self):
        b = self._make_backend()
        b._lazy_init = True
        b._store_initialized = False
        b.store = None

        result = b.exists(["k1", "k2"])

        self.assertEqual(result, [0, 0])


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
        result = b.get([], [], [])
        self.assertEqual(result, [])
        b._hetero_client.mget_h2d.assert_not_called()

    def test_get(self):
        b = self._make_backend()
        b._hetero_client.mget_h2d.return_value = []
        result = b.get(["k1"], [[100]], [[10]])
        self.assertEqual(result, [0])
        b._hetero_client.mget_h2d.assert_called_once()

    def test_get_partial_failure(self):
        b = self._make_backend()
        b._hetero_client.mget_h2d.return_value = ["k2"]
        result = b.get(["k1", "k2", "k3"], [[100], [200], [300]], [[10], [20], [30]])
        self.assertEqual(result, [0, 1, 0])

    def test_get_failed_keys(self):
        b = self._make_backend()
        b._hetero_client.mget_h2d.return_value = ["k1"]
        result = b.get(["k1"], [[100]], [[10]])  # Should log error
        self.assertEqual(result, [1])

    def test_get_exception(self):
        b = self._make_backend()
        b._hetero_client.mget_h2d.side_effect = Exception("fail")
        result = b.get(["k1"], [[100]], [[10]])
        self.assertIsNone(result)

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

    def test_register_buffer_noop_when_remote_h2d_disabled(self):
        b = self._make_backend()
        b.register_buffer([100], [200])
        b._hetero_client.pre_register_device_memory.assert_not_called()

    def test_register_buffer_when_remote_h2d_enabled(self):
        b = self._make_backend()
        b.config.enable_remote_h2d = True
        b.register_buffer([100], [200])
        b._hetero_client.pre_register_device_memory.assert_called_once_with([100], [200])

    def test_register_buffer_noop_on_a2(self):
        # A2 must not register (opposite of memcache_backend's _is_a2 gating).
        b = self._make_backend()
        b._is_a2 = True
        b.config.enable_remote_h2d = True
        b.register_buffer([100], [200])
        b._hetero_client.pre_register_device_memory.assert_not_called()

    def test_register_buffer_idempotent(self):
        b = self._make_backend()
        b.config.enable_remote_h2d = True
        b.register_buffer([100], [200])
        b.register_buffer([300], [400])
        b._hetero_client.pre_register_device_memory.assert_called_once_with([100], [200])

    def test_register_buffers_if_needed_no_buffers(self):
        b = self._make_backend()
        b.config.enable_remote_h2d = True
        b._registered_buffers = None
        b._register_buffers_if_needed()
        b._hetero_client.pre_register_device_memory.assert_not_called()

    def test_register_buffers_if_needed_already_registered(self):
        b = self._make_backend()
        b.config.enable_remote_h2d = True
        b._registered_buffers = ([100], [200])
        b._buffers_registered = True
        b._register_buffers_if_needed()
        b._hetero_client.pre_register_device_memory.assert_not_called()

    def test_register_buffers_if_needed_disabled(self):
        b = self._make_backend()
        b.config.enable_remote_h2d = False
        b._registered_buffers = ([100], [200])
        b._register_buffers_if_needed()
        b._hetero_client.pre_register_device_memory.assert_not_called()

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

    def test_init_creates_hetero_client_and_calls_init(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import YuanrongBackend

        blob_cls = MagicMock()
        blob_list_cls = MagicMock()
        hetero_client_cls = MagicMock()
        set_param = MagicMock()
        set_param_cls = MagicMock(return_value=set_param)
        write_mode = MagicMock()
        write_mode.NONE_L2_CACHE_EVICT = "none_l2_cache_evict"
        config = YuanrongConfig(
            worker_addr="127.0.0.1:8765",
            enable_exclusive_connection=True,
            enable_remote_h2d=False,
        )

        with (
            patch("yr.datasystem.hetero_client.Blob", new=blob_cls, create=True),
            patch("yr.datasystem.hetero_client.DeviceBlobList", new=blob_list_cls, create=True),
            patch("yr.datasystem.hetero_client.HeteroClient", new=hetero_client_cls, create=True),
            patch("yr.datasystem.kv_client.SetParam", new=set_param_cls, create=True),
            patch("yr.datasystem.object_client.WriteMode", new=write_mode, create=True),
            patch.object(YuanrongConfig, "load_from_env", return_value=config),
        ):
            backend = YuanrongBackend(MagicMock())

        hetero_client_cls.assert_called_once_with(
            "127.0.0.1",
            8765,
            enable_exclusive_connection=True,
            enable_remote_h2d=False,
        )
        hetero_client_cls.return_value.init.assert_called_once_with()
        self.assertIs(backend._helper._blob_cls, blob_cls)
        self.assertIs(backend._helper._blob_list_cls, blob_list_cls)
        self.assertEqual(backend._ds_set_param.write_mode, "none_l2_cache_evict")

    def test_set_device_sets_helper_device_id(self):
        b = self._make_backend()
        device = MagicMock()
        world_group = MagicMock(local_rank=3)

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend.get_world_group",
                return_value=world_group,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend.torch.device",
                return_value=device,
            ) as mock_device,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend.torch.npu.set_device"
            ) as mock_set_device,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend.torch.npu.current_device",
                return_value=3,
            ),
        ):
            b.set_device()

        mock_device.assert_called_once_with("npu:3")
        mock_set_device.assert_called_once_with(device)
        self.assertEqual(b._helper._device_id, 3)


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
            backend._is_a2 = False
            backend._registered_buffers = None
            backend._buffers_registered = False
            return backend

    def test_exists(self):
        b = self._make_backend()
        b.store.batch_is_exist.return_value = [1]
        self.assertEqual(b.exists(["k1"]), [1])

    def test_set_device_uses_local_rank(self):
        b = self._make_backend()
        b.local_rank = 2
        device = MagicMock()

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.torch.device",
                return_value=device,
            ) as mock_device,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.torch.npu.set_device"
            ) as mock_set_device,
        ):
            b.set_device()

        mock_device.assert_called_once_with("npu:2")
        mock_set_device.assert_called_once_with(device)

    def test_mmc_direct_enum_values(self):
        self.assertEqual(MmcDirect.COPY_L2G.value, 0)
        self.assertEqual(MmcDirect.COPY_G2L.value, 1)
        self.assertEqual(MmcDirect.COPY_G2H.value, 2)
        self.assertEqual(MmcDirect.COPY_H2G.value, 3)

    def test_register_buffer(self):
        b = self._make_backend()
        b._is_a2 = True
        b.register_buffer([100], [200])
        b.store.register_buffer.assert_called_once()

    def test_get(self):
        b = self._make_backend()
        b.store.batch_get_into_layers.return_value = [0]
        b.get(["k1"], [[100]], [[10]])
        b.store.batch_get_into_layers.assert_called_once()

    def test_get_uses_copy_g2l_enum(self):
        b = self._make_backend()
        b.store.batch_get_into_layers.return_value = [0]

        b.get(["k1"], [[100]], [[10]])

        b.store.batch_get_into_layers.assert_called_once_with(["k1"], [[100]], [[10]], MmcDirect.COPY_G2L.value)

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

    def test_put_uses_copy_l2g_enum(self):
        b = self._make_backend()
        b.store.batch_put_from_layers.return_value = [0]

        b.put(["k1"], [[100]], [[10]])

        b.store.batch_put_from_layers.assert_called_once_with(["k1"], [[100]], [[10]], MmcDirect.COPY_L2G.value)

    def test_put_error(self):
        b = self._make_backend()
        b.store.batch_put_from_layers.return_value = [1]
        b.put(["k1"], [[100]], [[10]])

    def test_put_exception(self):
        b = self._make_backend()
        b.store.batch_put_from_layers.side_effect = Exception("fail")
        b.put(["k1"], [[100]], [[10]])

    def test_init_non_a2_lazy_init_defers_store_setup(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.get_world_group",
                return_value=MagicMock(local_rank=4),
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.get_ascend_device_type",
                return_value=None,
            ),
            patch.object(MemcacheBackend, "_setup_store", return_value=MagicMock()) as mock_setup,
        ):
            backend = MemcacheBackend(MagicMock(), lazy_init=True)

        self.assertTrue(backend._lazy_init)
        self.assertFalse(backend._store_initialized)
        self.assertIsNone(backend.store)
        mock_setup.assert_not_called()

    def test_init_a2_ignores_lazy_init_and_initializes_store(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend

        store = MagicMock()
        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.get_world_group",
                return_value=MagicMock(local_rank=1),
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.get_ascend_device_type",
                return_value="A2",
            ),
            patch.object(MemcacheBackend, "_setup_store", return_value=store) as mock_setup,
        ):
            backend = MemcacheBackend(MagicMock(), lazy_init=True)

        self.assertFalse(backend._lazy_init)
        self.assertTrue(backend._store_initialized)
        self.assertIs(backend.store, store)
        mock_setup.assert_called_once_with()

    def test_register_buffer_waits_for_a2_lazy_store_initialization(self):
        b = self._make_backend()
        b._is_a2 = True
        b._store_initialized = False
        b.store = MagicMock()

        b.register_buffer([100, 200], [10, 20])

        b.store.register_buffer.assert_not_called()
        self.assertEqual(b._registered_buffers, ([100, 200], [10, 20]))

        b._store_initialized = True
        b._register_buffers_if_needed()

        self.assertEqual(b.store.register_buffer.call_count, 2)
        self.assertEqual(b.store.register_buffer.call_args_list[0].args, (100, 10))
        self.assertEqual(b.store.register_buffer.call_args_list[1].args, (200, 20))
        self.assertTrue(b._buffers_registered)


if __name__ == "__main__":
    unittest.main()
