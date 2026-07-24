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
    def _write_config(self, **overrides):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(overrides, f)
        self.addCleanup(os.remove, path)
        return path

    def test_from_file(self):
        path = self._write_config(
            worker_addr="host:1234",
            enable_remote_h2d=False,
            remote_h2d_transport_backend="HIXL",
            connect_timeout_ms=12000,
            request_timeout_ms=8000,
            get_sub_timeout_ms=3000,
        )
        cfg = YuanrongConfig.from_file(path)
        self.assertEqual(cfg.worker_addr, "host:1234")
        self.assertFalse(cfg.enable_remote_h2d)
        self.assertEqual(cfg.remote_h2d_transport_backend, "HIXL")
        self.assertFalse(cfg.enable_fabric_mem)
        self.assertEqual(cfg.connect_timeout_ms, 12000)
        self.assertEqual(cfg.request_timeout_ms, 8000)
        self.assertEqual(cfg.get_sub_timeout_ms, 3000)

    def test_from_file_defaults(self):
        path = self._write_config(worker_addr="h:1")
        cfg = YuanrongConfig.from_file(path)
        self.assertFalse(cfg.enable_remote_h2d)
        self.assertEqual(cfg.remote_h2d_transport_backend, "HIXL")
        self.assertFalse(cfg.enable_fabric_mem)
        self.assertEqual(cfg.connect_timeout_ms, 9000)
        self.assertEqual(cfg.request_timeout_ms, 0)
        self.assertEqual(cfg.get_sub_timeout_ms, 0)

    def test_from_file_fabric_mem_with_hixl(self):
        path = self._write_config(
            worker_addr="h:1",
            remote_h2d_transport_backend="HIXL",
            enable_fabric_mem=True,
        )
        cfg = YuanrongConfig.from_file(path)
        self.assertTrue(cfg.enable_fabric_mem)


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


# =========================================================================
# YuanrongBackend (mocked store)
# =========================================================================
class TestYuanrongBackendMethods(unittest.TestCase):
    def _make_backend(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import YuanrongBackend

        with patch.object(YuanrongBackend, "__init__", lambda self, pc: None):
            backend = YuanrongBackend.__new__(YuanrongBackend)
            backend.store = MagicMock()
            backend.store.mget_h2d_from_multi_buffers.return_value = []
            backend.store.mset_d2h_from_multi_buffers.return_value = None
            backend.store.batch_is_exist.return_value = [1, 0]
            backend._ds_set_param = MagicMock()
            backend._needs_dev_mem_pregister = False
            backend._registered_buffers = None
            backend._buffers_registered = False
            backend.config = YuanrongConfig(
                worker_addr="127.0.0.1:0",
                enable_remote_h2d=False,
                remote_h2d_transport_backend="P2P_TRANSFER",
                enable_fabric_mem=False,
                get_sub_timeout_ms=1234,
            )
            backend.rank = 0
            return backend

    def test_exists(self):
        b = self._make_backend()
        b.store.batch_is_exist.return_value = [1, 0]
        result = b.exists(["k1", "k2"])
        self.assertEqual(result, [1, 0])
        b.store.batch_is_exist.assert_called_once_with(["k1", "k2"])

    def test_exists_exception(self):
        b = self._make_backend()
        b.store.batch_is_exist.side_effect = Exception("fail")
        result = b.exists(["k1"])
        self.assertEqual(result, [0])

    def test_get(self):
        b = self._make_backend()
        b.store.mget_h2d_from_multi_buffers.return_value = []
        result = b.get(["k1"], [[100]], [[10]])
        self.assertEqual(result, [0])
        b.store.mget_h2d_from_multi_buffers.assert_called_once_with(["k1"], [[100]], [[10]], 1234)

    def test_get_partial_failure(self):
        b = self._make_backend()
        b.store.mget_h2d_from_multi_buffers.return_value = ["k2"]
        result = b.get(["k1", "k2", "k3"], [[100], [200], [300]], [[10], [20], [30]])
        self.assertEqual(result, [0, 1, 0])

    def test_get_failed_keys(self):
        b = self._make_backend()
        b.store.mget_h2d_from_multi_buffers.return_value = ["k1"]
        result = b.get(["k1"], [[100]], [[10]])  # Should log error
        self.assertEqual(result, [1])

    def test_get_exception(self):
        b = self._make_backend()
        b.store.mget_h2d_from_multi_buffers.side_effect = Exception("fail")
        result = b.get(["k1"], [[100]], [[10]])
        self.assertIsNone(result)

    def test_put(self):
        b = self._make_backend()
        b.put(["k1"], [[100]], [[10]])
        b.store.mset_d2h_from_multi_buffers.assert_called_once_with(["k1"], [[100]], [[10]], b._ds_set_param)

    def test_put_exception(self):
        b = self._make_backend()
        b.store.mset_d2h_from_multi_buffers.side_effect = Exception("fail")
        b.put(["k1"], [[100]], [[10]])

    def test_register_buffer_noop_when_remote_h2d_disabled(self):
        b = self._make_backend()
        b.register_buffer([100], [200])
        b.store.pre_register_device_memory.assert_not_called()

    def test_register_buffer_when_remote_h2d_enabled_hixl(self):
        b = self._make_backend()
        b._needs_dev_mem_pregister = True
        b.register_buffer([100], [200])
        b.store.pre_register_device_memory.assert_called_once_with([100], [200])

    def test_register_buffer_noop_when_p2p_transfer_link(self):
        # P2P-Transfer RoCE transport backend does not use device memory pre-registration.
        b = self._make_backend()
        b.config.enable_remote_h2d = True
        b.config.remote_h2d_transport_backend = "P2P_TRANSFER"
        b._needs_dev_mem_pregister = False
        b.register_buffer([100], [200])
        b.store.pre_register_device_memory.assert_not_called()

    def test_register_buffer_noop_when_fabric_mem(self):
        # FabricMem mode relies on HIXL OPTION_ENABLE_USE_FABRIC_MEM for
        # automatic Fabric handle exchange; no client-side MEM_DEVICE
        # pre-registration. Mirrors the __init__ gating expression.
        b = self._make_backend()
        b.config.enable_remote_h2d = True
        b.config.remote_h2d_transport_backend = "HIXL"
        b.config.enable_fabric_mem = True
        b._needs_dev_mem_pregister = (
            b.config.enable_remote_h2d
            and b.config.remote_h2d_transport_backend == "HIXL"
            and not b.config.enable_fabric_mem
        )
        b.register_buffer([100], [200])
        b.store.pre_register_device_memory.assert_not_called()

    def test_register_buffer_idempotent(self):
        b = self._make_backend()
        b._needs_dev_mem_pregister = True
        b.register_buffer([100], [200])
        b.register_buffer([300], [400])
        b.store.pre_register_device_memory.assert_called_once_with([100], [200])

    def test_register_buffers_if_needed_no_buffers(self):
        b = self._make_backend()
        b._needs_dev_mem_pregister = True
        b._registered_buffers = None
        b._register_buffers_if_needed()
        b.store.pre_register_device_memory.assert_not_called()

    def test_register_buffers_if_needed_already_registered(self):
        b = self._make_backend()
        b._needs_dev_mem_pregister = True
        b._registered_buffers = ([100], [200])
        b._buffers_registered = True
        b._register_buffers_if_needed()
        b.store.pre_register_device_memory.assert_not_called()

    def test_register_buffers_if_needed_disabled(self):
        b = self._make_backend()
        b._needs_dev_mem_pregister = False
        b._registered_buffers = ([100], [200])
        b._register_buffers_if_needed()
        b.store.pre_register_device_memory.assert_not_called()


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
