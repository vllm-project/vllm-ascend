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
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_layerwise_protocol,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    DEFAULT_TENANT_ID,
    MooncakeBackend,
    MooncakeStoreConfig,
    _convert_to_bytes,
    _parse_global_segment_size,
    _ssd_setup_kwargs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import (
    YuanrongConfig,
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


# =========================================================================
# Backend layerwise protocol registry
# =========================================================================
class TestLayerwiseProtocolRegistry(unittest.TestCase):
    """``get_layerwise_protocol`` is the generic layers' only knowledge of
    layerwise support: it resolves the protocol module registered under the
    backend name (normalized), or None when the entry carries none."""

    def test_get_layerwise_protocol_resolves_module(self):
        protocol = get_layerwise_protocol("memcache")
        self.assertIsNotNone(protocol)
        self.assertTrue(hasattr(protocol, "GVAKeyFactory"))

    def test_get_layerwise_protocol_normalizes_name(self):
        for backend_name in ("MEMCACHE", " Memcache "):
            with self.subTest(backend=backend_name):
                self.assertIsNotNone(get_layerwise_protocol(backend_name))

    def test_get_layerwise_protocol_returns_none_without_protocol(self):
        for backend_name in ("mooncake", "yuanrong", "nonexistent"):
            with self.subTest(backend=backend_name):
                self.assertIsNone(get_layerwise_protocol(backend_name))


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
        self.assertEqual(defaults.tenant_id, DEFAULT_TENANT_ID)

        ssd_path = TestMooncakeStoreConfig._writable_ssd_path()
        self.addCleanup(lambda: os.rmdir(ssd_path))
        ssd = _make_mooncake_store_config(enable_ssd_offload=True, ssd_offload_path=ssd_path)
        self.assertEqual(ssd.ssd_offload_path, ssd_path)

    def test_from_file_normalizes_tenant_id(self):
        for value, expected in (
            (None, DEFAULT_TENANT_ID),
            ("", DEFAULT_TENANT_ID),
            ("   ", DEFAULT_TENANT_ID),
            ("tenant-a", "tenant-a"),
            ("  tenant-a  ", "tenant-a"),
        ):
            with self.subTest(value=value):
                cfg = _make_mooncake_store_config(tenant_id=value)
                self.assertEqual(cfg.tenant_id, expected)

    def test_from_file_rejects_non_string_tenant_id(self):
        with self.assertRaisesRegex(TypeError, "tenant_id must be a string or null"):
            _make_mooncake_store_config(tenant_id=False)

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
            enable_dev_mem_pregister=True,
        )
        cfg = YuanrongConfig.from_file(path)
        self.assertEqual(cfg.worker_addr, "host:1234")
        self.assertFalse(cfg.enable_remote_h2d)
        self.assertEqual(cfg.remote_h2d_transport_backend, "HIXL")
        self.assertFalse(cfg.enable_fabric_mem)
        self.assertEqual(cfg.connect_timeout_ms, 12000)
        self.assertEqual(cfg.request_timeout_ms, 8000)
        self.assertEqual(cfg.get_sub_timeout_ms, 3000)
        self.assertTrue(cfg.enable_dev_mem_pregister)

    def test_from_file_defaults(self):
        path = self._write_config(worker_addr="h:1")
        cfg = YuanrongConfig.from_file(path)
        self.assertFalse(cfg.enable_remote_h2d)
        self.assertEqual(cfg.remote_h2d_transport_backend, "HIXL")
        self.assertFalse(cfg.enable_fabric_mem)
        self.assertEqual(cfg.connect_timeout_ms, 9000)
        self.assertEqual(cfg.request_timeout_ms, 0)
        self.assertEqual(cfg.get_sub_timeout_ms, 0)
        self.assertFalse(cfg.enable_dev_mem_pregister)

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
class TestMooncakeBackendSetup(unittest.TestCase):
    _MODULE_PATH = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend"

    def _make_backend(
        self,
        *,
        config: MooncakeStoreConfig,
        use_fabric_mem: bool,
        contribute_memory: bool = True,
    ) -> MooncakeBackend:
        backend = MooncakeBackend.__new__(MooncakeBackend)
        backend.parallel_config = MagicMock()
        backend.config = config
        backend.local_seg = None
        backend._use_fabric_mem = use_fabric_mem
        backend._contribute_memory = contribute_memory
        return backend

    def _setup_store(self, backend: MooncakeBackend, store: MagicMock):
        transfer_engine = MagicMock()
        transfer_engine.get_rpc_port.return_value = 50052
        fake_store_module = sys.modules["mooncake.store"]
        with (
            patch.object(
                fake_store_module,
                "MooncakeDistributedStore",
                return_value=store,
                create=True,
            ),
            patch(f"{self._MODULE_PATH}.get_ip", return_value="10.0.0.7"),
            patch(f"{self._MODULE_PATH}.global_te") as mock_global_te,
            patch(f"{self._MODULE_PATH}.get_global_rank", return_value=3),
            patch(
                f"{self._MODULE_PATH}._mooncake_setup_supports_ssd_offload",
                return_value=True,
            ),
        ):
            mock_global_te.get_transfer_engine.return_value = transfer_engine
            return backend._setup_store()

    def test_setup_omits_default_tenant_for_all_memory_paths(self):
        for use_fabric_mem in (False, True):
            with self.subTest(use_fabric_mem=use_fabric_mem):
                backend = self._make_backend(
                    config=_make_mooncake_store_config(),
                    use_fabric_mem=use_fabric_mem,
                )
                store = MagicMock()
                store.setup.return_value = 0

                result = self._setup_store(backend, store)

                self.assertIs(result, store)
                self.assertNotIn("tenant_id", store.setup.call_args.kwargs)

    def test_setup_forwards_tenant_for_all_memory_paths(self):
        for use_fabric_mem in (False, True):
            with self.subTest(use_fabric_mem=use_fabric_mem):
                backend = self._make_backend(
                    config=_make_mooncake_store_config(tenant_id="  tenant-a  "),
                    use_fabric_mem=use_fabric_mem,
                )
                store = MagicMock()
                store.setup.return_value = 0

                self._setup_store(backend, store)

                self.assertEqual(store.setup.call_args.kwargs["tenant_id"], "tenant-a")

    def test_setup_preserves_ssd_kwargs_with_tenant(self):
        with tempfile.TemporaryDirectory(prefix="mooncake_ssd_ut_") as ssd_path:
            config = _make_mooncake_store_config(
                tenant_id="tenant-a",
                enable_ssd_offload=True,
                ssd_offload_path=ssd_path,
            )
            for use_fabric_mem in (False, True):
                with self.subTest(use_fabric_mem=use_fabric_mem):
                    backend = self._make_backend(
                        config=config,
                        use_fabric_mem=use_fabric_mem,
                    )
                    store = MagicMock()
                    store.setup.return_value = 0

                    self._setup_store(backend, store)

                    setup_kwargs = store.setup.call_args.kwargs
                    self.assertEqual(setup_kwargs["tenant_id"], "tenant-a")
                    self.assertIs(setup_kwargs["enable_ssd_offload"], True)
                    self.assertEqual(setup_kwargs["ssd_offload_path"], os.path.join(ssd_path, "rank_3"))

    def test_scheduler_client_forwards_tenant(self):
        config = _make_mooncake_store_config(tenant_id="tenant-a")
        for use_fabric_mem in (False, True):
            with self.subTest(use_fabric_mem=use_fabric_mem):
                backend = self._make_backend(
                    config=config,
                    use_fabric_mem=use_fabric_mem,
                    contribute_memory=False,
                )
                store = MagicMock()
                store.setup.return_value = 0

                self._setup_store(backend, store)

                setup_kwargs = store.setup.call_args.kwargs
                self.assertEqual(setup_kwargs["tenant_id"], "tenant-a")
                self.assertEqual(setup_kwargs["global_segment_size"], 0)
                self.assertEqual(setup_kwargs["local_buffer_size"], 0)

    def test_non_default_tenant_preserves_setup_type_error(self):
        setup_error = TypeError("setup(): incompatible function arguments")
        backend = self._make_backend(
            config=_make_mooncake_store_config(tenant_id="tenant-a"),
            use_fabric_mem=False,
        )
        store = MagicMock()
        store.setup.side_effect = setup_error

        with self.assertRaises(TypeError) as context:
            self._setup_store(backend, store)

        self.assertIs(context.exception, setup_error)


class TestMooncakeBackendMethods(unittest.TestCase):
    def _make_backend(self):
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
                enable_dev_mem_pregister=False,
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
        b.store.mget_h2d_from_multi_buffers.side_effect = RuntimeError("backend fail")
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend.logger"
        ) as mock_logger:
            result = b.get(["k1"], [[100]], [[10]])
        error_log = _format_log_call(mock_logger.error.call_args)
        self.assertIsNone(result)
        self.assertIn("RuntimeError", error_log)
        self.assertIn("backend fail", error_log)

    def test_put(self):
        b = self._make_backend()
        b.put(["k1"], [[100]], [[10]])
        b.store.mset_d2h_from_multi_buffers.assert_called_once_with(["k1"], [[100]], [[10]], b._ds_set_param)

    def test_put_exception(self):
        b = self._make_backend()
        b.store.mset_d2h_from_multi_buffers.side_effect = RuntimeError("backend fail")
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend.logger"
        ) as mock_logger:
            b.put(["k1"], [[100]], [[10]])
        error_log = _format_log_call(mock_logger.error.call_args)
        self.assertIn("RuntimeError", error_log)
        self.assertIn("backend fail", error_log)

    def test_register_buffer_noop_when_remote_h2d_disabled(self):
        b = self._make_backend()
        b.register_buffer([100], [200])
        b.store.pre_register_device_memory.assert_not_called()

    def test_register_buffer_noop_when_pregister_toggle_off(self):
        # HIXL conditions all met, but the enable_dev_mem_pregister toggle is
        # false by default -> pre-registration is skipped. Mirrors the
        # __init__ gating expression that ANDs in the toggle.
        b = self._make_backend()
        b.config.enable_remote_h2d = True
        b.config.remote_h2d_transport_backend = "HIXL"
        b.config.enable_fabric_mem = False
        b.config.enable_dev_mem_pregister = False
        b._needs_dev_mem_pregister = (
            b.config.enable_remote_h2d
            and b.config.remote_h2d_transport_backend == "HIXL"
            and not b.config.enable_fabric_mem
            and b.config.enable_dev_mem_pregister
        )
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

    def test_batch_write_finish(self):
        b = self._make_backend()
        b.store.batch_write_finish.return_value = [0]

        self.assertEqual(b.batch_write_finish(["k1"], [0]), [0])
        b.store.batch_write_finish.assert_called_once_with(["k1"], [0])

    def test_batch_write_finish_supports_legacy_store(self):
        b = self._make_backend()
        b.store = object()

        self.assertEqual(b.batch_write_finish(["k1"], [0]), [0])

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
        b.store.batch_get_into_layers.side_effect = RuntimeError("backend fail")
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.logger"
        ) as mock_logger:
            b.get(["k1"], [[100]], [[10]])
        error_log = _format_log_call(mock_logger.error.call_args)
        self.assertIn("RuntimeError", error_log)
        self.assertIn("backend fail", error_log)

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
        b.store.batch_put_from_layers.side_effect = RuntimeError("backend fail")
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.logger"
        ) as mock_logger:
            b.put(["k1"], [[100]], [[10]])
        error_log = _format_log_call(mock_logger.error.call_args)
        self.assertIn("RuntimeError", error_log)
        self.assertIn("backend fail", error_log)


if __name__ == "__main__":
    unittest.main()
