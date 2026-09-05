# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_backend as module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    DEFAULT_TENANT_ID,
    MooncakeBackend,
    MooncakeStoreConfig,
    _convert_to_bytes,
    _parse_global_segment_size,
    _ssd_setup_kwargs,
    _validate_store_qos,
)


@pytest.fixture(autouse=True)
def clear_sdk_signature_cache():
    module._mooncake_setup_supports_ssd_offload.cache_clear()
    yield
    module._mooncake_setup_supports_ssd_offload.cache_clear()


@pytest.fixture
def sdk_runtime(monkeypatch, tmp_path):
    path = tmp_path / "mooncake.json"
    path.write_text(json.dumps({"metadata_server": "meta", "master_server_address": "master"}))
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    monkeypatch.delenv("ASCEND_GLOBAL_RESOURCE_CONFIG", raising=False)
    monkeypatch.delenv("ASCEND_ENABLE_USE_FABRIC_MEM", raising=False)
    monkeypatch.setattr(module, "get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(module, "get_world_group", lambda: SimpleNamespace(local_rank=2))
    transfer_engine = MagicMock()
    transfer_engine.get_transfer_engine.return_value.get_rpc_port.return_value = 5555
    monkeypatch.setattr(module, "global_te", transfer_engine)
    factory = sys.modules["mooncake.store"].MooncakeDistributedStore
    factory.return_value.setup.return_value = 0
    factory.return_value.batch_put_from_multi_buffers.return_value = [0]
    return factory, path, transfer_engine


@pytest.mark.parametrize("supports", [False, True])
def test_ssd_signature_detection(supports):
    class OldSDK:
        def setup(self, hostname):
            pass

    class NewSDK:
        def setup(self, hostname, enable_ssd_offload=False):
            pass

    sys.modules["mooncake.store"].MooncakeDistributedStore = NewSDK if supports else OldSDK
    assert module._mooncake_setup_supports_ssd_offload() is supports


@pytest.mark.parametrize("doc,expected", [(None, False), ("setup(host)", False), ("setup(enable_ssd_offload)", True)])
def test_ssd_pybind_signature_falls_back_to_doc(monkeypatch, doc, expected):
    import inspect

    setup = MagicMock()
    setup.__doc__ = doc
    sys.modules["mooncake.store"].MooncakeDistributedStore.setup = setup
    monkeypatch.setattr(inspect, "signature", MagicMock(side_effect=ValueError("pybind")))
    assert module._mooncake_setup_supports_ssd_offload() is expected


@pytest.mark.parametrize("fabric", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_constructor_lazy_state_and_first_put(sdk_runtime, monkeypatch, fabric, lazy):
    factory, _, transfer_engine = sdk_runtime
    monkeypatch.setenv("ASCEND_ENABLE_USE_FABRIC_MEM", str(int(fabric)))
    backend = MooncakeBackend(None, lazy_init=lazy)
    assert backend._lazy_init is (fabric and lazy)
    if fabric and lazy:
        factory.assert_not_called()
        assert backend.exists(["a", "b"]) == [0, 0]
        assert backend.get(["a"], [[100]], [[8]]) is None
    backend.put(["a"], [[100]], [[8]])
    backend.ensure_initialized()
    assert backend._store_initialized
    assert factory.call_count == 1
    factory.return_value.batch_put_from_multi_buffers.assert_called_once()
    assert backend.local_seg == ("127.0.0.1" if fabric else "127.0.0.1:5555")
    backend.set_device()
    assert str(torch.npu.set_device.call_args.args[0]) == "npu:2"


def test_scheduler_does_not_contribute_memory(sdk_runtime):
    factory, _, _ = sdk_runtime
    backend = MooncakeBackend.create_scheduler_client(None)
    assert not backend._contribute_memory
    assert factory.return_value.setup.call_args.kwargs["global_segment_size"] == 0
    assert factory.return_value.setup.call_args.kwargs["local_buffer_size"] == 0
    torch.npu.set_device.assert_called_once_with(0)


def test_rejects_unsupported_protocol_before_sdk_setup(sdk_runtime):
    factory, path, _ = sdk_runtime
    path.write_text(json.dumps({"protocol": "tcp"}))
    with pytest.raises(NotImplementedError, match="tcp"):
        MooncakeBackend(None)
    factory.assert_not_called()


def test_sdk_setup_failure_does_not_mark_lazy_store_ready(sdk_runtime, monkeypatch):
    factory, _, _ = sdk_runtime
    monkeypatch.setenv("ASCEND_ENABLE_USE_FABRIC_MEM", "1")
    backend = MooncakeBackend(None, lazy_init=True)
    factory.return_value.setup.return_value = 4
    with pytest.raises(RuntimeError, match="Initialize mooncake failed"):
        backend.ensure_initialized()
    assert backend.store is None
    assert not backend._store_initialized


def test_missing_mooncake_sdk_is_actionable(sdk_runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "mooncake.store", None)
    with pytest.raises(ImportError, match="Please install mooncake"):
        MooncakeBackend(None)


def test_independent_te_registration_reports_failed_buffers(sdk_runtime, monkeypatch):
    factory, _, transfer_engine = sdk_runtime
    monkeypatch.setenv("ASCEND_GLOBAL_RESOURCE_CONFIG", "{}")
    logger = MagicMock()
    monkeypatch.setattr(module, "logger", logger)
    backend = MooncakeBackend(None)
    factory.return_value.register_buffer.side_effect = [0, -1]
    backend.register_buffer([100, 200], [8, 16])
    assert [c.args for c in factory.return_value.register_buffer.call_args_list] == [(100, 8), (200, 16)]
    logger.error.assert_called_once()
    transfer_engine.register_buffer.assert_not_called()


@pytest.mark.parametrize("result,error", [([-7], None), (None, RuntimeError("put failed"))])
def test_lazy_put_reports_failures(sdk_runtime, monkeypatch, result, error):
    factory, _, _ = sdk_runtime
    monkeypatch.setenv("ASCEND_ENABLE_USE_FABRIC_MEM", "1")
    logger = MagicMock()
    monkeypatch.setattr(module, "logger", logger)
    backend = MooncakeBackend(None, lazy_init=True)
    factory.return_value.batch_put_from_multi_buffers.return_value = result
    factory.return_value.batch_put_from_multi_buffers.side_effect = error
    backend.put(["a"], [[100]], [[8]])
    logger.error.assert_called_once()
    logger.warning.assert_called_once()


def test_positive_get_byte_counts_are_normalized_to_success(sdk_runtime):
    factory, _, _ = sdk_runtime
    backend = MooncakeBackend(None)
    factory.return_value.batch_get_into_multi_buffers.return_value = [8, -3, 0]
    assert backend.get(["a", "b", "c"], [[1], [2], [3]], [[8], [8], [8]]) == [0, -3, 0]


def test_initialization_rechecks_ready_state_under_lock(sdk_runtime, monkeypatch):
    factory, _, _ = sdk_runtime
    monkeypatch.setenv("ASCEND_ENABLE_USE_FABRIC_MEM", "1")
    backend = MooncakeBackend(None, lazy_init=True)

    @contextmanager
    def other_thread_finishes():
        backend._store_initialized = True
        yield

    backend._store_init_lock = other_thread_finishes()
    backend.ensure_initialized()
    factory.assert_not_called()


def test_ssd_directory_creation_failure_preserves_cause(sdk_runtime, monkeypatch, tmp_path):
    factory, path, _ = sdk_runtime
    path.write_text(json.dumps({"enable_ssd_offload": True, "ssd_offload_path": str(tmp_path / "ssd")}))
    factory.setup = lambda enable_ssd_offload=False: None
    monkeypatch.setattr(module, "get_global_rank", lambda config: 2)
    monkeypatch.setattr(module.os, "makedirs", MagicMock(side_effect=PermissionError("denied")))
    with pytest.raises(RuntimeError, match="Failed to create per-rank SSD offload directory.*denied"):
        MooncakeBackend(None)
    factory.return_value.setup.assert_not_called()


def test_non_mapping_global_resource_configuration_has_no_store_qos(monkeypatch):
    monkeypatch.setenv("ASCEND_GLOBAL_RESOURCE_CONFIG", "[]")
    _validate_store_qos()


def test_size_overflow_has_configuration_error():
    with pytest.raises(ValueError, match="Storage size too large"):
        _convert_to_bytes("inf", 1024, "inf KB")


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
        backend._use_store_independent_te = False
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
            backend._use_store_independent_te = False
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

    def test_register_buffer_with_store_independent_te(self):
        b = self._make_backend()
        b._use_store_independent_te = True
        b.store.register_buffer.return_value = 0
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te"
        ) as mock_te:
            b.register_buffer([100, 101], [200, 201])
            b.store.register_buffer.assert_any_call(100, 200)
            b.store.register_buffer.assert_any_call(101, 201)
            self.assertEqual(b.store.register_buffer.call_count, 2)
            mock_te.register_buffer.assert_not_called()

    def test_register_buffer_skips_fabric_mem(self):
        b = self._make_backend()
        b._use_fabric_mem = True
        b.register_buffer([100], [200])
        b.store.register_buffer.assert_not_called()

    def test_setup_store_with_store_independent_te(self):
        b = self._make_backend()
        b._use_store_independent_te = True
        b._contribute_memory = True
        b.config = SimpleNamespace(
            metadata_server="P2PHANDSHAKE",
            global_segment_size=1024,
            local_buffer_size=2048,
            protocol="ascend",
            device_name="",
            master_server_address="127.0.0.1:50088",
            enable_ssd_offload=False,
            tenant_id="default",
        )
        with (
            patch.object(sys.modules["mooncake.store"], "MooncakeDistributedStore", create=True) as mock_store_cls,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.global_te"
            ) as mock_te,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend.get_ip",
                return_value="127.0.0.1",
            ),
        ):
            mock_store = mock_store_cls.return_value
            mock_store.setup.return_value = 0
            self.assertIs(b._setup_store(), mock_store)
            mock_te.get_transfer_engine.assert_not_called()
            mock_store.setup.assert_called_once()
            setup_kwargs = mock_store.setup.call_args.kwargs
            self.assertEqual(setup_kwargs["local_hostname"], "127.0.0.1")
            self.assertEqual(setup_kwargs["local_buffer_size"], 0)
            self.assertNotIn("engine", setup_kwargs)


class TestMooncakeStoreQosValidation(unittest.TestCase):
    _ENV = "ASCEND_GLOBAL_RESOURCE_CONFIG"

    @staticmethod
    def _store_qos_env(qos) -> dict:
        return {"ASCEND_GLOBAL_RESOURCE_CONFIG": json.dumps({"store": {"comm_resource_config": {"qos": qos}}})}

    def test_unset_env_passes(self):
        with patch.dict(os.environ, {}, clear=True):
            _validate_store_qos()

    def test_valid_qos_values_pass(self):
        for qos in (0, 1, 2, 3, 4):
            with self.subTest(qos=qos), patch.dict(os.environ, self._store_qos_env(qos)):
                _validate_store_qos()

    def test_out_of_range_qos_rejected(self):
        for qos in (5, 6, 7, -1):
            with (
                self.subTest(qos=qos),
                patch.dict(os.environ, self._store_qos_env(qos)),
                self.assertRaisesRegex(ValueError, r"\[0, 4\]"),
            ):
                _validate_store_qos()

    def test_non_integer_qos_rejected(self):
        for qos in ("3", 2.5, True, None, [3]):
            with (
                self.subTest(qos=qos),
                patch.dict(os.environ, self._store_qos_env(qos)),
                self.assertRaisesRegex(ValueError, "QoS must be an integer"),
            ):
                _validate_store_qos()

    def test_malformed_json_rejected(self):
        with patch.dict(os.environ, {self._ENV: "not-json"}), self.assertRaisesRegex(ValueError, "not valid JSON"):
            _validate_store_qos()

    def test_missing_store_qos_passes(self):
        # The top-level comm_resource_config.qos is not used by the KV pool and
        # must not be validated here.
        configs: tuple[dict, ...] = (
            {},
            {"comm_resource_config": {"qos": 7}},
            {"store": {}},
            {"store": {"comm_resource_config": {}}},
            {"fabric_memory": {"max_capacity": 32}},
        )
        for config in configs:
            with self.subTest(config=config), patch.dict(os.environ, {self._ENV: json.dumps(config)}):
                _validate_store_qos()

    def test_init_rejects_invalid_store_qos(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"metadata_server": "192.168.0.1:2379"}, f)
            path = f.name
        try:
            env = {
                "MOONCAKE_CONFIG_PATH": path,
                "ASCEND_GLOBAL_RESOURCE_CONFIG": json.dumps({"store": {"comm_resource_config": {"qos": 6}}}),
            }
            with (
                patch.dict(os.environ, env),
                patch.object(MooncakeBackend, "_setup_store"),
                self.assertRaisesRegex(ValueError, r"\[0, 4\]"),
            ):
                MooncakeBackend(MagicMock())
        finally:
            os.unlink(path)


class TestParseGlobalSegmentSizeContract(unittest.TestCase):
    def test_int_input(self):
        self.assertEqual(_parse_global_segment_size(1024), 1024)
        self.assertEqual(_parse_global_segment_size(0), 0)

    def test_gb_unit(self):
        self.assertEqual(_parse_global_segment_size("2GB"), 2 * 1024**3)
        self.assertEqual(_parse_global_segment_size("1.5GB"), int(1.5 * 1024**3))
        self.assertEqual(_parse_global_segment_size(" 2 GB "), 2 * 1024**3)

    def test_gb_unit_edge_cases(self):
        with self.assertRaises(ValueError):
            _parse_global_segment_size("GB")
        with self.assertRaises(ValueError):
            _parse_global_segment_size("abcGB")

    def test_mb_unit(self):
        self.assertEqual(_parse_global_segment_size("512MB"), 512 * 1024**2)
        self.assertEqual(_parse_global_segment_size("0.5MB"), int(0.5 * 1024**2))
        self.assertEqual(_parse_global_segment_size("1024MB"), 1024 * 1024**2)

    def test_kb_unit(self):
        self.assertEqual(_parse_global_segment_size("256KB"), 256 * 1024)
        self.assertEqual(_parse_global_segment_size("1.25KB"), int(1.25 * 1024))

    def test_b_unit(self):
        self.assertEqual(_parse_global_segment_size("4096B"), 4096)
        self.assertEqual(_parse_global_segment_size("1024b"), 1024)

    def test_no_unit(self):
        self.assertEqual(_parse_global_segment_size("2048"), 2048)
        self.assertEqual(_parse_global_segment_size("0"), 0)

    def test_non_string_non_int_input(self):
        self.assertEqual(_parse_global_segment_size(2048.0), 2048)
        self.assertEqual(_parse_global_segment_size(True), 1)

        with self.assertRaises(TypeError):
            _parse_global_segment_size(None)

        with self.assertRaises(TypeError):
            _parse_global_segment_size({"size": 1024})


class TestConvertToBytesContract(unittest.TestCase):
    def test_valid_conversion(self):
        self.assertEqual(_convert_to_bytes("10", 1, "10"), 10)
        self.assertEqual(_convert_to_bytes("1.5", 1024, "1.5KB"), int(1.5 * 1024))
        self.assertEqual(_convert_to_bytes("0", 1024**3, "0GB"), 0)

    def test_invalid_numbers(self):
        with self.assertRaises(ValueError):
            _convert_to_bytes("abc", 1, "abc")

        with self.assertRaises(ValueError):
            _convert_to_bytes("1.2.3", 1024, "1.2.3KB")
