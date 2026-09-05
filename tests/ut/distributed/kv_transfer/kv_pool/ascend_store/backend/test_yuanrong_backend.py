# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import yuanrong_backend as module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import (
    YuanrongBackend,
    YuanrongConfig,
)


@pytest.mark.parametrize("layerwise,init_calls", [(False, 0), (True, 1)])
def test_scheduler_client_initializes_only_layerwise_data_plane(tmp_path, monkeypatch, layerwise, init_calls):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"worker_addr": "127.0.0.1:31501", "use_layerwise": layerwise, "enable_remote_h2d": True})
    )
    monkeypatch.setenv("YR_CONFIG_PATH", str(path))
    client = sys.modules["yr.datasystem.hetero_client"].HeteroClient
    backend = YuanrongBackend.create_scheduler_client(None)
    assert client.return_value.init.call_count == init_calls
    assert client.call_args.kwargs["enable_remote_h2d"] is False
    assert backend._needs_dev_mem_pregister is False


def test_lazy_initialization_warns_and_remains_eager(tmp_path, monkeypatch):
    _make_full_backend(tmp_path, monkeypatch)
    logger = MagicMock()
    monkeypatch.setattr(module, "logger", logger)
    backend = YuanrongBackend(None, lazy_init=True)
    logger.warning.assert_called_once()
    assert backend.store is not None
    monkeypatch.setattr(module, "get_world_group", lambda: SimpleNamespace(local_rank=1))
    backend.set_device()
    assert str(torch.npu.set_device.call_args.args[0]) == "npu:1"


def test_invalid_worker_address_is_reported_with_context(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"worker_addr": "invalid-address"}))
    monkeypatch.setenv("YR_CONFIG_PATH", str(path))
    with pytest.raises(ValueError, match="Invalid worker_addr invalid-address"):
        YuanrongBackend(None)


def test_missing_datasystem_dependency_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "yr.datasystem.hetero_client", None)
    with pytest.raises(ImportError, match="Please install openyuanrong-datasystem"):
        YuanrongBackend(None)


def _format_log_call(call):
    args = call.args
    return args[0] % args[1:]


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


def test_yuanrong_put_does_not_require_exists_check():
    assert YuanrongBackend.requires_exists_before_put is False


def _make_backend():
    backend = YuanrongBackend.__new__(YuanrongBackend)
    backend._ds_set_param = object()
    backend.config = SimpleNamespace(get_sub_timeout_ms=1234)
    backend.store = MagicMock()
    backend.store.mget_h2d_from_multi_buffers.return_value = []
    backend.store.mset_d2h_from_multi_buffers.return_value = None
    backend.store.batch_is_exist.return_value = [1, 0]
    return backend


def test_get_and_put_use_multi_buffer_apis():
    backend = _make_backend()
    keys = ["Qwen2.5-7B@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@abcdef"]
    addrs = [[100, 200]]
    sizes = [[10, 20]]

    assert backend.get(keys, addrs, sizes) == [0]
    backend.put(keys, addrs, sizes)

    backend.store.mget_h2d_from_multi_buffers.assert_called_once_with(keys, addrs, sizes, 1234)
    backend.store.mset_d2h_from_multi_buffers.assert_called_once_with(keys, addrs, sizes, backend._ds_set_param)
    assert backend.store.mget_h2d_from_multi_buffers.call_args.args[0] is keys
    assert backend.store.mset_d2h_from_multi_buffers.call_args.args[0] is keys


def test_get_forwards_empty_keys_to_sdk():
    backend = _make_backend()
    backend.store.mget_h2d_from_multi_buffers.return_value = []
    assert backend.get([], [], []) == []
    backend.store.mget_h2d_from_multi_buffers.assert_called_once_with([], [], [], 1234)


def test_get_marks_failed_keys():
    backend = _make_backend()
    keys = ["k1", "k2", "k3"]
    addrs = [[100], [200], [300]]
    sizes = [[10], [20], [30]]
    backend.store.mget_h2d_from_multi_buffers.return_value = ["k2"]

    assert backend.get(keys, addrs, sizes) == [0, 1, 0]
    backend.store.mget_h2d_from_multi_buffers.assert_called_once_with(keys, addrs, sizes, 1234)


def test_get_returns_none_on_exception():
    backend = _make_backend()
    backend.store.mget_h2d_from_multi_buffers.side_effect = Exception("fail")
    assert backend.get(["k1"], [[100]], [[10]]) is None


def test_put_forwards_empty_keys_to_sdk():
    backend = _make_backend()
    backend.put([], [], [])
    backend.store.mset_d2h_from_multi_buffers.assert_called_once_with([], [], [], backend._ds_set_param)


def test_put_logs_on_exception():
    backend = _make_backend()
    backend.store.mset_d2h_from_multi_buffers.side_effect = Exception("fail")
    backend.put(["k1"], [[100]], [[10]])  # Should log but not raise


def test_exists_returns_native_int_list():
    backend = _make_backend()
    keys = ["Qwen2.5-key0", "Qwen2.5-key1"]
    result = backend.exists(keys)

    assert result == [1, 0]
    backend.store.batch_is_exist.assert_called_once_with(keys)
    assert backend.store.batch_is_exist.call_args.args[0] is keys


def test_exists_forwards_empty_keys_to_sdk():
    backend = _make_backend()
    backend.store.batch_is_exist.return_value = []
    assert backend.exists([]) == []
    backend.store.batch_is_exist.assert_called_once_with([])


def test_exists_exception_returns_zeros():
    backend = _make_backend()
    backend.store.batch_is_exist.side_effect = Exception("fail")
    assert backend.exists(["k1", "k2"]) == [0, 0]


def test_yuanrong_config_loads_from_file(tmp_path):
    cfg_path = tmp_path / "yuanrong.json"
    cfg_path.write_text(
        json.dumps(
            {
                "worker_addr": "127.0.0.1:31501",
                "enable_remote_h2d": False,
                "connect_timeout_ms": 12000,
                "request_timeout_ms": 8000,
                "get_sub_timeout_ms": 3000,
                "enable_dev_mem_pregister": True,
            }
        )
    )

    cfg = YuanrongConfig.from_file(str(cfg_path))

    assert cfg.worker_addr == "127.0.0.1:31501"
    assert cfg.enable_remote_h2d is False
    assert cfg.connect_timeout_ms == 12000
    assert cfg.request_timeout_ms == 8000
    assert cfg.get_sub_timeout_ms == 3000
    assert cfg.enable_dev_mem_pregister is True


def test_yuanrong_config_defaults_from_file(tmp_path):
    cfg_path = tmp_path / "yuanrong.json"
    cfg_path.write_text(json.dumps({"worker_addr": "h:1"}))

    cfg = YuanrongConfig.from_file(str(cfg_path))

    assert cfg.enable_remote_h2d is False
    assert cfg.connect_timeout_ms == 9000
    assert cfg.request_timeout_ms == 0
    assert cfg.get_sub_timeout_ms == 0
    assert cfg.enable_dev_mem_pregister is False


@pytest.mark.parametrize("invalid_config", [None, []])
def test_yuanrong_config_rejects_non_object_json(tmp_path, invalid_config):
    cfg_path = tmp_path / "yuanrong.json"
    cfg_path.write_text(json.dumps(invalid_config))

    with pytest.raises(ValueError, match="expected a dictionary/object"):
        YuanrongConfig.from_file(str(cfg_path))


def test_yuanrong_config_load_from_env_requires_path(monkeypatch):
    monkeypatch.delenv("YR_CONFIG_PATH", raising=False)

    with pytest.raises(ValueError, match="YR_CONFIG_PATH"):
        YuanrongConfig.load_from_env()


def test_backend_forwards_configured_timeouts(tmp_path, monkeypatch):
    cfg_path = tmp_path / "yuanrong.json"
    cfg_path.write_text(
        json.dumps(
            {
                "worker_addr": "127.0.0.1:31501",
                "connect_timeout_ms": 12000,
                "request_timeout_ms": 8000,
                "get_sub_timeout_ms": 3000,
            }
        )
    )
    monkeypatch.setenv("YR_CONFIG_PATH", str(cfg_path))

    native_client = MagicMock()
    native_client.mget_h2d_from_multi_buffers.return_value = []
    hetero_client = MagicMock(return_value=native_client)
    monkeypatch.setattr(sys.modules["yr.datasystem.hetero_client"], "HeteroClient", hetero_client)
    backend_module = sys.modules[YuanrongBackend.__module__]
    monkeypatch.setattr(backend_module, "split_host_port", lambda _: ("127.0.0.1", 31501))

    backend = YuanrongBackend(MagicMock())

    hetero_client.assert_called_once_with(
        "127.0.0.1",
        31501,
        connect_timeout_ms=12000,
        req_timeout_ms=8000,
        enable_remote_h2d=False,
    )
    native_client.init.assert_called_once_with()
    backend.get(["k1"], [[100]], [[10]])
    native_client.mget_h2d_from_multi_buffers.assert_called_once_with(["k1"], [[100]], [[10]], 3000)


def _make_full_backend(tmp_path, monkeypatch, **overrides):
    """Build a real YuanrongBackend (with mocked HeteroClient) from JSON config."""
    cfg = {
        "worker_addr": "127.0.0.1:31501",
        "enable_remote_h2d": True,
        "remote_h2d_transport_backend": "HIXL",
        "enable_fabric_mem": False,
        "enable_dev_mem_pregister": False,
    }
    cfg.update(overrides)
    cfg_path = tmp_path / "yuanrong.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setenv("YR_CONFIG_PATH", str(cfg_path))

    native_client = MagicMock()
    hetero_client = MagicMock(return_value=native_client)
    monkeypatch.setattr(sys.modules["yr.datasystem.hetero_client"], "HeteroClient", hetero_client)
    backend_module = sys.modules[YuanrongBackend.__module__]
    monkeypatch.setattr(backend_module, "split_host_port", lambda _: ("127.0.0.1", 31501))
    return YuanrongBackend(MagicMock())


def test_backend_uses_nx_existence_for_put(tmp_path, monkeypatch):
    backend = _make_full_backend(tmp_path, monkeypatch)

    existence_opt = sys.modules["yr.datasystem.kv_client"].ExistenceOpt
    assert backend._ds_set_param.existence is existence_opt.NX


def test_dev_mem_pregister_disabled_by_default(tmp_path, monkeypatch):
    # All HIXL conditions met, but the new toggle defaults to false -> no pre-registration.
    backend = _make_full_backend(tmp_path, monkeypatch)
    assert backend._needs_dev_mem_pregister is False


def test_dev_mem_pregister_enabled_triggers_registration(tmp_path, monkeypatch):
    backend = _make_full_backend(tmp_path, monkeypatch, enable_dev_mem_pregister=True)
    assert backend._needs_dev_mem_pregister is True
    backend.register_buffer([100], [200])
    backend.store.pre_register_device_memory.assert_called_once_with([100], [200])


def test_dev_mem_pregister_skipped_when_fabric_mem(tmp_path, monkeypatch):
    # FabricMem mode still skips pre-registration even if the toggle is on.
    backend = _make_full_backend(tmp_path, monkeypatch, enable_dev_mem_pregister=True, enable_fabric_mem=True)
    assert backend._needs_dev_mem_pregister is False
    backend.register_buffer([100], [200])
    backend.store.pre_register_device_memory.assert_not_called()


def test_dev_mem_pregister_skipped_for_p2p_transfer(tmp_path, monkeypatch):
    # P2P_TRANSFER transport never uses device memory pre-registration.
    backend = _make_full_backend(
        tmp_path, monkeypatch, enable_dev_mem_pregister=True, remote_h2d_transport_backend="P2P_TRANSFER"
    )
    assert backend._needs_dev_mem_pregister is False
    backend.register_buffer([100], [200])
    backend.store.pre_register_device_memory.assert_not_called()
