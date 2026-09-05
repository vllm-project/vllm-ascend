# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


import os
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import memcache_backend as module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    _validate_device_ub_qos,
    extract_layout_config,
    make_full_key,
    make_hit_check_keys,
    make_partial_key,
)


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    factory = sys.modules["memcache_hybrid"].DistributedObjectStore
    factory.return_value.init.return_value = 0
    factory.return_value.batch_put_from_layers.return_value = [0]
    monkeypatch.setattr(module, "get_world_group", lambda: SimpleNamespace(local_rank=2))
    monkeypatch.setattr(module.time, "sleep", MagicMock())
    monkeypatch.delenv("MF_DEVICE_UB_QOS", raising=False)
    config = tmp_path / "mmc.conf"
    config.write_text("ock.mmc.local_service.protocol = device_sdma\n")
    monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", str(config))
    return factory


@pytest.mark.parametrize(
    "content,expected",
    [
        ("\n# comment\n; comment\ninvalid\nother=value\nock.mmc.local_service.protocol = device_sdma\n", True),
        ("ock.mmc.local_service.protocol=host\n", False),
        ("other=value\n", False),
    ],
)
def test_sdma_protocol_parser(monkeypatch, tmp_path, content, expected):
    config = tmp_path / "config"
    config.write_text(content)
    monkeypatch.setenv("MMC_LOCAL_CONFIG_PATH", str(config))
    assert module._is_device_sdma() is expected


def test_sdma_parser_requires_config_path(monkeypatch):
    monkeypatch.delenv("MMC_LOCAL_CONFIG_PATH", raising=False)
    with pytest.raises(ValueError, match="MMC_LOCAL_CONFIG_PATH"):
        module._is_device_sdma()


def test_eager_setup_and_scheduler_client_use_distinct_initialization(sdk):
    worker = MemcacheBackend(None)
    sdk.return_value.init.assert_called_once_with(2, init_bm=True)
    assert worker._store_initialized
    worker.set_device()
    assert str(torch.npu.set_device.call_args.args[0]) == "npu:2"
    sdk.reset_mock()
    scheduler = MemcacheBackend.create_scheduler_client(None)
    sdk.return_value.init.assert_called_once_with(0, init_bm=False)
    assert scheduler.local_rank == 0
    scheduler.init_store()
    assert sdk.call_count == 1


def test_lazy_init_defers_registration_and_lookup_until_first_write(sdk):
    backend = MemcacheBackend(None, lazy_init=True)
    pointers, lengths = [100, 200], [16, 32]
    backend.register_buffer(pointers, lengths)
    pointers.append(300)
    assert backend.exists(["a", "b"]) == [0, 0]
    assert backend.batch_get_key_info(["a"]) == []
    assert backend.get(["a"], [[100]], [[16]]) is None
    sdk.assert_not_called()
    backend.put(["a"], [[100]], [[16]])
    assert backend._store_initialized
    assert backend._pending_buffers is None
    assert [call.args for call in sdk.return_value.register_buffer.call_args_list] == [(100, 16), (200, 32)]
    backend.ensure_initialized()
    assert sdk.call_count == 1


def test_explicit_lazy_store_initialization_and_registration(sdk):
    backend = MemcacheBackend(None, lazy_init=True)
    backend.init_store(init_bm=False)
    backend.register_buffer([100], [8])
    sdk.return_value.init.assert_called_once_with(2, init_bm=False)
    sdk.return_value.register_buffer.assert_called_once_with(100, 8)
    assert backend._pending_buffers is None


def test_initialization_rechecks_state_after_acquiring_lock(sdk):
    backend = MemcacheBackend(None, lazy_init=True)

    @contextmanager
    def other_thread_finishes():
        backend._store_initialized = True
        yield

    backend._store_init_lock = other_thread_finishes()
    backend.ensure_initialized()
    sdk.assert_not_called()


@pytest.mark.parametrize("failure", [ValueError("config"), RuntimeError("SDK")])
def test_setup_failure_is_propagated_without_marking_store_ready(sdk, failure):
    backend = MemcacheBackend(None, lazy_init=True)
    sdk.return_value.init.side_effect = failure
    with pytest.raises(type(failure), match=str(failure)):
        backend.ensure_initialized()
    assert backend.store is None
    assert not backend._store_initialized


def test_nonzero_setup_result_is_rejected(sdk):
    sdk.return_value.init.return_value = 7
    with pytest.raises(AssertionError):
        MemcacheBackend(None)


def test_missing_sdk_reports_installation_requirement(sdk, monkeypatch):
    monkeypatch.setitem(sys.modules, "memcache_hybrid", None)
    with pytest.raises(ImportError, match="Please install memcache"):
        MemcacheBackend(None)


def test_layerwise_protocol_forwards_parameters_and_legacy_finish(sdk):
    backend = MemcacheBackend(None)
    store = sdk.return_value
    assert backend.batch_get_key_info(["a"]) is store.batch_get_key_info.return_value
    store.batch_get_key_info.assert_called_once_with(["a"])
    assert backend.batch_alloc(["a"], [8]) is store.batch_alloc.return_value
    store.batch_alloc.assert_called_once_with(["a"], [8])
    assert backend.batch_add_lease(["a"], 42) is store.batch_add_lease.return_value
    store.batch_add_lease.assert_called_once_with(["a"], 42)
    assert backend.batch_remove_lease(["a"]) is store.batch_remove_lease.return_value
    store.batch_remove_lease.assert_called_once_with(["a"])
    assert backend.batch_write_finish(["a"], [0]) is store.batch_write_finish.return_value
    store.batch_write_finish.assert_called_once_with(["a"], [0])
    store.batch_write_finish = None
    assert backend.batch_write_finish(["a", "b"], [0, 0]) == [0, 0]


@pytest.mark.parametrize("result,error", [([7], None), (None, RuntimeError("put failed"))])
def test_lazy_first_write_failures_are_logged(sdk, monkeypatch, result, error):
    logger = MagicMock()
    monkeypatch.setattr(module, "logger", logger)
    sdk.return_value.batch_put_from_layers.return_value = result
    sdk.return_value.batch_put_from_layers.side_effect = error
    backend = MemcacheBackend(None, lazy_init=True)
    backend.put(["a"], [[100]], [[8]])
    logger.error.assert_called_once()
    logger.warning.assert_called_once()


def _format_log_call(call):
    args = call.args
    return args[0] % args[1:]


class TestExtractLayoutConfig(unittest.TestCase):
    """The protocol owns the layerwise opt-in check of the layout layer."""

    def test_returns_config_when_opted_in(self):
        extra_config = {"use_layerwise": True, "layerwise_num_shared_buffers": 2}
        self.assertIs(extract_layout_config(extra_config), extra_config)

    def test_returns_none_when_not_opted_in(self):
        self.assertIsNone(extract_layout_config({}))
        self.assertIsNone(extract_layout_config({"use_layerwise": False}))


class TestLayerwiseKeyFormats(unittest.TestCase):
    """Byte-for-byte snapshots of the layerwise key formats.

    These strings are wire formats shared with deployed clusters: a single
    character of drift turns hits into misses after an upgrade. The
    expectations are transcribed from the pre-refactor pool_worker /
    pool_scheduler implementations.
    """

    def test_full_key_single_group_keeps_pr_11585_format(self):
        self.assertEqual(
            make_full_key("model", 0, "hash0", 3, 1),
            "model@hash0@3",
        )

    def test_full_key_multi_group_includes_group_id(self):
        self.assertEqual(
            make_full_key("model", 2, "hash0", 3, 4),
            "model@2@hash0@3",
        )

    def test_partial_key_format(self):
        self.assertEqual(
            make_partial_key("model", "r1", 0, 1, 20, 3),
            "model@partial@r1@0@1@20@3",
        )

    def test_hit_check_keys_single_group_one_key_per_rank(self):
        self.assertEqual(
            make_hit_check_keys("model", 0, "hash0", 4, 1),
            ["model@hash0@0", "model@hash0@1", "model@hash0@2", "model@hash0@3"],
        )

    def test_hit_check_keys_multi_group_one_key_per_rank(self):
        self.assertEqual(
            make_hit_check_keys("model", 1, "hash0", 2, 3),
            ["model@1@hash0@0", "model@1@hash0@1"],
        )

    def test_hit_check_keys_empty_when_no_ranks(self):
        self.assertEqual(make_hit_check_keys("model", 0, "hash0", 0, 1), [])

    def test_full_key_and_hit_check_key_share_rank_format(self):
        """The hit-check key of rank r must equal that rank's full key."""
        for num_groups in (1, 2):
            for rank in range(3):
                with self.subTest(num_groups=num_groups, rank=rank):
                    self.assertEqual(
                        make_hit_check_keys("model", 0, "hash0", 3, num_groups)[rank],
                        make_full_key("model", 0, "hash0", rank, num_groups),
                    )


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


class TestMemcacheQosValidation(unittest.TestCase):
    _ENV = "MF_DEVICE_UB_QOS"

    def test_unset_or_empty_env_passes(self):
        with patch.dict(os.environ, {}, clear=True):
            _validate_device_ub_qos()
        with patch.dict(os.environ, {self._ENV: ""}):
            _validate_device_ub_qos()

    def test_valid_qos_values_pass(self):
        for qos in ("0", "1", "2", "3", "4", " 3 "):
            with self.subTest(qos=qos), patch.dict(os.environ, {self._ENV: qos}):
                _validate_device_ub_qos()

    def test_out_of_range_qos_rejected(self):
        for qos in ("5", "7", "-1", "100"):
            with (
                self.subTest(qos=qos),
                patch.dict(os.environ, {self._ENV: qos}),
                self.assertRaisesRegex(ValueError, r"\[0, 4\]"),
            ):
                _validate_device_ub_qos()

    def test_non_integer_qos_rejected(self):
        for qos in ("abc", "3.5", "0x3"):
            with (
                self.subTest(qos=qos),
                patch.dict(os.environ, {self._ENV: qos}),
                self.assertRaisesRegex(ValueError, "QoS must be an integer"),
            ):
                _validate_device_ub_qos()

    def test_init_rejects_invalid_qos(self):
        with patch.dict(os.environ, {self._ENV: "7"}), self.assertRaisesRegex(ValueError, r"\[0, 4\]"):
            MemcacheBackend(MagicMock())
