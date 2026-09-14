# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import base as backend_base
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    Backend,
    parse_qos_from_extra_config,
)


class TestBackendABC(unittest.TestCase):
    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            Backend(MagicMock())  # type: ignore[abstract]


class ConcreteBackend(Backend):
    def __init__(self, parallel_config, lazy_init=False):
        self.parallel_config = parallel_config
        self.store = MagicMock()

    def set_device(self):
        return self.store.set_device()

    def register_buffer(self, ptrs, lengths):
        return self.store.register_buffer(ptrs, lengths)

    def exists(self, keys):
        return self.store.exists(keys)

    def put(self, keys, addrs, sizes):
        return self.store.put(keys, addrs, sizes)

    def get(self, keys, addrs, sizes):
        return self.store.get(keys, addrs, sizes)


def test_scheduler_factory_and_exists_alias_preserve_contract():
    config = object()
    backend = ConcreteBackend.create_scheduler_client(config)
    assert isinstance(backend, ConcreteBackend)
    assert backend.parallel_config is config
    backend.store.exists.return_value = [1, 0]
    assert backend.batch_is_exist(["a", "b"]) == [1, 0]
    backend.store.exists.assert_called_once_with(["a", "b"])


@pytest.mark.parametrize(
    "name,args",
    [
        ("batch_get_key_info", (["a"],)),
        ("batch_alloc", (["a"], [8])),
        ("batch_add_lease", (["a"], 10)),
        ("batch_remove_lease", (["a"],)),
        ("batch_write_finish", (["a"], [0])),
    ],
)
def test_optional_layerwise_protocol_fails_explicitly(name, args):
    backend = ConcreteBackend(None)
    with pytest.raises(NotImplementedError, match=f"ConcreteBackend does not support {name}"):
        getattr(backend, name)(*args)
    assert backend.store.mock_calls == []


class TestExtraConfigQos(unittest.TestCase):
    def test_parse_qos_from_extra_config(self):
        self.assertIsNone(parse_qos_from_extra_config(None))
        self.assertIsNone(parse_qos_from_extra_config({}))
        self.assertIsNone(parse_qos_from_extra_config({"backend": "mooncake"}))
        for qos in (0, 1, 2, 3, 4):
            with self.subTest(qos=qos):
                self.assertEqual(parse_qos_from_extra_config({"qos_priority": qos}), qos)

    def test_parse_qos_from_extra_config_rejects_invalid(self):
        for qos in (5, -1, "3", 2.5, True, None, [3]):
            with (
                self.subTest(qos=qos),
                self.assertRaisesRegex(ValueError, "kv_connector_extra_config"),
            ):
                parse_qos_from_extra_config({"qos_priority": qos})


class TestBackendDeviceBinding(unittest.TestCase):
    def test_scheduler_device_id_does_not_bind_assigned_device(self):
        npu = MagicMock()
        parallel_config = SimpleNamespace(assigned_physical_gpu_ids=[5])
        with (
            patch.object(backend_base.torch, "npu", npu),
            patch.object(backend_base, "set_assigned_physical_gpu_ids") as set_ids,
            patch.object(
                backend_base.current_platform,
                "logical_device_id_to_visible_device_id",
                return_value=2,
            ),
        ):
            device_id = backend_base.get_scheduler_device_id(parallel_config)  # type: ignore[arg-type]

        self.assertEqual(device_id, 2)
        set_ids.assert_called_once_with([5])
        npu.current_device.assert_not_called()
        npu.set_device.assert_not_called()

    def test_scheduler_device(self):
        for assigned_ids, expected in (([5], 2), (None, 3)):
            with self.subTest(assigned_ids=assigned_ids):
                npu = MagicMock()
                npu.current_device.return_value = 3
                parallel_config = SimpleNamespace(assigned_physical_gpu_ids=assigned_ids)
                with (
                    patch.object(backend_base.torch, "npu", npu),
                    patch.object(backend_base, "set_assigned_physical_gpu_ids") as set_ids,
                    patch.object(
                        backend_base.current_platform,
                        "logical_device_id_to_visible_device_id",
                        return_value=2,
                    ),
                ):
                    backend_base.set_scheduler_device(parallel_config)  # type: ignore[arg-type]

                npu.set_device.assert_called_once_with(expected)
                if assigned_ids is not None:
                    set_ids.assert_called_once_with(assigned_ids)
                    npu.current_device.assert_not_called()
