# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


import unittest
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend


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
