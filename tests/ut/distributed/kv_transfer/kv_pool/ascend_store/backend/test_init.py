# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


import unittest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    backend_map,
    get_layerwise_protocol,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

_PROTOCOL_FUNCTIONS = ("make_full_key", "make_partial_key", "make_hit_check_keys", "extract_layout_config")
_LAYERWISE_STORE_METHODS = (
    "batch_get_key_info",
    "batch_alloc",
    "batch_add_lease",
    "batch_remove_lease",
    "batch_write_finish",
)


class TestLayerwiseProtocolRegistry(unittest.TestCase):
    """``get_layerwise_protocol`` is the generic layers' only knowledge of
    layerwise support: it resolves the backend module carrying the layerwise
    protocol functions (registered under the normalized backend name), or
    None when the entry carries no protocol marker."""

    def test_get_layerwise_protocol_resolves_module(self):
        protocol = get_layerwise_protocol("memcache")
        self.assertIsNotNone(protocol)
        for func_name in ("make_full_key", "make_partial_key", "make_hit_check_keys", "extract_layout_config"):
            with self.subTest(func=func_name):
                self.assertTrue(callable(getattr(protocol, func_name, None)))

    def test_get_layerwise_protocol_normalizes_name(self):
        for backend_name in ("MEMCACHE", " Memcache "):
            with self.subTest(backend=backend_name):
                self.assertIsNotNone(get_layerwise_protocol(backend_name))

    def test_get_layerwise_protocol_returns_none_without_protocol(self):
        for backend_name in ("mooncake", "yuanrong", "nonexistent"):
            with self.subTest(backend=backend_name):
                self.assertIsNone(get_layerwise_protocol(backend_name))


class TestLayerwiseProtocolMemcacheExclusivity(unittest.TestCase):
    """The memcache backend is the only layerwise protocol carrier.

    Three views of the same fact must agree for every registered backend:
    the module exposes the protocol functions, the class overrides the
    five layerwise store calls (python's MRO: an override wins over the
    inherited NotImplementedError stub), and the registry entry carries
    the ``layerwise_protocol`` marker.
    """

    def _backend_entries(self):
        import importlib

        for name, entry in backend_map.items():
            module = importlib.import_module(entry["path"])
            yield name, entry, module, getattr(module, entry["name"])

    def test_protocol_functions_store_overrides_and_registry_marker_agree(self):
        for name, entry, module, backend_class in self._backend_entries():
            with self.subTest(backend=name):
                exposes_protocol = all(callable(getattr(module, func, None)) for func in _PROTOCOL_FUNCTIONS)
                owns_overrides = all(
                    any(method in vars(cls) for cls in backend_class.__mro__ if cls is not Backend)
                    for method in _LAYERWISE_STORE_METHODS
                )
                self.assertEqual(exposes_protocol, name == "memcache")
                self.assertEqual(owns_overrides, name == "memcache")
                self.assertEqual(exposes_protocol, bool(entry.get("layerwise_protocol")))
                self.assertEqual(owns_overrides, exposes_protocol)
