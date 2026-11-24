# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-prefix parsing with real upstream connector configuration."""

import unittest

from tests.ut.distributed.ascend_store._mock_deps import _MOCK_VLLM_DEPS

if not _MOCK_VLLM_DEPS:
    from vllm.config import KVTransferConfig

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import get_cache_prefix


def make_config(extra_config):
    return KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )


@unittest.skipIf(_MOCK_VLLM_DEPS, "requires real vLLM; mock results are not config validation")
class TestRealFixedCachePrefixConfig(unittest.TestCase):
    def test_default_and_literal_values(self):
        self.assertEqual(get_cache_prefix(make_config({}).kv_connector_extra_config), "")
        values: tuple[tuple[object, str], ...] = (
            ("", ""),
            ("release-v1", "release-v1"),
            ("AUTO", "AUTO"),
            (None, "None"),
            (False, "False"),
            (0, "0"),
        )
        for value, expected in values:
            with self.subTest(value=value):
                config = make_config({"cache_prefix": value})
                self.assertEqual(get_cache_prefix(config.kv_connector_extra_config), expected)

    def test_auto_is_not_supported(self):
        config = make_config({"cache_prefix": "auto"})
        with self.assertRaisesRegex(ValueError, "auto.*not supported"):
            get_cache_prefix(config.kv_connector_extra_config)

    def test_other_backends_keep_default_and_reject_nonempty_prefix(self):
        for backend in ("memcache", "yuanrong"):
            with self.subTest(backend=backend):
                config = make_config({"backend": backend})
                self.assertEqual(get_cache_prefix(config.kv_connector_extra_config), "")
                config.kv_connector_extra_config["cache_prefix"] = "release-v1"
                with self.assertRaisesRegex(ValueError, "only.*Mooncake"):
                    get_cache_prefix(config.kv_connector_extra_config)
