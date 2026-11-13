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

"""Unit tests for the AscendStoreConnector(memcache) driver version precheck."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    precheck_driver_version,
    precheck_memcache_driver_version,
    should_precheck_memcache_backend,
)

# isort: on


def _write_version_info(content: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".info", delete=False, encoding="utf-8") as version_file:
        version_file.write(content)
        return version_file.name


def _kv_transfer_config(connector=None, extra=None):
    config = MagicMock()
    config.kv_connector = connector
    config.kv_connector_extra_config = extra if extra is not None else {}
    return config


def _vllm_config(kv_transfer_config):
    config = MagicMock()
    config.kv_transfer_config = kv_transfer_config
    return config


class TestPrecheckDriverVersion(unittest.TestCase):
    def tearDown(self):
        for attr in ("version_info_path",):
            path = getattr(self, attr, None)
            if path and Path(path).exists():
                Path(path).unlink()

    def test_parses_innerversion(self):
        self.version_info_path = _write_version_info("Version=V100R001\nInnerversion=V100R001C23SPC008B220\n")
        self.assertEqual(precheck_driver_version(self.version_info_path), "V100R001C23SPC008B220")

    def test_key_is_case_insensitive(self):
        self.version_info_path = _write_version_info("innerversion=V100R001C21B035\n")
        self.assertEqual(precheck_driver_version(self.version_info_path), "V100R001C21B035")

    def test_missing_file_raises_with_mount_hint(self):
        with self.assertRaises(RuntimeError) as context:
            precheck_driver_version("/nonexistent/driver/version.info")
        self.assertIn("mount", str(context.exception))

    def test_empty_file_raises(self):
        self.version_info_path = _write_version_info("")
        with self.assertRaises(RuntimeError) as context:
            precheck_driver_version(self.version_info_path)
        self.assertIn("is empty", str(context.exception))

    def test_whitespace_file_raises(self):
        self.version_info_path = _write_version_info("  \n\t\n")
        with self.assertRaises(RuntimeError) as context:
            precheck_driver_version(self.version_info_path)
        self.assertIn("is empty", str(context.exception))

    def test_content_without_innerversion_returns_none(self):
        self.version_info_path = _write_version_info("Version=V100R001C23SPC008B220\n")
        self.assertIsNone(precheck_driver_version(self.version_info_path))


class TestShouldPrecheckMemcacheBackend(unittest.TestCase):
    def test_no_kv_transfer_config(self):
        self.assertFalse(should_precheck_memcache_backend(_vllm_config(None)))

    def test_top_level_memcache_connector(self):
        config = _kv_transfer_config("AscendStoreConnector", {"backend": "memcache", "lookup_rpc_port": 0})
        self.assertTrue(should_precheck_memcache_backend(_vllm_config(config)))

    def test_memcache_nested_in_multi_connector(self):
        extra = {
            "connectors": [
                {"kv_connector": "MooncakeConnector", "kv_connector_extra_config": {}},
                {"kv_connector": "AscendStoreConnector", "kv_connector_extra_config": {"backend": "memcache"}},
            ]
        }
        config = _kv_transfer_config("MultiConnector", extra)
        self.assertTrue(should_precheck_memcache_backend(_vllm_config(config)))

    def test_ascend_store_default_backend_is_not_memcache(self):
        config = _kv_transfer_config("AscendStoreConnector", {})
        self.assertFalse(should_precheck_memcache_backend(_vllm_config(config)))

    def test_unrelated_connector(self):
        config = _kv_transfer_config("MooncakeConnector", {})
        self.assertFalse(should_precheck_memcache_backend(_vllm_config(config)))

    def test_sub_connectors_without_ascend_store(self):
        extra = {"connectors": [{"kv_connector": "MooncakeConnector", "kv_connector_extra_config": {}}]}
        config = _kv_transfer_config("MultiConnector", extra)
        self.assertFalse(should_precheck_memcache_backend(_vllm_config(config)))


class TestPrecheckMemcacheDriverVersion(unittest.TestCase):
    def test_skips_non_memcache_deployments(self):
        config = _kv_transfer_config("MooncakeConnector", {})
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.precheck_driver_version"
        ) as precheck:
            precheck_memcache_driver_version(_vllm_config(config))
        precheck.assert_not_called()

    def test_runs_for_memcache_deployments(self):
        config = _kv_transfer_config("AscendStoreConnector", {"backend": "memcache"})
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.precheck_driver_version"
        ) as precheck:
            precheck_memcache_driver_version(_vllm_config(config))
        precheck.assert_called_once_with()


class TestSetupStoreError(unittest.TestCase):
    def _backend_without_init(self):
        backend = MemcacheBackend.__new__(MemcacheBackend)
        backend.device_id = 0
        backend._init_bm = False
        backend._lazy_init = False
        return backend

    def test_nonzero_return_code_raises_with_context(self):
        memcache_module = MagicMock()
        store = memcache_module.DistributedObjectStore.return_value
        store.init.return_value = 1
        with (
            patch.dict(sys.modules, {"memcache_hybrid": memcache_module}),
            self.assertRaises(RuntimeError) as context,
        ):
            self._backend_without_init()._setup_store()
        self.assertIn("return code 1", str(context.exception))

    def test_successful_init_returns_store(self):
        memcache_module = MagicMock()
        store = memcache_module.DistributedObjectStore.return_value
        store.init.return_value = 0
        with patch.dict(sys.modules, {"memcache_hybrid": memcache_module}):
            backend = self._backend_without_init()
            self.assertIs(backend._setup_store(), store)


if __name__ == "__main__":
    unittest.main()
