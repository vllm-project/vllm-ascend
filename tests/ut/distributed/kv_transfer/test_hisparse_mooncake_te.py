# SPDX-License-Identifier: Apache-2.0
"""Exercise the real adapter without requiring Torch, CANN or Mooncake."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestHiSparseMooncakeTE(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parents[4] / "vllm_ascend"
        utils = root / "distributed/kv_transfer/utils"
        common = load_source("_hisparse_common_te", utils / "mooncake_transfer_engine.py")
        package = ModuleType("vllm_ascend")
        package.envs = load_source("_hisparse_envs", root / "envs.py")
        mooncake = ModuleType("mooncake")
        engine_module = ModuleType("mooncake.engine")
        self.raw = MagicMock()
        self.raw.initialize.return_value = 0
        self.raw.get_rpc_port.return_value = 23456
        self.raw.register_memory.return_value = 0
        self.raw.batch_transfer_sync_read.return_value = 0
        engine_module.TransferEngine = MagicMock(return_value=self.raw)
        self.factory = engine_module.TransferEngine
        modules = patch.dict(
            sys.modules,
            {
                "vllm_ascend": package,
                "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine": common,
                "mooncake": mooncake,
                "mooncake.engine": engine_module,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)
        env = patch.dict(os.environ, {"MC_USE_TENT": "1", "MF_PROTOCOL": "device_urma"})
        env.start()
        self.addCleanup(env.stop)
        self.module = common
        self.manager = self.module.HiSparseMooncakeTE()

    def test_initialize_register_and_read_original_buffers(self):
        engine = self.manager.get_transfer_engine("192.0.2.1", device_name="")
        self.raw.initialize.assert_called_once_with("192.0.2.1", "P2PHANDSHAKE", "memfabric", "")
        self.manager.register_buffer([1000, 3000], [1024, 512])
        self.assertEqual(engine.get_rpc_port(), 23456)
        self.assertEqual(engine.batch_transfer_sync_read("peer:123", [1100], [8000], [64]), 0)
        self.raw.batch_transfer_sync_read.assert_called_once_with(
            "peer:123", [1100], [8000], [64], transport_hint="memfabric"
        )
        self.assertEqual(self.raw.register_memory.call_count, 2)
        self.assertIs(self.manager.get_transfer_engine("192.0.2.1", device_name=""), engine)
        self.factory.assert_called_once()

    def test_rejects_sdma_or_non_tent_before_constructing_engine(self):
        for config in ({"MF_PROTOCOL": "device_sdma"}, {"MF_PROTOCOL": ""}, {"MC_USE_TENT": "0"}):
            with self.subTest(config=config), patch.dict(os.environ, config), self.assertRaises(ValueError):
                self.manager.get_transfer_engine("192.0.2.1", device_name="")
        self.factory.assert_not_called()

    def test_failed_initialization_does_not_publish_and_cannot_retry(self):
        self.raw.initialize.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            self.manager.get_transfer_engine("192.0.2.1", device_name="")
        with self.assertRaisesRegex(RuntimeError, "previously failed"):
            self.manager.get_transfer_engine("192.0.2.1", device_name="")
        self.assertIsNone(self.manager.transfer_engine)

    def test_failed_read_preserves_failure_and_never_uses_write(self):
        engine = self.manager.get_transfer_engine("192.0.2.1", device_name="")
        self.raw.batch_transfer_sync_read.return_value = -7
        self.assertEqual(engine.batch_transfer_sync_read("peer", [1000], [2000], [16]), -7)
        self.raw.batch_transfer_sync_write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
