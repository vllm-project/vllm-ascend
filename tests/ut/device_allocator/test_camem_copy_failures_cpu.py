# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU-only ACL failure injection; no allocation or native copy is executed.

Run: python tests/ut/device_allocator/test_camem_copy_failures_cpu.py
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


class AllocatorMemcpyTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/device_allocator/camem.py"
        spec = importlib.util.spec_from_file_location("tested_camem", path)
        assert spec is not None and spec.loader is not None
        camem = importlib.util.module_from_spec(spec)
        self.backup = Mock()
        self.backup.data_ptr.return_value = 0x2000
        torch = MagicMock()
        torch.empty.return_value = self.backup
        self.memcpy = Mock(return_value=0)
        allocator_api = SimpleNamespace(
            AllocationData=object, HandleType=tuple, DEFERRABLE_TAGS={"weights", "kv_cache"}
        )
        with patch.dict(
            "sys.modules",
            {
                "torch": torch,
                "torch_npu": MagicMock(),
                "acl": MagicMock(),
                "acl.rt": SimpleNamespace(memcpy=self.memcpy),
                "vllm.device_allocator": allocator_api,
                "vllm.device_allocator.cumem": SimpleNamespace(CuMemAllocator=object),
                "vllm.logger": MagicMock(),
                "vllm.utils.system_utils": MagicMock(),
                "vllm_ascend.vllm_ascend_C": MagicMock(),
            },
        ):
            spec.loader.exec_module(camem)
        self.unmap = camem.python_unmap_and_release
        self.create = camem.python_create_and_map
        self.data = SimpleNamespace(handle=(0, 64, 0x1000, 9), tag="weights", is_asleep=False, cpu_backup_tensor=None)
        self.allocator = object.__new__(camem.CaMemAllocator)
        self.allocator.pointer_to_data = {0x1000: self.data}

    def test_sleep_copy_error_keeps_device_data_live(self):
        self.memcpy.return_value = 507000
        with self.assertRaisesRegex(RuntimeError, "507000"):
            self.allocator.sleep(("weights",))
        self.unmap.assert_not_called()
        self.assertFalse(self.data.is_asleep)
        self.assertIsNone(self.data.cpu_backup_tensor)

    def test_failed_sleep_can_retry_with_existing_backup(self):
        self.data.cpu_backup_tensor = self.backup
        self.memcpy.side_effect = (507000, 0)
        with self.assertRaisesRegex(RuntimeError, "507000"):
            self.allocator.sleep(("weights",))
        self.unmap.assert_not_called()
        self.assertFalse(self.data.is_asleep)
        self.assertIs(self.data.cpu_backup_tensor, self.backup)

        self.allocator.sleep(("weights",))
        self.unmap.assert_called_once_with(*self.data.handle)
        self.assertTrue(self.data.is_asleep)
        self.assertIs(self.data.cpu_backup_tensor, self.backup)

    def test_wakeup_copy_error_preserves_backup_and_can_retry(self):
        self.data.is_asleep = True
        self.data.cpu_backup_tensor = self.backup
        self.memcpy.return_value = 507000
        with self.assertRaisesRegex(RuntimeError, "507000"):
            self.allocator.wake_up()
        self.assertTrue(self.data.is_asleep)
        self.assertIs(self.data.cpu_backup_tensor, self.backup)
        self.unmap.assert_called_once_with(*self.data.handle)
        self.memcpy.return_value = 0
        self.allocator.wake_up()
        self.assertFalse(self.data.is_asleep)
        self.assertIs(self.data.cpu_backup_tensor, self.backup)
        self.assertEqual(self.create.call_count, 2)

    def test_successful_roundtrip_uses_destination_capacity_in_bytes(self):
        self.allocator.sleep(("weights",))
        self.memcpy.assert_called_once_with(0x2000, 64, 0x1000, 64, 2)
        self.assertTrue(self.data.is_asleep)
        self.assertIs(self.data.cpu_backup_tensor, self.backup)
        self.memcpy.reset_mock()
        self.allocator.wake_up()
        self.memcpy.assert_called_once_with(0x1000, 64, 0x2000, 64, 1)
        self.assertFalse(self.data.is_asleep)
        self.assertIs(self.data.cpu_backup_tensor, self.backup)


if __name__ == "__main__":
    unittest.main()
