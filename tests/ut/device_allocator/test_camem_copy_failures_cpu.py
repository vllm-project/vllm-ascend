# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU-only ACL failure injection; no allocation or native copy is executed.

Run: python tests/ut/device_allocator/test_camem_copy_failures_cpu.py
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


class AllocatorMemcpyTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/device_allocator/camem.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CaMemAllocator")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("sleep", "wake_up")]
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods],
            type_ignores=[],
        )
        self.backup = Mock()
        self.backup.numel.return_value = 64
        self.backup.element_size.return_value = 1
        self.backup.data_ptr.return_value = 0x2000
        self.memcpy = Mock(return_value=0)
        self.unmap = Mock()
        self.create = Mock()
        namespace = {
            "torch": SimpleNamespace(
                empty=Mock(return_value=self.backup), uint8="uint8", npu=Mock(), accelerator=Mock()
            ),
            "memcpy": self.memcpy,
            "unmap_and_release": self.unmap,
            "create_and_map": self.create,
            "gc": Mock(),
            "logger": Mock(),
            "CaMemAllocator": SimpleNamespace(default_tag="default"),
        }
        self.hardware = patch.dict("sys.modules", {"torch_npu": MagicMock(), "acl": MagicMock(), "acl.rt": MagicMock()})
        self.hardware.start()
        self.addCleanup(self.hardware.stop)
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        self.sleep = namespace["sleep"]
        self.wake_up = namespace["wake_up"]
        self.data = SimpleNamespace(handle=(0, 64, 0x1000, 9), tag="weights", is_asleep=False, cpu_backup_tensor=None)
        self.allocator = SimpleNamespace(pointer_to_data={0x1000: self.data})

    def test_sleep_copy_error_keeps_device_data_live(self):
        self.memcpy.return_value = 507000
        with self.assertRaisesRegex(RuntimeError, "507000"):
            self.sleep(self.allocator, ("weights",))
        self.unmap.assert_not_called()
        self.assertFalse(self.data.is_asleep)
        self.assertIsNone(self.data.cpu_backup_tensor)

    def test_wakeup_copy_error_preserves_backup_and_can_retry(self):
        self.data.is_asleep = True
        self.data.cpu_backup_tensor = self.backup
        self.memcpy.return_value = 507000
        with self.assertRaisesRegex(RuntimeError, "507000"):
            self.wake_up(self.allocator)
        self.assertTrue(self.data.is_asleep)
        self.assertIs(self.data.cpu_backup_tensor, self.backup)
        self.unmap.assert_called_once_with(self.data.handle)
        self.memcpy.return_value = 0
        self.wake_up(self.allocator)
        self.assertFalse(self.data.is_asleep)
        self.assertIsNone(self.data.cpu_backup_tensor)
        self.assertEqual(self.create.call_count, 2)

    def test_successful_roundtrip_uses_destination_capacity_in_bytes(self):
        self.sleep(self.allocator, ("weights",))
        self.memcpy.assert_called_once_with(0x2000, 64, 0x1000, 64, 2)
        self.assertTrue(self.data.is_asleep)
        self.assertIs(self.data.cpu_backup_tensor, self.backup)
        self.memcpy.reset_mock()
        self.wake_up(self.allocator)
        self.memcpy.assert_called_once_with(0x1000, 64, 0x2000, 64, 1)
        self.assertFalse(self.data.is_asleep)
        self.assertIsNone(self.data.cpu_backup_tensor)


if __name__ == "__main__":
    unittest.main()
