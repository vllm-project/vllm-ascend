# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU-only Platform classmethod contract regression.

Run directly: python3 tests/ut/test_platform_compute_units_contract.py
The class skeleton contains the production method and its exact decorators.
Pinned upstream Platform.num_compute_units is a classmethod (vLLM commit
24420b7a341af37a9bc3186e6fc6062512e0aa58, interface.py:1438-1445).
"""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SOURCE = Path(__file__).resolve().parents[2] / "vllm_ascend/platform.py"


def load_platform(npu):
    tree = ast.parse(SOURCE.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUPlatform")
    method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "num_compute_units")
    cls = ast.ClassDef(name="NPUPlatform", bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"torch": SimpleNamespace(npu=npu)}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["NPUPlatform"]


class PlatformComputeUnitsTest(unittest.TestCase):
    def setUp(self):
        self.npu = MagicMock()
        self.npu.get_device_properties.return_value = SimpleNamespace(cube_core_num=32)
        self.platform = load_platform(self.npu)
        mock_hardware = patch.dict(sys.modules, {"torch_npu": self.npu, "acl": MagicMock()})
        mock_hardware.start()
        self.addCleanup(mock_hardware.stop)

    def test_class_keyword_call_matches_platform_contract(self):
        self.assertEqual(self.platform.num_compute_units(device_id=3), 32)
        self.npu.get_device_properties.assert_called_once_with(3)

    def test_class_positional_call_preserves_device_id(self):
        self.assertEqual(self.platform.num_compute_units(3), 32)
        self.npu.get_device_properties.assert_called_once_with(3)

    def test_instance_call_still_works(self):
        self.assertEqual(self.platform().num_compute_units(device_id=3), 32)
        self.npu.get_device_properties.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
