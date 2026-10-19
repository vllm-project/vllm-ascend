# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


class TestFLASharedMemPatch(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/patch_fla_shared_mem.py"
        spec = importlib.util.spec_from_file_location("fla_patch_under_test", path)
        self.patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.patch)
        self.original_meta_path = sys.meta_path[:]
        self.original_sys_path = sys.path[:]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def tearDown(self):
        sys.meta_path[:] = self.original_meta_path
        sys.path[:] = self.original_sys_path

    def _fake_fla(self, backend):
        # Reproduce the real ordering: importing ops executes its __init__,
        # which imports chunk_o, which imports utils and immediately queries it.
        root = Path(self.temp.name)
        for name in ["vllm", "vllm/third_party", "vllm/third_party/flash_linear_attention"]:
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "__init__.py").write_text("")
        ops = root / "vllm/third_party/flash_linear_attention/ops"
        ops.mkdir()
        (ops / "__init__.py").write_text("from .chunk_o import BKV_LIST\n")
        (ops / "chunk_o.py").write_text(
            "from .utils import check_shared_mem\nBKV_LIST = [64, 128] if check_shared_mem() else [32, 64]\n"
        )
        (ops / "utils.py").write_text(
            f"device = {backend!r}\n"
            "calls = []\n"
            "def check_shared_mem(arch='none', tensor_idx=0):\n"
            "    calls.append((arch, tensor_idx))\n"
            "    if device == 'npu':\n"
            "        raise RuntimeError('unexpected NPU initialization')\n"
            "    return True\n"
        )
        sys.path.insert(0, str(root))

    def _import_fla(self, backend, install=True):
        self._fake_fla(backend)
        with patch.dict(sys.modules):
            for name in list(sys.modules):
                if name == "vllm" or name.startswith("vllm."):
                    del sys.modules[name]
            if install:
                self.patch.install_fla_shared_mem_patch()
                self.patch.install_fla_shared_mem_patch()
                self.assertEqual(sum(isinstance(f, self.patch._FLAUtilsFinder) for f in sys.meta_path), 1)
            ops = __import__("vllm.third_party.flash_linear_attention.ops", fromlist=["BKV_LIST"])
            utils = sys.modules[self.patch._FLA_UTILS_MODULE]
            if install:
                self.assertFalse(any(isinstance(f, self.patch._FLAUtilsFinder) for f in sys.meta_path))
            return ops.BKV_LIST, utils.calls

    def test_npu_import_does_not_query_device(self):
        self.assertEqual(self._import_fla("npu"), ([32, 64], []))

    def test_unpatched_import_reproduces_initialization(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected NPU initialization"):
            self._import_fla("npu", install=False)

    def test_non_npu_behavior_is_preserved(self):
        self.assertEqual(self._import_fla("cuda"), ([64, 128], [("none", 0)]))

    def test_already_loaded_module_is_patched_without_import(self):
        module = ModuleType(self.patch._FLA_UTILS_MODULE)
        module.device = "npu"
        with patch.dict(sys.modules, {self.patch._FLA_UTILS_MODULE: module}):
            self.patch.install_fla_shared_mem_patch()
            self.assertFalse(module.check_shared_mem(arch="hopper", tensor_idx=7))

    def test_unrelated_import_is_not_intercepted(self):
        self.assertIsNone(self.patch._FLAUtilsFinder().find_spec("unrelated_module"))


if __name__ == "__main__":
    unittest.main()
