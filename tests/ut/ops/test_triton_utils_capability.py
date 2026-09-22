"""Stdlib-only regression checks for the cache-only vector-core getter."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "vllm_ascend" / "ops" / "triton" / "triton_utils.py"


class VectorCoreGetterTests(unittest.TestCase):
    def test_getter_reads_cache_without_initialization(self):
        fake_vllm = types.ModuleType("vllm")
        fake_vllm_triton = types.ModuleType("vllm.triton_utils")
        fake_vllm_triton.HAS_TRITON = False
        fake_vllm_triton.tl = None
        fake_vllm_triton.triton = None
        fake_vllm.triton_utils = fake_vllm_triton
        fake_ascend = types.ModuleType("vllm_ascend")
        fake_ascend.envs = types.SimpleNamespace(VLLM_ASCEND_ROPE_UB_SIZE_KB=0)
        fake_torch = types.ModuleType("torch")
        saved = {
            name: sys.modules.get(name)
            for name in ("torch", "vllm", "vllm.triton_utils", "vllm_ascend")
        }
        sys.modules.update(
            {
                "torch": fake_torch,
                "vllm": fake_vllm,
                "vllm.triton_utils": fake_vllm_triton,
                "vllm_ascend": fake_ascend,
            }
        )
        name = "pr1_triton_utils"
        spec = importlib.util.spec_from_file_location(name, PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            module._NUM_VECTORCORE = -1
            self.assertIsNone(module.try_get_vectorcore_num())
            module._NUM_VECTORCORE = 40
            self.assertEqual(module.try_get_vectorcore_num(), 40)
            module._NUM_VECTORCORE = True
            self.assertIsNone(module.try_get_vectorcore_num())
        finally:
            sys.modules.pop(name, None)
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value


if __name__ == "__main__":
    unittest.main()
