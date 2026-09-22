"""Stdlib-only PR1 selector checks (no torch, Triton, device, or pytest)."""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "vllm_ascend" / "ops" / "triton" / "layernorm_gated_dispatch.py"


def load_selector():
    name = "pr1_layernorm_dispatch"
    spec = importlib.util.spec_from_file_location(name, PATH)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


class SelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_selector()
        cls.params = cls.mod.DispatchParams(
            bm_small=16,
            bm_multi=32,
            k_persist_num=1,
            k_persist_den=4,
            n_persist_min=128,
            hoist_qualified=True,
            persist_single_qualified=True,
            persist_multi_qualified=False,
        )

    def test_unknown_p_and_unqualified_device_keep_base64(self):
        m = self.mod
        self.assertEqual(
            m._select_layernorm_launch(20449, 128, 1, None, self.params),
            m.LaunchSpec("FT_BASE", 64),
        )
        self.assertEqual(
            m._select_layernorm_launch(
                20449, 128, 1, 40, self.params, qualified=False
            ),
            m.LaunchSpec("FT_BASE", 64),
        )

    def test_a2_calibrated_boundaries_are_formula_based(self):
        m = self.mod
        expected = {
            288: m.LaunchSpec("FT_BASE", 16),
            289: m.LaunchSpec("FT_PERSIST", 32),
            20448: m.LaunchSpec("FT_PERSIST", 32),
            20449: m.LaunchSpec("FT_PERSIST_HOIST", 32),
        }
        for rows, spec in expected.items():
            self.assertEqual(
                m._select_layernorm_launch(rows, 128, 1, 40, self.params), spec
            )

    def test_multi_group_and_wide_n_are_base_fallbacks(self):
        m = self.mod
        self.assertEqual(
            m._select_layernorm_launch(2048, 128, 2, 40, self.params),
            m.LaunchSpec("FT_BASE", 32),
        )
        self.assertEqual(
            m._select_layernorm_launch(2048, 127, 2, 40, self.params),
            m.LaunchSpec("FT_BASE", 16),
        )
        self.assertEqual(
            m._select_layernorm_launch(2048, 256, 1, 40, self.params),
            m.LaunchSpec("FT_BASE", 64),
        )

    def test_invalid_policy_and_shape_fail_closed(self):
        m = self.mod
        with self.assertRaises(m.DispatchConfigError):
            m._select_layernorm_launch(0, 128, 1, 40, self.params)
        with self.assertRaises(m.DispatchConfigError):
            m._select_layernorm_launch(
                64,
                32,
                1,
                40,
                m.DispatchParams(
                    bm_small=16,
                    bm_multi=32,
                    k_persist_num=1,
                    k_persist_den=4,
                    n_persist_min=128,
                    hoist_qualified=True,
                    persist_single_qualified=False,
                ),
            )


if __name__ == "__main__":
    unittest.main()
