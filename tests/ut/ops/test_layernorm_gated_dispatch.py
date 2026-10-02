"""Stdlib-only PR1 selector checks (no torch, Triton, device, or pytest)."""

import importlib.util
import sys
import unittest
from pathlib import Path
from typing import Any, ClassVar

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "vllm_ascend" / "ops" / "triton" / "layernorm_gated_dispatch.py"


def load_selector():
    name = "pr1_layernorm_dispatch"
    spec = importlib.util.spec_from_file_location(name, PATH)
    assert spec is not None and spec.loader is not None
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
    mod: ClassVar[Any]

    @classmethod
    def setUpClass(cls):
        cls.mod = load_selector()

    def test_unknown_p_keeps_base64(self):
        self.assertEqual(
            self.mod._select_layernorm_launch(20449, 128, 1, None),
            self.mod.LaunchSpec("FT_BASE", 64),
        )

    def test_single_group_uses_integer_quarter_wave_boundary(self):
        m = self.mod
        for rows in (128, 288):
            with self.subTest(rows=rows):
                self.assertEqual(
                    m._select_layernorm_launch(rows, 128, 1, 40),
                    m.LaunchSpec("FT_BASE", 16),
                )
        for rows in (289, 290, 639, 640, 641, 20449, 65536):
            with self.subTest(rows=rows):
                self.assertEqual(
                    m._select_layernorm_launch(rows, 128, 1, 40),
                    m.LaunchSpec("FT_PERSIST_HOIST", 32),
                )

    def test_boundary_scales_with_runtime_vector_core_count(self):
        m = self.mod
        for rows, expected in (
            (352, m.LaunchSpec("FT_BASE", 16)),
            (353, m.LaunchSpec("FT_PERSIST_HOIST", 32)),
            (354, m.LaunchSpec("FT_PERSIST_HOIST", 32)),
        ):
            with self.subTest(rows=rows):
                self.assertEqual(m._select_layernorm_launch(rows, 128, 1, 48), expected)

    def test_quarter_wave_boundary_uses_exact_integer_tiles(self):
        m = self.mod
        for runtime_p in (1, 3, 7, 39, 41):
            tiles_needed = (runtime_p + 3) // 4
            boundary = (tiles_needed - 1) * m.BM_HOIST + 1
            with self.subTest(runtime_p=runtime_p):
                self.assertEqual(
                    m._select_layernorm_launch(boundary, 128, 1, runtime_p),
                    m.LaunchSpec("FT_PERSIST_HOIST", 32),
                )
                if boundary > 1:
                    self.assertEqual(
                        m._select_layernorm_launch(boundary - 1, 128, 1, runtime_p),
                        m.LaunchSpec("FT_BASE", 16),
                    )

    def test_small_group_multi_group_and_wide_n_routes(self):
        m = self.mod
        self.assertEqual(m._select_layernorm_launch(2048, 128, 2, 40), m.LaunchSpec("FT_BASE", 32))
        self.assertEqual(m._select_layernorm_launch(2048, 127, 2, 40), m.LaunchSpec("FT_BASE", 16))
        self.assertEqual(m._select_layernorm_launch(2048, 129, 1, 40, ub_bytes=196608), m.LaunchSpec("FT_BASE", 16))
        self.assertEqual(m._select_layernorm_launch(2048, 192, 1, 40, ub_bytes=196608), m.LaunchSpec("FT_BASE", 16))
        self.assertEqual(m._select_layernorm_launch(2048, 256, 1, 40, ub_bytes=196608), m.LaunchSpec("FT_BASE", 16))
        self.assertEqual(m._select_layernorm_launch(2048, 384, 1, 40, ub_bytes=196608), m.LaunchSpec("FT_BASE", 16))
        self.assertEqual(m._select_layernorm_launch(2048, 512, 1, 40, ub_bytes=196608), m.LaunchSpec("FT_BASE", 16))
        self.assertEqual(m._select_layernorm_launch(2048, 513, 1, 40, ub_bytes=196608), m.LaunchSpec("FT_BASE", 64))

    def test_ft16_requires_qualified_ub_and_initialized_p(self):
        m = self.mod
        for ub_bytes in (None, 196607):
            with self.subTest(ub_bytes=ub_bytes):
                self.assertEqual(
                    m._select_layernorm_launch(65, 192, 1, 40, ub_bytes=ub_bytes),
                    m.LaunchSpec("FT_BASE", 64),
                )
        self.assertEqual(
            m._select_layernorm_launch(65, 192, 1, None, ub_bytes=196608),
            m.LaunchSpec("FT_BASE", 64),
        )

    def test_invalid_shape_and_resource_inputs_fail_closed(self):
        m = self.mod
        with self.assertRaises(m.DispatchConfigError):
            m._select_layernorm_launch(0, 128, 1, 40)
        with self.assertRaisesRegex(m.DispatchConfigError, "ub_bytes"):
            m._select_layernorm_launch(65, 192, 1, 40, ub_bytes=0)
        with self.assertRaisesRegex(m.DispatchConfigError, "runtime_p"):
            m._select_layernorm_launch(65, 128, 1, 0)


if __name__ == "__main__":
    unittest.main()
