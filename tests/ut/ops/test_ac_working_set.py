"""Self-contained integer resource contracts; no Torch, NumPy, compiler or reports."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

OP_DIR = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/linearnorm"


class WorkingSetTests(unittest.TestCase):
    module: ClassVar[ModuleType]

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("_test_ac_working_set", OP_DIR / "ac_working_set.py")
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def stages(self, *args):
        local: dict[str, Any] = {}
        function = self.module.working_set
        previous = sys.gettrace()

        def trace(frame, event, _arg):
            if frame.f_code is function.__code__ and event == "return":
                local.update(frame.f_locals)
            return trace

        sys.settrace(trace)
        try:
            value = function(*args)
        finally:
            sys.settrace(previous)
        return value, local

    def test_calibrated_phase_budgets(self):
        cases = (
            ((2, 8, 2, 256, 64, True), (50156, 92444, 92380, 54644)),
            ((4, 8, 2, 256, 64, True), (83668, 181556, 181428, 54644)),
            ((2, 8, 8, 256, 64, True), (76316, 110924, 111436, 74660)),
            ((2, 24, 4, 256, 64, True), (76316, 139628, 139532, 147460)),
            ((4, 6, 1, 256, 64, True), (61348, 134372, 134276, 40540)),
            ((4, 4, 1, 256, 64, True), (46468, 93348, 93284, 29772)),
            ((2, 8, 2, 128, 128, False), (49280, 67424, 67424, 27960)),
            ((2, 8, 2, 256, 64, True, True), (91628, 121116, 121052, 58228)),
            ((2, 2, 1, 512, 192, False), (53104, 59952, 59952, 34952)),
        )
        for geometry, expected in cases:
            with self.subTest(geometry=geometry):
                value, local = self.stages(*geometry)
                self.assertEqual(tuple(local[key] for key in ("init", "bulk", "output", "tail")), expected)
                self.assertEqual(value, max(expected))
                self.assertIs(type(value), int)

    def test_init_packed_predicate_negative_mutant(self):
        _, local = self.stages(2, 8, 2, 128, 128, False)
        # Reusing h32's fixed 8M term at h64 misses a real 16-byte predicate pool.
        mutant = max(24 * 64 + 4 * 2, local["bulk_control_bytes"] + 2 * 64 * 2 + 8 * 2)
        self.assertEqual(local["init_control_peak_bytes"] - mutant, 16)
        self.assertEqual(local["init"] - 16, 49264)

    def test_k8_m4_rejected_by_resource_model_itself(self):
        m4, local = self.stages(4, 8, 8, 256, 64, True)
        self.assertEqual(m4, 219540)
        self.assertGreater(m4, 196608)
        self.assertEqual(local["k_raw_bytes"], 16384)
        self.assertEqual(local["k_output_live_bytes"], local["k_raw_bytes"] + 2 * local["k_compute_bytes"])
        self.assertEqual(self.module.working_set(2, 8, 8, 256, 64, True), 111436)

    def test_budget_floor_padding_and_scratch_minimum(self):
        for small, padded in (((3, 2), (128, 64)), ((129, 66), (256, 128))):
            with self.subTest(geometry=small):
                value, local = self.stages(2, 2, 1, *small, False)
                self.assertEqual(value, self.module.working_set(2, 2, 1, *padded, False))
                self.assertEqual((local["D"], local["h"]), (padded[0], padded[1] // 2))
                self.assertGreaterEqual(local["bulk_mul_q"], 256)
                self.assertGreater(local["tail_control_bytes"], 0)
        _, local = self.stages(2, 2, 1, 512, 512, False)
        self.assertEqual(local["Sinit"], 512)

    def test_bias_and_mask_roles_are_separate_increments(self):
        _, base = self.stages(2, 8, 2, 256, 64, True)
        _, bias = self.stages(2, 8, 2, 256, 64, True, True)
        self.assertEqual(bias["prepared_parameter_bytes"] - base["prepared_parameter_bytes"], 28672)
        self.assertEqual(bias["init_parameter_aux_bytes"] - base["init_parameter_aux_bytes"], 12800)
        self.assertEqual(bias["tail"] - base["tail"], 3584)
        _, masked = self.stages(2, 8, 2, 256, 64, True, False, False)
        self.assertEqual(masked["mask_extra"], 1200)
        self.assertEqual(masked["tail_mask_extra"], 984)
        # Non-interleaved preparation carriers are not resident through bulk/output.
        self.assertEqual(masked["init"] - base["init"], 1200)
        self.assertEqual(masked["tail"] - base["tail"], 984)
        self.assertEqual(masked["bulk"], base["bulk"])
        self.assertEqual(masked["output"], base["output"])
        self.assertEqual(
            base["output_aux_bytes"],
            max(2 * 8 * 256 * 2, 16 * 4 + 2 * 16 * 32 * 4 + 512, 4 * 16 * 32 * 4, base["k_output_aux_bytes"]),
        )

    def test_total_integer_for_large_heads_and_strict_errors(self):
        for q in (1, 2, 6, 12, 24, 36, 48, 72, 120):
            for k in (1, 2, 8, 41, 80):
                value = self.module.working_set(2, q, k, 129, 66, True, True, False)
                self.assertIs(type(value), int)
                self.assertGreater(value, 0)
        for args, error in (
            ((2, True, 1, 128, 64, False), TypeError),
            ((2, 2, 1, 0, 2, False), ValueError),
            ((2, 2, 1, 128, 65, False), ValueError),
            ((2, 2, 1, 128, 130, False), ValueError),
            ((2, 16, 1, 128, 64, False), ValueError),
            ((2, 2, 1, 128, 64, False, 1), TypeError),
        ):
            with self.subTest(args=args), self.assertRaises(error):
                self.module.working_set(*args)


if __name__ == "__main__":
    unittest.main()
