# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from pathlib import Path

_path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/gdn_checkpoint_plan.py"
_spec = importlib.util.spec_from_file_location("gdn_checkpoint_plan", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate_checkpoint_plan = _module.validate_checkpoint_plan


class CheckpointPlanTest(unittest.TestCase):
    def test_ragged_batch_accepts_valid_internal_boundaries(self):
        validate_checkpoint_plan([129, 65, 193], [128, 0, 64])

    def test_no_checkpoint_and_empty_batch(self):
        validate_checkpoint_plan([1, 64], [0, 0])
        validate_checkpoint_plan([], [])

    def test_rejects_end_boundary_and_unaligned_offsets(self):
        for lengths, offsets in [([64], [64]), ([65], [1]), ([0], [0]), ([65], [-64]), ([65], [])]:
            with self.subTest(lengths=lengths, offsets=offsets), self.assertRaises(ValueError):
                validate_checkpoint_plan(lengths, offsets)


if __name__ == "__main__":
    unittest.main()
