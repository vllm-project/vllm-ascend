"""CPU-only checks of actual M2/M4 math, complete tails and ownership."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import ModuleType
from typing import ClassVar
from unittest import mock

import numpy as np

from tests.ut.ops import test_split_qkv_rmsnorm_mrope_dispatch as host_helpers

ROOT = Path(__file__).resolve().parents[3]
OP_REL = "vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py"


class M4ProbeTests(unittest.TestCase):
    host: ClassVar[ModuleType]
    current: ClassVar[str]

    @classmethod
    def setUpClass(cls):
        cls.host = host_helpers
        cls.current = (ROOT / OP_REL).read_text()

    def _run(self, heads, tokens, cores, block_m, *, source=None, launch=None):
        host, case = self.host, self
        base, extra = divmod(tokens, cores)
        packed_width = (2 * heads[0] + 2 * heads[1]) * 256

        class OwnedLanguage(host_helpers._Language):
            def _assert_rows(self, offsets, mask, width):
                offsets, mask = np.broadcast_arrays(offsets, mask)
                rows = offsets[mask] // width
                start = self.core * base + min(self.core, extra)
                end = start + base + (self.core < extra)
                case.assertTrue(np.all((rows >= start) & (rows < end)), "cross-core row access")

            def load(self, pointer, mask=True, other=0):
                if pointer.values.size == tokens * packed_width:
                    self._assert_rows(pointer.offsets, mask, packed_width)
                return super().load(pointer, mask, other)

            def store(self, pointer, values, mask=True):
                self._assert_rows(pointer.offsets, mask, pointer.values.size // tokens)
                return super().store(pointer, values, mask)

        body = host.SplitQKVMRoPEBodyTests()
        with mock.patch.object(host, "_Language", OwnedLanguage):
            return body._check_body(
                *heads, tokens=tokens, cores=cores, block_m=block_m, source_text=source, launch=launch
            )

    def test_m4_real_body_remainders_multiple_tiles_and_core_ownership(self):
        for heads, tokens, cores in (((6, 1), 17, 2), ((24, 4), 21, 2), ((4, 1), 101, 40)):
            with self.subTest(heads=heads, tokens=tokens):
                self._run(heads, tokens, cores, 4)

    def test_m2_real_body_remainders_multiple_tiles_and_ownership(self):
        # _run checks the independent mathematical reference and exact ownership.
        for heads, tokens, cores in (((6, 1), 17, 2), ((24, 4), 21, 2), ((4, 1), 101, 40)):
            with self.subTest(heads=heads, tokens=tokens):
                self._run(heads, tokens, cores, 2)

    def test_m4_keeps_complete_head_tile_rejection_even_without_full_blocks(self):
        with self.assertRaisesRegex(AssertionError, "M2 requires complete Q-head tiles"):
            self._run((16, 4), 3, 1, 4)

    def test_m4_outputs_equal_m2_on_the_same_input(self):
        for heads, tokens, cores in (((6, 1), 17, 2), ((24, 4), 21, 2)):
            current, _ = self._run(heads, tokens, cores, 4)
            m2, _ = self._run(heads, tokens, cores, 2)
            for index, (actual, expected) in enumerate(zip(current, m2)):
                if index < 2:
                    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
                else:
                    self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_dropped_tail_or_wrong_tail_start_mutations_are_detected(self):
        mutations = (
            ("n_tail = loop_num - num_blocks * BLOCK_M", "n_tail = (loop_num - num_blocks * BLOCK_M) % 2"),
            (
                "tail_row = block_offset + num_blocks * BLOCK_M + tail_index",
                "tail_row = num_blocks * BLOCK_M + tail_index",
            ),
        )
        for original, replacement in mutations:
            with self.subTest(replacement=replacement):
                self.assertEqual(self.current.count(original), 1)
                with self.assertRaises((AssertionError, IndexError)):
                    self._run((24, 4), 21, 2, 4, source=self.current.replace(original, replacement))


if __name__ == "__main__":
    unittest.main()
