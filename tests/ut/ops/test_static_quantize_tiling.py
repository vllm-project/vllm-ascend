# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import unittest

from vllm_ascend.ops.triton.static_quantize_tiling import (
    _cann_ub_bytes,
    _cann_ub_factors,
    _quantize_tiling,
    _quantize_ub_bytes,
)


def covered_indices(plan, rows, width):
    for core in range(plan["NUM_CORES"]):
        row_start = core // plan["COLUMN_CORES"] * plan["ROW_FACTOR"]
        column_start = core % plan["COLUMN_CORES"] * plan["COLUMN_FACTOR"]
        row_end = min(row_start + plan["ROW_FACTOR"], rows)
        column_end = min(column_start + plan["COLUMN_FACTOR"], width)
        for column_tile in range(0, plan["COLUMN_FACTOR"], plan["BLOCK_COLS"]):
            for row_batch in range(row_start, row_end, plan["BASE_N"]):
                for row in range(row_batch, min(row_batch + plan["BASE_N"], row_end)):
                    for column in range(
                        column_start + column_tile, min(column_start + column_tile + plan["BLOCK_COLS"], column_end)
                    ):
                        yield row * width + column


class QuantizeTilingTests(unittest.TestCase):
    def make_plan(
        self,
        rows,
        width,
        element_bytes=2,
        parameter_bytes=2,
        contiguous=True,
        parameter_contiguous=True,
        per_tensor=False,
        cores=48,
        ub=196608,
        line=512,
    ):
        return _quantize_tiling(
            rows, width, element_bytes, parameter_bytes, contiguous, parameter_contiguous, per_tensor, cores, ub, line
        )

    def test_strided_layouts_keep_original_cann_core_partition(self):
        # CANN v9.1 source / trace: hardware pool 48 gives 6 rows x 8
        # column cores for these shapes. This is an original-source fixture,
        # not a model-specific policy in the production planner.
        for width in (8192, 2048):
            for contiguous, parameter_contiguous in itertools.product((False, True), repeat=2):
                if contiguous and parameter_contiguous:
                    continue
                original = self.make_plan(6, width, contiguous=contiguous, parameter_contiguous=parameter_contiguous)
                self.assertEqual(
                    (
                        original["NUM_CORES"],
                        original["ROW_FACTOR"],
                        original["COLUMN_CORES"],
                        original["COLUMN_FACTOR"],
                    ),
                    (48, 1, 8, width // 8),
                )
                self.assertEqual(
                    (original["BASE_N"], original["BLOCK_ROWS"], original["BLOCK_COLS"]), (1, 1, width // 8)
                )
        row = self.make_plan(67, 17)
        self.assertEqual((row["NUM_CORES"], row["ROW_FACTOR"], row["COLUMN_CORES"]), (34, 2, 1))
        column = self.make_plan(1, 4097, contiguous=False)
        self.assertEqual((column["NUM_CORES"], column["ROW_FACTOR"], column["COLUMN_CORES"]), (17, 1, 17))
        legacy = self.make_plan(10, 17, line=256)
        self.assertEqual((legacy["NUM_CORES"], legacy["ROW_FACTOR"]), (5, 2))
        self.assertEqual(set(covered_indices(legacy, 10, 17)), set(range(170)))

    def test_small_contiguous_rows_use_resource_sized_tasks(self):
        # Expected consequences of the measured 196608B UB / 48-core resource
        # plan, not shape guards. A wider row requires more UB-sized pieces.
        for rows, width, cores, column_factor in (
            (6, 8192, 6, 8192),
            (6, 2048, 6, 2048),
            (7, 16384, 14, 8192),
            (1, 65536, 8, 8192),
        ):
            plan = self.make_plan(rows, width)
            self.assertEqual((plan["NUM_CORES"], plan["COLUMN_FACTOR"]), (cores, column_factor))
            self.assertEqual(sorted(covered_indices(plan, rows, width)), list(range(rows * width)))
        # Capacity, rather than an element-count constant, changes the pool.
        normal = self.make_plan(3, 16384)
        larger_ub = self.make_plan(3, 16384, ub=393216)
        self.assertEqual((normal["NUM_CORES"], larger_ub["NUM_CORES"]), (6, 3))
        self.assertGreater(larger_ub["BLOCK_COLS"], normal["BLOCK_COLS"])
        for cores in (7, 13, 48):
            plan = self.make_plan(5, 8193, cores=cores)
            self.assertLessEqual(plan["NUM_CORES"], cores)
            self.assertEqual(sorted(covered_indices(plan, 5, 8193)), list(range(5 * 8193)))

    def test_cann_dtype_specific_buffers_not_one_bytes_per_element_constant(self):
        # Independent InitBuffer totals from the v9.1 FP16/BF16/FP32 headers.
        # BF16/FP16: batch x/y + one FP32 work row + raw/FP32 parameters.
        self.assertEqual(_cann_ub_bytes(3, 256, 2, 2, False), (3 * 3 + 4 + 12) * 256)
        # FP32: batch input is reused as its work buffer, parameters need no cast.
        self.assertEqual(_cann_ub_bytes(3, 256, 4, 4, False), (3 * 5 + 8) * 256)
        # Scalar BF16 parameters occupy two 32B buffers; FP32 reads GM scalars.
        self.assertEqual(_cann_ub_bytes(1, 256, 2, 2, True), 7 * 256 + 64)
        self.assertEqual(_cann_ub_bytes(1, 256, 4, 4, True), 5 * 256)

    def test_original_calc_ub_factor_reference(self):
        factors = _cann_ub_factors
        # These are evaluated from the original CANN formulas with its actual
        # 196608-byte UB and 2048-byte reserve, before Triton carrier rounding.
        for dtype_bytes, per_tensor, expected in (
            (2, False, 10240),
            (4, False, 14848),
            (2, True, 27648),
            (4, True, 38912),
        ):
            self.assertEqual(factors(1, 1 << 20, dtype_bytes, per_tensor, False, 196608, 512), (1, expected))
        self.assertEqual(factors(11, 17, 2, False, False, 196608, 512), (11, 256))
        self.assertEqual(factors(3, 8192, 2, False, False, 196608, 512), (1, 8192))
        self.assertEqual(factors(1, 1024, 2, False, True, 196608, 512), (1, 1024))

    def test_smaller_ub_adds_tiles_without_changing_core_ownership(self):
        large = self.make_plan(257, 16387, cores=7)
        small = self.make_plan(257, 16387, cores=7, ub=8192)
        for field in ("NUM_CORES", "ROW_FACTOR", "COLUMN_FACTOR", "COLUMN_CORES"):
            self.assertEqual(large[field], small[field])
        self.assertLess(small["BLOCK_COLS"], large["BLOCK_COLS"])
        self.assertGreater(small["COLUMN_FACTOR"], small["BLOCK_COLS"])
        many_rows = self.make_plan(513, 17, cores=1)
        self.assertEqual(many_rows["ROW_FACTOR"], 513)
        self.assertEqual(set(covered_indices(many_rows, 513, 17)), set(range(513 * 17)))
        self.assertGreater(many_rows["BASE_N"], 1)
        self.assertGreaterEqual(many_rows["BLOCK_ROWS"], many_rows["BASE_N"])
        with self.assertRaisesRegex(RuntimeError, "UB cannot hold"):
            self.make_plan(1, 17, ub=1)

    def test_core_and_ub_tails_cover_every_output_once(self):
        cases = ((1, 31), (2, 33), (3, 257), (7, 1001), (67, 17), (5, 8193), (1, 65539))
        for rows, width in cases:
            for cores, element_bytes, contiguous, parameter_contiguous, ub in itertools.product(
                (1, 7, 48), (2, 4), (True, False), (True, False), (8192, 196608)
            ):
                with self.subTest(
                    rows=rows,
                    width=width,
                    cores=cores,
                    dtype_bytes=element_bytes,
                    contiguous=contiguous,
                    parameter_contiguous=parameter_contiguous,
                    ub=ub,
                ):
                    scalar = rows == 1
                    plan = self.make_plan(
                        rows, width, element_bytes, element_bytes, contiguous, parameter_contiguous, scalar, cores, ub
                    )
                    covered = list(covered_indices(plan, rows, width))
                    self.assertEqual(len(covered), rows * width)
                    self.assertEqual(set(covered), set(range(rows * width)))
                    self.assertLessEqual(plan["NUM_CORES"], cores)
                    self.assertGreater(plan["NUM_CORES"], 0)
                    self.assertEqual(plan["BLOCK_COLS"] & (plan["BLOCK_COLS"] - 1), 0)
                    self.assertEqual(plan["BLOCK_COLS"] % 32, 0)
                    self.assertLessEqual(
                        _cann_ub_bytes(plan["BLOCK_ROWS"], plan["BLOCK_COLS"], element_bytes, element_bytes, scalar),
                        ub - 2048,
                    )
                    self.assertLessEqual(
                        _quantize_ub_bytes(
                            plan["BLOCK_ROWS"],
                            plan["BLOCK_COLS"],
                            element_bytes,
                            element_bytes,
                            contiguous,
                            parameter_contiguous,
                            scalar,
                        ),
                        ub - 2048,
                    )
