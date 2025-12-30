import random
import unittest

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

torch.set_printoptions(threshold=float("inf"))


class TestMatrixMultiplication(unittest.TestCase):

    def compute_golden(self, a, b, res1, m, n):
        """Compute reference result (golden)"""
        torch.bmm(a.transpose(0, 1),
                  b,
                  out=res1.view(-1, m, n).transpose(0, 1))

    def assert_tensors_almost_equal(self, actual, expected, dtype):
        """Check if two tensors are approximately equal (considering floating point errors)"""
        self.assertEqual(actual.shape, expected.shape, "Shape mismatch")

        # Check for NaN
        self.assertFalse(
            torch.isnan(actual).any(), "Actual result contains NaN")
        self.assertFalse(
            torch.isnan(expected).any(), "Expected result contains NaN")

        # Check for Inf
        self.assertFalse(
            torch.isinf(actual).any(), "Actual result contains Inf")
        self.assertFalse(
            torch.isinf(expected).any(), "Expected result contains Inf")

        # Set different tolerances based on data type
        if dtype == torch.float16:
            rtol, atol = 1e-5, 1e-5
        else:  # bfloat16
            rtol, atol = 1.5e-5, 1.5e-5

        # Compare values
        diff = torch.abs(actual - expected)
        max_diff = diff.max().item()
        max_expected = torch.abs(expected).max().item()

        # Check relative and absolute errors
        if max_expected > 0:
            relative_diff = max_diff / max_expected
            self.assertLessEqual(
                relative_diff,
                rtol,
                f"Relative error too large: {relative_diff} > {rtol}. Max difference: {max_diff}",
            )

        self.assertLessEqual(max_diff, atol,
                             f"Absolute error too large: {max_diff} > {atol}")

    def test_boundary_conditions(self):
        """Test boundary conditions"""
        test_cases = [
            # (b, m, k, n)
            (1, 1, 1, 1),  # Minimum size
            (1, 10, 1, 1),  # b=1
            (10, 1, 1, 10),  # m=1
            (5, 5, 1, 5),  # k=1
            (2, 2, 2, 1),  # n=1
            (100, 1, 1, 100),  # Flat case
            (1, 100, 100, 1),  # Flat case
            (2, 3, 4, 5),  # Random small size
            (10, 20, 30, 40),  # Medium size
            (36, 128, 512, 128),  # target case
            (8, 160, 512, 128),
        ]

        dtypes = [torch.float16, torch.bfloat16]

        for dtype in dtypes:
            for b, m, k, n in test_cases:
                with self.subTest(dtype=dtype, shape=f"({b}, {m}, {k}, {n})"):
                    a = torch.randn(b, m, k, dtype=dtype, device="npu")
                    b_tensor = torch.randn(m, k, n, dtype=dtype, device="npu")
                    res1 = torch.empty((b, m * n), dtype=dtype, device="npu")
                    res2 = torch.empty((b, m, n), dtype=dtype, device="npu")

                    self.compute_golden(a, b_tensor, res1, m, n)
                    torch.ops._C_ascend.batch_matmul_transpose(
                        a, b_tensor, res2)

                    self.assert_tensors_almost_equal(res1.view(-1, m, n), res2,
                                                     dtype)

    def test_random_shapes(self):
        """Test randomly generated shapes"""
        num_tests = 1
        dtypes = [torch.float16, torch.bfloat16]

        for dtype in dtypes:
            for _ in range(num_tests):
                # Generate reasonable random sizes
                b = random.randint(1, 500)
                m = random.randint(1, 500)
                k = random.randint(1, 500)
                n = random.randint(1, 500)

                with self.subTest(dtype=dtype,
                                  shape=f"Random ({b}, {m}, {k}, {n})"):
                    a = torch.randn(b, m, k, dtype=dtype, device="npu")
                    b_tensor = torch.randn(m, k, n, dtype=dtype, device="npu")
                    res1 = torch.empty((b, m * n), dtype=dtype, device="npu")
                    res2 = torch.empty((b, m, n), dtype=dtype, device="npu")

                    self.compute_golden(a, b_tensor, res1, m, n)
                    torch.ops._C_ascend.batch_matmul_transpose(
                        a, b_tensor, res2)
                    self.assert_tensors_almost_equal(res1.view(-1, m, n), res2,
                                                     dtype)

    def test_zero_values(self):
        """Test zero input values"""
        dtypes = [torch.float16, torch.bfloat16]
        b, m, k, n = 5, 4, 3, 2

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                a = torch.zeros(b, m, k, dtype=dtype, device="npu")
                b_tensor = torch.zeros(m, k, n, dtype=dtype, device="npu")
                res1 = torch.empty((b, m * n), dtype=dtype, device="npu")
                res2 = torch.empty((b, m, n), dtype=dtype, device="npu")

                self.compute_golden(a, b_tensor, res1, m, n)
                torch.ops._C_ascend.batch_matmul_transpose(a, b_tensor, res2)

                self.assert_tensors_almost_equal(res1.view(-1, m, n), res2,
                                                 dtype)
                self.assertTrue(torch.all(res2 == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
