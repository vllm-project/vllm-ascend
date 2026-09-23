# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Complementary accuracy coverage for the fused rms_norm_cast op.

The base test (test_rms_norm_cast.py) pins one hidden size and a few token
counts. The optimized kernel has additional paths that need their own
coverage:

- The row pipeline (double-buffered slots, deferred event flags) takes
  different branches for 1, 2 and 3 rows per core, for cores with mixed
  row counts and for idle cores.
- The rstd vector broadcast handles hidden sizes that are not multiples
  of the 64-element repeat, and the copies handle non-block-aligned
  sizes.
- The op contract allows any input rank >= 1 and any non-negative
  epsilon.
- Every event flag must be consumed before the kernel exits; a leftover
  flag poisons the next kernel on the same core. Sequential launches in
  one process guard that contract.
- The tiling rejects oversized hidden sizes and negative epsilon, and
  the adapter rejects mismatched x/gamma dtypes.
"""

import pytest
import torch
import torch_npu


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    # Independent FP32 reductions can land on adjacent BF16 values.
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    return tolerance, tolerance


def _assert_matches_reference(x: torch.Tensor, gamma: torch.Tensor,
                              epsilon: float) -> None:
    actual, actual_fp32 = torch.ops._C_ascend.npu_rms_norm_cast(
        x, gamma, epsilon)
    expected, _ = torch_npu.npu_rms_norm(x, gamma, epsilon)
    rtol, atol = _tolerances(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    # Routing must consume the widened, already-rounded RMSNorm result.
    torch.testing.assert_close(actual_fp32, actual.float(), rtol=0, atol=0)


# 2 and 3 tokens give a single row on a couple of cores; 41 splits into
# cores with 2 rows, one core with a single row and idle cores; 80 and 120
# give exactly 2 and 3 rows per core, the boundaries where the pipeline
# starts and stops deferring its event flags.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [2, 3, 41, 80, 120])
def test_rms_norm_cast_row_pipeline_boundaries(dtype: torch.dtype,
                                               num_tokens: int):
    torch.manual_seed(3)
    x = torch.randn(num_tokens, 7168, dtype=dtype, device="npu")
    gamma = torch.randn(7168, dtype=dtype, device="npu")
    _assert_matches_reference(x, gamma, 1e-6)


# 64 is a single broadcast repeat; 100 is neither 64- nor 16-aligned and
# exercises the masked broadcast tail plus padded copies; 7184 leaves a
# 16-element tail next to the production hidden size.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64, 100, 7184])
def test_rms_norm_cast_hidden_size_tail(dtype: torch.dtype,
                                        hidden_size: int):
    torch.manual_seed(4)
    x = torch.randn(37, hidden_size, dtype=dtype, device="npu")
    gamma = torch.randn(hidden_size, dtype=dtype, device="npu")
    _assert_matches_reference(x, gamma, 1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_cast_high_rank_input(dtype: torch.dtype):
    torch.manual_seed(5)
    x = torch.randn(2, 3, 7168, dtype=dtype, device="npu")
    gamma = torch.randn(7168, dtype=dtype, device="npu")
    _assert_matches_reference(x, gamma, 1e-6)


@pytest.mark.parametrize("epsilon", [0.0, 1e-3])
def test_rms_norm_cast_epsilon_values(epsilon: float):
    torch.manual_seed(6)
    x = torch.randn(16, 7168, dtype=torch.bfloat16, device="npu")
    gamma = torch.randn(7168, dtype=torch.bfloat16, device="npu")
    _assert_matches_reference(x, gamma, epsilon)


def test_rms_norm_cast_input_extremes():
    torch.manual_seed(6)
    gamma = torch.randn(7168, dtype=torch.bfloat16, device="npu")
    # All-zero rows make rstd = 1/sqrt(epsilon); large magnitudes stress
    # the fp32 reduction accumulation.
    zeros = torch.zeros(8, 7168, dtype=torch.bfloat16, device="npu")
    _assert_matches_reference(zeros, gamma, 1e-6)
    large = torch.randn(8, 7168, dtype=torch.bfloat16,
                        device="npu") * 100.0
    _assert_matches_reference(large, gamma, 1e-6)


def test_rms_norm_cast_sequential_launches():
    """Back-to-back launches must not corrupt per-core event state.

    The pipelined kernel must consume every event flag it raises before
    exiting; a leftover flag hangs the next kernel on the same core.
    Interleave shapes with different rows-per-core and dtypes, then run
    the reference op afterwards and verify every result.
    """
    torch.manual_seed(7)
    x_a = torch.randn(41, 7168, dtype=torch.bfloat16, device="npu")
    gamma_a = torch.randn(7168, dtype=torch.bfloat16, device="npu")
    x_b = torch.randn(80, 100, dtype=torch.float16, device="npu")
    gamma_b = torch.randn(100, dtype=torch.float16, device="npu")
    x_c = torch.randn(3, 7184, dtype=torch.bfloat16, device="npu")
    gamma_c = torch.randn(7184, dtype=torch.bfloat16, device="npu")

    y_a, y_a_fp32 = torch.ops._C_ascend.npu_rms_norm_cast(x_a, gamma_a, 1e-6)
    y_b, y_b_fp32 = torch.ops._C_ascend.npu_rms_norm_cast(x_b, gamma_b, 1e-6)
    ref_a, _ = torch_npu.npu_rms_norm(x_a, gamma_a, 1e-6)
    y_c, y_c_fp32 = torch.ops._C_ascend.npu_rms_norm_cast(x_c, gamma_c, 1e-6)
    torch.npu.synchronize()
    ref_b, _ = torch_npu.npu_rms_norm(x_b, gamma_b, 1e-6)
    ref_c, _ = torch_npu.npu_rms_norm(x_c, gamma_c, 1e-6)

    for actual, actual_fp32, expected in ((y_a, y_a_fp32, ref_a),
                                          (y_b, y_b_fp32, ref_b),
                                          (y_c, y_c_fp32, ref_c)):
        rtol, atol = _tolerances(actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        torch.testing.assert_close(actual_fp32, actual.float(), rtol=0,
                                   atol=0)


@pytest.mark.parametrize("case", ["oversized_hidden", "negative_epsilon",
                                  "gamma_dtype_mismatch"])
def test_rms_norm_cast_rejects_invalid_inputs(case: str):
    torch.manual_seed(8)
    x = torch.randn(4, 7168, dtype=torch.bfloat16, device="npu")
    gamma = torch.randn(7168, dtype=torch.bfloat16, device="npu")
    if case == "oversized_hidden":
        # 22B/col bf16 far exceeds any supported UB budget.
        x = torch.randn(4, 16384, dtype=torch.bfloat16, device="npu")
        gamma = torch.randn(16384, dtype=torch.bfloat16, device="npu")
        epsilon = 1e-6
    elif case == "negative_epsilon":
        epsilon = -1.0
    else:
        gamma = torch.randn(7168, dtype=torch.float16, device="npu")
        epsilon = 1e-6
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.npu_rms_norm_cast(x, gamma, epsilon)
