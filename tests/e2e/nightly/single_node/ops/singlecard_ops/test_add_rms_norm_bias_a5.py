# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""A5 AddRmsNormBias correctness gates on Ascend 950.

These tests call the custom operator directly, without a Python fallback. The
small-D and D > 16384 cases are intended to cover full-load and SplitD paths;
actual tiling keys still require target-runtime evidence. Beta can change the
UB budget and hence the selected path. No timing or key selection is inferred
from these shapes. Beta is added to the FP32 normalized product before the
final output cast; low-precision npu_add_rms_norm followed by add_ has different
rounding semantics.
"""

import math

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import check_ascend_device_type, is_950
from vllm_ascend.utils import bootstrap_custom_op_env

EPSILON = 1e-6
SEED = 20260907
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Keep the FP32 statistics tolerance separate from the output dtype tolerance.
RSTD_RTOL = 2e-5
RSTD_ATOL = 2e-6

pytestmark = pytest.mark.e2e_coverage(
    arch="",
    feature="",
    parallel="",
    deploy="",
    hardware="A5",
    quantization="",
    graph_mode="eager",
)


@pytest.fixture(scope="module", autouse=True)
def require_a5_custom_operator():
    if not torch.npu.is_available():
        pytest.skip("A5 AddRmsNormBias requires an Ascend 950 device")
    if not is_950():
        pytest.skip("A5 AddRmsNormBias is tested with an Ascend 950 build")
    check_ascend_device_type()
    bootstrap_custom_op_env()
    # Load the operator explicitly without changing the model's runtime gates.
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

    assert hasattr(torch.ops._C_ascend, "npu_add_rms_norm_bias")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::npu_add_rms_norm_bias", "PrivateUse1")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::npu_add_rms_norm_bias", "Meta")


def _reference(x1, x2, gamma, beta, epsilon=EPSILON):
    """CPU reference: independent FP64 reduction over the FP32 residual sum.

    The reduction is deliberately not a replay of the kernel's reduction tree.
    Add beta before the final output cast. Keep the normalization and beta
    addition in FP64 here to provide an independent reference for the kernel's
    FP32 arithmetic. Do not use the rounded residual output for normalization.
    """
    normalized_dims = tuple(range(x1.ndim - gamma.ndim, x1.ndim))
    residual_fp32 = x1.float() + x2.float()
    residual_fp64 = residual_fp32.double()
    rstd_fp64 = torch.rsqrt(residual_fp64.square().mean(normalized_dims, keepdim=True) + epsilon)
    y = residual_fp64 * rstd_fp64 * gamma.double()
    if beta is not None:
        y = y + beta.double()
    return y.to(x1.dtype), rstd_fp64.float(), residual_fp32.to(x1.dtype)


def _output_tolerance(dtype):
    if dtype == torch.float16:
        return 2e-3, 2e-3
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    return 2e-5, 2e-6


def _assert_outputs(actual, expected, source, *, compare_y=True):
    y, rstd, residual = actual
    ref_y, ref_rstd, ref_residual = expected
    for name, value in zip(("y", "rstd", "residual"), actual):
        assert torch.isfinite(value).all(), f"{source}: {name} contains non-finite values"
    if compare_y:
        rtol, atol = _output_tolerance(ref_y.dtype)
        torch.testing.assert_close(y, ref_y, rtol=rtol, atol=atol, msg=f"{source}: y")
    torch.testing.assert_close(rstd, ref_rstd, rtol=RSTD_RTOL, atol=RSTD_ATOL, msg=f"{source}: rstd")
    # Residual is an elementwise FP32 sum followed by one output cast, with no
    # reduction-order ambiguity for these finite inputs.
    torch.testing.assert_close(residual, ref_residual, rtol=0, atol=0, msg=f"{source}: residual")


def _run_and_compare(x1, x2, gamma, beta):
    expected = _reference(x1, x2, gamma, beta)
    x1_npu, x2_npu, gamma_npu = (value.npu() for value in (x1, x2, gamma))
    beta_npu = None if beta is None else beta.npu()
    actual_npu = torch.ops._C_ascend.npu_add_rms_norm_bias(x1_npu, x2_npu, gamma_npu, beta_npu, EPSILON)
    actual = tuple(value.cpu() for value in actual_npu)
    _assert_outputs(actual, expected, "independent FP64 reference")

    # Record the installed operator's provenance and SDK version. Its rstd and
    # residual remain comparable for every beta. Its y is comparable only when
    # beta is absent: adding beta to an already rounded eager y is a different
    # low-precision contract from adding beta directly to the FP32 product.
    eager_y, eager_rstd, eager_residual = torch_npu.npu_add_rms_norm(x1_npu, x2_npu, gamma_npu, epsilon=EPSILON)
    eager = tuple(value.cpu() for value in (eager_y, eager_rstd, eager_residual))
    _assert_outputs(actual, eager, "installed npu_add_rms_norm", compare_y=beta is None)
    return actual


def _make_beta(gamma_shape, dtype, beta_kind):
    if beta_kind == "none":
        return None
    if beta_kind == "zero":
        return torch.zeros(gamma_shape, dtype=dtype)
    # Both signs and distinct columns detect omission and wrong broadcasting.
    return torch.linspace(-0.75, 0.5, math.prod(gamma_shape)).reshape(gamma_shape).to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("beta_kind", ("none", "zero", "signed"))
@pytest.mark.parametrize(
    ("shape", "gamma_shape"),
    (
        pytest.param((65, 1025), (1025,), id="small-d-odd-rows-column-tail"),
        pytest.param((3, 7, 33), (7, 33), id="multidimensional-normalization"),
        pytest.param((7, 8192), (8192,), id="beta-ub-budget"),
        pytest.param((65, 32768), (32768,), id="large-d-aligned"),
        pytest.param((3, 32771), (32771,), id="large-d-column-tail"),
    ),
)
@torch.inference_mode()
def test_a5_add_rms_norm_bias_all_outputs(shape, gamma_shape, dtype, beta_kind):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    x1 = torch.randn(shape, generator=generator).to(dtype)
    x2 = torch.randn(shape, generator=generator).to(dtype)
    gamma = torch.linspace(-1.25, 1.25, math.prod(gamma_shape)).reshape(gamma_shape).to(dtype)
    beta = _make_beta(gamma_shape, dtype, beta_kind)
    _run_and_compare(x1, x2, gamma, beta)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("beta_kind", ("none", "signed"))
@pytest.mark.parametrize(
    ("columns", "input_kind"),
    (
        pytest.param(1025, "cancellation", id="small-d-cancellation"),
        pytest.param(32771, "near-zero", id="large-d-near-zero"),
    ),
)
@torch.inference_mode()
def test_a5_add_rms_norm_bias_sensitive_inputs(columns, input_kind, dtype, beta_kind):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    shape = (3, columns)
    if input_kind == "cancellation":
        x1 = torch.randn(shape, generator=generator).to(dtype)
        perturbation = torch.randn(shape, generator=generator) * 0.01
        x2 = (-x1.float() + perturbation).to(dtype)
        # Include a completely cancelled row, where epsilon controls rstd.
        x2[0] = -x1[0]
    else:
        x1 = (torch.randn(shape, generator=generator) * 1e-4).to(dtype)
        x2 = (torch.randn(shape, generator=generator) * 1e-4).to(dtype)
    gamma = torch.linspace(-1.25, 1.25, columns).to(dtype)
    beta = _make_beta((columns,), dtype, beta_kind)
    _run_and_compare(x1, x2, gamma, beta)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("columns", (256, 32768))
@torch.inference_mode()
def test_a5_beta_is_added_in_fp32_before_output_cast(dtype, columns):
    # Alternating 1 and 2 gives an exactly representable mean square of 2.5.
    # Beta cancels the rounded normalization output, so the old staged cast/add
    # returns zero. Direct FP32 product + beta retains the discarded fraction:
    # about +1.31e-4/+2.62e-4 for FP16, -3.57e-4/-7.13e-4 for BF16.
    x1 = torch.tensor((1.0, 2.0), dtype=dtype).repeat(3, columns // 2)
    x2 = torch.zeros_like(x1)
    gamma = torch.ones(columns, dtype=dtype)
    unbiased_y, _, _ = _reference(x1, x2, gamma, None)
    beta = -unbiased_y[0]
    expected_y, _, _ = _reference(x1, x2, gamma, beta)
    staged_y = (unbiased_y.float() + beta.float()).to(dtype)
    assert torch.count_nonzero(staged_y) == 0

    y, _, _ = _run_and_compare(x1, x2, gamma, beta)
    # Allow small FP32-vs-FP64 arithmetic differences, while keeping the
    # absolute tolerance over 65 times smaller than the smallest expected y.
    fp32_add_atol = 2e-6
    assert torch.all(expected_y.abs() > fp32_add_atol)
    assert torch.all(y != 0)
    torch.testing.assert_close(y, expected_y, rtol=0, atol=fp32_add_atol)


@pytest.mark.parametrize("invalid_beta", ("shape", "dtype"))
@torch.inference_mode()
def test_a5_add_rms_norm_bias_rejects_invalid_beta(invalid_beta):
    columns = 256
    x1 = torch.ones((3, columns), dtype=torch.float16, device="npu")
    x2 = torch.zeros_like(x1)
    gamma = torch.ones(columns, dtype=torch.float16, device="npu")
    if invalid_beta == "shape":
        beta = torch.zeros(columns - 1, dtype=torch.float16, device="npu")
    else:
        beta = torch.zeros(columns, dtype=torch.float32, device="npu")
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.npu_add_rms_norm_bias(x1, x2, gamma, beta, EPSILON)


def _run_and_compare_in_row_chunks(shape, gamma_shape, dtype, beta_kind):
    """Run one full NPU shape; bound the independent CPU reference by rows.

    Never split a normalization row: each FP64 reference sees the entire D.
    Inputs, custom output and installed eager output are full-size NPU tensors.
    Only input generation and CPU comparisons use bounded row blocks.
    """
    columns = math.prod(gamma_shape)
    assert tuple(shape[-len(gamma_shape) :]) == tuple(gamma_shape)
    rows = math.prod(shape) // columns
    # The largest D may itself exceed this budget; retain at least one full row.
    rows_per_chunk = max(1, (1024 * 1024) // columns)
    row_view_shape = (rows, *gamma_shape)
    rstd_row_view_shape = (rows, *((1,) * len(gamma_shape)))

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    x1_npu = torch.empty(shape, dtype=dtype, device="npu")
    x2_npu = torch.empty_like(x1_npu)
    x1_rows = x1_npu.reshape(row_view_shape)
    x2_rows = x2_npu.reshape(row_view_shape)
    for first in range(0, rows, rows_per_chunk):
        last = min(first + rows_per_chunk, rows)
        chunk_shape = (last - first, *gamma_shape)
        x1_cpu = torch.randn(chunk_shape, generator=generator, dtype=torch.float32).to(dtype)
        x2_cpu = torch.randn(chunk_shape, generator=generator, dtype=torch.float32).to(dtype)
        x1_rows[first:last].copy_(x1_cpu)
        x2_rows[first:last].copy_(x2_cpu)
        del x1_cpu, x2_cpu

    gamma = torch.linspace(-1.25, 1.25, columns).reshape(gamma_shape).to(dtype)
    beta = _make_beta(gamma_shape, dtype, beta_kind)
    gamma_npu = gamma.npu()
    beta_npu = None if beta is None else beta.npu()
    actual = torch.ops._C_ascend.npu_add_rms_norm_bias(x1_npu, x2_npu, gamma_npu, beta_npu, EPSILON)
    torch.npu.synchronize()
    expected_rstd_shape = (*shape[: -len(gamma_shape)], *((1,) * len(gamma_shape)))
    assert tuple(actual[0].shape) == tuple(shape)
    assert tuple(actual[1].shape) == expected_rstd_shape
    assert tuple(actual[2].shape) == tuple(shape)
    actual_rows = (
        actual[0].reshape(row_view_shape),
        actual[1].reshape(rstd_row_view_shape),
        actual[2].reshape(row_view_shape),
    )

    # Replay the original CPU inputs rather than reading possibly modified NPU
    # inputs after the custom call. Same seed, chunk shapes and call order.
    generator.manual_seed(SEED)
    # First check the custom output independently, before allocating eager output.
    for first in range(0, rows, rows_per_chunk):
        last = min(first + rows_per_chunk, rows)
        chunk_shape = (last - first, *gamma_shape)
        x1_cpu = torch.randn(chunk_shape, generator=generator, dtype=torch.float32).to(dtype)
        x2_cpu = torch.randn(chunk_shape, generator=generator, dtype=torch.float32).to(dtype)
        expected = _reference(x1_cpu, x2_cpu, gamma, beta)
        actual_cpu = tuple(value[first:last].cpu() for value in actual_rows)
        _assert_outputs(actual_cpu, expected, f"large shape {shape}, rows [{first}:{last}), FP64 reference")
        del x1_cpu, x2_cpu, expected, actual_cpu

    # The installed operator is still run on the complete shape. Its y is only
    # comparable without beta; the existing helper keeps that contract intact.
    eager = torch_npu.npu_add_rms_norm(x1_npu, x2_npu, gamma_npu, epsilon=EPSILON)
    torch.npu.synchronize()
    assert tuple(eager[0].shape) == tuple(shape)
    assert tuple(eager[1].shape) == expected_rstd_shape
    assert tuple(eager[2].shape) == tuple(shape)
    eager_rows = (
        eager[0].reshape(row_view_shape),
        eager[1].reshape(rstd_row_view_shape),
        eager[2].reshape(row_view_shape),
    )
    for first in range(0, rows, rows_per_chunk):
        last = min(first + rows_per_chunk, rows)
        actual_cpu = tuple(value[first:last].cpu() for value in actual_rows)
        eager_cpu = tuple(value[first:last].cpu() for value in eager_rows)
        _assert_outputs(
            actual_cpu,
            eager_cpu,
            f"large shape {shape}, rows [{first}:{last}), installed npu_add_rms_norm",
            compare_y=beta is None,
        )
        del actual_cpu, eager_cpu


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("beta_kind", ("none", "signed"))
@pytest.mark.parametrize(
    ("shape", "gamma_shape"),
    (
        pytest.param((8192, 8192), (8192,), id="large-many-rows"),
        pytest.param((4096, 32768), (32768,), id="large-split-d"),
        pytest.param((512, 131072), (131072,), id="large-wide-d"),
        pytest.param((64, 1048576), (1048576,), id="large-million-d"),
        pytest.param((8193, 8191), (8191,), id="large-many-rows-tails"),
        pytest.param((4095, 32769), (32769,), id="large-split-d-tails"),
        # Shape/dtype scenarios borrowed from fixed upstream A5 Host UT; no
        # upstream tiling key is assumed for local SDK/UB/beta combinations.
        pytest.param((4096, 1, 2304), (2304,), id="upstream-many-row-full-shape"),
        pytest.param((1024, 1, 12288), (12288,), id="upstream-many-row-split-shape"),
        pytest.param((65, 1048575), (1048575,), id="large-million-d-tails"),
        pytest.param((257, 16385), (16385,), id="large-over-full-load-width"),
        pytest.param((17, 31, 16, 2048), (16, 2048), id="large-multidimensional-gamma"),
        pytest.param((3, 257, 7, 2049), (7, 2049), id="large-multidimensional-gamma-tails"),
    ),
)
@torch.inference_mode()
def test_a5_add_rms_norm_bias_large_shapes(shape, gamma_shape, dtype, beta_kind):
    _run_and_compare_in_row_chunks(shape, gamma_shape, dtype, beta_kind)
