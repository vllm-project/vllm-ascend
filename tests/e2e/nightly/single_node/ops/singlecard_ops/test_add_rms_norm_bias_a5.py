# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import math

import pytest
import torch

from vllm_ascend.device.device_config import check_ascend_device_type, is_950
from vllm_ascend.utils import bootstrap_custom_op_env

EPSILON = 1e-6

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
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

    assert hasattr(torch.ops._C_ascend, "npu_add_rms_norm_bias")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::npu_add_rms_norm_bias", "PrivateUse1")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::npu_add_rms_norm_bias", "Meta")


def npu_add_rms_norm_bias_golden(x1, x2, gamma, beta, epsilon=EPSILON):
    # Normalize the FP32 sum before rounding the residual output. Use FP64 for
    # an independent reference, and add beta before the final output cast.
    normalized_dims = tuple(range(x1.ndim - gamma.ndim, x1.ndim))
    residual_fp32 = x1.float() + x2.float()
    residual_fp64 = residual_fp32.double()
    rstd = torch.rsqrt(residual_fp64.square().mean(normalized_dims, keepdim=True) + epsilon)
    y = residual_fp64 * rstd * gamma.double()
    if beta is not None:
        y = y + beta.double()
    return y.to(x1.dtype), rstd.float(), residual_fp32.to(x1.dtype)


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 1e-2, 1e-2),
        (torch.float32, 2e-6, 2e-5),
    ],
)
@pytest.mark.parametrize("has_beta", [False, True])
@pytest.mark.parametrize(
    "shape, gamma_shape",
    [
        ((1, 6144), (6144,)),
        ((65, 1025), (1025,)),
        ((3, 7, 33), (7, 33)),
        ((7, 8192), (8192,)),
        ((65, 32768), (32768,)),
        ((3, 32771), (32771,)),
        ((1024, 6144), (6144,)),
    ],
)
@torch.inference_mode()
def test_add_rms_norm_bias_a5(shape, gamma_shape, has_beta, dtype, atol, rtol):
    generator = torch.Generator(device="cpu").manual_seed(45)
    x1 = torch.randn(shape, generator=generator).to(dtype)
    x2 = torch.randn(shape, generator=generator).to(dtype)
    gamma = torch.linspace(-1.25, 1.25, math.prod(gamma_shape)).reshape(gamma_shape).to(dtype)
    beta = torch.linspace(-0.75, 0.5, math.prod(gamma_shape)).reshape(gamma_shape).to(dtype) if has_beta else None

    y_ref, rstd_ref, residual_ref = npu_add_rms_norm_bias_golden(x1, x2, gamma, beta)
    y, rstd, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
        x1.npu(), x2.npu(), gamma.npu(), beta.npu() if has_beta else None, EPSILON
    )

    torch.testing.assert_close(y.cpu(), y_ref, atol=atol, rtol=rtol)
    # Statistics are always FP32; the residual has no reduction-order error.
    torch.testing.assert_close(rstd.cpu(), rstd_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(residual.cpu(), residual_ref, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("columns", [256, 32768])
@torch.inference_mode()
def test_a5_beta_is_added_in_fp32_before_output_cast(dtype, columns):
    x1 = torch.tensor((1.0, 2.0), dtype=dtype).repeat(3, columns // 2)
    x2 = torch.zeros_like(x1)
    gamma = torch.ones(columns, dtype=dtype)
    unbiased_y, _, _ = npu_add_rms_norm_bias_golden(x1, x2, gamma, None)
    beta = -unbiased_y[0]
    y_ref, _, _ = npu_add_rms_norm_bias_golden(x1, x2, gamma, beta)
    # Adding beta after rounding would incorrectly produce zero in this case.
    assert torch.count_nonzero((unbiased_y.float() + beta.float()).to(dtype)) == 0
    assert torch.all(y_ref.abs() > 2e-6)

    y, _, _ = torch.ops._C_ascend.npu_add_rms_norm_bias(x1.npu(), x2.npu(), gamma.npu(), beta.npu(), EPSILON)
    torch.testing.assert_close(y.cpu(), y_ref, atol=2e-6, rtol=0)


@pytest.mark.parametrize("invalid_beta", ["shape", "dtype"])
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
