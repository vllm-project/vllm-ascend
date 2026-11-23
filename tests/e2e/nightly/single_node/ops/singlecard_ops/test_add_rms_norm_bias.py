# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import gc
import math
import random

import numpy as np
import pytest
import torch

from vllm_ascend.device.device_config import check_ascend_device_type, get_ascend_device_type, is_950
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import bootstrap_custom_op_env

EPSILON = 1e-6

seed = 45
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def npu_add_rms_norm_bias_golden(input_x1, input_x2, input_gamma, input_beta, kernelType, epsilon=0.000001):
    ori_x_shape = input_x1.shape
    ori_gamma_shape = input_gamma.shape
    xlength = len(ori_x_shape)
    gammaLength = len(ori_gamma_shape)
    torchType32 = torch.float32
    rstdShape = []
    rstdSize = 1
    for i in range(xlength):
        if i < (xlength - gammaLength):
            rstdShape.append(ori_x_shape[i])
            rstdSize = rstdSize * ori_x_shape[i]
        else:
            rstdShape.append(1)

    n = xlength - gammaLength
    gammaSize = np.multiply.reduce(np.array(ori_gamma_shape))
    input_gamma = input_gamma.reshape(gammaSize)
    input_beta = input_beta.reshape(gammaSize)
    x1_shape = ori_x_shape[0:n] + input_gamma.shape
    input_x1 = input_x1.reshape(x1_shape)
    input_x2 = input_x2.reshape(x1_shape)

    if kernelType == 1:
        oriType = torch.float16
        xOut = input_x1.to(oriType) + input_x2.to(oriType)
    elif kernelType == 2:
        oriType = torch.bfloat16
        x_fp32 = input_x1.to(torchType32) + input_x2.to(torchType32)
        xOut = x_fp32.to(oriType)
    else:
        oriType = torch.float32
        xOut = input_x1.to(torchType32) + input_x2.to(torchType32)
    x_fp32 = xOut.to(torchType32)
    avgFactor = 1 / gammaSize
    x_2 = torch.pow(x_fp32, 2)
    x_2_mean = x_2 * avgFactor
    tmp_sum = torch.sum(x_2_mean, axis=-1, keepdims=True)
    tmp_add_eps = tmp_sum + epsilon
    std = torch.sqrt(tmp_add_eps)
    rstd = 1 / std
    result_mid = x_fp32 * rstd
    if kernelType == 1:
        result_mid_ori = result_mid.to(oriType)
        y_array = result_mid_ori * input_gamma.to(oriType)
        y_array = y_array + input_beta.to(oriType)
    elif kernelType == 2:
        result_mid_ori = result_mid.to(oriType)
        y_array = result_mid_ori.to(torchType32) * input_gamma.to(torchType32)
        y_array = y_array + input_beta.to(torchType32)
    else:
        y_array = result_mid.to(torchType32) * input_gamma.to(torchType32)
        y_array = y_array + input_beta.to(torchType32)
    rstdOut = rstd.reshape(rstdShape).to(torchType32)
    yOut = y_array.reshape(ori_x_shape).to(oriType)
    xOut = x_fp32.reshape(ori_x_shape).to(oriType)
    return yOut, rstdOut, xOut


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Legacy AddRmsNormBias reference targets A2/A3; A5 has a separate reference",
)
@pytest.mark.parametrize(
    "row, col",
    [(1, 8), (1000, 15000), (255, 8), (1000, 128), (77, 3000), (128, 7168)],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol, kernelType",
    [
        (torch.float16, 0.0010986328125, 0.0010986328125, 1),
        (torch.bfloat16, 0.0079345703125, 0.0079345703125, 2),
        (torch.float32, 0.000244140625, 0.000244140625, 3),
    ],
)
def test_quant_fpx_linear(row: int, col: int, dtype, atol, rtol, kernelType):
    shape_x = [row, col]
    shape_gamma = [col]

    dataType = dtype

    input_x1 = np.random.uniform(1, 10, size=tuple(shape_x)).astype(np.float32)
    input_x1_tensor = torch.tensor(input_x1).type(dataType)

    input_x2 = np.random.uniform(1, 10, size=tuple(shape_x)).astype(np.float32)
    input_x2_tensor = torch.tensor(input_x2).type(dataType)

    input_gamma = np.random.uniform(1, 10, size=tuple(shape_gamma)).astype(np.float32)
    input_gamma_tensor = torch.tensor(input_gamma).type(dataType)

    input_beta = np.random.uniform(1, 10, size=tuple(shape_gamma)).astype(np.float32)
    grad_bias = torch.tensor(input_beta).type(dataType)
    y, rstd, x = torch.ops._C_ascend.npu_add_rms_norm_bias(
        input_x1_tensor.npu(), input_x2_tensor.npu(), input_gamma_tensor.npu(), grad_bias.npu(), 1e-6
    )

    y = y.cpu()
    rstd = rstd.cpu()
    x = x.cpu()

    y1, rstd1, x1 = npu_add_rms_norm_bias_golden(
        input_x1_tensor, input_x2_tensor, input_gamma_tensor, grad_bias, kernelType, epsilon=0.000001
    )

    a = y1 > 1
    a1 = y1 <= 1
    b = rstd1 > 1
    b1 = rstd1 <= 1
    c = x1 > 1
    c1 = x1 <= 1
    torch.testing.assert_close(y * a, y1 * a, atol=atol, rtol=100)
    torch.testing.assert_close(y * a1, y1 * a1, rtol=rtol, atol=100)
    torch.testing.assert_close(rstd * b, rstd1 * b, atol=atol, rtol=100)
    torch.testing.assert_close(rstd * b1, rstd1 * b1, rtol=rtol, atol=100)
    torch.testing.assert_close(x * c, x1 * c, atol=atol, rtol=100)
    torch.testing.assert_close(x * c1, x1 * c1, rtol=rtol, atol=100)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.fixture(scope="module")
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


def npu_add_rms_norm_bias_golden_a5(x1, x2, gamma, beta, epsilon=EPSILON):
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


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="AddRmsNormBias A5 reference targets Ascend A5",
)
@pytest.mark.usefixtures("require_a5_custom_operator")
@pytest.mark.e2e_coverage(
    arch="",
    feature="",
    parallel="",
    deploy="",
    hardware="A5",
    quantization="",
    graph_mode="eager",
)
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

    y_ref, rstd_ref, residual_ref = npu_add_rms_norm_bias_golden_a5(x1, x2, gamma, beta)
    y, rstd, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
        x1.npu(), x2.npu(), gamma.npu(), beta.npu() if has_beta else None, EPSILON
    )

    torch.testing.assert_close(y.cpu(), y_ref, atol=atol, rtol=rtol)
    # Statistics are always FP32; the residual has no reduction-order error.
    torch.testing.assert_close(rstd.cpu(), rstd_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(residual.cpu(), residual_ref, atol=0, rtol=0)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="AddRmsNormBias A5 reference targets Ascend A5",
)
@pytest.mark.usefixtures("require_a5_custom_operator")
@pytest.mark.e2e_coverage(
    arch="",
    feature="",
    parallel="",
    deploy="",
    hardware="A5",
    quantization="",
    graph_mode="eager",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("columns", [256, 32768])
@torch.inference_mode()
def test_a5_beta_is_added_in_fp32_before_output_cast(dtype, columns):
    x1 = torch.tensor((1.0, 2.0), dtype=dtype).repeat(3, columns // 2)
    x2 = torch.zeros_like(x1)
    gamma = torch.ones(columns, dtype=dtype)
    unbiased_y, _, _ = npu_add_rms_norm_bias_golden_a5(x1, x2, gamma, None)
    beta = -unbiased_y[0]
    y_ref, _, _ = npu_add_rms_norm_bias_golden_a5(x1, x2, gamma, beta)
    # Adding beta after rounding would incorrectly produce zero in this case.
    assert torch.count_nonzero((unbiased_y.float() + beta.float()).to(dtype)) == 0
    assert torch.all(y_ref.abs() > 2e-6)

    y, _, _ = torch.ops._C_ascend.npu_add_rms_norm_bias(x1.npu(), x2.npu(), gamma.npu(), beta.npu(), EPSILON)
    torch.testing.assert_close(y.cpu(), y_ref, atol=2e-6, rtol=0)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="AddRmsNormBias A5 reference targets Ascend A5",
)
@pytest.mark.usefixtures("require_a5_custom_operator")
@pytest.mark.e2e_coverage(
    arch="",
    feature="",
    parallel="",
    deploy="",
    hardware="A5",
    quantization="",
    graph_mode="eager",
)
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
