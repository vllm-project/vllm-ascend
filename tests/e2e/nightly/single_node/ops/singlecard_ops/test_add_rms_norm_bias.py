import gc
import random

import numpy as np
import pytest
import torch

from vllm_ascend.utils import AscendDeviceType, bootstrap_custom_op_env, get_ascend_device_type

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
    get_ascend_device_type() == AscendDeviceType.A5,
    reason="A5 has a separate aligned-width kernel and FP32 affine reference below.",
)
@pytest.mark.parametrize(
    "row",
    [1, 16, 64, 77, 128, 255, 1000],
)
@pytest.mark.parametrize(
    "col",
    [
        8,
        16,
        128,
        3000,
        7168,
        15000,
    ],
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


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires Ascend A5")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [16, 128, 1024, 6144])
@pytest.mark.parametrize("with_bias", [False, True])
def test_a5_add_rms_norm_bias_outputs(dtype, width, with_bias):
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

    generator = torch.Generator().manual_seed(45)
    x = torch.randn(3, width, dtype=dtype, generator=generator)
    residual = torch.randn(3, width, dtype=dtype, generator=generator)
    weight = torch.randn(width, dtype=dtype, generator=generator)
    bias = torch.randn(width, dtype=dtype, generator=generator) if with_bias else None
    y, rstd, summed = torch.ops._C_ascend.npu_add_rms_norm_bias(
        x.npu(), residual.npu(), weight.npu(), bias.npu() if bias is not None else None, 1e-6
    )
    rounded_sum = (x.float() + residual.float()).to(dtype).float()
    expected_rstd = torch.rsqrt(rounded_sum.square().mean(-1, keepdim=True) + 1e-6)
    expected = rounded_sum * expected_rstd * weight.float()
    if bias is not None:
        expected += bias.float()
    tolerance = {torch.bfloat16: 0.008, torch.float16: 0.001, torch.float32: 2e-5}[dtype]
    torch.testing.assert_close(summed.cpu(), rounded_sum.to(dtype), atol=0, rtol=0)
    torch.testing.assert_close(rstd.cpu(), expected_rstd, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(y.cpu(), expected.to(dtype), atol=tolerance, rtol=tolerance)
