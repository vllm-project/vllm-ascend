import gc
import random

import numpy as np
import pytest
import torch

# Strict-assertion companion to test_add_rms_norm_bias.py.
#
# The base file's tolerance structure (rtol/atol=100 on one branch of each
# comparison) cannot catch row-mixing or large-magnitude numeric errors, so
# the optimization work on this operator is gated here instead:
#   - y: per-dtype assert_close tolerances (one output-ulp flip of headroom
#     for fp32 summation-order noise between kernel and golden),
#   - rstd: fp32 tolerances of 1e-3 - rows use independent random data, so
#     every row's rstd differs by several percent and any row-mixing bug
#     (e.g. a wrong compaction Gather index repeating row values) fails,
#   - x: bit-exact (a single fp32 add plus one rounding in both sides).
# Shape matrix additionally covers the compaction tail lengths of the
# pipelined NORMAL path (rows per core 8/52/64+1 via rows=313/2064/2600) and
# the UB cost-model boundary col=8192 (bf16 NORMAL falls back to SPLIT_D).

seed = 45
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def npu_add_rms_norm_bias_golden(input_x1, input_x2, input_gamma, input_beta, kernelType, epsilon=0.000001):
    # Mirrors the golden in test_add_rms_norm_bias.py (kept self-contained on
    # purpose so the two files cannot drift through import paths).
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


KERNEL_TYPE = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
}
# y tolerance = 3 output ulp: rstd differs from the golden by ~1e-6 relative
# (fp32 summation order), and both sides double-round y (round x_norm, then
# scale/affine, then round again), so each rounding can flip on that noise -
# observed up to 2 fp16 ulp and 2 bf16 ulp. bf16 ulp is 2^-7 relative, fp16
# 2^-10. Real defects (row-mixing rstd, wrong scale) shift y by far more.
Y_ATOL_RTOL = {
    torch.float16: (0.003, 0.003),
    torch.bfloat16: (0.024, 0.024),
    torch.float32: (0.000244140625, 0.000244140625),
}


def run_case(row, col, dtype, use_beta: bool):
    atol, rtol = Y_ATOL_RTOL[dtype]
    kernelType = KERNEL_TYPE[dtype]
    shape_x = [row, col]
    shape_gamma = [col]

    input_x1 = np.random.uniform(1, 10, size=tuple(shape_x)).astype(np.float32)
    input_x1_tensor = torch.tensor(input_x1).type(dtype)
    input_x2 = np.random.uniform(1, 10, size=tuple(shape_x)).astype(np.float32)
    input_x2_tensor = torch.tensor(input_x2).type(dtype)
    input_gamma = np.random.uniform(1, 10, size=tuple(shape_gamma)).astype(np.float32)
    input_gamma_tensor = torch.tensor(input_gamma).type(dtype)
    if use_beta:
        input_beta = np.random.uniform(1, 10, size=tuple(shape_gamma)).astype(np.float32)
        beta_arg = torch.tensor(input_beta).type(dtype).npu()
    else:
        input_beta = np.zeros(tuple(shape_gamma)).astype(np.float32)
        beta_arg = None
    input_beta_tensor = torch.tensor(input_beta).type(dtype)

    y, rstd, x = torch.ops._C_ascend.npu_add_rms_norm_bias(
        input_x1_tensor.npu(), input_x2_tensor.npu(), input_gamma_tensor.npu(), beta_arg, 1e-6
    )
    y = y.cpu()
    rstd = rstd.cpu()
    x = x.cpu()

    y1, rstd1, x1 = npu_add_rms_norm_bias_golden(
        input_x1_tensor, input_x2_tensor, input_gamma_tensor, input_beta_tensor, kernelType, epsilon=0.000001
    )

    torch.testing.assert_close(y, y1, atol=atol, rtol=rtol)
    torch.testing.assert_close(rstd, rstd1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x, x1, rtol=0, atol=0)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    "row",
    [1, 16, 64, 255, 313, 1000, 2064, 2600],
)
@pytest.mark.parametrize(
    "col",
    [8, 128, 3000, 7168, 8192, 15000],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_strict_with_beta(row: int, col: int, dtype):
    run_case(row, col, dtype, use_beta=True)


@pytest.mark.parametrize(
    "row, col, dtype",
    [
        (1000, 7168, torch.bfloat16),
        (1000, 10240, torch.bfloat16),
        (1000, 7168, torch.float16),
        (16, 128, torch.bfloat16),
    ],
)
def test_strict_without_beta(row: int, col: int, dtype):
    # beta=None dispatches the nullptrBeta kernel paths and the no-beta
    # branch of the tiling UB cost model; the golden adds a zero beta.
    run_case(row, col, dtype, use_beta=False)


def test_epsilon_must_be_finite():
    # NaN/Inf epsilon must be rejected by tiling (legacy path alignment with
    # the A5 finite-and-nonnegative constraint).
    x1 = torch.ones(4, 128, dtype=torch.bfloat16, device="npu")
    x2 = torch.ones_like(x1)
    gamma = torch.ones(128, dtype=torch.bfloat16, device="npu")
    beta = torch.zeros_like(gamma)
    with pytest.raises(Exception):
        torch.ops._C_ascend.npu_add_rms_norm_bias(x1, x2, gamma, beta, float("nan"))
    with pytest.raises(Exception):
        torch.ops._C_ascend.npu_add_rms_norm_bias(x1, x2, gamma, beta, float("inf"))


def test_beta_shape_must_match_gamma():
    # A short beta would make the kernel read out of bounds; tiling must
    # reject it (legacy path alignment with the A5 tiling contract).
    x1 = torch.ones(4, 128, dtype=torch.bfloat16, device="npu")
    x2 = torch.ones_like(x1)
    gamma = torch.ones(128, dtype=torch.bfloat16, device="npu")
    short_beta = torch.ones(64, dtype=torch.bfloat16, device="npu")
    with pytest.raises(Exception):
        torch.ops._C_ascend.npu_add_rms_norm_bias(x1, x2, gamma, short_beta, 1e-6)


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
)
def test_extreme_scale_overflow_domain(dtype):
    # x ~ 3e17 with numCol=7168: per-element (x^2 * 1/N) stays finite in fp32
    # while the raw sum of squares (9e34 * 7168) would overflow. The golden
    # scales per element before summing, and the kernel must match - this
    # pins the pre-reduce scaling contract that the avgFactor fold broke
    # (fold reverted for fp32/bf16 in d7132b0d5).
    row, col = 64, 7168
    x = (torch.full((row, col), 3.0e17) * torch.linspace(0.9, 1.1, col)).to(dtype)
    zeros = torch.zeros_like(x)
    gamma = torch.ones(col, dtype=dtype)
    beta = torch.zeros(col, dtype=dtype)
    y, rstd, x_out = torch.ops._C_ascend.npu_add_rms_norm_bias(
        x.npu(), zeros.npu(), gamma.npu(), beta.npu(), 1e-6
    )
    y = y.cpu()
    rstd = rstd.cpu()
    x_out = x_out.cpu()
    y1, rstd1, x1 = npu_add_rms_norm_bias_golden(
        x, zeros, gamma, beta, KERNEL_TYPE[dtype], epsilon=0.000001
    )
    # the whole point: rstd must be finite (the raw-sum fold would give Inf -> 0)
    assert torch.isfinite(rstd).all(), "rstd must stay finite at extreme input scale"
    torch.testing.assert_close(rstd, rstd1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x_out, x1, rtol=0, atol=0)
    atol, rtol = Y_ATOL_RTOL[dtype]
    torch.testing.assert_close(y, y1, atol=atol, rtol=rtol)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    "row, col, dtype",
    [
        (2600, 7168, torch.bfloat16),
        (128, 3000, torch.bfloat16),
    ],
)
def test_row_contrast_rstd_separation(row: int, col: int, dtype):
    # Rows with deliberately very different RMS (amplitude cycling 1 / 100 /
    # 0.01 -> rstd differing by ~100x between adjacent rows): any row-mixing
    # in the rstd path fails the 1e-3 check by orders of magnitude, which
    # uniform random rows cannot guarantee.
    shape_x = [row, col]
    amp = torch.tensor([1.0, 100.0, 0.01]).repeat(row // 3 + 1)[:row].unsqueeze(1)
    base = torch.empty(shape_x).uniform_(1.0, 2.0)
    x1 = (base * amp).to(dtype)
    x2 = torch.zeros_like(x1)
    gamma = torch.ones(col, dtype=dtype)
    beta = torch.zeros(col, dtype=dtype)
    y, rstd, x_out = torch.ops._C_ascend.npu_add_rms_norm_bias(
        x1.npu(), x2.npu(), gamma.npu(), beta.npu(), 1e-6
    )
    y = y.cpu()
    rstd = rstd.cpu()
    x_out = x_out.cpu()
    y1, rstd1, x1g = npu_add_rms_norm_bias_golden(
        x1, x2, gamma, beta, KERNEL_TYPE[dtype], epsilon=0.000001
    )
    torch.testing.assert_close(rstd, rstd1, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x_out, x1g, rtol=0, atol=0)
    atol, rtol = Y_ATOL_RTOL[dtype]
    torch.testing.assert_close(y, y1, atol=atol, rtol=rtol)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
