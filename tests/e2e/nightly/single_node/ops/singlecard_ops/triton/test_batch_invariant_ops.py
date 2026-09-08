# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.batch_invariant.matmul import (
    addmm_batch_invariant,
    bmm_batch_invariant,
    linear_batch_invariant,
    linear_persistent,
    matmul_batch_invariant,
    matmul_persistent,
    mm_batch_invariant,
)
from vllm_ascend.ops.triton.batch_invariant.mean import mean_batch_invariant, mean_dim
from vllm_ascend.ops.triton.batch_invariant.rmsnorm import rms_norm
from vllm_ascend.ops.triton.batch_invariant.softmax import softmax_batch_invariant
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

SEED = 42
DEVICE = "npu"
TOLERANCES = {
    torch.float16: (2e-3, 2e-2),
    torch.bfloat16: (2e-2, 5e-2),
    torch.float32: (1e-4, 1e-4),
}


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties():
    init_device_properties_triton()


def assert_kernel_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare in FP32 so the configured tolerances have consistent meaning."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    rtol, atol = TOLERANCES[actual.dtype]
    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("m", "k", "n", "with_bias"),
    [
        pytest.param(1, 64, 128, False, id="single-row-no-bias"),
        pytest.param(17, 65, 129, True, id="non-aligned-with-bias"),
        pytest.param(129, 257, 131, True, id="multi-block-with-bias"),
    ],
)
def test_matmul_bias_persistent_kernel_precision(dtype, m, k, n, with_bias):
    """Guard x @ y and the optional fused bias against an FP32 golden."""
    torch.manual_seed(SEED)
    x_cpu = torch.randn((m, k), dtype=torch.float32) * 0.2
    y_cpu = torch.randn((k, n), dtype=torch.float32) * 0.2
    bias_cpu = torch.randn((n,), dtype=torch.float32) * 0.1 if with_bias else None

    x = x_cpu.to(dtype=dtype, device=DEVICE)
    y = y_cpu.to(dtype=dtype, device=DEVICE)
    bias = bias_cpu.to(dtype=dtype, device=DEVICE) if bias_cpu is not None else None

    actual = matmul_persistent(x, y, bias)
    expected = x.cpu().float() @ y.cpu().float()
    if bias is not None:
        expected += bias.cpu().float()
    expected = expected.to(dtype)

    assert_kernel_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        pytest.param(7, 63, 129, id="small-non-aligned"),
        pytest.param(257, 257, 131, id="medium-m-greater-than-n"),
        pytest.param(1025, 65, 257, id="large-m-multi-block"),
    ],
)
def test_linear_persistent_kernel_precision(dtype, m, k, n):
    """Guard F.linear's x @ weight.T calculation against an FP32 golden."""
    torch.manual_seed(SEED)
    x = (torch.randn((m, k), dtype=torch.float32) * 0.2).to(dtype=dtype, device=DEVICE)
    weight = (torch.randn((n, k), dtype=torch.float32) * 0.2).to(dtype=dtype, device=DEVICE)

    actual = linear_persistent(x, weight)
    expected = (x.cpu().float() @ weight.cpu().float().T).to(dtype)

    assert_kernel_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("shape", "dim", "keepdim"),
    [
        pytest.param((3, 1031, 17), 1, False, id="middle-dim-multi-block"),
        pytest.param((5, 7, 2051), -1, True, id="negative-last-dim-keepdim"),
        pytest.param((2051, 3), 0, True, id="first-dim-keepdim"),
    ],
)
def test_mean_kernel_precision(dtype, shape, dim, keepdim):
    """Guard reduction tails, negative dimensions, and keepdim behavior."""
    torch.manual_seed(SEED)
    input_ = (torch.randn(shape, dtype=torch.float32) * 0.5).to(dtype=dtype, device=DEVICE)

    actual = mean_dim(input_, dim=dim, keepdim=keepdim, dtype=dtype)
    expected = input_.cpu().float().mean(dim=dim, keepdim=keepdim).to(dtype)

    assert_kernel_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("shape", "eps"),
    [
        pytest.param((1, 7, 1025), 1e-6, id="non-aligned-hidden-size"),
        pytest.param((2, 65, 4096), 1e-5, id="multi-program-multi-block"),
    ],
)
def test_rms_norm_kernel_precision(dtype, shape, eps):
    """Guard row scheduling and hidden-size reduction against an FP32 golden."""
    torch.manual_seed(SEED)
    input_ = (torch.randn(shape, dtype=torch.float32) * 0.5).to(dtype=dtype, device=DEVICE)
    weight = (torch.randn((shape[-1],), dtype=torch.float32) * 0.2 + 1.0).to(
        dtype=dtype,
        device=DEVICE,
    )

    actual = rms_norm(input_, weight, eps=eps)
    input_fp32 = input_.cpu().float()
    expected = input_fp32 * torch.rsqrt(input_fp32.square().mean(dim=-1, keepdim=True) + eps)
    expected = (expected * weight.cpu().float()).to(dtype)

    assert_kernel_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        pytest.param((7, 127), -1, id="non-aligned-last-dim"),
        pytest.param((3, 1025, 5), 1, id="middle-dim"),
        pytest.param((2, 32001), -1, id="model-vocab-size"),
    ],
)
def test_softmax_batch_invariant_precision(dtype, shape, dim):
    """Guard common dimensions and long reduction tails against an FP32 golden."""
    torch.manual_seed(SEED)
    input_ = (torch.randn(shape, dtype=torch.float32) * 3.0).to(dtype=dtype, device=DEVICE)

    actual = softmax_batch_invariant(input_, dim=dim)
    expected = torch.softmax(input_.cpu().float(), dim=dim).to(dtype)

    assert_kernel_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_softmax_batch_invariant_numerical_stability(dtype):
    """Large logits must remain finite after max subtraction."""
    input_ = torch.tensor(
        [[-10000.0, -1000.0, 0.0, 1000.0, 10000.0]],
        dtype=dtype,
        device=DEVICE,
    )

    actual = softmax_batch_invariant(input_, dim=-1)
    expected = torch.softmax(input_.cpu().float(), dim=-1).to(dtype)

    assert torch.isfinite(actual).all()
    assert_kernel_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_wrappers_2d_precision(dtype):
    """Guard mm, addmm, and linear wrappers, including their bias paths."""
    torch.manual_seed(SEED)
    x = (torch.randn((17, 65), dtype=torch.float32) * 0.2).to(dtype=dtype, device=DEVICE)
    weight = (torch.randn((131, 65), dtype=torch.float32) * 0.2).to(dtype=dtype, device=DEVICE)
    bias = (torch.randn((131,), dtype=torch.float32) * 0.1).to(dtype=dtype, device=DEVICE)

    mm_actual = mm_batch_invariant(x, weight.T)
    addmm_actual = addmm_batch_invariant(bias, x, weight.T)
    linear_actual = linear_batch_invariant(x, weight, bias)
    mm_expected = (x.cpu().float() @ weight.cpu().float().T).to(dtype)
    biased_expected = (mm_expected.float() + bias.cpu().float()).to(dtype)

    assert_kernel_close(mm_actual, mm_expected)
    assert_kernel_close(addmm_actual, biased_expected)
    assert_kernel_close(linear_actual, biased_expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [
        pytest.param((3, 7, 65), (3, 65, 17), id="3d-by-3d"),
        pytest.param((3, 7, 65), (65, 17), id="3d-by-2d"),
        pytest.param((7, 65), (3, 65, 17), id="2d-by-3d"),
        pytest.param((2, 3, 7, 65), (2, 3, 65, 17), id="4d-by-4d"),
    ],
)
def test_matmul_batch_invariant_supported_shapes(dtype, a_shape, b_shape):
    """Guard every dimensionality combination currently declared by the wrapper."""
    torch.manual_seed(SEED)
    a = (torch.randn(a_shape, dtype=torch.float32) * 0.2).to(dtype=dtype, device=DEVICE)
    b = (torch.randn(b_shape, dtype=torch.float32) * 0.2).to(dtype=dtype, device=DEVICE)

    actual = matmul_batch_invariant(a, b)
    expected = torch.matmul(a.cpu().float(), b.cpu().float()).to(dtype)

    assert_kernel_close(actual, expected)


@pytest.mark.parametrize("use_matmul", [False, True], ids=["bmm", "matmul"])
def test_batched_matmul_out_parameter(use_matmul):
    """The out overload must write into and return the supplied tensor."""
    torch.manual_seed(SEED)
    a = torch.randn((3, 7, 65), dtype=torch.float16, device=DEVICE) * 0.2
    b = torch.randn((3, 65, 17), dtype=torch.float16, device=DEVICE) * 0.2
    out = torch.full((3, 7, 17), torch.nan, dtype=torch.float16, device=DEVICE)

    if use_matmul:
        result = matmul_batch_invariant(a, b, out=out)
    else:
        result = bmm_batch_invariant(a, b, out=out)
    expected = torch.bmm(a.cpu().float(), b.cpu().float()).to(torch.float16)

    assert result is out
    assert_kernel_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("keepdim", [False, True])
def test_mean_batch_invariant_multiple_dims(dtype, keepdim):
    """Guard the wrapper's multi-dimension reduction path."""
    torch.manual_seed(SEED)
    input_ = (torch.randn((3, 17, 65), dtype=torch.float32) * 0.5).to(dtype=dtype, device=DEVICE)

    actual = mean_batch_invariant(input_, dim=[1, 2], keepdim=keepdim, dtype=None)
    expected = input_.cpu().float().mean(dim=(1, 2), keepdim=keepdim).to(dtype)

    assert_kernel_close(actual, expected)


def test_softmax_batch_invariant_honors_float32_dtype():
    """A requested FP32 output must not be silently cast back to BF16."""
    torch.manual_seed(SEED)
    input_ = torch.randn((7, 1025), dtype=torch.bfloat16, device=DEVICE)

    actual = softmax_batch_invariant(input_, dim=-1, dtype=torch.float32)
    expected = torch.softmax(input_.cpu().float(), dim=-1)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=TOLERANCES[torch.float32][0],
        atol=TOLERANCES[torch.float32][1],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((65,), id="1d"),
        pytest.param((3, 7, 65), id="3d"),
        pytest.param((2, 3, 7, 65), id="4d"),
    ],
)
def test_linear_batch_invariant_nd_precision(dtype, input_shape):
    """Linear must preserve arbitrary leading dimensions and apply bias."""
    torch.manual_seed(SEED)
    input_ = (torch.randn(input_shape, dtype=torch.float32) * 0.2).to(
        dtype=dtype,
        device=DEVICE,
    )
    weight = (torch.randn((131, 65), dtype=torch.float32) * 0.2).to(
        dtype=dtype,
        device=DEVICE,
    )
    bias = (torch.randn((131,), dtype=torch.float32) * 0.1).to(
        dtype=dtype,
        device=DEVICE,
    )

    actual = linear_batch_invariant(input_, weight, bias)
    expected = torch.matmul(input_.cpu().float(), weight.cpu().float().T)
    expected = (expected + bias.cpu().float()).to(dtype)

    assert_kernel_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        pytest.param(0.5, 2.0, id="scaled-matmul-and-input"),
        pytest.param(0.0, 1.0, id="input-only"),
        pytest.param(1.0, 0.0, id="matmul-only"),
    ],
)
def test_addmm_batch_invariant_alpha_beta_precision(dtype, alpha, beta):
    """Non-default alpha and beta must follow torch.addmm semantics."""
    torch.manual_seed(SEED)
    mat1 = (torch.randn((17, 65), dtype=torch.float32) * 0.2).to(
        dtype=dtype,
        device=DEVICE,
    )
    mat2 = (torch.randn((65, 131), dtype=torch.float32) * 0.2).to(
        dtype=dtype,
        device=DEVICE,
    )
    input_ = (torch.randn((131,), dtype=torch.float32) * 0.1).to(
        dtype=dtype,
        device=DEVICE,
    )

    actual = addmm_batch_invariant(input_, mat1, mat2, alpha=alpha, beta=beta)
    expected = alpha * (mat1.cpu().float() @ mat2.cpu().float())
    expected = (expected + beta * input_.cpu().float()).to(dtype)

    assert_kernel_close(actual, expected)
