# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU metadata checks against the real extension from an A5 build.

Run this file alone in a fresh process:
    python3 -m pytest -q tests/test_add_rms_norm_bias_meta.py

It lives outside tests/ut, whose conftest mocks _build_info as A2, and outside
nightly/ops, whose conftest initializes the NPU through Triton device queries.
Do not collect it together with UT files in the same pytest process.

Build the A5 extension and _build_info.py first. Other hardware builds skip
these checks; an A5 build without its extension or Meta registration fails.
Only Meta and CPU FakeTensors are used, with real-kernel fallback disabled.
No NPU tensors or kernels are created.
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from vllm_ascend import _build_info
from vllm_ascend.device.device_config import is_950
from vllm_ascend.utils import bootstrap_custom_op_env


@pytest.fixture(scope="module")
def add_rms_norm_bias():
    assert getattr(_build_info, "__file__", None), (
        "A real _build_info.py is required. Run this file alone in a fresh pytest process, without tests/ut."
    )
    if not is_950():
        pytest.skip("These CPU checks require an extension built for A5")
    bootstrap_custom_op_env()
    # This operator's Meta implementation is registered by the C++ extension.
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::npu_add_rms_norm_bias", "Meta")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("_C_ascend::npu_add_rms_norm_bias", "PrivateUse1")
    return torch.ops._C_ascend.npu_add_rms_norm_bias.default


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("has_beta", (False, True))
@pytest.mark.parametrize(
    ("shape", "gamma_shape", "rstd_shape"),
    (
        pytest.param((65, 1025), (1025,), (65, 1), id="last-dimension"),
        pytest.param((3, 7, 33), (7, 33), (3, 1, 1), id="multiple-normalized-dimensions"),
        pytest.param((7, 33), (7, 33), (1, 1), id="all-dimensions-normalized"),
    ),
)
def test_a5_add_rms_norm_bias_meta_outputs(add_rms_norm_bias, shape, gamma_shape, rstd_shape, dtype, has_beta):
    x1 = torch.empty(shape, dtype=dtype, device="meta")
    x2 = torch.empty_like(x1)
    gamma = torch.empty(gamma_shape, dtype=dtype, device="meta")
    if has_beta:
        beta = torch.empty_like(gamma)
        outputs = add_rms_norm_bias(x1, x2, gamma, beta, 1e-6)
    else:
        # Exercise the existing schema defaults for both beta and epsilon.
        outputs = add_rms_norm_bias(x1, x2, gamma)

    y, rstd, residual = outputs
    for value, expected_shape, expected_dtype in (
        (y, shape, dtype),
        (rstd, rstd_shape, torch.float32),
        (residual, shape, dtype),
    ):
        assert value.shape == expected_shape
        assert value.dtype == expected_dtype
        assert value.device.type == "meta"
        assert value.is_contiguous()
    assert y is not x1 and residual is not x1 and y is not residual


@pytest.mark.parametrize("has_beta", (False, True))
def test_a5_add_rms_norm_bias_fake_preserves_symbolic_shapes(add_rms_norm_bias, has_beta):
    mode = FakeTensorMode(shape_env=ShapeEnv(), allow_fallback_kernels=False)
    # All concrete dimensions exceed one so the leading dimension can remain
    # symbolic. The production C++ Meta must use sym_sizes/empty_symint.
    x1 = mode.from_tensor(torch.empty((3, 7, 33), dtype=torch.bfloat16), static_shapes=False)
    x2 = mode.from_tensor(torch.empty((3, 7, 33), dtype=torch.bfloat16), static_shapes=False)
    gamma = mode.from_tensor(torch.empty((7, 33), dtype=torch.bfloat16), static_shapes=False)
    beta = mode.from_tensor(torch.empty((7, 33), dtype=torch.bfloat16), static_shapes=False) if has_beta else None

    assert isinstance(x1.shape[0], torch.SymInt)
    with mode:
        y, rstd, residual = add_rms_norm_bias(x1, x2, gamma, beta, 1e-6)

    for value in (y, rstd, residual):
        assert isinstance(value, FakeTensor)
        assert value.fake_mode is mode
        assert value.device.type == "cpu"
        assert isinstance(value.shape[0], torch.SymInt)
    assert y.shape == x1.shape
    assert residual.shape == x1.shape
    assert rstd.shape == (x1.shape[0], 1, 1)
    assert y.dtype == residual.dtype == torch.bfloat16
    assert rstd.dtype == torch.float32


def test_a5_add_rms_norm_bias_fake_preserves_two_leading_dimensions(add_rms_norm_bias):
    mode = FakeTensorMode(shape_env=ShapeEnv(), allow_fallback_kernels=False)
    x1 = mode.from_tensor(torch.empty((3, 5, 7, 33), dtype=torch.float16), static_shapes=False)
    x2 = mode.from_tensor(torch.empty((3, 5, 7, 33), dtype=torch.float16), static_shapes=False)
    gamma = mode.from_tensor(torch.empty((7, 33), dtype=torch.float16), static_shapes=False)

    leading_dims = x1.shape[:2]
    assert all(isinstance(dim, torch.SymInt) for dim in leading_dims)
    # Different sizes must remain independent symbols, not one reused dimension.
    assert leading_dims[0].node.expr != leading_dims[1].node.expr
    with mode:
        y, rstd, residual = add_rms_norm_bias(x1, x2, gamma)

    for value in (y, rstd, residual):
        assert isinstance(value, FakeTensor)
        assert value.fake_mode is mode
        assert value.device.type == "cpu"
        for actual, expected in zip(value.shape[:2], leading_dims):
            assert isinstance(actual, torch.SymInt)
            assert actual.node.expr == expected.node.expr
    assert y.shape == residual.shape == x1.shape
    assert rstd.shape == (*leading_dims, 1, 1)
    assert y.dtype == residual.dtype == torch.float16
    assert rstd.dtype == torch.float32
