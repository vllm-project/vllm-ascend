# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_npu


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    # Independent FP32 reductions can land on adjacent BF16 values.
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    return tolerance, tolerance


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 16, 128])
def test_rms_norm_cast(dtype: torch.dtype, num_tokens: int):
    torch.manual_seed(7)
    hidden_size = 7168
    epsilon = 1e-6
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="npu")
    gamma = torch.randn(hidden_size, dtype=dtype, device="npu")

    expected, _ = torch_npu.npu_rms_norm(x, gamma, epsilon)
    actual, actual_fp32 = torch.ops._C_ascend.npu_rms_norm_cast(x, gamma, epsilon)

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    # Routing must consume the widened, already-rounded RMSNorm result.
    torch.testing.assert_close(actual_fp32, actual.float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_cast_npu_graph(dtype: torch.dtype):
    torch.manual_seed(11)
    x = torch.randn(16, 7168, dtype=dtype, device="npu")
    gamma = torch.randn(7168, dtype=dtype, device="npu")
    expected, _ = torch_npu.npu_rms_norm(x, gamma, 1e-6)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        capture_error_mode="thread_local",
        auto_dispatch_capture=True,
    ):
        actual, actual_fp32 = torch.ops._C_ascend.npu_rms_norm_cast(
            x,
            gamma,
            1e-6,
        )
    graph.replay()

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_fp32, actual.float(), rtol=0, atol=0)
