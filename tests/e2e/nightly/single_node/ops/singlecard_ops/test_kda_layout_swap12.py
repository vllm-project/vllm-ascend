# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import gc

import pytest
import torch


def _cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _assert_close(name, actual, expected, rtol=5e-2, atol=5e-2):
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=rtol, atol=atol, msg=name)


def _layout_swap12_reference(x):
    return x.transpose(1, 2).contiguous()


def test_kda_layout_swap12_output_metadata():
    raw_gate = torch.zeros((1, 64, 2, 128), device="npu", dtype=torch.bfloat16)
    swapped = torch.ops._C_ascend.kda_layout_swap12(raw_gate)
    assert swapped.shape == (1, 2, 64, 128)
    assert swapped.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("shape", "dtype", "with_dependency"),
    [
        ((1, 64, 2, 128), torch.float32, False),
        ((1, 64, 2, 128), torch.float16, True),
        ((1, 64, 2, 128), torch.bfloat16, False),
    ],
)
@torch.inference_mode()
def test_kda_layout_swap12_matches_reference(shape, dtype, with_dependency):
    torch.manual_seed(20260720 + len(shape) + shape[-1])

    x = (torch.randn(*shape, dtype=dtype) * 0.04).npu()
    expected = _layout_swap12_reference(x.cpu())
    dependency = torch.empty_like(expected).npu() if with_dependency else None

    got = torch.ops._C_ascend.kda_layout_swap12(x, dependency=dependency)

    assert torch.isfinite(got).all().item()
    _assert_close("layout_swap12", got, expected, rtol=0, atol=0)
    _cleanup_npu()
