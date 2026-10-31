# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

HC_MULT = 4
HIDDEN_SIZE = 4096
EXTENDED_HIDDEN_SIZE = 7168
BF16_RTOL = 1e-2
BF16_ATOL = 1e-3


def _hc_post_cpu(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    # comb[..., i, j] routes residual stream i into output stream j.
    mixed_residual = torch.einsum("bsij,bsid->bsjd", comb.float(), residual.float())
    post_term = post.float().unsqueeze(-1) * x.float().unsqueeze(-2)
    return (mixed_residual + post_term).to(residual.dtype)


def _compare_hc_post_with_cpu(batch: int, seq_len: int, hidden_size: int, mode: str):
    generator = torch.Generator().manual_seed(1024)
    x = torch.randn(batch, seq_len, hidden_size, generator=generator).bfloat16()
    residual = torch.randn(batch, seq_len, HC_MULT, hidden_size, generator=generator).bfloat16()
    post = 2 * torch.sigmoid(torch.randn(batch, seq_len, HC_MULT, generator=generator))
    # Non-symmetric matrices make an incorrect mixing transpose observable.
    comb = torch.randn(batch, seq_len, HC_MULT, HC_MULT, generator=generator).softmax(-1)
    if mode == "residual_only":
        post.zero_()
    elif mode == "post_only":
        comb.zero_()

    inputs = (x, residual, post, comb)
    expected = _hc_post_cpu(*inputs)
    npu_inputs = tuple(tensor.npu() for tensor in inputs)
    actual = torch.ops._C_ascend.npu_hc_post(*npu_inputs)

    assert actual.shape == residual.shape
    assert actual.dtype == residual.dtype
    assert actual.device == npu_inputs[0].device
    # Compare every element after FP32 accumulation and BF16 output rounding.
    torch.testing.assert_close(actual.cpu(), expected, rtol=BF16_RTOL, atol=BF16_ATOL)
    for npu_input, original in zip(npu_inputs, inputs):
        torch.testing.assert_close(npu_input.cpu(), original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch,seq_len,hidden_size,mode",
    [
        pytest.param(1, 1, HIDDEN_SIZE, "mixed", id="decode"),
        pytest.param(1, 17, HIDDEN_SIZE, "mixed", id="prefill"),
        pytest.param(2, 3, HIDDEN_SIZE, "mixed", id="multi_batch"),
        pytest.param(1, 4, EXTENDED_HIDDEN_SIZE, "mixed", id="extended_hidden_size"),
        pytest.param(1, 2, HIDDEN_SIZE, "residual_only", id="residual_only"),
        pytest.param(1, 2, HIDDEN_SIZE, "post_only", id="post_only"),
    ],
)
@torch.inference_mode()
def test_npu_hc_post_bf16(batch, seq_len, hidden_size, mode):
    _compare_hc_post_with_cpu(batch, seq_len, hidden_size, mode)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
