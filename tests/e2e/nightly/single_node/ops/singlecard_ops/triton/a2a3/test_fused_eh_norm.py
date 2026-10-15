# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.linearnorm.fused_eh_norm import fused_eh_norm

TOLERANCES = {
    torch.float16: (2e-3, 2e-2),
    torch.bfloat16: (2e-2, 5e-2),
    torch.float32: (1e-4, 1e-4),
}


def fused_eh_norm_ref(positions, embeds, previous_hidden, enorm_w, hnorm_w, eps):
    """CPU fp32 reference using PyTorch RMSNorm for each output half."""
    masked_embeds = torch.where(positions[:, None] == 0, 0.0, embeds.float())
    shape = (embeds.shape[-1],)
    e_norm = F.rms_norm(masked_embeds, shape, enorm_w.float(), eps)
    h_norm = F.rms_norm(previous_hidden.float(), shape, hnorm_w.float(), eps)
    return torch.cat((e_norm, h_norm), dim=-1).to(embeds.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_rows", [False, True])
@pytest.mark.parametrize(
    ("positions_list", "hidden_size", "eps"),
    [
        pytest.param([7], 4096, 1e-5, id="decode"),
        pytest.param([0, 1, 0, 9], 4096, 1e-5, id="mixed-start-positions"),
        pytest.param(list(range(17)), 4096, 1e-5, id="prefill"),
        pytest.param([0, 2, 5], 513, 1e-6, id="masked-hidden-tail"),
    ],
)
@torch.inference_mode()
def test_fused_eh_norm_matches_reference(dtype, strided_rows, positions_list, hidden_size, eps):
    # 4096 and 1e-5 are the GLM text config defaults; 513 exercises tail masks.
    generator = torch.Generator().manual_seed(42)
    num_tokens = len(positions_list)
    positions = torch.tensor(positions_list, dtype=torch.int64)
    embeds = torch.randn(num_tokens, hidden_size, generator=generator).to(dtype)
    previous_hidden = torch.randn(num_tokens, hidden_size, generator=generator).to(dtype)
    enorm_w = torch.randn(hidden_size, generator=generator).to(dtype)
    hnorm_w = torch.randn(hidden_size, generator=generator).to(dtype)
    if num_tokens > 1:
        # Exercise epsilon on zero rows at a nonzero position as well.
        embeds[1].zero_()
        previous_hidden[1].zero_()

    def to_npu_rows(x, padding):
        backing = torch.full((num_tokens, hidden_size + padding), 19.0, dtype=dtype, device="npu")
        backing[:, :hidden_size].copy_(x)
        return backing[:, :hidden_size]

    # Different row strides detect accidental reuse of the embedding stride.
    embeds_npu = to_npu_rows(embeds, 16 if strided_rows else 0)
    previous_npu = to_npu_rows(previous_hidden, 32 if strided_rows else 0)
    actual = fused_eh_norm(positions.to("npu"), embeds_npu, previous_npu, enorm_w.to("npu"), hnorm_w.to("npu"), eps)
    expected = fused_eh_norm_ref(positions, embeds, previous_hidden, enorm_w, hnorm_w, eps)

    assert actual.shape == (num_tokens, 2 * hidden_size)
    assert actual.dtype == dtype
    assert actual.device == embeds_npu.device
    actual_cpu = actual.cpu()
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(actual_cpu, expected, rtol=rtol, atol=atol)
    assert torch.count_nonzero(actual_cpu[positions == 0, :hidden_size]) == 0
    torch.testing.assert_close(embeds_npu.cpu(), embeds, rtol=0, atol=0)
    torch.testing.assert_close(previous_npu.cpu(), previous_hidden, rtol=0, atol=0)
