# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.mark.parametrize("num_tokens", [1, 2048])
@torch.inference_mode()
def test_deepseek_gate_bf16_matmul_returns_fp32(num_tokens):
    generator = torch.Generator().manual_seed(7)
    hidden_states = torch.randn(num_tokens, 5120, generator=generator).bfloat16()
    weight = (torch.randn(384, 5120, generator=generator) * 0.025).bfloat16()
    expected = torch.mm(hidden_states.float(), weight.float().t())

    logits = torch.mm(hidden_states.npu(), weight.npu().t(), out_dtype=torch.float32)

    torch.testing.assert_close(logits.cpu(), expected, rtol=2e-5, atol=3e-5)
