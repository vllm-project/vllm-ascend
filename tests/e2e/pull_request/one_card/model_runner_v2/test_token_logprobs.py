# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset", [10000.0, -10000.0])
@pytest.mark.parametrize("vocab_size", [97, 32000])
def test_token_logprobs_large_common_offset(dtype, offset, vocab_size):
    """Subtract the row maximum before the small log-normalization term."""
    logits = ((torch.arange(vocab_size, device="cpu") % 17).float() / 8).repeat(3, 1)
    logits[0].zero_()
    logits[2].zero_()
    logits[2, -1] = 16
    logits = (logits + offset).to(dtype)
    token_ids = torch.tensor([[0, 1, vocab_size // 2, vocab_size - 1]], device="cpu").expand(3, -1).contiguous()

    # Normalize the quantized input, so input rounding is not attributed to the kernel.
    expected = logits.double().log_softmax(dim=-1).gather(1, token_ids).float()
    actual = compute_token_logprobs(logits.npu(), token_ids.npu())
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
