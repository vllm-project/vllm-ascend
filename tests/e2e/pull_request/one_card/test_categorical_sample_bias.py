# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.sample.categorical_sample import categorical_sample
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

pytestmark = pytest.mark.e2e_coverage(
    arch="", feature="dspark", parallel="", deploy="", hardware="A2", quantization="", graph_mode="eager"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_markov_bias_sampling_and_draft_cache_on_npu(dtype, with_bias):
    """Exercise the real kernel, row mapping and fine/coarse tail boundaries."""
    init_device_properties_triton()
    logits_cpu = torch.zeros(2, 8193, dtype=dtype)
    logits_cpu[:, 0] = 1
    bias_cpu = torch.zeros_like(logits_cpu)
    bias_cpu[0, 8192] = 4
    bias_cpu[1, 1024] = 4
    expected = (logits_cpu.float() + bias_cpu.float()).to(dtype) if with_bias else logits_cpu
    logits = logits_cpu.to("npu")
    bias = bias_cpu.to("npu") if with_bias else None
    cache = torch.full((3, 3, 8193), -33, dtype=dtype, device="npu")
    sampled = categorical_sample(
        logits,
        torch.tensor([1, 0], dtype=torch.int32, device="npu"),
        torch.zeros(3, device="npu"),
        torch.tensor([3, 7, 11], dtype=torch.int64, device="npu"),
        torch.tensor([8, 12], dtype=torch.int64, device="npu"),
        apply_temperature=True,
        is_drafting=True,
        logits_cache=cache,
        logits_cache_col=torch.tensor([1, 2], dtype=torch.int32, device="npu"),
        bias=bias,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(sampled.cpu(), expected.argmax(dim=-1), rtol=0, atol=0)
    torch.testing.assert_close(cache[1, 1].cpu(), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(cache[0, 2].cpu(), expected[1], rtol=0, atol=0)
    torch.testing.assert_close(cache[2].cpu(), torch.full((3, 8193), -33, dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(logits.cpu(), logits_cpu, rtol=0, atol=0)
