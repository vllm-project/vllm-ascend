# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.quantize_mxfp4_indexer import quantize_mxfp4_indexer


def reference(query):
    groups = query.float().unflatten(-1, (4, 32))
    scale = (groups.abs().amax(dim=-1).clamp_min(6 * 2.0**-126) / 6).log2().ceil()
    normalized = groups / torch.exp2(scale).unsqueeze(-1)
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    code = torch.bucketize(normalized.abs(), thresholds, out_int32=True).to(torch.uint8)
    code |= (normalized < 0).to(torch.uint8) << 3
    pairs = code.flatten(-2).unflatten(-1, (64, 2))
    packed = pairs[..., 0] | (pairs[..., 1] << 4)
    return packed, (scale + 127).to(torch.uint8)


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="MXFP4 indexer query path is A5-specific")
@pytest.mark.parametrize("tokens", [0, 1, 256, 16384])
def test_quantize_mxfp4_indexer_matches_reference(tokens):
    generator = torch.Generator().manual_seed(42)
    query = torch.randn((tokens, 32, 128), generator=generator, dtype=torch.bfloat16)
    if tokens:
        # Different scales in all four groups, including maxima from either
        # nibble, exact ties, tiny negative numbers and a fully zero group.
        groups = query[0, 0].unflatten(0, (4, 32))
        groups.zero_()
        groups[0, :10] = torch.tensor([6.0, -0.125, -0.0, 0.25, -0.75, 1.25, -1.75, 2.5, 3.5, -5.0])
        groups[1, :3] = torch.tensor([0.5, -12.0, 7.0])
        groups[2, :3] = torch.tensor([3.0, 1.75, -0.5])

    expected_data, expected_scale = reference(query)
    data, scale = quantize_mxfp4_indexer(query.npu())
    torch.testing.assert_close(data.cpu(), expected_data, rtol=0, atol=0)
    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=0, atol=0)
