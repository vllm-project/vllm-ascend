# SPDX-License-Identifier: Apache-2.0
"""Wire-format and payload checks for rectangular and fallback DCP packing."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.dcp.dcp_a2a import pack_dcp_output_lse


@pytest.mark.parametrize(
    "tokens,heads,dim,world",
    [
        (1, 64, 512, 8),
        (7, 64, 512, 8),
        (114, 64, 512, 8),
        (127, 64, 512, 8),
        (128, 64, 512, 8),
        (129, 64, 512, 8),
        (144, 96, 512, 8),
        (152, 96, 512, 8),
        (65, 128, 512, 8),
        (129, 12, 512, 4),
        (65, 7, 512, 1),
        (5, 96, 64, 8),
        (5, 96, 128, 8),
        (5, 96, 384, 8),
        (5, 96, 1024, 8),
        (41, 32, 256, 8),
        (41, 32, 2048, 8),
        (3, 96, 512, 8),
        (4, 96, 512, 8),
        (5, 96, 512, 8),
        (5, 96, 513, 8),
        (5, 96, 769, 8),
        (5, 96, 1536, 8),
        (5, 96, 2047, 8),
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "head_major", "feature_stride"])
@pytest.mark.parametrize("from_stats", [False, True])
def test_dcp_pack_payload_and_lse_wire_format(tokens, heads, dim, world, layout, from_stats):
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(20878)
    output = torch.randn((tokens, heads, dim), generator=generator).bfloat16()
    output[0, 0, :5] = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.0, -0.0], dtype=torch.bfloat16)
    lse = torch.randn((tokens, heads, 1), generator=generator) * 8
    lse.flatten()[:8] = torch.tensor([0.0, -0.0, torch.inf, -torch.inf, torch.nan, 70_000.0, -70_000.0, 1e-40])
    sums = torch.rand((tokens, heads, 1), generator=generator) + 0.5
    if layout == "head_major":
        device_output = output.transpose(0, 1).contiguous().npu().transpose(0, 1)
        device_lse = lse.transpose(0, 1).contiguous().npu().transpose(0, 1)
    elif layout == "feature_stride":
        device_output = torch.stack((output, output), -1).npu()[..., 0]
        device_lse = torch.stack((lse, lse), -1).npu()[..., 0]
    else:
        device_output, device_lse = output.npu(), lse.npu()
    device_sums = torch.stack((sums, sums), -1).npu()[..., 0] if from_stats else None
    packed = pack_dcp_output_lse(device_output, device_lse, world, 1, softmax_sum=device_sums)
    # Undo only the documented head-scatter layout, without using a GPU decoder.
    wire = packed.cpu().reshape(heads, tokens, dim + 4).transpose(0, 1).contiguous()
    assert torch.equal(wire[..., :dim].contiguous().view(torch.int16), output.view(torch.int16))

    fields = wire[..., dim:].float()
    code, high, middle, low = fields.unbind(-1)
    significand = high.double() * 65536 + middle.double() * 256 + low.double()
    decoded = code.sign().double() * torch.ldexp(significand, code.abs().int() - 151)
    expected = lse[..., 0].double()
    if from_stats:
        expected = expected + sums[..., 0].double().log()
    finite = expected.isfinite()
    assert torch.equal(code == 0, ~finite)
    torch.testing.assert_close(decoded[finite], expected[finite], rtol=2**-10, atol=2**-16)
    if not from_stats:
        # Independent CPU bit decomposition checks exact FP32 wire fields,
        # including subnormals, signed zero and non-finite sentinels.
        raw = lse[..., 0].contiguous().view(torch.int32).long() & 0xFFFFFFFF
        exponent = (raw >> 23) & 255
        fraction = raw & 0x7FFFFF
        significand = fraction | torch.where(exponent == 0, 0, 0x800000)
        significand = torch.where(finite, significand, 0)
        expected_code = (exponent.clamp_min(1) + 1) * torch.where(raw >> 31 == 0, 1, -1)
        expected_code = torch.where(raw & 0x7FFFFFFF == 0, 128, expected_code)
        expected_code = torch.where(finite, expected_code, 0)
        expected_fields = torch.stack(
            (expected_code, (significand >> 16) & 255, (significand >> 8) & 255, significand & 255), -1
        ).float()
        assert torch.equal(fields, expected_fields)
