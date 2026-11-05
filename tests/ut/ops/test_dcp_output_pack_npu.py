# SPDX-License-Identifier: Apache-2.0
"""Independent wire-format and graph-replay checks for output/LSE packing."""

import pytest
import torch

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("NPU required", allow_module_level=True)

from vllm_ascend.ops.triton.dcp.dcp_a2a import pack_dcp_output_lse
from vllm_ascend.ops.triton.dcp.dcp_a2a_batched import _encode_dcp_lse_kernel


def wire_fields(values):
    bits = values.float().contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    significand = (bits & 0x7FFFFF) | torch.where(exponent == 0, 0, 0x800000)
    significand = torch.where(exponent < 255, significand, 0)
    code = exponent.clamp_min(1) + 1
    code = torch.where(bits < 0, -code, code)
    code = torch.where((bits & 0x7FFFFFFF) == 0, 128, code)
    code = torch.where(exponent < 255, code, 0)
    return torch.cat((code, (significand >> 16) & 255, (significand >> 8) & 255, significand & 255), -1)


def decoded(fields):
    code, hi, mid, lo = fields.double().unbind(-1)
    significand = hi * 65536 + mid * 256 + lo
    return torch.sign(code) * torch.ldexp(significand, (code.abs() - 151).int())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (16, 48, 6, 512, 1),
        (8, 128, 12, 512, 1),
        (17, 49, 1, 256, 1),
        (3, 9, 2, 32, 1),
        (1, 1, 1, 512, 1),
        (4, 16, 3, 512, 0),
        (4, 17, 7, 1024, 1),
        (8, 32, 3, 2048, 1),
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "feature_stride", "row_stride"])
def test_pack_wire_and_changed_input_graph(dtype, shape, layout):
    world, tokens, heads, dim, scatter = shape
    total_heads = world * heads if scatter == 1 else heads
    generator = torch.Generator().manual_seed(55)
    values = torch.randn(tokens, total_heads, dim, generator=generator).to(dtype)
    lse = torch.randn(tokens, total_heads, 1, generator=generator) * 12
    special = torch.tensor([0, -0.0, float("nan"), float("inf"), -float("inf"), 1e30, -1e30])
    lse.reshape(-1)[: min(special.numel(), lse.numel())] = special[: min(special.numel(), lse.numel())]
    if layout == "feature_stride":
        device_values = torch.empty((tokens, total_heads, dim * 2), device="npu", dtype=dtype)[..., ::2]
    elif layout == "row_stride":
        device_values = torch.empty((tokens * 2, total_heads, dim), device="npu", dtype=dtype)[::2]
    else:
        device_values = torch.empty_like(values, device="npu")
    device_values.copy_(values)
    device_lse = lse.npu()

    def expected():
        if dtype == torch.float32:
            fields = lse
        else:
            fields = wire_fields(lse).to(dtype)
        packed = torch.cat((values, fields), -1)
        if scatter == 1:
            return packed.reshape(tokens, world, heads, -1).permute(1, 2, 0, 3).contiguous()
        return packed.reshape(world, tokens // world, total_heads, -1)

    for _ in range(3):
        actual = pack_dcp_output_lse(device_values, device_lse, world, scatter)
    torch.testing.assert_close(actual.cpu(), expected(), rtol=0, atol=0, equal_nan=True)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            captured = pack_dcp_output_lse(device_values, device_lse, world, scatter)
        for iteration in range(2):
            values.add_(1)
            lse.add_(0.5)
            device_values.copy_(values)
            device_lse.copy_(lse)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(captured.cpu(), expected(), rtol=0, atol=0, equal_nan=True)
    finally:
        graph.reset()


@pytest.mark.parametrize("from_stats", [False, True])
def test_encoder_ieee_patterns_and_stats(from_stats):
    torch.manual_seed(57)
    words = torch.randint(-(1 << 31), (1 << 31) - 1, (8192, 1), dtype=torch.int32)
    patterns = [
        0,
        -2147483648,
        1,
        -2147483647,
        0x7FFFFF,
        0x800000,
        0x7F7FFFFF,
        -8388609,
        0x7F800000,
        -8388608,
        0x7FC00001,
    ]
    words[: len(patterns), 0] = torch.tensor(patterns, dtype=torch.int32)
    maxima = words.view(torch.float32)
    sums = (torch.rand_like(maxima) + 0.125) if from_stats else torch.ones_like(maxima)
    encoded = torch.full((8192, 4), -999.0, device="npu", dtype=torch.float32)
    _encode_dcp_lse_kernel[(40,)](
        maxima.npu(), sums.npu(), encoded, 1, 1, 1, 1, 8192, 1, from_stats, 256, multibuffer=False
    )
    actual = encoded.cpu()
    if not from_stats:
        torch.testing.assert_close(actual, wire_fields(maxima).float(), rtol=0, atol=0)
    else:
        expected = maxima.double() + sums.double().log()
        finite = torch.isfinite(expected[:, 0])
        torch.testing.assert_close(decoded(actual)[finite], expected[:, 0][finite], rtol=1e-5, atol=2e-6)
        assert torch.all(actual[~finite] == 0)


def test_public_pack_from_stats_and_graph():
    torch.manual_seed(58)
    maxima = torch.randn(48, 96, 1)
    sums = torch.rand_like(maxima) + 0.125
    sums.reshape(-1)[:4] = torch.tensor([0.0, -1.0, float("inf"), float("nan")])
    values = torch.randn(48, 96, 512).bfloat16()
    device_values, device_max, device_sum = values.npu(), maxima.npu(), sums.npu()
    for _ in range(3):
        pack_dcp_output_lse(device_values, device_max, 16, 1, device_sum)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            result = pack_dcp_output_lse(device_values, device_max, 16, 1, device_sum)
        for _ in range(2):
            maxima.add_(0.5)
            device_max.copy_(maxima)
            graph.replay()
            torch.npu.synchronize()
            actual = result.cpu().permute(2, 0, 1, 3).reshape(48, 96, 516)
            torch.testing.assert_close(actual[..., :512], values, rtol=0, atol=0)
            expected = (maxima.double() + sums.double().log())[..., 0]
            finite = torch.isfinite(expected)
            torch.testing.assert_close(decoded(actual[..., 512:])[finite], expected[finite], rtol=1e-5, atol=2e-6)
            assert torch.all(actual[..., 512:][~finite] == 0)
    finally:
        graph.reset()
