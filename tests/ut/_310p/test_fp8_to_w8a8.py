# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend._310p.quantization.methods.fp8_to_w8a8 import (
    decode_e4m3fn,
    requantize_block_fp8_to_int8,
)


def test_decode_e4m3fn_known_values() -> None:
    codes = torch.tensor([0x00, 0x01, 0x38, 0x3C, 0x7E, 0xB8], dtype=torch.uint8)
    encoded = codes.view(torch.float8_e4m3fn)
    decoded = decode_e4m3fn(encoded)
    expected = torch.tensor([0.0, 2.0**-9, 1.0, 1.5, 448.0, -1.0])
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


def test_block_fp8_requantization_matches_reference() -> None:
    torch.manual_seed(9)
    weight_fp32 = torch.randn(5, 11).clamp(-4, 4)
    encoded = weight_fp32.to(torch.float8_e4m3fn)
    block_scale = torch.tensor([[0.5, 2.0], [1.5, 0.25]], dtype=torch.float32)

    qweight, row_scale = requantize_block_fp8_to_int8(
        encoded,
        block_scale,
        block_shape=(3, 6),
        rows_per_chunk=2,
    )

    expanded_scale = block_scale.repeat_interleave(3, 0).repeat_interleave(6, 1)[:5, :11]
    dequantized = encoded.float() * expanded_scale
    expected_scale = dequantized.abs().amax(-1) / 127.0
    safe_scale = torch.where(expected_scale > 0, expected_scale, torch.ones_like(expected_scale))
    expected_weight = torch.round(dequantized / safe_scale[:, None]).clamp(-127, 127).to(torch.int8)

    torch.testing.assert_close(qweight, expected_weight)
    torch.testing.assert_close(row_scale, safe_scale)
    assert qweight._base is None
