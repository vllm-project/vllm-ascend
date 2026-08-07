# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend._310p.quantization.methods.mxfp4_to_w8a8 import (
    decode_e8m0,
    requantize_mxfp4_to_int8,
    unpack_mxfp4_groups,
)


def test_decode_e8m0_exact_powers_of_two() -> None:
    encoded = torch.tensor([0, 126, 127, 128, 129], dtype=torch.uint8)
    decoded = decode_e8m0(encoded)
    expected = torch.tensor([2.0**-127, 0.5, 1.0, 2.0, 4.0], dtype=torch.float32)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


def test_unpack_mxfp4_preserves_per_byte_nibble_order() -> None:
    low_codes = torch.arange(16, dtype=torch.uint8)
    high_codes = torch.arange(15, -1, -1, dtype=torch.uint8)
    packed = (low_codes | (high_codes << 4)).reshape(1, 1, 16)
    unpacked = unpack_mxfp4_groups(packed).reshape(-1)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    expected = torch.stack((table, table.flip(0)), dim=-1).flatten()
    torch.testing.assert_close(unpacked, expected)


def test_requantize_mxfp4_to_int8_preserves_shape_and_relative_values() -> None:
    # Each byte stores two consecutive values. Low and high nibbles use the
    # same code here, and E8M0=127 gives a group scale of one.
    codes = torch.arange(16, dtype=torch.uint8)
    packed = (codes | (codes << 4)).reshape(1, 1, 16)
    scales = torch.tensor([[[127]]], dtype=torch.uint8)

    qweight, row_scale = requantize_mxfp4_to_int8(packed, scales, rows_per_chunk=1)

    assert qweight.shape == (1, 1, 32)
    assert row_scale.shape == (1, 1, 1)
    torch.testing.assert_close(row_scale, torch.tensor([[[6.0 / 127.0]]]))
    assert qweight[0, 0, 14].item() == 127
    assert qweight[0, 0, 15].item() == 127
    assert qweight[0, 0, 30].item() == -127
    assert qweight[0, 0, 31].item() == -127
    torch.testing.assert_close(qweight[..., 0::2], qweight[..., 1::2])


def test_requantize_chunking_is_stable() -> None:
    generator = torch.Generator().manual_seed(7)
    packed = torch.randint(0, 256, (2, 5, 32), dtype=torch.uint8, generator=generator)
    scales = torch.randint(120, 132, (2, 5, 2), dtype=torch.uint8, generator=generator)
    q1, s1 = requantize_mxfp4_to_int8(packed, scales, rows_per_chunk=1)
    q2, s2 = requantize_mxfp4_to_int8(packed, scales, rows_per_chunk=64)
    torch.testing.assert_close(q1, q2)
    torch.testing.assert_close(s1, s2)


def test_requantize_allocates_the_final_logical_shape() -> None:
    packed = torch.zeros((2, 3, 16), dtype=torch.uint8)
    scales = torch.full((2, 3, 1), 127, dtype=torch.uint8)
    qweight, row_scale = requantize_mxfp4_to_int8(packed, scales)

    assert qweight.shape == (2, 3, 32)
    assert row_scale.shape == (2, 3, 1)
    # A view of a flattened allocation would retain a 2D storage descriptor
    # on Ascend. Allocating the final shape directly is required before NZ
    # conversion for grouped matmul.
    assert qweight._base is None
    assert row_scale._base is None
