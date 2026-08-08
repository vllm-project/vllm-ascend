# SPDX-License-Identifier: Apache-2.0
"""Software MXFP4 to per-row INT8 conversion for Ascend 310P.

DeepSeek V4 stores expert weights as two E2M1 values per byte and one E8M0
scale per group of 32 logical weights. Ascend 310P cannot consume the custom
FP4 dtype directly, so the experimental backend converts each local EP shard
to the existing 310P W8A8 representation during weight finalization.
"""

from __future__ import annotations

import torch

MXFP4_GROUP_SIZE = 32
MXFP4_VALUES_PER_BYTE = 2

# OCP MX E2M1 values. Each value is represented exactly in float32.
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_E2M1_TABLE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 scale bytes to float32 exactly.

    E8M0 encodes ``2 ** (byte - 127)``. Byte zero is the denormal value
    ``2 ** -127``, which the bit construction below handles explicitly.
    ``scale`` may be a uint8 tensor or a float8_e8m0fnu tensor.
    """
    scale_bytes = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
    scale_i32 = scale_bytes.to(torch.int32)
    bits = scale_i32 << 23
    bits = torch.where(scale_i32 == 0, torch.full_like(bits, 0x00400000), bits)
    return bits.contiguous().view(torch.float32)


def unpack_mxfp4_groups(packed: torch.Tensor) -> torch.Tensor:
    """Unpack ``[..., groups, 16]`` bytes into ``[..., groups, 32]`` E2M1 values.

    OCP MXFP4 stores the first 16 values in the low nibbles and the next 16
    values in the high nibbles of the same 16-byte group.
    """
    if packed.dtype not in (torch.uint8, torch.int8):
        raise TypeError(f"MXFP4 packed weights must use uint8/int8 storage, got {packed.dtype}.")
    if packed.shape[-1] != MXFP4_GROUP_SIZE // MXFP4_VALUES_PER_BYTE:
        raise ValueError(f"Expected 16 packed bytes per MXFP4 group, got shape {tuple(packed.shape)}.")

    packed_u8 = packed.view(torch.uint8)
    low = packed_u8 & 0x0F
    high = packed_u8 >> 4
    # compressed-tensors packs consecutive FP4 values into each byte:
    # value[2*i] in the low nibble and value[2*i+1] in the high nibble.
    # Preserve that per-byte interleaving when expanding 16 bytes to 32 values.
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    key = (packed.device.type, packed.device.index)
    table = _E2M1_TABLE_CACHE.get(key)
    if table is None:
        table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
        _E2M1_TABLE_CACHE[key] = table
    return table[codes.to(torch.long)]


def requantize_mxfp4_to_int8(
    packed_weight: torch.Tensor,
    e8m0_scale: torch.Tensor,
    *,
    rows_per_chunk: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert packed MXFP4 weights to symmetric per-row W8A8 weights.

    Args:
        packed_weight: Tensor shaped ``[..., output_rows, input_size / 2]``.
        e8m0_scale: Tensor shaped ``[..., output_rows, input_size / 32]``.
        rows_per_chunk: Number of flattened output rows converted at once.

    Returns:
        ``(int8_weight, float32_scale)`` where the weight has logical input
        width and the scale has shape ``[..., output_rows, 1]``.
    """
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive.")
    if packed_weight.shape[:-1] != e8m0_scale.shape[:-1]:
        raise ValueError(
            "Packed weight and E8M0 scale leading dimensions differ: "
            f"{tuple(packed_weight.shape)} vs {tuple(e8m0_scale.shape)}."
        )

    groups = e8m0_scale.shape[-1]
    expected_packed_width = groups * (MXFP4_GROUP_SIZE // MXFP4_VALUES_PER_BYTE)
    if packed_weight.shape[-1] != expected_packed_width:
        raise ValueError(
            f"Packed width {packed_weight.shape[-1]} does not match {groups} MXFP4 groups "
            f"({expected_packed_width} bytes)."
        )

    leading_shape = packed_weight.shape[:-1]
    num_rows = 1
    for dim in leading_shape:
        num_rows *= dim

    packed_rows = packed_weight.reshape(num_rows, groups, MXFP4_GROUP_SIZE // MXFP4_VALUES_PER_BYTE)
    scale_rows = e8m0_scale.reshape(num_rows, groups)
    logical_shape = (*leading_shape, groups * MXFP4_GROUP_SIZE)
    scale_shape = (*leading_shape, 1)
    quantized = torch.empty(
        logical_shape,
        dtype=torch.int8,
        device=packed_weight.device,
    )
    row_scales = torch.empty(scale_shape, dtype=torch.float32, device=packed_weight.device)
    quantized_rows = quantized.reshape(num_rows, groups * MXFP4_GROUP_SIZE)
    row_scale_rows = row_scales.reshape(num_rows, 1)

    for start in range(0, num_rows, rows_per_chunk):
        end = min(start + rows_per_chunk, num_rows)
        fp4_values = unpack_mxfp4_groups(packed_rows[start:end])
        group_scales = decode_e8m0(scale_rows[start:end]).unsqueeze(-1)
        dequantized = (fp4_values * group_scales).reshape(end - start, -1)

        max_abs = dequantized.abs().amax(dim=-1, keepdim=True)
        scale = max_abs / 127.0
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        qweight = torch.round(dequantized / safe_scale).clamp_(-127, 127).to(torch.int8)

        quantized_rows[start:end].copy_(qweight)
        row_scale_rows[start:end].copy_(safe_scale)

    return quantized, row_scales
