# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent raw-byte decoders, with no NPU operator dependency."""

import math

MX_BLOCK_SIZE = 32
E4M3_BIAS = 7
E4M3_MANTISSA_BITS = 3
E8M0_BIAS = 127
BYTE_VALUES = 256


def decode_e4m3fn(byte: int) -> float:
    if not 0 <= byte < BYTE_VALUES:
        raise ValueError("FP8 code must be a byte")
    sign = -1.0 if byte & 0x80 else 1.0
    exponent = (byte >> E4M3_MANTISSA_BITS) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    if exponent == 0:
        return sign * math.ldexp(float(mantissa), 1 - E4M3_BIAS - E4M3_MANTISSA_BITS)
    return sign * math.ldexp(1.0 + mantissa / (1 << E4M3_MANTISSA_BITS), exponent - E4M3_BIAS)


def decode_e8m0(byte: int) -> float:
    if not 0 <= byte < BYTE_VALUES:
        raise ValueError("E8M0 code must be a byte")
    return math.nan if byte == 0xFF else math.ldexp(1.0, byte - E8M0_BIAS)
