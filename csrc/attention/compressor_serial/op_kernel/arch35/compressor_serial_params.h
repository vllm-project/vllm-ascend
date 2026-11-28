// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// Shared, fixed-width ABI. Metadata values stay on device during graph replay.
struct CompressorSerialParams {
    int32_t batch, sequence, tokens, dim, ratio, coff, cacheMode, blockSize;
    int32_t cacheStride, cacheBlocks, tableWidth, outputRows, layout, rotaryMode;
    int32_t flags, epsBits, projectionStride, gateOffset;
    int32_t reserved[6];
    uint32_t ropeOffsets[64];
    float ropeSigns[64];
};

constexpr int32_t HAS_TABLE = 1;
constexpr int32_t HAS_CU = 2;
constexpr int32_t HAS_USED = 4;
constexpr int32_t HAS_START = 8;
