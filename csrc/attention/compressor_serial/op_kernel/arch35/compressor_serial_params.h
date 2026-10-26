// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// Shared, fixed-width ABI. Metadata values stay on device during graph replay.
struct CompressorSerialParams {
    int64_t batch, sequence, tokens, dim, ratio, coff, cacheMode, blockSize;
    int64_t cacheStride, cacheBlocks, tableWidth, outputRows, layout, rotaryMode;
    int64_t flags, epsBits, projectionStride, gateOffset;
};

constexpr int64_t HAS_TABLE = 1;
constexpr int64_t HAS_CU = 2;
constexpr int64_t HAS_USED = 4;
constexpr int64_t HAS_START = 8;
