// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace optiling {
constexpr int64_t KDA_STATE_COPY_TILE_BYTES = 8192;
struct KdaStateCopyTilingData {
    int64_t cacheRows, selectedRows, cacheStrideBytes, payloadBytes;
    uint32_t toCache, hasInitialState;
};
}
