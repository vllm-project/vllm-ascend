// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace optiling {
struct KdaRmsNormGatedTilingData {
    int64_t tokens, heads, xTokenStride, gateTokenStride, tileTokens;
    float epsilon;
    uint32_t reserved;
};
}
