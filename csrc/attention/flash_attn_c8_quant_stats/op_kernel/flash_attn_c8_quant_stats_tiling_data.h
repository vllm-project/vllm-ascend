// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "flash_attn_c8_quant_common.h"
namespace optiling {
struct FlashAttnC8QuantStatsTilingData {
    int64_t tokens, heads, keyTokenStride, keyHeadStride, valueTokenStride, valueHeadStride;
};
}
