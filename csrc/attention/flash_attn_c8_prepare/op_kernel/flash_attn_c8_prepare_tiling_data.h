// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "../../flash_attn_c8_quant_stats/op_kernel/flash_attn_c8_quant_common.h"
namespace optiling {
struct FlashAttnC8PrepareTilingData {
    int64_t queryTokens, keyTokens, heads;
    int64_t queryTokenStride, queryHeadStride, keyTokenStride, keyHeadStride;
    int64_t valueTokenStride, valueHeadStride, ropeTokenStride, ropeHeadStride;
};
}
