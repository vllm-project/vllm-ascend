// SPDX-License-Identifier: Apache-2.0
#ifndef FLASH_ATTN_C8_TILING_DATA_H_
#define FLASH_ATTN_C8_TILING_DATA_H_
#include <cstdint>
#include "../../a5_mla_common/op_kernel/arch35/c8_pipeline/flash_mla_with_kvcache_tiling_data.h"
namespace optiling {
struct FlashAttnC8TilingData {
    FlashMlaWithKvcacheNoQuantTilingArch35 baseTiling;
};
}
#endif
