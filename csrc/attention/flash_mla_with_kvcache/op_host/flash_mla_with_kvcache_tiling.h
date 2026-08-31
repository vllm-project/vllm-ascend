/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_mla_with_kvcache_tiling.h
 * \brief FlashMlaWithKvcache Tiling
 */

#ifndef FLASH_MLA_WITH_KVCACHE_TILING_H_
#define FLASH_MLA_WITH_KVCACHE_TILING_H_

#include <cstdint>
#include <register/op_impl_registry.h>
#include "../op_kernel/arch35/flash_mla_with_kvcache_tiling_data.h"
#include "flash_mla_with_kvcache_tiling_common.h"

namespace optiling {

// FlashMlaWithKvcache 使用common中的公共结构
// using FlashMlaWithKvcacheTilingData = FlashMlaWithKvcacheSimplifiedTilingData;

ASCENDC_EXTERN_C ge::graphStatus TilingFlashMlaWithKvcache(gert::TilingContext *context);
ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashMlaWithKvcache(gert::TilingParseContext *context);

}  // namespace optiling

#endif  // FLASH_MLA_WITH_KVCACHE_TILING_H_
