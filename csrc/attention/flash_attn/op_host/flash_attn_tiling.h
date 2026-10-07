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
 * \file flash_attn_tiling.h
 * \brief FlashAttn Tiling
 */

#ifndef FLASH_ATTN_TILING_H_
#define FLASH_ATTN_TILING_H_

#include <cstdint>
#include <register/op_impl_registry.h>
#include "../op_kernel/arch35/flash_attn_tiling_data.h"
#include "flash_attn_tiling_common.h"

namespace optiling {

// FlashAttn 使用common中的公共结构
// using FlashAttnTilingData = FlashAttnSimplifiedTilingData;

ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttn(gert::TilingContext *context);
ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttn(gert::TilingParseContext *context);

} // namespace optiling

#endif // FLASH_ATTN_TILING_H_
