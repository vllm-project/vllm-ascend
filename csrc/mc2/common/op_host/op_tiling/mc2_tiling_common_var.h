/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_tiling_common_var.h
 * \brief
 */

#ifndef __MC2_TILING_COMMON_VAR_H__
#define __MC2_TILING_COMMON_VAR_H__

#include "op_host/tiling_type.h"

namespace optiling {
constexpr int64_t MAX_HCCL_HANDLE_LIMIT = 32;
constexpr uint32_t FP32_DATASIZE = 4;
constexpr uint32_t FP16_DATASIZE = 2;
constexpr uint32_t ALIGN32 = 32;
constexpr uint32_t ALIGN16 = 16;
constexpr uint32_t ARR_LENGTH = 128;
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4; // 只通信不计算
} // namespace optiling

#endif
