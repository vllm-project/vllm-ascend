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
 * \file op_api_def.h
 * \brief
 */

#ifndef Transformer_COMMON_OP_API_DEF_H
#define Transformer_COMMON_OP_API_DEF_H

namespace op {
    constexpr size_t MAX_SUPPORT_DIMS_NUMS = 8;
    constexpr size_t BN_MIN_SUPPORT_DIMS_NUMS = 2;
    constexpr int8_t FP16FP32_KEEP_DTYPE = -1;
    constexpr int8_t KEEP_DTYPE = 0;
    constexpr int8_t ALLOW_FP32_DOWN_PRECISION = 1;
    constexpr int8_t USE_FP16 = 2;
    constexpr int8_t USE_HF32 = 3;
    constexpr size_t MAX_MASK_LEN64 = 64;
} // namespace op
#endif // Transformer_COMMON_OP_API_DEF_H