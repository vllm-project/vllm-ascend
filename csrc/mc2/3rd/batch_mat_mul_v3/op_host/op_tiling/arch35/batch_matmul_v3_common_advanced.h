/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_matmul_v3_common_advanced.h
 * \brief
 */
#ifndef __OP_HOST_BATCH_MATMUL_V3_COMMON_ADVANCED_H__
#define __OP_HOST_BATCH_MATMUL_V3_COMMON_ADVANCED_H__

#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"

namespace optiling {
namespace Mc2batch_matmul_v3_advanced {

constexpr uint64_t FINAL_SHAPE_DIM = 1;
constexpr uint64_t NO_BATCH_SHAPE_DIM = 2;
constexpr uint64_t ONE_BATCH_SHAPE_DIM = 3;
constexpr uint64_t TWO_BATCH_SHAPE_DIM = 4;
constexpr uint64_t THREE_BATCH_SHAPE_DIM = 5;
constexpr uint64_t FOUR_BATCH_SHAPE_DIM = 6;
constexpr uint64_t ND_NZ_DIM_DIFF = 2;
constexpr uint64_t BATCH_DIM_MAX = 6;
} // namespace Mc2batch_matmul_v3_advanced
} // namespace optiling
#endif // __OP_HOST_BATCH_MATMUL_V3_COMMON_ADVANCED_H__
