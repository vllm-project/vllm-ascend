/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file load_to_l0_utils.h
 * \brief
 */

#ifndef MATMUL_TILE_LOAD_DATA_LOAD_TO_L0_UTILS_H
#define MATMUL_TILE_LOAD_DATA_LOAD_TO_L0_UTILS_H

namespace Act {
namespace Gemm {
namespace Tile {
constexpr uint16_t HW_N0 = 16;                  ///< Hardware configuration parameter for N0
constexpr uint16_t HW_M0 = 16;                  ///< Hardware configuration parameter for M0
constexpr uint16_t ALIGN_NUM = 16;              ///< Alignment number for data processing
constexpr uint64_t M_POS_BIT = 48;              ///< Bit position for M index
constexpr uint64_t K_POS_BIT = 32;              ///< Bit position for K index
constexpr uint64_t M_STEP_BIT = 16;             ///< Bit step for M index
constexpr uint8_t INDEX_SHIFT = 2;              ///< Shift value for index manipulation
constexpr uint8_t K_STEP_MIN_VAL_B32 = 2;       ///< Minimum step value for K index in B32
constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};   ///< Padding list for data alignment

} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif