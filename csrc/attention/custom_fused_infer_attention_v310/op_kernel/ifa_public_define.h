/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ifa_public_define.h
 * \brief
 */
#ifndef IFA_PUBLIC_DEFINE_H
#define IFA_PUBLIC_DEFINE_H

#include "kernel_operator.h"

using namespace AscendC;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;

constexpr int32_t FLOAT_VECTOR_SIZE_I = 64;
constexpr int32_t VECTOR_SIZE_I = 128;
constexpr int32_t BLOCK_SIZE_I = 16;
constexpr int32_t BLOCK_SIZE_FLOAT = 8;
constexpr uint32_t L0AB_HALF_BUF_SIZE_I = 16384;
constexpr uint32_t CUBE_MATRIX_SIZE_I = 256;

constexpr uint32_t L1_UINT8_BLOCK_SIZE = 131072;
constexpr int32_t UB_UINT8_BLOCK_SIZE_I = 32768;
constexpr int32_t UB_UINT8_LINE_SIZE_I = 1024;
constexpr int32_t UB_FLOAT_LINE_SIZE_I = 256;
constexpr int32_t UB_HALF_LINE_SIZE_I = 512;
constexpr uint32_t MAX_LEN_64_BYTES = 64;
constexpr uint32_t DEC_UB_UINT8_BLOCK_SIZE = 8192;

enum class CalcMode : uint8_t {
    CALC_MODE_DEFAULT = 0,
    CALC_MODE_PREFILL = 1,
};

enum class LAYOUT {
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

enum class AMLAMODE {
    NORMAL = 0,
    AMLA = 1,
    AMLA_3BUF = 2
};

template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T, const bool PAGE_ATTENTION = false,
          const bool FLASH_DECODE = false, LAYOUT LAYOUT_T = LAYOUT::BSH, const uint8_t ANTIQUANT_MODE = 0,
          const bool SHARED_PREFIX = false, LAYOUT KV_LAYOUT_T = LAYOUT::BSH, const AMLAMODE AMLA = AMLAMODE::NORMAL,
          const bool BALANCE = false, typename TILING_T = IncreFlashAttentionTilingDataV2, typename... Args>
struct IFAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    using orginalType = ORIGIN_T;
    using TilingType = TILING_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr LAYOUT layout = LAYOUT_T;
    static constexpr uint8_t antiquantMode = ANTIQUANT_MODE;
    static constexpr bool sharedPrefix = SHARED_PREFIX;
    static constexpr LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr AMLAMODE isAMla = AMLA;
    static constexpr bool isBalance = BALANCE;
};

#endif // IFA_PUBLIC_DEFINE_H
