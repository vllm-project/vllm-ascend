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
 * \file basic_block_config.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONFIG_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONFIG_H

#include "../tool_arch35.h"
#include "../../weight_quant_batch_matmul_v2_constant.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35 {

constexpr static uint16_t WEIGHT_F16_UB_NZ_STRIDE = 65;
constexpr int16_t SHIFT_FOR_BF16 = 1;

struct WqmmConfig {
    bool aTrans;
    bool bTrans;
    Mc2QuantType antiQuantType;
    bool hasAntiQuantOffset;
    Mc2QuantType quantType;
    CubeFormat weightFormat;
};

struct BasicBlockOffsetParam {
    uint64_t mL1Size;
    uint64_t kaL1Size;
    uint64_t kbL1Size;
    uint64_t nL1Size;

    uint64_t mOffset;
    uint64_t nOffset;

    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t kAlign;

    GM_ADDR yGmAddr;
};

struct VecAntiQuantConfig {
    uint64_t ubMte2BufferNum = 2;
    uint64_t ubMte2InnerSize = 512;
};

struct UbConsumeConfig {
    uint64_t ubVfBufferNum;
    uint64_t l1RequireVfComputeRealK;
    uint64_t l1RequireVfComputeRealN;
    uint64_t kWeightLowBitUbOffset;
    uint64_t nWeightLowBitUbOffset;
};

struct L1ConsumeConfig {
    uint64_t l1SplitTwoVecExternalOffset;
    uint64_t l1RealExternalLen;
};

struct UbBufferInfo {
    uint64_t ubWeightOutputHighBitBufferNum;
    uint64_t weightInputLowbitUbTotalSize;
    uint64_t highBitDataUbTotalSize;
    uint64_t antiQuantScaleUbTotalSize;
    uint64_t antiQuantScaleAfterCastUbTotalSize;
    uint64_t antiQuantOffsetUbTotalSize;
    uint64_t weightInputLowBitUbSingleBufferSize;
    uint64_t antiQuantScaleUbSingleBufferSize;
    uint64_t antiQuantScaleAfterCastUbSingleBufferSize;
    uint64_t antiQuantOffsetUbSingleBufferSize;
    uint64_t highBitDataUbSingleBufferSize;
    uint32_t antiQuantScaleMaskBufferSize;
};

template <const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr UbBufferInfo GetNzBufferInfo()
{
    return {.ubWeightOutputHighBitBufferNum = QUADRUPLE_BUFFER_NUM,
            .weightInputLowbitUbTotalSize = 112 * GetKBUnit<int8_t>(), // 112KB
            .highBitDataUbTotalSize = 128 * GetKBUnit<half>(),         // 128KB
            .antiQuantScaleUbTotalSize = 4 * GetKBUnit<half>(),        // 4KB
            .antiQuantScaleAfterCastUbTotalSize = 0,
            .antiQuantOffsetUbTotalSize = 4 * GetKBUnit<half>(), // 4KB
            .weightInputLowBitUbSingleBufferSize = 112 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleUbSingleBufferSize = 4 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleAfterCastUbSingleBufferSize = 0,
            .antiQuantOffsetUbSingleBufferSize = 4 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .highBitDataUbSingleBufferSize = 128 * GetKBUnit<half>() / QUADRUPLE_BUFFER_NUM,
            .antiQuantScaleMaskBufferSize = 0};
}

template <const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr UbBufferInfo GetS8S4NzBufferInfo()
{
    return {.ubWeightOutputHighBitBufferNum = QUADRUPLE_BUFFER_NUM,
            .weightInputLowbitUbTotalSize = 96 * GetKBUnit<int8_t>(), // 96KB
            .highBitDataUbTotalSize = 128 * GetKBUnit<int8_t>(),      // 128KB
            .antiQuantScaleUbTotalSize = 12 * GetKBUnit<half>(),      // 12KB
            .antiQuantScaleAfterCastUbTotalSize = 0,
            .antiQuantOffsetUbTotalSize = 0,
            .weightInputLowBitUbSingleBufferSize = 96 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum, // 32KB
            .antiQuantScaleUbSingleBufferSize = 12 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,      // 4KB
            .antiQuantScaleAfterCastUbSingleBufferSize = 0,
            .antiQuantOffsetUbSingleBufferSize = 0,
            .highBitDataUbSingleBufferSize = 128 * GetKBUnit<int8_t>() / QUADRUPLE_BUFFER_NUM, // 32KB
            .antiQuantScaleMaskBufferSize = 32 / sizeof(uint64_t)};                            // 32B (4个uint64)
}

template <const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr UbBufferInfo GetNdBufferInfo()
{
    return {.ubWeightOutputHighBitBufferNum = DOUBLE_BUFFER_NUM,
            .weightInputLowbitUbTotalSize = 174 * GetKBUnit<int8_t>(), // 174KB
            .highBitDataUbTotalSize = 66 * GetKBUnit<half>(),          // 66KB
            .antiQuantScaleUbTotalSize = 4 * GetKBUnit<half>(),        // 4KB
            .antiQuantScaleAfterCastUbTotalSize = 0,
            .antiQuantOffsetUbTotalSize = 4 * GetKBUnit<half>(), // 4KB
            .weightInputLowBitUbSingleBufferSize = 174 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleUbSingleBufferSize = 4 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleAfterCastUbSingleBufferSize = 0,
            .antiQuantOffsetUbSingleBufferSize = 4 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .highBitDataUbSingleBufferSize = 66 * GetKBUnit<half>() / DOUBLE_BUFFER_NUM,
            .antiQuantScaleMaskBufferSize = 0};
}

template <const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr UbBufferInfo GetMxFp4NdBufferInfo()
{
    return {.ubWeightOutputHighBitBufferNum = DOUBLE_BUFFER_NUM,
            .weightInputLowbitUbTotalSize = 128 * GetKBUnit<int8_t>(),    // 128KB
            .highBitDataUbTotalSize = 66 * GetKBUnit<half>(),             // 66KB
            .antiQuantScaleUbTotalSize = 8 * GetKBUnit<int8_t>(),         // 8KB
            .antiQuantScaleAfterCastUbTotalSize = 32 * GetKBUnit<half>(), // 32KB
            .antiQuantOffsetUbTotalSize = 0,
            .weightInputLowBitUbSingleBufferSize = 128 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleUbSingleBufferSize = 8 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleAfterCastUbSingleBufferSize = 32 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .antiQuantOffsetUbSingleBufferSize = 0,
            .highBitDataUbSingleBufferSize = 66 * GetKBUnit<half>() / DOUBLE_BUFFER_NUM,
            .antiQuantScaleMaskBufferSize = 0};
}

template <const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr UbBufferInfo GetMxFp4NzBufferInfo()
{
    return {.ubWeightOutputHighBitBufferNum = QUADRUPLE_BUFFER_NUM,
            .weightInputLowbitUbTotalSize = 64 * GetKBUnit<int8_t>(),     // 64KB
            .highBitDataUbTotalSize = 128 * GetKBUnit<half>(),            // 128KB
            .antiQuantScaleUbTotalSize = 8 * GetKBUnit<int8_t>(),         // 8KB
            .antiQuantScaleAfterCastUbTotalSize = 16 * GetKBUnit<half>(), // 16KB
            .antiQuantOffsetUbTotalSize = 0,
            .weightInputLowBitUbSingleBufferSize = 64 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleUbSingleBufferSize = 8 * GetKBUnit<int8_t>() / vecConfig.ubMte2BufferNum,
            .antiQuantScaleAfterCastUbSingleBufferSize = 16 * GetKBUnit<half>() / vecConfig.ubMte2BufferNum,
            .antiQuantOffsetUbSingleBufferSize = 0,
            .highBitDataUbSingleBufferSize = 128 * GetKBUnit<half>() / QUADRUPLE_BUFFER_NUM,
            .antiQuantScaleMaskBufferSize = 0};
}

template <typename xType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr UbBufferInfo GetBufferConfig()
{
    if constexpr (wqmmConfig.antiQuantType == Mc2QuantType::MX) {
        if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ) {
            return GetMxFp4NdBufferInfo<vecConfig>();
        } else {
            return GetMxFp4NzBufferInfo<vecConfig>();
        }
    }

    if constexpr (wqmmConfig.weightFormat == CubeFormat::ND) {
        return GetNdBufferInfo<vecConfig>();
    } else {
        if constexpr (IsSameType<xType, int8_t>::value) {
            return GetS8S4NzBufferInfo<vecConfig>();
        } else {
            return GetNzBufferInfo<vecConfig>();
        }
    }
}

struct VfConfig {
    uint64_t vfNStandardLen;
    uint64_t vfKStandardLen;
};

template <typename xType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ constexpr VfConfig GetVfConfig()
{
    if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ && wqmmConfig.bTrans) {
        return {.vfNStandardLen = 64, .vfKStandardLen = 256};
    } else if constexpr (wqmmConfig.weightFormat != CubeFormat::NZ && !wqmmConfig.bTrans) {
        return {.vfNStandardLen = 256, .vfKStandardLen = 64};
    } else {
        // NZ transB=False
        if constexpr (wqmmConfig.antiQuantType == Mc2QuantType::MX) {
            return {.vfNStandardLen = 32 * GetKBUnit<half>() / vecConfig.ubMte2InnerSize,
                    .vfKStandardLen = vecConfig.ubMte2InnerSize};
        } else if constexpr (IsSameType<xType, int8_t>::value) {
            return {.vfNStandardLen = 64, .vfKStandardLen = 512};
        } else {
            return {.vfNStandardLen = 64, .vfKStandardLen = 256};
        }
    }
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONFIG_H
