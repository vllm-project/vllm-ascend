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
 * \file indices_sort_utils.h
 * \brief indices_sort_utils
 */
#ifndef ASCENDC_SCATTER_COMMON_INDICES_SORT_UTILS_H_
#define ASCENDC_SCATTER_COMMON_INDICES_SORT_UTILS_H_

#include "op_kernel/math_util.h"

#define LAST_DIM_SIZE_LIMIT 512
#define INDICES_BUCKETS_SIZE 256
#define FNV_PRIME_B32 0x01000193UL
#define FNV_PRIME_B64 0x0100000001B3UL
#define FNV_OFFSET_BIASIS_B32 0x811C9DC5UL
#define FNV_OFFSET_BIASIS_B64 0xCBF29CE484222325UL

using namespace AscendC;
static constexpr MicroAPI::CastTrait castTraitInt16ToFp32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename INDICE_TYPE>
__simd_vf__ inline void IndexStatisticInt32Vf(__ubuf__ uint32_t* srcM, __ubuf__ float* dstLocalAddr,
                                              int32_t lastDimShift, uint16_t mainLoop, uint32_t tailNum,
                                              uint16_t tailLoop)
{
    using namespace AscendC::MicroAPI;
    MaskReg patAllB32 = CreateMask<uint32_t, MaskPattern::ALL>();
    MaskReg patAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg patAllB8 = CreateMask<uint8_t, MaskPattern::ALL>();

    RegTensor<float> maxCntFp32;
    RegTensor<uint16_t> histVector0;
    RegTensor<uint16_t> histVector1;
    Duplicate(histVector0, (uint16_t)0);
    Duplicate(histVector1, (uint16_t)0);

    RegTensor<uint32_t> xorOffset;
    Duplicate(xorOffset, FNV_OFFSET_BIASIS_B32);
    RegTensor<int32_t> lastDimShiftAmount;
    Duplicate(lastDimShiftAmount, lastDimShift);

    for (uint16_t i = 0; i < mainLoop; i++) {
        RegTensor<uint32_t> vectorIndex0, vectorIndex1, vectorIndex2, vectorIndex3;
        RegTensor<uint16_t> vectorB16Tmp0, vectorB16Tmp1, vectorB16Tmp2, vectorB16Tmp3;
        RegTensor<uint8_t> vectorB8Hash0, vectorB8Hash1;

        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vectorIndex0, srcM, 64);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vectorIndex1, srcM, 64);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vectorIndex2, srcM, 64);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vectorIndex3, srcM, 64);

        ShiftRight(vectorIndex0, vectorIndex0, lastDimShiftAmount, patAllB32);
        ShiftRight(vectorIndex1, vectorIndex1, lastDimShiftAmount, patAllB32);
        ShiftRight(vectorIndex2, vectorIndex2, lastDimShiftAmount, patAllB32);
        ShiftRight(vectorIndex3, vectorIndex3, lastDimShiftAmount, patAllB32);

        Xor(vectorIndex0, vectorIndex0, xorOffset, patAllB32);
        Xor(vectorIndex1, vectorIndex1, xorOffset, patAllB32);
        Xor(vectorIndex2, vectorIndex2, xorOffset, patAllB32);
        Xor(vectorIndex3, vectorIndex3, xorOffset, patAllB32);

        Muls(vectorIndex0, vectorIndex0, FNV_PRIME_B32, patAllB32);
        Muls(vectorIndex1, vectorIndex1, FNV_PRIME_B32, patAllB32);
        Muls(vectorIndex2, vectorIndex2, FNV_PRIME_B32, patAllB32);
        Muls(vectorIndex3, vectorIndex3, FNV_PRIME_B32, patAllB32);

        DeInterleave(vectorB16Tmp0, vectorB16Tmp1, (RegTensor<uint16_t>&)vectorIndex0,
                     (RegTensor<uint16_t>&)vectorIndex1);
        DeInterleave(vectorB16Tmp2, vectorB16Tmp3, (RegTensor<uint16_t>&)vectorIndex2,
                     (RegTensor<uint16_t>&)vectorIndex3);
        DeInterleave(vectorB8Hash0, vectorB8Hash1, (RegTensor<uint8_t>&)vectorB16Tmp0,
                     (RegTensor<uint8_t>&)vectorB16Tmp2);

        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(histVector0, vectorB8Hash0,
                                                                                          patAllB8);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::FREQUENCY>(histVector1, vectorB8Hash0,
                                                                                          patAllB8);
    }

    for (uint16_t i = 0; i < tailLoop; i++) {
        MaskReg maskReg = UpdateMask<uint32_t>(tailNum);
        RegTensor<uint32_t> vectorIndex0;
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vectorIndex0, srcM, 64);
        ShiftRight(vectorIndex0, vectorIndex0, lastDimShiftAmount, patAllB32);
        Xor(vectorIndex0, vectorIndex0, xorOffset, patAllB32);
        Muls(vectorIndex0, vectorIndex0, FNV_PRIME_B32, patAllB32);

        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(
            histVector0, (RegTensor<uint8_t>&)vectorIndex0, maskReg);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::FREQUENCY>(
            histVector1, (RegTensor<uint8_t>&)vectorIndex0, maskReg);
    }

    Max(histVector0, histVector0, histVector1, patAllB16);
    ReduceMax(histVector0, histVector0, patAllB16);

    Cast<float, int16_t, castTraitInt16ToFp32>(maxCntFp32, (RegTensor<int16_t>&)histVector0, patAllB16);
    DataCopy<float, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_FIRST_ELEMENT_B32>(dstLocalAddr, maxCntFp32, 1,
                                                                                      patAllB32);
}

template <typename INDICE_TYPE>
__simd_vf__ inline void IndexStatisticInt64Vf(__ubuf__ uint64_t* srcM, __ubuf__ float* dstLocalAddr,
                                              int64_t lastDimShift, uint32_t dataLen, uint16_t loopSize)
{
    using namespace AscendC::MicroAPI;
    MaskReg patAllB32 = CreateMask<uint32_t, MaskPattern::ALL>();
    MaskReg patAllB16 = CreateMask<uint16_t, MaskPattern::ALL>();

    RegTensor<float> maxCntFp32;
    RegTensor<uint16_t> histVector0, histVector1, maxValue;
    Duplicate(histVector0, (uint16_t)0);
    Duplicate(histVector1, (uint16_t)0);

    RegTensor<uint64_t> xorOffset;
    RegTensor<int64_t> lastDimShiftAmount;
    Duplicate(xorOffset, FNV_OFFSET_BIASIS_B64);
    Duplicate(lastDimShiftAmount, lastDimShift);

    RegTensor<uint64_t> vectorIndex0;
    for (uint16_t i = 0; i < loopSize; i++) {
        MaskReg maskReg = UpdateMask<uint64_t>(dataLen);
        DataCopy(vectorIndex0, srcM);
        ShiftRight(vectorIndex0, vectorIndex0, lastDimShiftAmount, maskReg);
        Xor(vectorIndex0, vectorIndex0, xorOffset, maskReg);
        Muls(vectorIndex0, vectorIndex0, FNV_PRIME_B64, maskReg);

        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(
            histVector0, (RegTensor<uint8_t>&)vectorIndex0, maskReg);
        Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::FREQUENCY>(
            histVector1, (RegTensor<uint8_t>&)vectorIndex0, maskReg);
        Max(maxValue, histVector0, histVector1, patAllB16);
    }
    ReduceMax(maxValue, maxValue, patAllB16);
    Cast<float, int16_t, castTraitInt16ToFp32>(maxCntFp32, (RegTensor<int16_t>&)maxValue, patAllB16);
    DataCopy<float, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_FIRST_ELEMENT_B32>(dstLocalAddr, maxCntFp32, 1,
                                                                                      patAllB32);
}

/*
 * 用于计算
 */
template <typename INDICE_TYPE>
__aicore__ void IndexStatisticInt32(LocalTensor<INDICE_TYPE>& srcLocal, LocalTensor<float>& dstLocal, float& maxScore,
                                    int64_t rowLen, int64_t lastDim)
{
    __local_mem__ uint32_t* srcLocalAddr = (__local_mem__ uint32_t*)srcLocal.GetPhyAddr();
    __local_mem__ uint32_t* srcM = srcLocalAddr;
    __local_mem__ float* dstLocalAddr = (__local_mem__ float*)dstLocal.GetPhyAddr();

    uint64_t lastDimSize = lastDim * sizeof(INDICE_TYPE);
    int32_t lastDimShift = 0;
    if (lastDimSize < LAST_DIM_SIZE_LIMIT) {
        uint64_t divisor = LAST_DIM_SIZE_LIMIT / lastDimSize;
        auto lz = 64 - ScalarCountLeadingZero(divisor);
        auto bc1 = ScalarGetCountOfValue<1>(divisor);
        lastDimShift = (bc1 != 1) ? lz : lz - 1;
    }
    uint16_t mainLoop = rowLen / INDICES_BUCKETS_SIZE;
    uint32_t tailNum = rowLen % INDICES_BUCKETS_SIZE;
    uint16_t tailLoop = Ops::Base::CeilDiv(tailNum, static_cast<uint32_t>(64));
    IndexStatisticInt32Vf<INDICE_TYPE>((__ubuf__ uint32_t*)srcM, (__ubuf__ float*)dstLocalAddr, lastDimShift, mainLoop,
                                       tailNum, tailLoop);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    maxScore = dstLocalAddr[0] / rowLen;
}

template <typename INDICE_TYPE>
__aicore__ void IndexStatisticInt64(LocalTensor<INDICE_TYPE>& srcLocal, LocalTensor<float>& dstLocal, float& maxScore,
                                    int64_t rowLen, int64_t lastDim)
{
    __local_mem__ uint64_t* srcLocalAddr = (__local_mem__ uint64_t*)srcLocal.GetPhyAddr();
    __local_mem__ uint64_t* srcM = srcLocalAddr;
    __local_mem__ float* dstLocalAddr = (__local_mem__ float*)dstLocal.GetPhyAddr();

    int32_t lastDimSize = lastDim * sizeof(INDICE_TYPE);
    int64_t lastDimShift = 0;
    if (lastDimSize < LAST_DIM_SIZE_LIMIT) {
        uint64_t divisor = LAST_DIM_SIZE_LIMIT / lastDimSize;
        auto lz = 64 - ScalarCountLeadingZero(divisor);
        auto bc1 = ScalarGetCountOfValue<1>(divisor);
        lastDimShift = (bc1 != 1) ? lz : lz - 1;
    }
    uint32_t dataLen = static_cast<uint32_t>(rowLen);
    uint16_t loopSize = Ops::Base::CeilDiv(rowLen, 32L);
    IndexStatisticInt64Vf<INDICE_TYPE>((__ubuf__ uint64_t*)srcM, (__ubuf__ float*)dstLocalAddr, lastDimShift, dataLen,
                                       loopSize);

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    maxScore = dstLocalAddr[0] / rowLen;
}

#endif