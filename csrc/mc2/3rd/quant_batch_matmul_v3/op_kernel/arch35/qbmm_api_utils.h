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
 * \file qbmm_api_utils.h
 * \brief
 */
#ifndef MC2_QBMM_API_UTILS_H
#define MC2_QBMM_API_UTILS_H

#include "../quant_batch_matmul_v3_base.h"
#include "qbmm_asw_block.h"


namespace Mc2QuantBatchMatmulV3 {

using namespace AscendC;
using namespace matmul;

template <typename T, bool trans>
__aicore__ inline void CopyInA1(const Mc2QuantBmmAswBlock &block, uint32_t blockIdx, bool isMultiCore,
                                LocalTensor<T> &al1Local, const GlobalTensor<T> &aGlobal)
{
    auto &matmulTilingData = *block.tilingData_;
    uint64_t shapeM = matmulTilingData.matmulTiling.M;
    uint64_t shapeK = matmulTilingData.matmulTiling.Ka;
    uint64_t mCntIdx = blockIdx % block.params_.mCnt;
    uint64_t singleCoreM = isMultiCore && mCntIdx == block.params_.mCnt - 1UL ?
                               block.params_.mBaseTail :
                               static_cast<uint64_t>(matmulTilingData.matmulTiling.singleCoreM);
    uint64_t nDim = singleCoreM;
    uint64_t dDim = shapeK;
    uint64_t offsetA = mCntIdx * matmulTilingData.matmulTiling.singleCoreM * shapeK;
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;

    if constexpr (trans) {
        nDim = shapeK;
        dDim = singleCoreM;
        offsetA = mCntIdx * matmulTilingData.matmulTiling.singleCoreM;
        if constexpr (DequantBmm::IsFp4<T>()) {
            nd2nzParams.srcDValue = DequantBmm::CeilDiv(shapeM, 2); // 2: 将2个4bits合成1个8bits搬运
        } else {
            nd2nzParams.srcDValue = shapeM;
        }
    } else {
        if constexpr (DequantBmm::IsFp4<T>()) {
            nd2nzParams.srcDValue = DequantBmm::CeilDiv(shapeK, 2); // 2: 将2个4bits合成1个8bits搬运
        } else {
            nd2nzParams.srcDValue = shapeK;
        }
    }

    nd2nzParams.nValue = nDim;
    if constexpr (DequantBmm::IsFp4<T>()) {
        nd2nzParams.dValue = DequantBmm::CeilDiv(dDim, 2); // 2: 将2个4bits合成1个8bits搬运
    } else {
        nd2nzParams.dValue = dDim;
    }
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = DequantBmm::Align(nDim, BMM_BLOCK_NUM);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    if constexpr (DequantBmm::IsFp4<T>()) {
        GlobalTensor<int8_t> aGlobalInt8;
        aGlobalInt8.SetGlobalBuffer(((__gm__ int8_t *)(aGlobal.GetPhyAddr())), (nDim * dDim) >> 1);
        auto al1LocalImpl = al1Local.template ReinterpretCast<int8_t>();
        DataCopy(al1LocalImpl, aGlobalInt8[offsetA >> 1], nd2nzParams);
    } else {
        DataCopy(al1Local, aGlobal[offsetA], nd2nzParams);
    }
}

template <typename T, bool trans>
__aicore__ inline void CopyInScaleA(const Mc2QuantBmmAswBlock &block, uint32_t blockId, bool isMultiCore,
                                    LocalTensor<T> &scaleAl1Local, const GlobalTensor<T> &scaleAGlobal)
{
    auto &multiTilingData = *block.tilingData_;
    uint64_t shapeM = multiTilingData.matmulTiling.M;
    uint64_t shapeK =
        DequantBmm::Align(DequantBmm::CeilDiv(multiTilingData.matmulTiling.Ka, MXFP_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);
    uint64_t mCntIdx = blockId % block.params_.mCnt;
    uint64_t singleCoreM = isMultiCore && mCntIdx == block.params_.mCnt - 1UL ?
                               block.params_.mBaseTail :
                               static_cast<uint64_t>(multiTilingData.matmulTiling.singleCoreM);

    uint64_t nDim = singleCoreM;
    uint64_t dDim = shapeK;
    uint64_t offsetScaleA = 0;
    if constexpr (trans) {
        nDim = shapeK;
        dDim = singleCoreM;
        offsetScaleA = mCntIdx * multiTilingData.matmulTiling.singleCoreM;
    } else {
        offsetScaleA = mCntIdx * multiTilingData.matmulTiling.singleCoreM * shapeK;
    }

    Dn2NzParams dn2nzParams;
    dn2nzParams.dnNum = 1;
    dn2nzParams.dValue = nDim;
    dn2nzParams.nValue = DequantBmm::CeilDiv(dDim, 2); // 将两个float8e8m0合成一个half类型，一起搬运
    dn2nzParams.srcDnMatrixStride = 0;
    dn2nzParams.srcDValue = shapeK >> 1;
    dn2nzParams.dstNzC0Stride = DequantBmm::CeilDiv(dDim, 2); // 将两个float8e8m0合成一个half类型，一起搬运
    dn2nzParams.dstNzNStride = 1;
    dn2nzParams.dstNzMatrixStride = 0;

    GlobalTensor<half> aScaleGlobalB16;
    aScaleGlobalB16.SetGlobalBuffer(((__gm__ half *)(scaleAGlobal.GetPhyAddr())), (nDim * dDim) >> 1);
    auto scaleAl1LocalImpl = scaleAl1Local.template ReinterpretCast<half>();
    DataCopy(scaleAl1LocalImpl, aScaleGlobalB16[offsetScaleA >> 1], dn2nzParams);
}

template <typename T>
__aicore__ inline void ProcessWithBatch(Mc2QuantBmmAswBlock &block, T &object)
{
    uint64_t batchC3C4 = static_cast<uint64_t>(block.tilingData_->params.batchC3) * block.tilingData_->params.batchC4;
    uint64_t batchC2C3C4 = block.tilingData_->params.batchC2 * batchC3C4;
    uint64_t batchB3B4 = static_cast<uint64_t>(block.tilingData_->params.batchB3) * block.tilingData_->params.batchB4;
    uint64_t batchB2B3B4 = block.tilingData_->params.batchB2 * batchB3B4;
    uint64_t batchA3A4 = static_cast<uint64_t>(block.tilingData_->params.batchA3) * block.tilingData_->params.batchA4;
    uint64_t batchA2A3A4 = block.tilingData_->params.batchA2 * batchA3A4;
    uint32_t multiA1C1 = block.tilingData_->params.batchA1 / block.tilingData_->params.batchC1;
    uint32_t multiA2C2 = block.tilingData_->params.batchA2 / block.tilingData_->params.batchC2;
    uint32_t multiA3C3 = block.tilingData_->params.batchA3 / block.tilingData_->params.batchC3;
    uint32_t multiA4C4 = block.tilingData_->params.batchA4 / block.tilingData_->params.batchC4;
    uint32_t multiB1C1 = block.tilingData_->params.batchB1 / block.tilingData_->params.batchC1;
    uint32_t multiB2C2 = block.tilingData_->params.batchB2 / block.tilingData_->params.batchC2;
    uint32_t multiB3C3 = block.tilingData_->params.batchB3 / block.tilingData_->params.batchC3;
    uint32_t multiB4C4 = block.tilingData_->params.batchB4 / block.tilingData_->params.batchC4;
    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    for (uint64_t b1Index = 0; b1Index < block.tilingData_->params.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < block.tilingData_->params.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < block.tilingData_->params.batchC3; ++b3Index) {
                block.offset_.batchCOffset = batchC3Offset;
                block.offset_.batchAOffset = batchA3Offset;
                block.offset_.batchBOffset = batchB3Offset;
                for (uint64_t b4Index = 0; b4Index < block.tilingData_->params.batchC4; ++b4Index) {
                    block.ResetAddressOffsets();
                    object.ProcessWithoutBatch();
                    block.offset_.batchCOffset += 1;
                    block.offset_.batchAOffset += multiA4C4;
                    block.offset_.batchBOffset += multiB4C4;
                }
                batchC3Offset += block.tilingData_->params.batchC4;
                batchA3Offset += block.tilingData_->params.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += block.tilingData_->params.batchB4 * static_cast<uint64_t>(multiB3C3);
            }
            batchC2Offset += batchC3C4;
            batchA2Offset += batchA3A4 * multiA2C2;
            batchB2Offset += batchB3B4 * multiB2C2;
        }
        batchC1Offset += batchC2C3C4;
        batchA1Offset += batchA2A3A4 * multiA1C1;
        batchB1Offset += batchB2B3B4 * multiB1C1;
    }
}

} // namespace Mc2QuantBatchMatmulV3

#endif // QBMM_API_UTILS_H
