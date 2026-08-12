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
 * \file qbmm_perblock_api_param_utils.h
 * \brief
 */

#ifndef MC2_QBMM_PERBLOCK_API_PARAM_UTILS_H
#define MC2_QBMM_PERBLOCK_API_PARAM_UTILS_H

#include "qbmm_asw_block.h"
#include "../quant_batch_matmul_v3_base.h"

#define MATMUL_PERBLOCK_CLASS_TEM_PARAMS                                                                               \
    template <class aType, class bType, class scaleType, class biasType, class ptScaleType, class cType,               \
              CubeFormat aFormat, CubeFormat bFormat, CubeFormat cFormat, bool aTrans, bool bTrans, class l0cDtype,    \
              class blockType>

#define MATMUL_PERBLOCK_FUNC_PARAMS                                                                                    \
    aType, bType, scaleType, biasType, ptScaleType, cType, aFormat, bFormat, cFormat, aTrans, bTrans, l0cDtype,        \
        blockType

namespace Mc2QuantBatchMatmulV3 {
MATMUL_PERBLOCK_CLASS_TEM_PARAMS
class MatMulCommonParam {
public:
    __aicore__ inline MatMulCommonParam(){};
    __aicore__ inline void Init(blockType &block, const TCubeTiling &tilingData);
    __aicore__ inline uint64_t CalcAGMOffsetInnerLoop(const uint64_t &mOffset, const uint64_t &kOffset);
    __aicore__ inline uint64_t CalcBGMOffsetInnerLoop(const uint64_t &nOffset, const uint64_t &kOffset);
    __aicore__ inline void CalNd2NzParamA(AscendC::Nd2NzParams &nd2nzParam, const bool &isTailAL1);
    __aicore__ inline void CalNd2NzParamB(AscendC::Nd2NzParams &nd2nzParam, const bool &isTailBL1);
    __aicore__ inline uint32_t CalcAL1Offset(const uint64_t &mAL1Offset, const uint64_t &kAL1Offset,
                                             const bool &isKTail);
    __aicore__ inline uint32_t CalcBL1Offset(const uint64_t &nBL1Offset, const uint64_t &kBL1Offset,
                                             const bool &isKTail);
    __aicore__ inline void LoadData2dParamsA(LoadData2DParamsV2 &loadData2dParams, const uint64_t &kOffset,
                                             const bool &isTailAL1);
    __aicore__ inline void LoadData2dParamsB(LoadData2DParamsV2 &loadData2dParams, const uint64_t &kOffset,
                                             const bool &isTailBL1);

protected:
    blockType *block_;
    const TCubeTiling *matmulTiling_;
    uint64_t kA1_;
    uint64_t kB1_;
    uint64_t mA1C0_;
    uint64_t nB1C0_;
    uint64_t kB1C0_;
    uint64_t kA1C0_;
    uint64_t kA1Tail_;
    uint64_t kB1Tail_;
};

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulCommonParam<MATMUL_PERBLOCK_FUNC_PARAMS>::Init(blockType &block,
                                                                            const TCubeTiling &tilingData)
{
    matmulTiling_ = &tilingData;
    block_ = &block;
    if constexpr (bTrans) {
        nB1C0_ = BMM_BLOCK_NUM;
        kB1C0_ = K0_INT8;
    } else {
        nB1C0_ = K0_INT8;
        kB1C0_ = BMM_BLOCK_NUM;
    }
    if constexpr (aTrans) {
        kA1C0_ = BMM_BLOCK_NUM;
        mA1C0_ = K0_INT8;
    } else {
        kA1C0_ = K0_INT8;
        mA1C0_ = BMM_BLOCK_NUM;
    }
    kA1_ = matmulTiling_->baseK * matmulTiling_->stepKa;
    kB1_ = matmulTiling_->baseK * matmulTiling_->stepKb;
    kB1Tail_ = matmulTiling_->Kb % kB1_ == 0 ? kB1_ : matmulTiling_->Kb % kB1_;
    kA1Tail_ = matmulTiling_->Ka % kA1_ == 0 ? kA1_ : matmulTiling_->Ka % kA1_;
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline uint64_t
MatMulCommonParam<MATMUL_PERBLOCK_FUNC_PARAMS>::CalcAGMOffsetInnerLoop(const uint64_t &mOffset, const uint64_t &kOffset)
{
    uint64_t offsetA = block_->offset_.offsetA;
    if constexpr (aTrans) {
        offsetA += kOffset * matmulTiling_->M + mOffset;
    } else {
        offsetA += kOffset + mOffset * matmulTiling_->Ka;
    }
    return offsetA;
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline uint64_t
MatMulCommonParam<MATMUL_PERBLOCK_FUNC_PARAMS>::CalcBGMOffsetInnerLoop(const uint64_t &nOffset, const uint64_t &kOffset)
{
    uint64_t offsetB = block_->offset_.offsetB;
    if constexpr (bTrans) {
        offsetB += kOffset + nOffset * matmulTiling_->Kb;
    } else {
        offsetB += kOffset * matmulTiling_->N + nOffset;
    }
    return offsetB;
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulCommonParam<MATMUL_PERBLOCK_FUNC_PARAMS>::CalNd2NzParamA(AscendC::Nd2NzParams &nd2nzParam,
                                                                                      const bool &isTailAL1)
{
    uint64_t currentK = isTailAL1 ? kA1Tail_ : kA1_;
    nd2nzParam.ndNum = 1;
    nd2nzParam.srcNdMatrixStride = 0;
    nd2nzParam.dstNzNStride = 1;
    nd2nzParam.dstNzMatrixStride = 0;
    if constexpr (aTrans) {
        nd2nzParam.nValue = currentK;
        nd2nzParam.dValue = block_->params_.singleCoreM;
        nd2nzParam.srcDValue = matmulTiling_->M;
        nd2nzParam.dstNzC0Stride = DequantBmm::Align(currentK, static_cast<uint64_t>(DATA_BLOCK)); // 对齐到32
    } else {
        nd2nzParam.nValue = block_->params_.singleCoreM;
        nd2nzParam.dValue = currentK;
        nd2nzParam.srcDValue = matmulTiling_->Ka;
        nd2nzParam.dstNzC0Stride = DequantBmm::Align(block_->params_.singleCoreM, static_cast<uint64_t>(k0_FLOAT16));
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulCommonParam<MATMUL_PERBLOCK_FUNC_PARAMS>::CalNd2NzParamB(AscendC::Nd2NzParams &nd2nzParam,
                                                                                      const bool &isTailBL1)
{
    uint64_t currentK = isTailBL1 ? kB1Tail_ : kB1_;
    nd2nzParam.ndNum = 1;
    nd2nzParam.srcNdMatrixStride = 0;
    nd2nzParam.dstNzNStride = 1;
    nd2nzParam.dstNzMatrixStride = 0;
    if constexpr (bTrans) {
        nd2nzParam.nValue = block_->params_.singleCoreN;
        nd2nzParam.dValue = currentK;
        nd2nzParam.srcDValue = matmulTiling_->Kb;
        nd2nzParam.dstNzC0Stride = DequantBmm::Align(block_->params_.singleCoreN, static_cast<uint64_t>(k0_FLOAT16));
    } else {
        nd2nzParam.nValue = currentK;
        nd2nzParam.dValue = block_->params_.singleCoreN;
        nd2nzParam.srcDValue = matmulTiling_->N;
        nd2nzParam.dstNzC0Stride = DequantBmm::Align(currentK, static_cast<uint64_t>(DATA_BLOCK)); // 对齐到32
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline uint32_t MatMulCommonParam<MATMUL_PERBLOCK_FUNC_PARAMS>::CalcAL1Offset(const uint64_t &mAL1Offset,
                                                                                         const uint64_t &kAL1Offset,
                                                                                         const bool &isKTail)
{
    uint64_t kAL1 = isKTail ? kA1Tail_ : kA1_;
    kAL1 = DequantBmm::Align(kAL1, static_cast<uint64_t>(K0_INT8));
    uint64_t mAL1 = DequantBmm::Align(block_->params_.singleCoreM, mA1C0_);
    if constexpr (aTrans) {
        // (m1, k1, k0, m0)
        return DequantBmm::Align(mAL1Offset, mA1C0_) * kAL1 + kAL1Offset * mA1C0_;
    } else {
        // (k1, m1, m0, k0)
        return DequantBmm::Align(kAL1Offset, kA1C0_) * mAL1 + mAL1Offset * kA1C0_;
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline uint32_t MatMulCommonParam<MATMUL_PERBLOCK_FUNC_PARAMS>::CalcBL1Offset(const uint64_t &nBL1Offset,
                                                                                         const uint64_t &kBL1Offset,
                                                                                         const bool &isKTail)
{
    uint64_t kBL1 = isKTail ? kB1Tail_ : kB1_;
    kBL1 = DequantBmm::Align(kBL1, static_cast<uint64_t>(K0_INT8));
    uint64_t nBL1 = DequantBmm::Align(block_->params_.singleCoreN, nB1C0_);
    if constexpr (bTrans) {
        // (k1, n1, n0, k0)
        return DequantBmm::Align(kBL1Offset, kB1C0_) * nBL1 + nBL1Offset * kB1C0_;
    } else {
        // (n1, k1, k0, n0)
        return DequantBmm::Align(nBL1Offset, nB1C0_) * kBL1 + kBL1Offset * nB1C0_;
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void
MatMulCommonParam<MATMUL_PERBLOCK_FUNC_PARAMS>::LoadData2dParamsA(LoadData2DParamsV2 &loadData2dParams,
                                                                  const uint64_t &kOffset, const bool &isTailAL1)
{
    uint64_t currM = DequantBmm::Min(block_->params_.singleCoreM, static_cast<uint64_t>(matmulTiling_->baseM));
    uint64_t currK = DequantBmm::Min(static_cast<uint64_t>(matmulTiling_->Ka) - kOffset,
                                     static_cast<uint64_t>(matmulTiling_->baseK));
    if constexpr (aTrans) {
        // 转置场景对int8的输入需要2个16*32的分型做转置
        loadData2dParams.mStep = DequantBmm::Align(DequantBmm::CeilDiv(currK, static_cast<uint64_t>(k0_FLOAT16)), 2UL);
        loadData2dParams.kStep = DequantBmm::CeilDiv(currM, static_cast<uint64_t>(K0_INT8));
        loadData2dParams.srcStride =
            DequantBmm::Align(DequantBmm::CeilDiv(isTailAL1 ? kA1Tail_ : kA1_, k0_FLOAT16), 2UL);
        loadData2dParams.dstStride =
            DequantBmm::Align(DequantBmm::CeilDiv(currM, static_cast<uint64_t>(k0_FLOAT16)), 2UL);
        loadData2dParams.ifTranspose = true;
    } else {
        loadData2dParams.mStep = DequantBmm::CeilDiv(currM, static_cast<uint64_t>(k0_FLOAT16));
        loadData2dParams.kStep = DequantBmm::CeilDiv(currK, static_cast<uint64_t>(K0_INT8));
        loadData2dParams.srcStride =
            DequantBmm::CeilDiv(currM * matmulTiling_->stepM, static_cast<uint64_t>(k0_FLOAT16));
        loadData2dParams.dstStride = DequantBmm::CeilDiv(currM, static_cast<uint64_t>(k0_FLOAT16));
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void
MatMulCommonParam<MATMUL_PERBLOCK_FUNC_PARAMS>::LoadData2dParamsB(LoadData2DParamsV2 &loadData2dParams,
                                                                  const uint64_t &kOffset, const bool &isTailBL1)
{
    uint64_t currN = DequantBmm::Min(block_->params_.singleCoreN, static_cast<uint64_t>(matmulTiling_->baseN));
    uint64_t currK = DequantBmm::Min(matmulTiling_->Kb - kOffset, static_cast<uint64_t>(matmulTiling_->baseK));
    if constexpr (bTrans) {
        loadData2dParams.mStep = DequantBmm::CeilDiv(currN, static_cast<uint64_t>(k0_FLOAT16));
        loadData2dParams.kStep = DequantBmm::CeilDiv(currK, static_cast<uint64_t>(K0_INT8));
        loadData2dParams.srcStride =
            DequantBmm::CeilDiv(currN * matmulTiling_->stepN, static_cast<uint64_t>(k0_FLOAT16));
        loadData2dParams.dstStride = DequantBmm::CeilDiv(currN, static_cast<uint64_t>(k0_FLOAT16));
    } else {
        loadData2dParams.ifTranspose = true;
        // 转置场景对int8的输入需要2个16*32的分型做转置
        loadData2dParams.mStep = DequantBmm::Align(DequantBmm::CeilDiv(currK, static_cast<uint64_t>(k0_FLOAT16)), 2UL);
        loadData2dParams.kStep = DequantBmm::CeilDiv(currN, static_cast<uint64_t>(K0_INT8));
        loadData2dParams.srcStride =
            DequantBmm::Align(DequantBmm::CeilDiv(isTailBL1 ? kB1Tail_ : kB1_, k0_FLOAT16), 2UL);
        loadData2dParams.dstStride =
            DequantBmm::Align(DequantBmm::CeilDiv(currN, static_cast<uint64_t>(k0_FLOAT16)), 2UL);
    }
}
} // namespace Mc2QuantBatchMatmulV3
#endif // QBMM_PERBLOCK_API_PARAM_UTILS_H
