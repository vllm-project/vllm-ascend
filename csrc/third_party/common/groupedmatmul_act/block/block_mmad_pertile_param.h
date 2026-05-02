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
 * \file block_mmad_pertile_param.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_MMAD_PERTILE_PARAM_H
#define MATMUL_BLOCK_BLOCK_MMAD_PERTILE_PARAM_H

#include "kernel_operator.h"
#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/grouped_matmul_constant.h"

namespace Act {
namespace Gemm {
namespace Block {

using namespace Act::Gemm::GroupedMatmul;

template <bool aTrans, bool bTrans>
class MatMulCommonParam {
public:
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;              // m,n,k
    using TupleTileShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>; // m,n,ka,kb

public:
    __aicore__ inline MatMulCommonParam(){};
    __aicore__ inline void Init(const TupleShape& l0Shape, const TupleTileShape& tileL12L0);
    __aicore__ inline void UpdateForNextGroup(const TupleShape& problemShape);
    __aicore__ inline void UpdateNextBlockParams(const TupleShape& actualSingleShape);
    __aicore__ inline uint64_t CalcAGMOffsetInnerLoop(uint64_t mOffset, uint64_t kOffset);
    __aicore__ inline uint64_t CalcBGMOffsetInnerLoop(uint64_t nOffset, uint64_t kOffset);
    __aicore__ inline void CalNd2NzParamA(AscendC::Nd2NzParams& nd2nzParam, bool isTailAL1);
    __aicore__ inline void CalNd2NzParamB(AscendC::Nd2NzParams& nd2nzParam, bool isTailBL1);
    __aicore__ inline uint32_t CalcAL1Offset(uint64_t mAL1Offset, uint64_t kAL1Offset, bool isKTail);
    __aicore__ inline uint32_t CalcBL1Offset(uint64_t nBL1Offset, uint64_t kBL1Offset, bool isKTail);
    __aicore__ inline void LoadData2dParamsA(AscendC::LoadData2DParamsV2& loadData2dParams, uint64_t kOffset,
                                             bool isTailAL1);
    __aicore__ inline void LoadData2dParamsB(AscendC::LoadData2DParamsV2& loadData2dParams, uint64_t kOffset,
                                             bool isTailBL1);

protected:
    uint64_t kA1_;
    uint64_t kB1_;
    uint64_t mA1C0_;
    uint64_t nB1C0_;
    uint64_t kB1C0_;
    uint64_t kA1C0_;
    uint64_t kA1Tail_;
    uint64_t kB1Tail_;

private:
    TupleShape problemShape_;
    TupleShape actualSingleShape_;
    TupleShape l0Shape_;
    TupleTileShape tileL12L0_;
};

template <bool aTrans, bool bTrans>
__aicore__ inline void MatMulCommonParam<aTrans, bTrans>::Init(const TupleShape& l0Shape,
                                                               const TupleTileShape& tileL12L0)
{
    l0Shape_ = l0Shape;
    tileL12L0_ = tileL12L0;

    if constexpr (bTrans) {
        nB1C0_ = AscendC::BLOCK_CUBE;
        kB1C0_ = K0_B8;
    } else {
        nB1C0_ = K0_B8;
        kB1C0_ = AscendC::BLOCK_CUBE;
    }
    if constexpr (aTrans) {
        kA1C0_ = AscendC::BLOCK_CUBE;
        mA1C0_ = K0_B8;
    } else {
        kA1C0_ = K0_B8;
        mA1C0_ = AscendC::BLOCK_CUBE;
    }
    kA1_ = Get<MNK_K>(l0Shape_) * Get<2>(tileL12L0_); // 2: idx of stepKa in tileshape
    kB1_ = Get<MNK_K>(l0Shape_) * Get<3>(tileL12L0_); // 3: idx of stepKb in tileshape
}

template <bool aTrans, bool bTrans>
__aicore__ inline void MatMulCommonParam<aTrans, bTrans>::UpdateForNextGroup(const TupleShape& problemShape)
{
    problemShape_ = problemShape;

    kB1Tail_ = Get<MNK_K>(problemShape_) % kB1_ == 0 ? kB1_ : Get<MNK_K>(problemShape_) % kB1_;
    kA1Tail_ = Get<MNK_K>(problemShape_) % kA1_ == 0 ? kA1_ : Get<MNK_K>(problemShape_) % kA1_;
}

template <bool aTrans, bool bTrans>
__aicore__ inline void MatMulCommonParam<aTrans, bTrans>::UpdateNextBlockParams(const TupleShape& actualSingleShape)
{
    actualSingleShape_ = actualSingleShape;
}

template <bool aTrans, bool bTrans>
__aicore__ inline uint64_t MatMulCommonParam<aTrans, bTrans>::CalcAGMOffsetInnerLoop(uint64_t mOffset, uint64_t kOffset)
{
    if constexpr (aTrans) {
        return kOffset * Get<MNK_M>(problemShape_) + mOffset;
    } else {
        return kOffset + mOffset * Get<MNK_K>(problemShape_);
    }
}

template <bool aTrans, bool bTrans>
__aicore__ inline uint64_t MatMulCommonParam<aTrans, bTrans>::CalcBGMOffsetInnerLoop(uint64_t nOffset, uint64_t kOffset)
{
    if constexpr (bTrans) {
        return kOffset + nOffset * Get<MNK_K>(problemShape_);
    } else {
        return kOffset * Get<MNK_N>(problemShape_) + nOffset;
    }
}

template <bool aTrans, bool bTrans>
__aicore__ inline void MatMulCommonParam<aTrans, bTrans>::CalNd2NzParamA(AscendC::Nd2NzParams& nd2nzParam,
                                                                         bool isTailAL1)
{
    uint64_t currentK = isTailAL1 ? kA1Tail_ : kA1_;
    nd2nzParam.ndNum = 1;
    nd2nzParam.srcNdMatrixStride = 0;
    nd2nzParam.dstNzNStride = 1;
    nd2nzParam.dstNzMatrixStride = 0;
    if constexpr (aTrans) {
        nd2nzParam.nValue = currentK;
        nd2nzParam.dValue = Get<MNK_M>(actualSingleShape_);
        nd2nzParam.srcDValue = Get<MNK_M>(problemShape_);
        nd2nzParam.dstNzC0Stride = Align(currentK, static_cast<uint64_t>(GMM_DATA_BLOCK)); // Align to 32-byte boundary
    } else {
        nd2nzParam.nValue = Get<MNK_M>(actualSingleShape_);
        nd2nzParam.dValue = currentK;
        nd2nzParam.srcDValue = Get<MNK_K>(problemShape_);
        nd2nzParam.dstNzC0Stride = Align(Get<MNK_M>(actualSingleShape_), static_cast<uint64_t>(GMM_k0_FLOAT16));
    }
}

template <bool aTrans, bool bTrans>
__aicore__ inline void MatMulCommonParam<aTrans, bTrans>::CalNd2NzParamB(AscendC::Nd2NzParams& nd2nzParam,
                                                                         bool isTailBL1)
{
    uint64_t currentK = isTailBL1 ? kB1Tail_ : kB1_;
    nd2nzParam.ndNum = 1;
    nd2nzParam.srcNdMatrixStride = 0;
    nd2nzParam.dstNzNStride = 1;
    nd2nzParam.dstNzMatrixStride = 0;
    if constexpr (bTrans) {
        nd2nzParam.nValue = Get<MNK_N>(actualSingleShape_);
        nd2nzParam.dValue = currentK;
        nd2nzParam.srcDValue = Get<MNK_K>(problemShape_);
        nd2nzParam.dstNzC0Stride = Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(GMM_k0_FLOAT16));
    } else {
        nd2nzParam.nValue = currentK;
        nd2nzParam.dValue = Get<MNK_N>(actualSingleShape_);
        nd2nzParam.srcDValue = Get<MNK_N>(problemShape_);
        nd2nzParam.dstNzC0Stride = Align(currentK, static_cast<uint64_t>(GMM_DATA_BLOCK)); // Align to 32-byte boundary
    }
}

template <bool aTrans, bool bTrans>
__aicore__ inline uint32_t MatMulCommonParam<aTrans, bTrans>::CalcAL1Offset(uint64_t mAL1Offset, uint64_t kAL1Offset,
                                                                            bool isKTail)
{
    uint64_t kAL1 = isKTail ? kA1Tail_ : kA1_;
    kAL1 = Align(kAL1, static_cast<uint64_t>(K0_B8));
    uint64_t mAL1 = Align(Get<MNK_M>(actualSingleShape_), mA1C0_);
    if constexpr (aTrans) {
        // (m1, k1, k0, m0)
        return Align(mAL1Offset, mA1C0_) * kAL1 + kAL1Offset * mA1C0_;
    } else {
        // (k1, m1, m0, k0)
        return Align(kAL1Offset, kA1C0_) * mAL1 + mAL1Offset * kA1C0_;
    }
}

template <bool aTrans, bool bTrans>
__aicore__ inline uint32_t MatMulCommonParam<aTrans, bTrans>::CalcBL1Offset(uint64_t nBL1Offset, uint64_t kBL1Offset,
                                                                            bool isKTail)
{
    uint64_t kBL1 = isKTail ? kB1Tail_ : kB1_;
    kBL1 = Align(kBL1, static_cast<uint64_t>(K0_B8));
    uint64_t nBL1 = Align(Get<MNK_N>(actualSingleShape_), nB1C0_);
    if constexpr (bTrans) {
        // (k1, n1, n0, k0)
        return Align(kBL1Offset, kB1C0_) * nBL1 + nBL1Offset * kB1C0_;
    } else {
        // (n1, k1, k0, n0)
        return Align(nBL1Offset, nB1C0_) * kBL1 + kBL1Offset * nB1C0_;
    }
}

template <bool aTrans, bool bTrans>
__aicore__ inline void
MatMulCommonParam<aTrans, bTrans>::LoadData2dParamsA(AscendC::LoadData2DParamsV2& loadData2dParams, uint64_t kOffset,
                                                     bool isTailAL1)
{
    uint64_t currM = AscendC::Std::min(Get<MNK_M>(actualSingleShape_), Get<MNK_M>(l0Shape_));
    uint64_t currK =
        AscendC::Std::min(Get<MNK_K>(problemShape_) - kOffset, static_cast<uint64_t>(Get<MNK_K>(l0Shape_)));
    if constexpr (aTrans) {
        // For b8 input in transpose scenarios: use two 16x32 fractals
        loadData2dParams.mStep = Align(Act::Gemm::CeilDiv(currK, static_cast<uint64_t>(GMM_k0_FLOAT16)), 2UL);
        loadData2dParams.kStep = Act::Gemm::CeilDiv(currM, static_cast<uint64_t>(K0_B8));
        loadData2dParams.srcStride = Align(Act::Gemm::CeilDiv(isTailAL1 ? kA1Tail_ : kA1_, GMM_k0_FLOAT16), 2UL);
        loadData2dParams.dstStride = Align(Act::Gemm::CeilDiv(currM, static_cast<uint64_t>(GMM_k0_FLOAT16)), 2UL);
        loadData2dParams.ifTranspose = true;
    } else {
        loadData2dParams.mStep = Act::Gemm::CeilDiv(currM, static_cast<uint64_t>(GMM_k0_FLOAT16));
        loadData2dParams.kStep = Act::Gemm::CeilDiv(currK, static_cast<uint64_t>(K0_B8));
        loadData2dParams.srcStride =
            Act::Gemm::CeilDiv(currM * Get<0>(tileL12L0_), static_cast<uint64_t>(GMM_k0_FLOAT16));
        loadData2dParams.dstStride = Act::Gemm::CeilDiv(currM, static_cast<uint64_t>(GMM_k0_FLOAT16));
    }
}

template <bool aTrans, bool bTrans>
__aicore__ inline void
MatMulCommonParam<aTrans, bTrans>::LoadData2dParamsB(AscendC::LoadData2DParamsV2& loadData2dParams, uint64_t kOffset,
                                                     bool isTailBL1)
{
    uint64_t currN = AscendC::Std::min(Get<MNK_N>(actualSingleShape_), Get<MNK_N>(l0Shape_));
    uint64_t currK =
        AscendC::Std::min(Get<MNK_K>(problemShape_) - kOffset, static_cast<uint64_t>(Get<MNK_K>(l0Shape_)));
    if constexpr (bTrans) {
        loadData2dParams.mStep = Act::Gemm::CeilDiv(currN, static_cast<uint64_t>(GMM_k0_FLOAT16));
        loadData2dParams.kStep = Act::Gemm::CeilDiv(currK, static_cast<uint64_t>(K0_B8));
        loadData2dParams.srcStride =
            Act::Gemm::CeilDiv(currN * Get<1>(tileL12L0_), static_cast<uint64_t>(GMM_k0_FLOAT16));
        loadData2dParams.dstStride = Act::Gemm::CeilDiv(currN, static_cast<uint64_t>(GMM_k0_FLOAT16));
    } else {
        loadData2dParams.ifTranspose = true;
        // For b8 input in transpose scenarios: use two 16x32 fractals
        loadData2dParams.mStep = Align(Act::Gemm::CeilDiv(currK, static_cast<uint64_t>(GMM_k0_FLOAT16)), 2UL);
        loadData2dParams.kStep = Act::Gemm::CeilDiv(currN, static_cast<uint64_t>(K0_B8));
        loadData2dParams.srcStride = Align(Act::Gemm::CeilDiv(isTailBL1 ? kB1Tail_ : kB1_, GMM_k0_FLOAT16), 2UL);
        loadData2dParams.dstStride = Align(Act::Gemm::CeilDiv(currN, static_cast<uint64_t>(GMM_k0_FLOAT16)), 2UL);
    }
}
} // namespace Block
} // namespace Gemm
} // namespace Act

#endif