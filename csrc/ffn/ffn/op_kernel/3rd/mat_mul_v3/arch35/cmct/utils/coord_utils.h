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
 * \file coord_utils.h
 * \brief
 */

#pragma once

#include "common_utils.h"
#include "quant_batch_matmul_constant.h"
namespace Cmct {
namespace Gemm {

constexpr uint32_t OUTER_SIZE = 16;
constexpr int IDX_M_BASE_NORM_CNT = 0;
constexpr int IDX_M_BASE_TAIL_MAIN = 1;
constexpr int IDX_N_BASE_NORM_CNT = 2;
constexpr int IDX_N_BASE_TAIL_MAIN = 3;

using TupleL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

template <class BlockCoord_, class ProblemShape_, class ATensorType_, class BTensorType_, class CTensorType_>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t> GetOffset(BlockCoord_ blockCoord,
                                                                      ProblemShape_ problemShape, ATensorType_ aTensor,
                                                                      BTensorType_ bTensor, CTensorType_ cTensor,
                                                                      bool transA, bool transB)
{
    int64_t m = Get<MNK_M>(problemShape);
    int64_t n = Get<MNK_N>(problemShape);
    int64_t k = Get<MNK_K>(problemShape);
    AscendC::Coord<int64_t, int64_t> ACoord;
    if (!transA) {
        ACoord = AscendC::MakeCoord(Get<0>(blockCoord), Get<2>(blockCoord));
    } else {
        ACoord = AscendC::MakeCoord(Get<2>(blockCoord), Get<0>(blockCoord));
    }
    AscendC::Coord<int64_t, int64_t> BCoord;
    if (!transB) {
        BCoord = AscendC::MakeCoord(Get<2>(blockCoord), Get<1>(blockCoord));
    } else {
        BCoord = AscendC::MakeCoord(Get<1>(blockCoord), Get<2>(blockCoord));
    }
    AscendC::Coord<int64_t, int64_t> CCoord;
    CCoord = AscendC::MakeCoord(Get<0>(blockCoord), Get<1>(blockCoord));

    int64_t offsetA = aTensor.GetTensorTrait().GetLayout()(ACoord) + Get<3>(blockCoord) * m * k;
    int64_t offsetB = bTensor.GetTensorTrait().GetLayout()(BCoord) + Get<3>(blockCoord) * n * k;
    int64_t offsetC = cTensor.GetTensorTrait().GetLayout()(CCoord) + Get<3>(blockCoord) * m * n;

    return {offsetA, offsetB, offsetC};
}

template <class BlockCoord, class ProblemShape>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t, int64_t> GetOffsetForNDLayout(
    BlockCoord blockCoord, ProblemShape problemShape, bool transA, bool transB, bool isBias,
    AscendC::Shape<int64_t, int64_t, int64_t> nonContinuousParams, TupleL1L0Shape& blockShape, bool isSplitSingleK = 0)
{
    int64_t m = Get<MNK_M>(problemShape);
    int64_t n = Get<MNK_N>(problemShape);
    int64_t k = Get<MNK_K>(problemShape);
    uint64_t sliceM = Get<0>(nonContinuousParams);
    uint64_t srcNdStride = Get<1>(nonContinuousParams);
    int64_t innerBatch = AscendC::Std::max(Get<2>(nonContinuousParams), 1L);
    int64_t realM = m;
    int64_t realMOffset = Get<0>(blockCoord);
    if (srcNdStride != 1 && sliceM != 0) {
        int64_t oriM = srcNdStride / k;
        realM = (Get<MNK_M>(blockShape) / sliceM) * oriM; // ndNum * oriM
        realMOffset = Get<2>(blockCoord);
    }

    int64_t offsetA = Get<MNK_B>(blockCoord) * realM * k;
    int64_t offsetB = Get<MNK_B>(blockCoord) * n * k;
    int64_t offsetC = Get<MNK_B>(blockCoord) * m * n + Get<0>(blockCoord) * n + Get<1>(blockCoord);
    int64_t offsetBias = 0;
    int64_t kOffset = isSplitSingleK ? Get<2>(blockCoord) : 0;
    if (innerBatch > 1) {
        offsetA = transA ? Get<MNK_B>(blockCoord) * realM : Get<MNK_B>(blockCoord) * k;
        offsetB = transB ? Get<MNK_B>(blockCoord) * k : Get<MNK_B>(blockCoord) * n;
    }
    if (transA) {
        offsetA += Get<0>(blockCoord) + kOffset * realM;
    } else {
        // m, b, k
        offsetA += realMOffset * innerBatch * k + kOffset; // get mOffsetNew from kTileIdx
    }
    if (transB) {
        // n, b, k
        offsetB += Get<1>(blockCoord) * innerBatch * k + kOffset;
    } else {
        offsetB += Get<1>(blockCoord) + kOffset * n;
    }
    if (isBias) {
        offsetBias = Get<1>(blockCoord);
    }
    return {offsetA, offsetB, offsetC, offsetBias};
}

template <class BlockCoord, class ProblemShape, class B_T>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t, int64_t> GetOffsetForNZLayout(
    BlockCoord blockCoord, ProblemShape problemShape, bool transA, bool transB, bool isBias,
    AscendC::Shape<int64_t, int64_t, int64_t, int64_t> tileL1 = {0, 0, 0, 0},
    AscendC::Shape<int64_t, int64_t> SplitOffset = {0, 0},
    AscendC::Shape<int64_t, int64_t, int64_t, int64_t> tailParams = {0, 0, 0, 0})
{
    int64_t C0_SIZE = BLOCK_BYTE_SIZE / sizeof(B_T);
    int64_t m = Get<MNK_M>(problemShape);
    int64_t n = Get<MNK_N>(problemShape);
    int64_t k = Get<MNK_K>(problemShape);
    int64_t mL1 = Get<MNK_M>(tileL1);
    int64_t nL1 = Get<MNK_N>(tileL1);

    int64_t mSplitOffset_ = Get<MNK_M>(SplitOffset);
    int64_t nSplitOffset_ = Get<MNK_N>(SplitOffset);
    int64_t mL1NormCnt_ = Get<0>(tailParams);
    int64_t mL1TailMain_ = Get<1>(tailParams);
    int64_t nL1NormCnt_ = Get<2>(tailParams);
    int64_t nL1TailMain_ = Get<3>(tailParams);

    int64_t mOffset = Get<MNK_M>(blockCoord) * mL1 + mSplitOffset_;
    int64_t nOffset = Get<MNK_N>(blockCoord) * nL1 + nSplitOffset_;

    if (Get<MNK_M>(blockCoord) > mL1NormCnt_) {
        mOffset = mL1NormCnt_ * mL1 + (Get<MNK_M>(blockCoord) - mL1NormCnt_) * mL1TailMain_ + mSplitOffset_;
    }
    if (Get<MNK_N>(blockCoord) > nL1NormCnt_) {
        nOffset = nL1NormCnt_ * nL1 + (Get<MNK_N>(blockCoord) - nL1NormCnt_) * nL1TailMain_ + nSplitOffset_;
    }

    int64_t offsetA = Get<MNK_B>(blockCoord) * m * k;
    int64_t offsetB = Get<MNK_B>(blockCoord) * CeilAlign(n, OUTER_SIZE) * CeilAlign(k, C0_SIZE);
    int64_t offsetC = Get<MNK_B>(blockCoord) * m * n + mOffset * n + nOffset;
    int64_t offsetBias = 0;

    if (transA) {
        offsetA += mOffset;
    } else {
        offsetA += mOffset * k;
    }
    if (transB) {
        offsetB += nOffset * C0_SIZE;
    } else {
        offsetB += nOffset * CeilAlign(k, OUTER_SIZE);
    }
    if (isBias) {
        offsetBias = nOffset;
    }
    return {offsetA, offsetB, offsetC, offsetBias};
}

// GetOffsetWithoutLayout
template <class BlockCoord, class ProblemShape, CubeFormat LayoutB = CubeFormat::ND, class B_T>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t, int64_t> GetOffsetWithoutLayout(
    BlockCoord blockCoord, ProblemShape problemShape, bool transA, bool transB, bool isBias,
    AscendC::Shape<int64_t, int64_t, int64_t> nonContinuousParams, TupleL1L0Shape& blockShape,
    AscendC::Shape<int64_t, int64_t, int64_t, int64_t> tileL1 = {0, 0, 0, 0},
    AscendC::Shape<int64_t, int64_t> SplitOffset = {0, 0},
    AscendC::Shape<int64_t, int64_t, int64_t, int64_t> tailParams = {0, 0, 0, 0}, bool isSplitSingleK = 0)
{
    if constexpr (LayoutB == CubeFormat::ND) {
        return GetOffsetForNDLayout(blockCoord, problemShape, transA, transB, isBias, nonContinuousParams, blockShape,
                                    isSplitSingleK);
    } else {
        return GetOffsetForNZLayout<BlockCoord, ProblemShape, B_T>(blockCoord, problemShape, transA, transB, isBias,
                                                                   tileL1, SplitOffset, tailParams);
    }
}

// GetOffsetStreamK
template <class BlockCoord_, class ProblemShape_, CubeFormat LayoutB = CubeFormat::ND, class B_T>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t, int64_t> GetOffsetStreamK(
    BlockCoord_ blockCoord, ProblemShape_ problemShape, AscendC::Shape<int64_t, int64_t, int64_t, int64_t> tileL1,
    int64_t kSingleCore, bool transA, bool transB, bool isBias, int64_t splitSingleK = 0, int64_t splitSingleKIdx = 0)
{
    int64_t m = Get<MNK_M>(problemShape);
    int64_t n = Get<MNK_N>(problemShape);
    int64_t k = Get<MNK_K>(problemShape);
    int64_t mL1 = Get<MNK_M>(tileL1);
    int64_t nL1 = Get<MNK_N>(tileL1);

    int64_t offsetA = 0;
    int64_t offsetB = 0;
    int64_t offsetC = Get<MNK_B>(blockCoord) * m * n + Get<MNK_M>(blockCoord) * mL1 * n + Get<MNK_N>(blockCoord) * nL1;
    int64_t offsetBias = 0;
    int64_t C0_SIZE = BLOCK_BYTE_SIZE / sizeof(B_T);
    int64_t kOffset = splitSingleKIdx * splitSingleK;

    if (transA) {
        offsetA = Get<MNK_B>(blockCoord) * m * k + Get<MNK_M>(blockCoord) * mL1 +
                  (Get<MNK_K>(blockCoord) * kSingleCore + kOffset) * m;
    } else {
        offsetA = Get<MNK_B>(blockCoord) * m * k + Get<MNK_M>(blockCoord) * mL1 * k +
                  Get<MNK_K>(blockCoord) * kSingleCore + kOffset;
    }
    if constexpr (LayoutB == CubeFormat::ND) {
        if (transB) {
            offsetB = Get<MNK_B>(blockCoord) * n * k + Get<MNK_N>(blockCoord) * nL1 * k +
                      Get<MNK_K>(blockCoord) * kSingleCore + kOffset;
        } else {
            offsetB = Get<MNK_B>(blockCoord) * n * k + Get<MNK_N>(blockCoord) * nL1 +
                      (Get<MNK_K>(blockCoord) * kSingleCore + kOffset) * n;
        }
    } else {
        if (transB) {
            offsetB = Get<MNK_B>(blockCoord) * CeilAlign(n, C0_SIZE) * CeilAlign(k, C0_SIZE) +
                      Get<MNK_N>(blockCoord) * nL1 * C0_SIZE +
                      Get<MNK_K>(blockCoord) * kSingleCore * CeilAlign(n, OUTER_SIZE);
        } else {
            offsetB = Get<MNK_B>(blockCoord) * CeilAlign(n, C0_SIZE) * CeilAlign(k, C0_SIZE) +
                      Get<MNK_N>(blockCoord) * nL1 * CeilAlign(k, C0_SIZE) +
                      Get<MNK_K>(blockCoord) * kSingleCore * C0_SIZE;
        }
    }
    if (isBias) {
        offsetBias = Get<MNK_N>(blockCoord) * nL1;
    }

    return {offsetA, offsetB, offsetC, offsetBias};
}

// GetOffsetIterBatch
template <class BlockCoord, class ProblemShape, class ATensorType, class BTensorType, class CTensorType>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t> GetOffsetIterBatch(
    BlockCoord blockCoord, ProblemShape problemShape, ATensorType aTensor, BTensorType bTensor, CTensorType cTensor,
    int64_t innerBatch = 0, bool transB = false)
{
    int64_t m = Get<MNK_M>(problemShape);
    int64_t n = Get<MNK_N>(problemShape);
    int64_t k = Get<MNK_K>(problemShape);
    int64_t offsetA = Get<MNK_B>(blockCoord) * m * k;
    int64_t offsetB;
    if (innerBatch > 0) {
        if (transB) {
            offsetB = Get<MNK_B>(blockCoord) * k;
        } else {
            offsetB = Get<MNK_B>(blockCoord) * n;
        }
    } else {
        offsetB = Get<MNK_B>(blockCoord) * k * n;
    }

    int64_t offsetC = Get<MNK_B>(blockCoord) * m * n;
    return {offsetA, offsetB, offsetC};
}

template <bool isTransA_, bool isTransB_, CubeFormat layoutA_, CubeFormat layoutB_, CubeFormat layoutC_>
class Coordinate {
public:
    __aicore__ inline Coordinate(int64_t m, int64_t n, int64_t k, int64_t l1M, int64_t l1N, int64_t l1K)
        : m(m), n(n), k(k), l1M(l1M), l1N(l1N), l1K(l1K)
    {}

    static constexpr bool isTransA = isTransA_;
    static constexpr bool isTransB = isTransB_;
    static constexpr CubeFormat layoutB = layoutB_;

    __aicore__ inline int64_t GetAOffset(int64_t mTileIdx, int64_t kTileIdx, int64_t batchTileIdx = 0,
                                         int64_t mSplitOffset = 0)
    {
        if (isTransA) {
            return batchTileIdx * m * k + kTileIdx * l1K * m + (mTileIdx * l1M + mSplitOffset);
        }
        return batchTileIdx * m * k + (mTileIdx * l1M + mSplitOffset) * k + kTileIdx * l1K;
    }

    __aicore__ inline int64_t GetBOffset(int64_t nTileIdx, int64_t kTileIdx, int64_t batchTileIdx = 0, int32_t c0 = 0,
                                         int64_t nSplitOffset = 0)
    {
        if constexpr (layoutB == CubeFormat::NZ) {
            if (c0 == 0) {
                return 0;
            }
            if (isTransB) {
                return batchTileIdx * CeilAlign(n, OUTER_SIZE) * CeilAlign(k, c0) +
                       (nTileIdx * l1N + nSplitOffset) * c0 + kTileIdx * l1K * CeilAlign(n, OUTER_SIZE);
            }
            return batchTileIdx * CeilAlign(n, c0) * CeilAlign(k, OUTER_SIZE) + kTileIdx * l1K * c0 +
                   (nTileIdx * l1N + nSplitOffset) * CeilAlign(k, OUTER_SIZE);
        }
        if (isTransB) {
            return batchTileIdx * n * k + (nTileIdx * l1N + nSplitOffset) * k + kTileIdx * l1K;
        }
        return batchTileIdx * n * k + kTileIdx * l1K * n + (nTileIdx * l1N + nSplitOffset);
    }

    __aicore__ inline int64_t GetCOffset(int64_t mTileIdx, int64_t nTileIdx, int64_t batchTileIdx = 0,
                                         int64_t mSplitOffset = 0, int64_t nSplitOffset = 0)
    {
        return batchTileIdx * n * m + (mTileIdx * l1M + mSplitOffset) * n + (nTileIdx * l1N + nSplitOffset);
    }

    __aicore__ inline int64_t GetBiasOffset(int64_t nTileIdx, int64_t nSplitOffset = 0)
    {
        return nTileIdx * l1N + nSplitOffset;
    }

    template <QuantBatchMatmul::QuantMode aQuantMode>
    __aicore__ inline void CalOffsetOfAIV(
        int64_t mOffset, int64_t nOffset,
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& offset)
    {
        int64_t x1ScaleMOffset = mOffset;
        if constexpr (aQuantMode == QuantBatchMatmul::QuantMode::PERBLOCK_MODE) {
            x1ScaleMOffset = mOffset / PER_BLOCK_SIZE;
        }
        if constexpr (isTransA) {
            Get<2>(offset) = x1ScaleMOffset; // 2: idx of x1Scale
        } else {
            Get<2>(offset) = x1ScaleMOffset * CeilDiv(k, PER_BLOCK_SIZE); // 2: idx of x1Scale
        }
        if constexpr (isTransB) {
            Get<3>(offset) = nOffset / PER_BLOCK_SIZE * CeilDiv(k, PER_BLOCK_SIZE); // 3: idx of x2Scale
        } else {
            Get<3>(offset) = nOffset / PER_BLOCK_SIZE; // 3: idx of x2Scale
        }
    }

    template <class AType>
    __aicore__ inline void CalOffset4Weight(
        int64_t nOffset, AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& offset)
    {
        if constexpr (layoutB == CubeFormat::NZ) {
            if constexpr (isTransB) {
                int64_t c0Size = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value ? C0_SIZE_B4 : C0_SIZE_B8;
                Get<1>(offset) = nOffset * c0Size;
            } else {
                Get<1>(offset) = nOffset * CeilDiv(k, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE;
            }
        } else {
            if constexpr (isTransB) {
                Get<1>(offset) = nOffset * k;
            } else {
                Get<1>(offset) = nOffset;
            }
        }
    }

    template <QuantBatchMatmul::QuantMode aQuantMode, class AType, bool enableLoadBalance = false>
    __aicore__ inline AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> GetQuantOffset(
        int64_t mTileIdx, int64_t nTileIdx, int64_t mSplitOffset = 0, int64_t nSplitOffset = 0,
        const AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>& loadBalanceParam = {0u, 0u, 0u, 0u})
    {
        int64_t mOffset = mTileIdx * l1M + mSplitOffset;
        int64_t nOffset = nTileIdx * l1N + nSplitOffset;
        if constexpr (enableLoadBalance) {
            if constexpr (!isTransA) {
                if (mTileIdx > Get<IDX_M_BASE_NORM_CNT>(loadBalanceParam)) {
                    mOffset -= (mTileIdx - Get<IDX_M_BASE_NORM_CNT>(loadBalanceParam)) *
                               (l1M - Get<IDX_M_BASE_TAIL_MAIN>(loadBalanceParam));
                }
            }
            if constexpr (isTransB) {
                if (nTileIdx > Get<IDX_N_BASE_NORM_CNT>(loadBalanceParam)) {
                    nOffset -= (nTileIdx - Get<IDX_N_BASE_NORM_CNT>(loadBalanceParam)) *
                               (l1N - Get<IDX_N_BASE_TAIL_MAIN>(loadBalanceParam));
                }
            }
        }
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> offset{0, 0, 0, 0, 0, 0};
        if constexpr (isTransA) {
            Get<0>(offset) = mOffset;
        } else {
            Get<0>(offset) = mOffset * k;
        }
        CalOffset4Weight<AType>(nOffset, offset);

        Get<5>(offset) = mOffset * n + nOffset; // 5: idx of y
        if constexpr (aQuantMode == QuantBatchMatmul::QuantMode::PERGROUP_MODE ||
                      aQuantMode == QuantBatchMatmul::QuantMode::PERBLOCK_MODE) {
            if ASCEND_IS_AIV {
                this->CalOffsetOfAIV<aQuantMode>(mOffset, nOffset, offset);
            }
        } else if constexpr (aQuantMode == QuantBatchMatmul::QuantMode::MX_PERGROUP_MODE) {
            if constexpr (isTransA) {
                Get<2>(offset) = mOffset * MXFP_MULTI_BASE_SIZE; // 2: idx of x1Scale
            } else {
                Get<2>(offset) = mOffset * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE; // 2: idx of x1Scale
            }
            if constexpr (isTransB) {
                Get<3>(offset) = nOffset * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE; // 3: idx of x2Scale
            } else {
                Get<3>(offset) = nOffset * MXFP_MULTI_BASE_SIZE; // 3: idx of x2Scale
            }
        } else {
            Get<2>(offset) = mOffset; // 2: idx of x1Scale
            Get<3>(offset) = nOffset; // 3: idx of x2Scale
        }
        Get<4>(offset) = nOffset; // 4: idx of bias
        return offset;
    }

    int64_t m{0};
    int64_t n{0};
    int64_t k{0};
    int64_t l1M{0};
    int64_t l1N{0};
    int64_t l1K{0};
};
} // namespace Gemm
} // namespace Cmct
