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
 * \file coord_utils.h
 * \brief
 */

#ifndef UTILS_COORD_UTILS_H
#define UTILS_COORD_UTILS_H

#include "common_utils.h"
#include "grouped_matmul_constant.h"
namespace Act {
namespace Gemm {

constexpr uint32_t OUTER_SIZE = 16;

template <class BlockCoord_, class ProblemShape_, class ATensorType_, class BTensorType_, class CTensorType_>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t>
GetOffset(BlockCoord_ blockCoord, ProblemShape_ problemShape, ATensorType_ aTensor, BTensorType_ bTensor,
          CTensorType_ cTensor, bool transA, bool transB)
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
// GetOffsetWithoutLayout
template <class BlockCoord, class ProblemShape, class ATensorType, class BTensorType, class CTensorType>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t, int64_t>
GetOffsetWithoutLayout(BlockCoord blockCoord, ProblemShape problemShape, ATensorType aTensor, BTensorType bTensor,
                       CTensorType cTensor, bool transA, bool transB, bool isBias)
{
    int64_t m = Get<MNK_M>(problemShape);
    int64_t n = Get<MNK_N>(problemShape);
    int64_t k = Get<MNK_K>(problemShape);
    int64_t offsetA = Get<MNK_B>(blockCoord) * m * k;
    int64_t offsetB = Get<MNK_B>(blockCoord) * n * k;
    int64_t offsetC = Get<MNK_B>(blockCoord) * m * n + Get<0>(blockCoord) * n + Get<1>(blockCoord);
    int64_t offsetBias = 0;
    if (transA) {
        offsetA += Get<0>(blockCoord);
    } else {
        offsetA += Get<0>(blockCoord) * k;
    }
    if (transB) {
        offsetB += Get<1>(blockCoord) * k;
    } else {
        offsetB += Get<1>(blockCoord);
    }
    if (isBias) {
        offsetBias = Get<MNK_B>(blockCoord) * n + Get<1>(blockCoord);
    }

    return {offsetA, offsetB, offsetC, offsetBias};
}

// GetOffsetStreamK
template <class BlockCoord_, class ProblemShape_, class ATensorType_, class BTensorType_, class CTensorType_>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t, int64_t>
GetOffsetStreamK(BlockCoord_ blockCoord, ProblemShape_ problemShape,
                 AscendC::Shape<int64_t, int64_t, int64_t, int64_t> tileL1, int64_t kSingleCore,
                 ATensorType_ aTensor, BTensorType_ bTensor, CTensorType_ cTensor,
                 bool transA, bool transB, bool isBias)
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
    if (transA) {
        offsetA = Get<MNK_B>(blockCoord) * m * k +
                  Get<MNK_M>(blockCoord) * mL1 +
                  Get<MNK_K>(blockCoord) * kSingleCore * m;
    } else {
        offsetA = Get<MNK_B>(blockCoord) * m * k +
                  Get<MNK_M>(blockCoord) * mL1 * k +
                  Get<MNK_K>(blockCoord) * kSingleCore;
    }
    if (transB) {
        offsetB = Get<MNK_B>(blockCoord) * n * k +
                  Get<MNK_N>(blockCoord) * nL1 * k +
                  Get<MNK_K>(blockCoord) * kSingleCore;
    } else {
        offsetB = Get<MNK_B>(blockCoord) * n * k +
                  Get<MNK_N>(blockCoord) * nL1 +
                  Get<MNK_K>(blockCoord) * kSingleCore * n;
    }
    if (isBias) {
        offsetBias = Get<MNK_B>(blockCoord) * n + Get<MNK_N>(blockCoord) * nL1;
    }

    return {offsetA, offsetB, offsetC, offsetBias};
}

// GetOffsetIterBatch
template <class BlockCoord, class ProblemShape, class ATensorType, class BTensorType, class CTensorType>
__aicore__ inline AscendC::Coord<int64_t, int64_t, int64_t>
GetOffsetIterBatch(BlockCoord blockCoord, ProblemShape problemShape, ATensorType aTensor, BTensorType bTensor,
                   CTensorType cTensor)
{
    int64_t m = Get<MNK_M>(problemShape);
    int64_t n = Get<MNK_N>(problemShape);
    int64_t k = Get<MNK_K>(problemShape);
    int64_t offsetA = Get<MNK_B>(blockCoord) * m * k;
    int64_t offsetB = Get<MNK_B>(blockCoord) * k * n;
    int64_t offsetC = Get<MNK_B>(blockCoord) * m * n;
    return {offsetA, offsetB, offsetC};
}

template <bool isTransA_, bool isTransB_, CubeFormat layoutA_, CubeFormat layoutB_, CubeFormat layoutC_>
class Coordinate {
public:
    __aicore__ inline Coordinate(int64_t m, int64_t n, int64_t k, int64_t l1M, int64_t l1N, int64_t l1K) :
        m(m), n(n), k(k), l1M(l1M), l1N(l1N), l1K(l1K)
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
                return batchTileIdx * CeilAlign(n, OUTER_SIZE) * CeilAlign(k, c0) + (nTileIdx * l1N + nSplitOffset) * c0
                       + kTileIdx * l1K * CeilAlign(n, OUTER_SIZE);
            }
            return batchTileIdx * CeilAlign(n, c0) * CeilAlign(k, OUTER_SIZE) + kTileIdx * l1K * c0
                   + (nTileIdx * l1N + nSplitOffset) * CeilAlign(k, OUTER_SIZE);
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

    template <GroupedMatmul::QuantMode aQuantMode>
    __aicore__ inline AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
    GetQuantOffset(int64_t mTileIdx, int64_t nTileIdx, int64_t mSplitOffset = 0, int64_t nSplitOffset = 0)
    {
        int64_t mOffset = mTileIdx * l1M + mSplitOffset;
        int64_t nOffset = nTileIdx * l1N + nSplitOffset;
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> offset{0, 0, 0, 0, 0, 0};
        if constexpr (isTransA) {
            Get<0>(offset) = mOffset;
        } else {
            Get<0>(offset) = mOffset * k;
        }
        if constexpr (isTransB) {
            Get<1>(offset) = nOffset * k;
        } else {
            Get<1>(offset) = nOffset;
        }
        Get<5>(offset) = mOffset * n + nOffset; // 5: idx of y
        if constexpr (aQuantMode == GroupedMatmul::QuantMode::PERGROUP_MODE ||
                      aQuantMode == GroupedMatmul::QuantMode::PERBLOCK_MODE) {
            if ASCEND_IS_AIV {
                int64_t x1ScaleMOffset = (aQuantMode == GroupedMatmul::QuantMode::PERGROUP_MODE) ?
                                             mOffset :
                                             mOffset / PER_BLOCK_SIZE;
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
        } else if constexpr (aQuantMode == GroupedMatmul::QuantMode::MX_PERGROUP_MODE) {
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

    template <GroupedMatmul::QuantMode aQuantMode>
    __aicore__ inline AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
    GetQuantIOOffset(int64_t mTileIdx, int64_t nTileIdx, int64_t mSplitOffset = 0, int64_t nSplitOffset = 0)
    {
        int64_t mOffset = mTileIdx * l1M + mSplitOffset;
        int64_t nOffset = nTileIdx * l1N + nSplitOffset;
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> offset{0, 0, 0, 0, 0, 0, 0};
        if constexpr (!isTransA) {
            Get<0>(offset) = mOffset * k;
        } else {
            Get<0>(offset) = mOffset;
        }
        if constexpr (!isTransB) {
            Get<1>(offset) = nOffset;
        } else {
            Get<1>(offset) = nOffset * k;
        }
        Get<5>(offset) = mOffset * n / 2 + nOffset; // 5: idx of y
        if constexpr (aQuantMode == GroupedMatmul::QuantMode::PERGROUP_MODE ||
                      aQuantMode == GroupedMatmul::QuantMode::PERBLOCK_MODE) {
            if ASCEND_IS_AIV {
                if constexpr (!isTransA) {
                    Get<2>(offset) = mOffset * CeilDiv(k, PER_BLOCK_SIZE); // 2: idx of x1Scale
                } else {
                    Get<2>(offset) = mOffset; // 2: idx of x1Scale
                }
                if constexpr (!isTransB) {
                    Get<3>(offset) = CeilDiv(nOffset, PER_BLOCK_SIZE); // 3: idx of x2Scale
                } else {
                    Get<3>(offset) = CeilDiv(nOffset, PER_BLOCK_SIZE) * CeilDiv(k, PER_BLOCK_SIZE); // 3: idx of x2Scale
                }
            }
        } else if constexpr (aQuantMode == GroupedMatmul::QuantMode::MX_PERGROUP_MODE) {
            if constexpr (!isTransA) {
                Get<2>(offset) = mOffset * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE; // 2: idx of x1Scale
            } else {
                Get<2>(offset) = mOffset * MXFP_MULTI_BASE_SIZE; // 2: idx of x1Scale
            }
            if constexpr (!isTransB) {
                Get<3>(offset) = nOffset * MXFP_MULTI_BASE_SIZE; // 3: idx of x2Scale
            } else {
                Get<3>(offset) = nOffset * CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE; // 3: idx of x2Scale
            }
            auto scaleN = CeilDiv(n / 2, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            // 6: idx of yScale
            Get<6>(offset) = mOffset * scaleN + CeilDiv(nOffset, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
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
} // namespace Act
#endif