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
 * \file mat_mul_base_block.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_V3_BASE_BLOCK_H__
#define __OP_KERNEL_MATMUL_V3_BASE_BLOCK_H__

#include "mat_mul_v3_common.h"

using namespace AscendC;
using namespace matmul;

struct BlockOffset {
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetC = 0;
    uint64_t offsetBias = 0;
};

struct BaseBlockArgs {
    uint64_t index;
    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t totalTileCnt;
    uint64_t blockBaseM;
    uint64_t blockBaseN;
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t nBaseTail;
    uint64_t mBaseTail;
    uint64_t mTileCntL2;
    uint64_t nTileCntL2;
    uint64_t mCntTail;
    uint64_t nCntTail;
    uint64_t mTotalCnt;
    uint64_t nTotalCnt;
    uint64_t round;
    uint64_t realRound;
    uint64_t preCoreNum;
    uint64_t mCntUse;
    uint64_t nCntUse;
    uint32_t rowOrder;
    uint64_t blockIdxStart;
    uint64_t blockIdxEnd;

    uint64_t mTileAddrOffset;
    uint64_t nTileAddrOffset;

    uint64_t c0Size;
    uint64_t alignedOriM;
    uint64_t alignedOriN;
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;

    bool isTransposeA;
    bool isTransposeB;
    uint32_t isHf32;
};

class Mc2MatmulBaseBlock {
public:
    __aicore__ inline Mc2MatmulBaseBlock()
    {
    }
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void InitBlockIndex(uint64_t index);
    __aicore__ inline void UpdateBlockParams(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateBlockIndex();
    __aicore__ inline void UpdateBasicIndex(const uint64_t roundIdx);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex);

public:
    BlockOffset offset_;
    BaseBlockArgs params_;
    const Mc2MatmulV3TilingData *matmulTilingData_;
    bool indexInit_ = false;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatmulBaseBlock::Init(const void *tilingData)
{
    matmulTilingData_ = static_cast<const Mc2MatmulV3TilingData *>(tilingData);
    const Mc2L2cacheTilePara &tilingL2 = matmulTilingData_->tileL2cacheTiling;

    params_.isTransposeA = matmulTilingData_->matmulRunInfo.transA;
    params_.isTransposeB = matmulTilingData_->matmulRunInfo.transB;
    params_.isHf32 = matmulTilingData_->matmulRunInfo.isHf32;

    params_.blockBaseM = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseM);
    params_.blockBaseN = static_cast<uint64_t>(matmulTilingData_->matmulTiling.baseN);
    params_.mTileCntL2 = static_cast<uint64_t>(tilingL2.mTileCntL2); // M方向的Tile份数
    params_.nTileCntL2 = static_cast<uint64_t>(tilingL2.nTileCntL2); // N方向的Tile份数
    params_.mTotalCnt =
        (static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) + matmulTilingData_->matmulTiling.singleCoreM - 1) /
        matmulTilingData_->matmulTiling.singleCoreM; // 总的m方向base块个数
    params_.nTotalCnt =
        (static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) + matmulTilingData_->matmulTiling.singleCoreN - 1) /
        matmulTilingData_->matmulTiling.singleCoreN; // 总的n方向base块个数
    params_.nBaseTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.N) -
                        (params_.nTotalCnt - 1) * matmulTilingData_->matmulTiling.singleCoreN; // n方向上的base尾块
    params_.mBaseTail = static_cast<uint64_t>(matmulTilingData_->matmulTiling.M) -
                        (params_.mTotalCnt - 1) * matmulTilingData_->matmulTiling.singleCoreM; // m方向上的base尾块
    params_.singleCoreM = 0;
    params_.singleCoreN = 0;
    params_.blockIdxStart = 0;
    params_.blockIdxEnd = 0;
    params_.index = 0;
    // mCnt和nCnt需要添加约束，否则可能会地址超限，当前tiling保证mTileCntL2和nTileCntL2合法性
    // 需要保证mTileCntL2和nTileCntL2的切分策略正好满足整块+尾块的处理
    params_.mCnt = (params_.mTotalCnt + params_.mTileCntL2 - 1) / params_.mTileCntL2; // 每一份mTile包含的base块个数
    params_.nCnt = (params_.nTotalCnt + params_.nTileCntL2 - 1) / params_.nTileCntL2; // 每一份nTile包含的base块个数
    if (tilingL2.mTileBlock > 0 && tilingL2.nTileBlock > 0) {
        params_.mCnt = tilingL2.mTileBlock;
        params_.nCnt = tilingL2.nTileBlock;
    }
    params_.totalTileCnt = params_.mCnt * params_.nCnt;

    params_.mCntTail = params_.mTotalCnt - (params_.mTileCntL2 - 1) * params_.mCnt; // M方向上mTile尾块里的base块的个数
    params_.nCntTail = params_.nTotalCnt - (params_.nTileCntL2 - 1) * params_.nCnt; // M方向上nTile尾块里的base块的个数
    params_.round = (params_.totalTileCnt + matmulTilingData_->matmulTiling.usedCoreNum - 1) /
                    matmulTilingData_->matmulTiling.usedCoreNum;
    params_.realRound = 0;
    params_.preCoreNum = params_.totalTileCnt % matmulTilingData_->matmulTiling.usedCoreNum;
    params_.mCntUse = params_.mCnt;
    params_.nCntUse = params_.nCnt;
    // 当B矩阵比A矩阵大时，列优先输出能减少数据的重复替换

    params_.rowOrder = tilingL2.calOrder;

    params_.c0Size = 0;
    using A_T = typename A_TYPE::T;
    GetSizeC0<A_T>(params_.c0Size);
    params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, ALIGNED_H) * ALIGNED_H;
    params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, params_.c0Size) * params_.c0Size;
    params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, params_.c0Size) * params_.c0Size;
    params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;
    // A B矩阵都是对齐矩阵
    if (params_.isTransposeA) {
        params_.alignedOriM = MMV3DivCeil(matmulTilingData_->matmulTiling.M, params_.c0Size) * params_.c0Size;
        params_.alignedKaSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Ka, ALIGNED_H) * ALIGNED_H;
    }
    if (params_.isTransposeB) {
        params_.alignedOriN = MMV3DivCeil(matmulTilingData_->matmulTiling.N, ALIGNED_H) * ALIGNED_H;
        params_.alignedKbSize = MMV3DivCeil(matmulTilingData_->matmulTiling.Kb, params_.c0Size) * params_.c0Size;
    }
}

__aicore__ inline void Mc2MatmulBaseBlock::UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex)
{
    params_.mTileAddrOffset = mTileIndex * params_.mCnt * matmulTilingData_->matmulTiling.singleCoreM;
    params_.nTileAddrOffset = nTileIndex * params_.nCnt * matmulTilingData_->matmulTiling.singleCoreN;

    if ((mTileIndex == (params_.mTileCntL2 - 1)) && (nTileIndex == (params_.nTileCntL2 - 1))) {
        params_.totalTileCnt = params_.mCntTail * params_.nCntTail;
        params_.mCntUse = params_.mCntTail;
        params_.nCntUse = params_.nCntTail;
    } else if (mTileIndex == (params_.mTileCntL2 - 1)) {
        params_.totalTileCnt = params_.mCntTail * params_.nCnt;
        params_.mCntUse = params_.mCntTail;
        params_.nCntUse = params_.nCnt;
    } else if (nTileIndex == (params_.nTileCntL2 - 1)) {
        params_.totalTileCnt = params_.mCnt * params_.nCntTail;
        params_.mCntUse = params_.mCnt;
        params_.nCntUse = params_.nCntTail;
    } else {
        params_.totalTileCnt = params_.mCnt * params_.nCnt;
        params_.mCntUse = params_.mCnt;
        params_.nCntUse = params_.nCnt;
    }

    params_.round = DivCeil(params_.totalTileCnt, static_cast<uint64_t>(matmulTilingData_->matmulTiling.usedCoreNum));
    params_.preCoreNum = params_.totalTileCnt % matmulTilingData_->matmulTiling.usedCoreNum;
    if (params_.preCoreNum == 0) {
        params_.preCoreNum = static_cast<uint64_t>(matmulTilingData_->matmulTiling.usedCoreNum);
    }
}

__aicore__ inline void Mc2MatmulBaseBlock::UpdateBasicIndex(const uint64_t roundIdx)
{
    uint64_t newBlockIdx =
        (GetCurrentBlockIdx() + matmulTilingData_->matmulTiling.usedCoreNum - params_.blockIdxStart) %
            matmulTilingData_->matmulTiling.usedCoreNum +
        roundIdx * matmulTilingData_->matmulTiling.usedCoreNum;
    uint64_t mIdx = newBlockIdx % params_.mCntUse;
    uint64_t nIdx = 0;
    if (params_.mCntUse != 0 && params_.nCntUse != 0) {
        nIdx = (newBlockIdx + newBlockIdx / MMLcm(params_.mCntUse, params_.nCntUse)) % params_.nCntUse;
    }
    params_.index = mIdx * params_.nCntUse + nIdx;
}


__aicore__ inline void Mc2MatmulBaseBlock::InitBlockIndex(uint64_t index)
{
    if (indexInit_) {
        params_.blockIdxStart = params_.blockIdxEnd; // 开始运算时，首核的索引
    } else {
        params_.blockIdxStart =
            index * params_.preCoreNum % matmulTilingData_->matmulTiling.usedCoreNum; // 开始运算时，首核的索引
        indexInit_ = true;
    }
    params_.blockIdxEnd = (params_.blockIdxStart + params_.preCoreNum) %
                          matmulTilingData_->matmulTiling.usedCoreNum; // 结束运算时，尾核的索引
    uint64_t indexStart = params_.blockIdxStart;
    uint64_t indexEnd = params_.blockIdxEnd;

    // 利用roudCnt来解决尾块负载均衡问题
    if (indexStart < indexEnd) {
        // 正常排序, preCore在整个Cores的中间
        if (GetCurrentBlockIdx() < indexStart) {
            params_.index = GetCurrentBlockIdx() * (params_.round - 1);
            params_.realRound = params_.round - 1;
        } else if (GetCurrentBlockIdx() < indexEnd) {
            params_.index = indexStart * (params_.round - 1) + (GetCurrentBlockIdx() - indexStart) * params_.round;
            params_.realRound = params_.round;
        } else {
            params_.index = (indexStart * (params_.round - 1) + params_.preCoreNum * params_.round +
                             (GetCurrentBlockIdx() - indexEnd) * (params_.round - 1));
            params_.realRound = params_.round - 1;
        }
        if (params_.rowOrder == COL_FIRST) {
            // 列优先分配
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    } else if (indexEnd < indexStart) {
        // indexEnd会翻转
        if (GetCurrentBlockIdx() < indexEnd) {
            params_.index = GetCurrentBlockIdx() * params_.round;
            params_.realRound = params_.round;
        } else if (GetCurrentBlockIdx() < indexStart) {
            params_.index = indexEnd * params_.round + (GetCurrentBlockIdx() - indexEnd) * (params_.round - 1);
            params_.realRound = params_.round - 1;
        } else {
            params_.index = (indexEnd * params_.round + (indexStart - indexEnd) * (params_.round - 1) +
                             (GetCurrentBlockIdx() - indexStart) * params_.round);
            params_.realRound = params_.round;
        }
        if (params_.rowOrder == COL_FIRST) {
            // 列优先分配
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    } else {
        // 不存在尾核，基本块对齐
        params_.index = GetCurrentBlockIdx() * params_.round;
        params_.realRound = params_.round;
        if (params_.rowOrder == COL_FIRST) {
            // 列优先分配
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    }
}

__aicore__ inline void Mc2MatmulBaseBlock::UpdateBlockParams(uint64_t mTileIndex, uint64_t nTileIndex)
{
    if ((mTileIndex == (params_.mTileCntL2 - 1)) && (nTileIndex == (params_.nTileCntL2 - 1)) &&
        (params_.index == (params_.totalTileCnt - 1))) {
        params_.singleCoreM = params_.mBaseTail;
        params_.singleCoreN = params_.nBaseTail;
    } else if ((mTileIndex == (params_.mTileCntL2 - 1)) && (params_.index >= (params_.mCntUse - 1) * params_.nCntUse)) {
        params_.singleCoreM = params_.mBaseTail;
        params_.singleCoreN = matmulTilingData_->matmulTiling.singleCoreN;
    } else if ((nTileIndex == (params_.nTileCntL2 - 1)) && ((params_.index + 1) % params_.nCntUse == 0)) {
        params_.singleCoreM = matmulTilingData_->matmulTiling.singleCoreM;
        params_.singleCoreN = params_.nBaseTail;
    } else {
        params_.singleCoreM = matmulTilingData_->matmulTiling.singleCoreM;
        params_.singleCoreN = matmulTilingData_->matmulTiling.singleCoreN;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatmulBaseBlock::CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex)
{
    uint64_t mCntIndex = params_.index / params_.nCntUse;
    uint64_t nCntIndex = params_.index % params_.nCntUse;
    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeA) {
            offset_.offsetA = mCntIndex * matmulTilingData_->matmulTiling.singleCoreM + params_.mTileAddrOffset;
        } else {
            offset_.offsetA =
                mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * matmulTilingData_->matmulTiling.Ka +
                params_.mTileAddrOffset * matmulTilingData_->matmulTiling.Ka;
        }
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeA) {
            offset_.offsetA = mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * params_.alignedKaSize +
                              params_.mTileAddrOffset * params_.alignedKaSize;
        } else {
            offset_.offsetA = mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * params_.c0Size +
                              params_.mTileAddrOffset * params_.c0Size;
        }
    }
    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (params_.isTransposeB) {
            offset_.offsetB =
                nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * matmulTilingData_->matmulTiling.Kb +
                params_.nTileAddrOffset * matmulTilingData_->matmulTiling.Kb;
        } else {
            offset_.offsetB = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN + params_.nTileAddrOffset;
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (params_.isTransposeB) {
            offset_.offsetB = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * params_.c0Size +
                              params_.nTileAddrOffset * params_.c0Size;
        } else {
            offset_.offsetB = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * params_.alignedKbSize +
                              params_.nTileAddrOffset * params_.alignedKbSize;
        }
    }
    if constexpr (C_TYPE::format == CubeFormat::ND) {
        offset_.offsetC = (nCntIndex * matmulTilingData_->matmulTiling.singleCoreN +
                           mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * matmulTilingData_->matmulTiling.N +
                           (params_.mTileAddrOffset * matmulTilingData_->matmulTiling.N + params_.nTileAddrOffset));
    } else {
        offset_.offsetC =
            (nCntIndex * matmulTilingData_->matmulTiling.singleCoreN * matmulTilingData_->matmulTiling.M +
             mCntIndex * matmulTilingData_->matmulTiling.singleCoreM * params_.c0Size +
             (params_.mTileAddrOffset * params_.c0Size + params_.nTileAddrOffset * matmulTilingData_->matmulTiling.M));
    }
    if (matmulTilingData_->matmulTiling.isBias) {
        offset_.offsetBias = nCntIndex * matmulTilingData_->matmulTiling.singleCoreN + params_.nTileAddrOffset;
    }
}

__aicore__ inline void Mc2MatmulBaseBlock::UpdateBlockIndex()
{
    if (params_.rowOrder == ROW_FIRST) {
        params_.index += 1;
    } else if (params_.rowOrder == COL_FIRST) {
        params_.index += params_.nCntUse;
        if (params_.index >= params_.totalTileCnt) {
            params_.index = params_.index % params_.totalTileCnt + 1;
        }
    }
}

#endif // MMV3_MATMUL_BLOCK_H
