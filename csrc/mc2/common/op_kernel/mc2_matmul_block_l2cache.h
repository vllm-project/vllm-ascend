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
 * \file mc2_matmul_block_l2cache.h
 * \brief
 */

#ifndef MC2_MATMUL_BLOCK_L2CACHE_H
#define MC2_MATMUL_BLOCK_L2CACHE_H

#include "mc2_matmul_block.h"

namespace AscendC {
struct L2CacheTileArguments {
    uint32_t rankBaseTailM;
    uint32_t rankBaseTailN;
    uint32_t mBlockCnt;       // M方向的基本块个数
    uint32_t nBlockCnt;       // N方向的基本块个数
    uint32_t totalTileBlocks; //
    uint32_t mOneTileBlockCnt;
};

class MatmulBaseBlockL2Cache : public MatmulBaseBlockMC2 {
public:
    __aicore__ inline MatmulBaseBlockL2Cache()
    {
    }
    __aicore__ inline void UpdateBlockCnt(int32_t mTileIndex, int32_t nTileIndex);
    __aicore__ inline void UpdateBlockParams(int32_t mTileIndex = 0, int32_t nTileIndex = 0);
    __aicore__ inline void Init(Mc2Tiling::RCSTiling &cfg, TCubeTiling &tiling, Mc2Tiling::TileL2Tiling &l2Tiling,
                                uint32_t rankID = 0);
    __aicore__ inline void UpdateBlockOffset(int32_t mL2TileIndex, int32_t nL2TileIndex);
    __aicore__ inline void InitBlockIndex(uint32_t index = 0);

public:
    uint32_t rankID_;
    L2CacheTileArguments l2Args_;
    Mc2Tiling::TileL2Tiling l2Tiling_;
};

__aicore__ inline void MatmulBaseBlockL2Cache::Init(Mc2Tiling::RCSTiling &cfg, TCubeTiling &tiling,
                                                    Mc2Tiling::TileL2Tiling &l2Tiling, uint32_t rankID)
{
    MatmulBaseBlockMC2::Init(cfg, tiling, l2Tiling);
    l2Tiling_ = l2Tiling;
    rankID_ = rankID;
    args_.isRowOrder = true;
    if (tiling_.N > 2 * tiling_.M * l2Tiling_.rankTileNum) { // 2: ratio of rowOrder
        args_.isRowOrder = false;
    }
    l2Args_.mOneTileBlockCnt = DivCeil(tiling_.M, tiling_.baseM); // M方向分Base块个数
}

__aicore__ inline void MatmulBaseBlockL2Cache::UpdateBlockCnt(int32_t mTileIndex, int32_t nTileIndex)
{
    if ((mTileIndex == (l2Tiling_.mL2TileCnt - 1)) && (nTileIndex == (l2Tiling_.nL2TileCnt - 1))) {
        args_.mBlockCnt = l2Tiling_.mTailBlocks;
        args_.nBlockCnt = l2Tiling_.nTailBlocks;
    } else if (mTileIndex == (l2Tiling_.mL2TileCnt - 1)) {
        args_.mBlockCnt = l2Tiling_.mTailBlocks;
        args_.nBlockCnt = l2Tiling_.nTileBlocks;
    } else if (nTileIndex == (l2Tiling_.nL2TileCnt - 1)) {
        args_.mBlockCnt = l2Tiling_.mTileBlocks;
        args_.nBlockCnt = l2Tiling_.nTailBlocks;
    } else {
        args_.mBlockCnt = l2Tiling_.mTileBlocks;
        args_.nBlockCnt = l2Tiling_.nTileBlocks;
    }
}

__aicore__ inline void MatmulBaseBlockL2Cache::UpdateBlockParams(int32_t mTileIndex, int32_t nTileIndex)
{
    bool isLastTileM = (mTileIndex == (l2Tiling_.mL2TileCnt - 1));
    bool isLastTileN = (nTileIndex == (l2Tiling_.nL2TileCnt - 1));
    if (isLastTileM && isLastTileN && (args_.blockCurrIdx == (args_.totalBlockCnt - 1))) {
        args_.singleCoreM = args_.mBaseTail;
        args_.singleCoreN = args_.nBaseTail;
    } else if (isLastTileM && (args_.blockCurrIdx >= (args_.mBlockCnt - 1) * args_.nBlockCnt)) {
        // M方向最后一片矩阵的最后一行为尾块
        args_.singleCoreM = args_.mBaseTail;
        args_.singleCoreN = tiling_.baseN;
    } else if (isLastTileN && ((args_.blockCurrIdx + 1) % args_.nBlockCnt == 0)) {
        // N方向的最后一片矩阵的最后一列为尾块
        args_.singleCoreM = tiling_.baseM;
        args_.singleCoreN = args_.nBaseTail;
    } else {
        args_.singleCoreM = tiling_.baseM;
        args_.singleCoreN = tiling_.baseN;
    }

    // M方向单片矩阵的最后一行为尾块
    if (l2Tiling_.rankTileNum > 1) {
        uint32_t rankMBlockIdx = mTileIndex * l2Tiling_.mTileBlocks + args_.blockCurrIdx / args_.nBlockCnt;
        if ((rankMBlockIdx + 1) % l2Args_.mOneTileBlockCnt == 0) {
            args_.singleCoreM = args_.mBaseTail;
        }
    }
    UpdateBlockOffset(mTileIndex, nTileIndex);
}

__aicore__ inline void MatmulBaseBlockL2Cache::UpdateBlockOffset(int32_t mL2TileIndex, int32_t nL2TileIndex)
{
    uint32_t mL2TileBlockIdx = args_.blockCurrIdx / args_.nBlockCnt; // L2切分后矩阵的基本块所在的行索引
    uint32_t nL2TileBlockIdx = args_.blockCurrIdx % args_.nBlockCnt; // L2切分后矩阵的基本块所在的列索引
    uint32_t rankMBlockIdx = mL2TileIndex * l2Tiling_.mTileBlocks + mL2TileBlockIdx;
    auto mTileIndex = rankMBlockIdx / l2Args_.mOneTileBlockCnt;
    auto mBlockIndex = rankMBlockIdx % l2Args_.mOneTileBlockCnt;

    uint32_t stride = 0;
    if (l2Tiling_.rankTileNum == cfg_.rankDim - 1) { // allgather skip local rank
        stride = mTileIndex >= rankID_ ? cfg_.rankM : 0;
    }
    auto rankM = cfg_.rankM;
    if (l2Tiling_.rankTileNum == cfg_.rankDim) { // reducescatter
        rankM = cfg_.rankM / cfg_.rankDim;
    }

    args_.mBlockOffset = mBlockIndex * tiling_.baseM + mTileIndex * rankM + stride;
    args_.nBlockOffset = nL2TileBlockIdx * tiling_.baseN + nL2TileIndex * l2Tiling_.nTileBlocks * tiling_.baseN;
    args_.mCWorkOffset = args_.mBlockOffset;
}

__aicore__ inline void MatmulBaseBlockL2Cache::InitBlockIndex(uint32_t index)
{
    (void)index;
    args_.totalBlockCnt = args_.mBlockCnt * args_.nBlockCnt;
    args_.blockCnt = args_.totalBlockCnt / tiling_.usedCoreNum;
    args_.preCoreNum = args_.totalBlockCnt % tiling_.usedCoreNum;

    // 多分配1个基本块的核索引, 从上一次结束位置开始
    auto startIdx = args_.preCoreStartIdx;
    auto endIdx = (startIdx + args_.preCoreNum) % tiling_.usedCoreNum;
    args_.preCoreStartIdx = endIdx;
    GetBlockStartIdx(startIdx, endIdx);
}

enum SplitType {
    DEFAULT = 0,
    L2CACHE = 1
};

template <SplitType T>
struct BlockType {
    __aicore__ inline BlockType(){};
};

template <>
struct BlockType<DEFAULT> {
    __aicore__ inline BlockType(){};
    using PARAMS = MatmulBaseBlockMC2;
};

template <>
struct BlockType<L2CACHE> {
    __aicore__ inline BlockType(){};
    using PARAMS = MatmulBaseBlockL2Cache;
};
} // namespace AscendC
#endif // MC2_MATMUL_BLOCK_L2CACHE_H
