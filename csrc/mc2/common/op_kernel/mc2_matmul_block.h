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
 * \file mc2_matmul_block.h
 * \brief
 */

#ifndef MC2_MATMUL_BLOCK_H
#define MC2_MATMUL_BLOCK_H

namespace AscendC {

constexpr uint32_t C0_SIZE = 16;

struct BaseBlockOffset {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetBias;
};

struct BaseBlockArguments {
    bool isRowOrder;
    bool isAtomic;
    bool isTransA;
    bool isTransB;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t mBlockCnt;       // M方向的基本块个数
    uint32_t nBlockCnt;       // N方向的基本块个数
    uint32_t nBaseTail;       // N方向的尾块大小
    uint32_t mBaseTail;       // M方向的尾块大小
    uint32_t totalBlockCnt;   // C矩阵的全部基本块个数
    uint32_t blockCnt;        // 单核需要计算的基本块个数
    uint32_t blockStartIdx;   // 当前核需计算的基本块的起始位置索引
    uint32_t blockCurrIdx;    // 当前核需计算的基本块的当前位置索引
    uint32_t preCoreNum;      // 满核分配后剩余基本块数/需要预分配1个block的核数
    uint32_t preCoreStartIdx; // 多分配一个基本块的核起始位置
    uint64_t mBlockOffset;
    uint64_t nBlockOffset;
    uint64_t mCWorkOffset;
};

class MatmulBaseBlockMC2 {
public:
    __aicore__ inline MatmulBaseBlockMC2()
    {
    }
    __aicore__ inline void Init(Mc2Tiling::RCSTiling &cfg, TCubeTiling &tiling, Mc2Tiling::TileL2Tiling &l2Tiling,
                                uint32_t rankID = 0);
    __aicore__ inline void InitBlockIndex(uint32_t index = 0);
    __aicore__ inline void InitBlockWithoutIndex();
    __aicore__ inline void UpdateBlockIndex(uint32_t currPos);
    __aicore__ inline void UpdateBlockParams(int32_t mTileIndex = 0, int32_t nTileIndex = 0);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void CalcGMOffset();
    __aicore__ inline void GetBlockStartIdx(uint32_t startIdx, uint32_t endIdx);

public:
    BaseBlockOffset offset_;
    BaseBlockArguments args_;
    TCubeTiling tiling_;
    Mc2Tiling::RCSTiling cfg_;
};

__aicore__ inline void MatmulBaseBlockMC2::Init(Mc2Tiling::RCSTiling &cfg, TCubeTiling &tiling,
                                                Mc2Tiling::TileL2Tiling &l2Tiling, uint32_t rankID)
{
    (void)l2Tiling;
    (void)rankID;
    tiling_ = tiling;
    cfg_ = cfg;
    args_.preCoreStartIdx = 0;
    args_.mBlockCnt = DivCeil(tiling.M, tiling.baseM); // M方向分Base块个数
    args_.nBlockCnt = DivCeil(tiling.N, tiling.baseN); // N方向分Base块个数
    args_.nBaseTail = tiling.N - (args_.nBlockCnt - 1) * tiling.baseN;
    args_.mBaseTail = tiling.M - (args_.mBlockCnt - 1) * tiling.baseM;
    args_.totalBlockCnt = args_.mBlockCnt * args_.nBlockCnt;
    args_.isRowOrder = true;
    if (tiling_.N > 5 * tiling_.M) { // 5: ratio of rowOrder
        args_.isRowOrder = false;
    }
    args_.isTransA = cfg.isTransposeA > 0 ? true : false;
    args_.isTransB = cfg.isTransposeB > 0 ? true : false;
    args_.isAtomic = false;
    if (args_.isTransA) {
        args_.isAtomic = true;
    }
}

__aicore__ inline void MatmulBaseBlockMC2::InitBlockIndex(uint32_t index)
{
    args_.totalBlockCnt = args_.mBlockCnt * args_.nBlockCnt;
    args_.blockCnt = args_.totalBlockCnt / tiling_.usedCoreNum;
    args_.preCoreNum = args_.totalBlockCnt % tiling_.usedCoreNum;

    // 多分配1个基本块的核索引, 从上一次结束位置开始
    auto startIdx = index * args_.preCoreNum % tiling_.usedCoreNum;
    auto endIdx = (startIdx + args_.preCoreNum) % tiling_.usedCoreNum;
    GetBlockStartIdx(startIdx, endIdx);
}

__aicore__ inline void MatmulBaseBlockMC2::InitBlockWithoutIndex()
{
    args_.totalBlockCnt = args_.mBlockCnt * args_.nBlockCnt;
    args_.blockCnt = args_.totalBlockCnt / tiling_.usedCoreNum;
    args_.preCoreNum = args_.totalBlockCnt % tiling_.usedCoreNum;

    // 多分配1个基本块的核索引, 从上一次结束位置开始
    auto startIdx = args_.preCoreStartIdx;
    auto endIdx = (startIdx + args_.preCoreNum) % tiling_.usedCoreNum;
    args_.preCoreStartIdx = endIdx;
    GetBlockStartIdx(startIdx, endIdx);
}

__aicore__ inline void MatmulBaseBlockMC2::GetBlockStartIdx(uint32_t startIdx, uint32_t endIdx)
{
    if (startIdx > endIdx) {
        if (block_idx < endIdx) {
            args_.blockCnt += 1;
            args_.blockStartIdx = block_idx * args_.blockCnt;
        } else if (block_idx >= startIdx) {
            args_.blockCnt += 1;
            args_.blockStartIdx = block_idx * args_.blockCnt - (tiling_.usedCoreNum - args_.preCoreNum);
        } else {
            args_.blockStartIdx = block_idx * args_.blockCnt + endIdx;
        }
    } else {
        if (block_idx < startIdx) {
            args_.blockStartIdx = block_idx * args_.blockCnt;
        } else if (block_idx >= endIdx) {
            args_.blockStartIdx = block_idx * args_.blockCnt + args_.preCoreNum;
        } else {
            args_.blockCnt += 1;
            args_.blockStartIdx = block_idx * args_.blockCnt - startIdx;
        }
    }

    if (!args_.isRowOrder) {
        auto blockStart = args_.blockStartIdx;
        args_.blockStartIdx = blockStart / args_.mBlockCnt + blockStart % args_.mBlockCnt * args_.nBlockCnt;
    }
}

__aicore__ inline void MatmulBaseBlockMC2::UpdateBlockIndex(uint32_t currPos)
{
    // 按行取，计算第i个基本块的index
    if (args_.isRowOrder) {
        args_.blockCurrIdx = args_.blockStartIdx + currPos % args_.blockCnt;
        return;
    }

    args_.blockCurrIdx = args_.blockStartIdx + (currPos % args_.blockCnt) * args_.nBlockCnt;
    // 按列取，如果block超行，需计算下一列的位置
    if (args_.blockCurrIdx >= args_.totalBlockCnt) {
        args_.blockCurrIdx = args_.blockCurrIdx % args_.totalBlockCnt + args_.blockCurrIdx / args_.totalBlockCnt;
    }
    return;
}

__aicore__ inline void MatmulBaseBlockMC2::UpdateBlockParams(int32_t mTileIndex, int32_t nTileIndex)
{
    (void)mTileIndex;
    (void)nTileIndex;
    if (args_.blockCurrIdx == (args_.totalBlockCnt - 1)) {
        // 当前矩阵最后一块
        args_.singleCoreM = args_.mBaseTail;
        args_.singleCoreN = args_.nBaseTail;
    } else if (args_.blockCurrIdx >= (args_.mBlockCnt - 1) * args_.nBlockCnt) {
        // 当前矩阵最后一行
        args_.singleCoreM = args_.mBaseTail;
        args_.singleCoreN = tiling_.baseN;
    } else if ((args_.blockCurrIdx + 1) % args_.nBlockCnt == 0) {
        // 当前矩阵最后一列
        args_.singleCoreM = tiling_.baseM;
        args_.singleCoreN = args_.nBaseTail;
    } else {
        args_.singleCoreM = tiling_.baseM;
        args_.singleCoreN = tiling_.baseN;
    }

    // 更新基本块的地址偏移
    args_.mBlockOffset = args_.blockCurrIdx / args_.nBlockCnt * tiling_.baseM; // 基本块所在的行偏移
    args_.nBlockOffset = args_.blockCurrIdx % args_.nBlockCnt * tiling_.baseN; // 基本块所在的列偏移
    args_.mCWorkOffset = args_.mBlockOffset;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatmulBaseBlockMC2::CalcGMOffset()
{
    auto alignedKa = AlignUp(tiling_.Ka, C0_SIZE);
    auto alignedKb = AlignUp(tiling_.Kb, C0_SIZE);

    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (args_.isTransA) {
            offset_.offsetA = args_.mBlockOffset;
        } else {
            offset_.offsetA = args_.mBlockOffset * tiling_.Ka;
        }
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        if (args_.isTransA) {
            offset_.offsetA = args_.mBlockOffset * alignedKa;
        } else {
            offset_.offsetA = args_.mBlockOffset * C0_SIZE;
        }
    }

    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (args_.isTransB) {
            offset_.offsetB = args_.nBlockOffset * tiling_.Kb;
        } else {
            offset_.offsetB = args_.nBlockOffset;
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (args_.isTransB) {
            offset_.offsetB = args_.nBlockOffset * C0_SIZE;
        } else {
            offset_.offsetB = args_.nBlockOffset * alignedKb;
        }
    }

    // C矩阵和BIAS只支持ND
    if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
        offset_.offsetC = args_.nBlockOffset + args_.mCWorkOffset * tiling_.N;
    }
    if constexpr (BIAS_TYPE::format == CubeFormat::ND) {
        offset_.offsetBias = args_.nBlockOffset;
    }
}
} // namespace AscendC
#endif // MC2_MATMUL_BLOCK_H
