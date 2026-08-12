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
 * \file batch_mat_mul_v3_block.h
 * \brief
 */
#ifndef BATCH_MATMUL_V3_BLOCK_H
#define BATCH_MATMUL_V3_BLOCK_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../mat_mul_v3/op_kernel/mat_mul_v3_common.h"
#include "lib/matmul_intf.h"
#include "batch_mat_mul_v3_tiling_data.h"

using namespace AscendC;
using namespace matmul;

struct Mc2MulMultiBatchBaseBlockOffset {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetBias;
};

struct Mc2MulMultiBatchBaseBlockArgs {
    uint64_t singleASize;
    uint64_t singleASizeL1;
    uint64_t singleBSize;
    uint64_t singleCSize;
    uint64_t singleBiasSize;
    uint64_t mainLoopPreCoreBatchNum;
    uint64_t lastLoopAllBatchNum;
    uint64_t mainLoopPreCoreBatchNumB;
    uint64_t mainLoopBatchNumB;
    uint64_t lastLoopPreCoreBatchNum;
    uint64_t lastLoopBlockNum;
    uint64_t loopTimes;
    uint64_t batchIndex;
    uint64_t batchNum;
    uint64_t useCoreNum;
    uint64_t batchOffset;
    uint64_t batchOffsetA;
    uint64_t batchOffsetB;
    uint64_t mainLoopBatchNum;
    bool isTransposeA;
    bool isTransposeB;
    bool isHf32;
};

class Mc2BatchMatMulMultiBatchFullLoadBlock {
public:
    __aicore__ inline Mc2BatchMatMulMultiBatchFullLoadBlock()
    {
    }
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void SetC0(uint64_t c0Size)
    {
        c0Size_ = c0Size;
    }
    __aicore__ inline void GetMultiBatchInfo(uint64_t loopIndex);
    __aicore__ inline void CalcGMOffset();

public:
    Mc2MulMultiBatchBaseBlockOffset offset_;
    Mc2MulMultiBatchBaseBlockArgs params_;
    const Mc2BatchMatmulTilingData *batchMatmulTilingData_;

protected:
    uint64_t c0Size_;
};

__aicore__ inline void Mc2BatchMatMulMultiBatchFullLoadBlock::Init(const void *tilingData)
{
    batchMatmulTilingData_ = static_cast<const Mc2BatchMatmulTilingData *>(tilingData);
    params_.isTransposeA = batchMatmulTilingData_->matmulTiling.matmulRunInfo.transA;
    params_.isTransposeB = batchMatmulTilingData_->matmulTiling.matmulRunInfo.transB;
    params_.isHf32 = batchMatmulTilingData_->matmulTiling.matmulRunInfo.isHf32;

    uint32_t innerA = batchMatmulTilingData_->matmulTiling.matmulRunInfo.transA ?
                          batchMatmulTilingData_->matmulTiling.matmulTiling.M :
                          batchMatmulTilingData_->matmulTiling.matmulTiling.Ka;
    uint32_t outterA = batchMatmulTilingData_->matmulTiling.matmulRunInfo.transA ?
                           batchMatmulTilingData_->matmulTiling.matmulTiling.Ka :
                           batchMatmulTilingData_->matmulTiling.matmulTiling.M;

    // 用于计算GM上的地址偏移
    params_.singleASize = static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                          batchMatmulTilingData_->matmulTiling.matmulTiling.Ka;
    params_.singleBSize = static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.N) *
                          batchMatmulTilingData_->matmulTiling.matmulTiling.Kb;
    params_.singleCSize = static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                          batchMatmulTilingData_->matmulTiling.matmulTiling.N;

    // 用于计算A矩阵在L1上的地址偏移
    innerA = DivCeil(innerA, c0Size_) * c0Size_;
    outterA = DivCeil(outterA, ALIGNED_H) * ALIGNED_H;
    params_.singleASizeL1 = innerA * outterA;

    params_.singleBiasSize = static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.N);
    params_.useCoreNum = batchMatmulTilingData_->Mc2multiBatchInfo.batchUsedCoreNum;

    // 每次循环往L1搬入的A矩阵batch数和B矩阵的batch数
    params_.mainLoopPreCoreBatchNum = batchMatmulTilingData_->Mc2multiBatchInfo.aBatch;
    params_.mainLoopBatchNum = batchMatmulTilingData_->Mc2multiBatchInfo.aBatch * params_.useCoreNum;
    params_.mainLoopPreCoreBatchNumB = batchMatmulTilingData_->Mc2multiBatchInfo.bBatch;
    params_.mainLoopBatchNumB = batchMatmulTilingData_->Mc2multiBatchInfo.bBatch * params_.useCoreNum;

    // 计算外层循环次数
    params_.loopTimes = MMV3DivCeil(batchMatmulTilingData_->Mc2multiBatchInfo.cBatchDimAll, params_.mainLoopBatchNum);

    params_.lastLoopAllBatchNum = batchMatmulTilingData_->Mc2multiBatchInfo.cBatchDimAll % params_.mainLoopBatchNum;
    params_.lastLoopAllBatchNum =
        params_.lastLoopAllBatchNum == 0 ? params_.mainLoopBatchNum : params_.lastLoopAllBatchNum;

    params_.lastLoopPreCoreBatchNum = MMV3DivFloor(params_.lastLoopAllBatchNum, params_.useCoreNum);
    params_.lastLoopBlockNum = params_.lastLoopAllBatchNum % params_.useCoreNum;
    params_.batchIndex = 0;
    params_.batchNum = 1;
    params_.batchOffset = 0;
    params_.batchOffsetA = 0;
    params_.batchOffsetB = 0;
}

// 更新每次循环中，需要处理的batch数，以及每个batch在所有batch中的索引
__aicore__ inline void Mc2BatchMatMulMultiBatchFullLoadBlock::GetMultiBatchInfo(uint64_t loopIndex)
{
    // main loop
    if (loopIndex + 1 < params_.loopTimes) {
        params_.batchNum = params_.mainLoopPreCoreBatchNum;
        params_.batchIndex =
            loopIndex * params_.mainLoopBatchNum + GetCurrentBlockIdx() * params_.mainLoopPreCoreBatchNum;
        return;
    }

    // last loop
    if (GetCurrentBlockIdx() < params_.lastLoopBlockNum) {
        params_.batchNum = params_.lastLoopPreCoreBatchNum + 1;
        params_.batchIndex = loopIndex * params_.mainLoopBatchNum + GetCurrentBlockIdx() * params_.batchNum;
    } else {
        params_.batchNum = params_.lastLoopPreCoreBatchNum;
        params_.batchIndex = loopIndex * params_.mainLoopBatchNum +
                             params_.lastLoopBlockNum * (params_.lastLoopPreCoreBatchNum + 1) +
                             (GetCurrentBlockIdx() - params_.lastLoopBlockNum) * params_.lastLoopPreCoreBatchNum;
    }
}

__aicore__ inline void Mc2BatchMatMulMultiBatchFullLoadBlock::CalcGMOffset()
{
    offset_.offsetA = params_.batchIndex * params_.singleASize + params_.batchOffsetA * params_.singleASize;
    offset_.offsetB = params_.batchIndex * params_.singleBSize + params_.batchOffsetB * params_.singleBSize;
    offset_.offsetC = params_.batchIndex * params_.singleCSize + params_.batchOffset * params_.singleCSize;
    offset_.offsetBias = params_.batchIndex * params_.singleBiasSize + params_.batchOffset * params_.singleBiasSize;
}

class Mc2BatchMatMulMultiBatchBaseBlock {
public:
    __aicore__ inline Mc2BatchMatMulMultiBatchBaseBlock()
    {
    }
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void GetMultiBatchInfo(uint64_t loopIndex);
    __aicore__ inline void CalcGMOffset();

public:
    Mc2MulMultiBatchBaseBlockOffset offset_;
    Mc2MulMultiBatchBaseBlockArgs params_;
    const Mc2BatchMatmulTilingData *batchMatmulTilingData_;
};

__aicore__ inline void Mc2BatchMatMulMultiBatchBaseBlock::Init(const void *tilingData)
{
    batchMatmulTilingData_ = static_cast<const Mc2BatchMatmulTilingData *>(tilingData);
    params_.isTransposeA = batchMatmulTilingData_->matmulTiling.matmulRunInfo.transA;
    params_.isTransposeB = batchMatmulTilingData_->matmulTiling.matmulRunInfo.transB;
    params_.isHf32 = batchMatmulTilingData_->matmulTiling.matmulRunInfo.isHf32;

    params_.singleASize = static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                          batchMatmulTilingData_->matmulTiling.matmulTiling.Ka;
    params_.singleBSize = static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.N) *
                          batchMatmulTilingData_->matmulTiling.matmulTiling.Kb;
    params_.singleCSize = static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.M) *
                          batchMatmulTilingData_->matmulTiling.matmulTiling.N;
    params_.singleBiasSize = static_cast<uint64_t>(batchMatmulTilingData_->matmulTiling.matmulTiling.N);
    params_.useCoreNum = batchMatmulTilingData_->Mc2multiBatchInfo.batchUsedCoreNum;
    params_.mainLoopPreCoreBatchNum = batchMatmulTilingData_->Mc2multiBatchInfo.iterBatch;
    params_.mainLoopBatchNum = params_.mainLoopPreCoreBatchNum * params_.useCoreNum;

    params_.loopTimes = MMV3DivCeil(batchMatmulTilingData_->Mc2multiBatchInfo.cBatchDimAll, params_.mainLoopBatchNum);

    params_.lastLoopAllBatchNum = batchMatmulTilingData_->Mc2multiBatchInfo.cBatchDimAll % params_.mainLoopBatchNum;
    params_.lastLoopAllBatchNum =
        params_.lastLoopAllBatchNum == 0 ? params_.mainLoopBatchNum : params_.lastLoopAllBatchNum;

    params_.lastLoopPreCoreBatchNum = MMV3DivFloor(params_.lastLoopAllBatchNum, params_.useCoreNum);
    params_.lastLoopBlockNum = params_.lastLoopAllBatchNum % params_.useCoreNum;
    params_.batchIndex = 0;
    params_.batchNum = 1;
    params_.batchOffset = 0;
    params_.batchOffsetA = 0;
    params_.batchOffsetB = 0;
}

__aicore__ inline void Mc2BatchMatMulMultiBatchBaseBlock::GetMultiBatchInfo(uint64_t loopIndex)
{
    // main loop
    if (loopIndex + 1 < params_.loopTimes) {
        params_.batchNum = params_.mainLoopPreCoreBatchNum;
        params_.batchIndex =
            loopIndex * params_.mainLoopBatchNum + GetCurrentBlockIdx() * params_.mainLoopPreCoreBatchNum;
        return;
    }

    // last loop
    if (GetCurrentBlockIdx() < params_.lastLoopBlockNum) {
        params_.batchNum = params_.lastLoopPreCoreBatchNum + 1;
        params_.batchIndex = loopIndex * params_.mainLoopBatchNum + GetCurrentBlockIdx() * params_.batchNum;
    } else {
        params_.batchNum = params_.lastLoopPreCoreBatchNum;
        params_.batchIndex = loopIndex * params_.mainLoopBatchNum +
                             params_.lastLoopBlockNum * (params_.lastLoopPreCoreBatchNum + 1) +
                             (GetCurrentBlockIdx() - params_.lastLoopBlockNum) * params_.lastLoopPreCoreBatchNum;
    }
}

__aicore__ inline void Mc2BatchMatMulMultiBatchBaseBlock::CalcGMOffset()
{
    offset_.offsetA = params_.batchIndex * params_.singleASize + params_.batchOffsetA * params_.singleASize;
    offset_.offsetB = params_.batchIndex * params_.singleBSize + params_.batchOffsetB * params_.singleBSize;
    offset_.offsetC = params_.batchIndex * params_.singleCSize + params_.batchOffset * params_.singleCSize;
    offset_.offsetBias = params_.batchIndex * params_.singleBiasSize + params_.batchOffset * params_.singleBiasSize;
}


class Mc2BatchMatMulUnalignedMultiBatchBaseBlock : public Mc2BatchMatMulMultiBatchBaseBlock {
public:
    __aicore__ inline Mc2BatchMatMulUnalignedMultiBatchBaseBlock()
    {
    }
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void SetC0(uint64_t c0Size)
    {
        c0Size_ = c0Size;
    }
    __aicore__ inline void SetNd2nzFlag(uint64_t nd2nzFlag)
    {
        nd2nzFlag_ = nd2nzFlag;
    }
    __aicore__ inline void CalculateLoopTimes(uint64_t nd2nzBatchNum);
    __aicore__ inline void CalculateBatchOffset(uint64_t loopIndex, bool alignedA, bool alignedB);

protected:
    uint64_t c0Size_;
    int32_t nd2nzFlag_;
};

__aicore__ inline void Mc2BatchMatMulUnalignedMultiBatchBaseBlock::Init(const void *tilingData)
{
    Mc2BatchMatMulMultiBatchBaseBlock::Init(tilingData);

    uint32_t innerA = batchMatmulTilingData_->matmulTiling.matmulRunInfo.transA ?
                          batchMatmulTilingData_->matmulTiling.matmulTiling.M :
                          batchMatmulTilingData_->matmulTiling.matmulTiling.Ka;
    uint32_t outterA = batchMatmulTilingData_->matmulTiling.matmulRunInfo.transA ?
                           batchMatmulTilingData_->matmulTiling.matmulTiling.Ka :
                           batchMatmulTilingData_->matmulTiling.matmulTiling.M;
    uint32_t innerB = batchMatmulTilingData_->matmulTiling.matmulRunInfo.transB ?
                          batchMatmulTilingData_->matmulTiling.matmulTiling.Kb :
                          batchMatmulTilingData_->matmulTiling.matmulTiling.N;
    uint32_t outterB = batchMatmulTilingData_->matmulTiling.matmulRunInfo.transB ?
                           batchMatmulTilingData_->matmulTiling.matmulTiling.N :
                           batchMatmulTilingData_->matmulTiling.matmulTiling.Kb;

    innerA = DivCeil(innerA, c0Size_) * c0Size_;
    innerB = DivCeil(innerB, c0Size_) * c0Size_;
    outterA = DivCeil(outterA, ALIGNED_H) * ALIGNED_H;
    outterB = DivCeil(outterB, ALIGNED_H) * ALIGNED_H;
    if (nd2nzFlag_ == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) {
        params_.singleASize = innerA * outterA;
    } else if (nd2nzFlag_ == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) {
        params_.singleBSize = innerB * outterB;
    } else if (nd2nzFlag_ == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB)) {
        params_.singleASize = innerA * outterA;
        params_.singleBSize = innerB * outterB;
    }
    CalculateLoopTimes(batchMatmulTilingData_->Mc2multiBatchInfo.batchTileBlock);
}

__aicore__ inline void Mc2BatchMatMulUnalignedMultiBatchBaseBlock::CalculateLoopTimes(uint64_t nd2nzBatchNum)
{
    params_.loopTimes = MMV3DivCeil(nd2nzBatchNum, params_.mainLoopPreCoreBatchNum * params_.useCoreNum);

    params_.lastLoopAllBatchNum = nd2nzBatchNum % (params_.mainLoopPreCoreBatchNum * params_.useCoreNum);
    params_.lastLoopAllBatchNum = params_.lastLoopAllBatchNum == 0 ?
                                      params_.mainLoopPreCoreBatchNum * params_.useCoreNum :
                                      params_.lastLoopAllBatchNum;

    params_.lastLoopPreCoreBatchNum = MMV3DivFloor(params_.lastLoopAllBatchNum, params_.useCoreNum);
    params_.lastLoopBlockNum = params_.lastLoopAllBatchNum % params_.useCoreNum;
}

__aicore__ inline void Mc2BatchMatMulUnalignedMultiBatchBaseBlock::CalculateBatchOffset(uint64_t loopIndex,
                                                                                        bool alignedA, bool alignedB)
{
    params_.batchOffset = loopIndex * batchMatmulTilingData_->Mc2multiBatchInfo.batchTileBlock;
    params_.batchOffsetA = params_.batchOffset;
    params_.batchOffsetB = params_.batchOffset;
    if (((nd2nzFlag_ == static_cast<int32_t>(ND2NZ_SELECT::ONLY_A)) ||
         (nd2nzFlag_ == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB))) &&
        !alignedA) {
        // loopIndex % 2 == 1,pong时在workspace偏移一个batchTileBlock
        params_.batchOffsetA = batchMatmulTilingData_->Mc2multiBatchInfo.batchTileBlock * (loopIndex % 2);
    }
    if (((nd2nzFlag_ == static_cast<int32_t>(ND2NZ_SELECT::ONLY_B)) ||
         (nd2nzFlag_ == static_cast<int32_t>(ND2NZ_SELECT::BOTH_AB))) &&
        !alignedB) {
        // loopIndex % 2 == 1,pong时在workspace偏移一个batchTileBlock
        params_.batchOffsetB = batchMatmulTilingData_->Mc2multiBatchInfo.batchTileBlock * (loopIndex % 2);
    }
}

#endif // BATCH_MATMUL_V3_BLOCK_H
