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
 * \file quant_batch_matmul_v3_block.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_BLOCK_H
#define MC2_QUANT_BATCH_MATMUL_V3_BLOCK_H

#include "quant_batch_matmul_v3_base.h"

namespace AscendC {
class Mc2QuantBatchMatmulV3BaseBlock {
public:
    __aicore__ inline Mc2QuantBatchMatmulV3BaseBlock()
    {
    }
    __aicore__ inline void Init(const Mc2QuantBatchMatmulV3TilingData *tilingData);
    __aicore__ inline void Init(const Mc2L2cacheTileParam *tilingL2, const TCubeTiling *matmulTilingData);
    __aicore__ inline void UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void InitFirstTileBlockIndex();
    __aicore__ inline void InitBlockIndex();
    __aicore__ inline void UpdateBlockIndex();

public:
    QBmmBaseBlockArgs params_;
    bool indexInit_ = false;
    uint32_t rowOrder_;
    uint64_t mCnt_;
    uint64_t nCnt_;
    uint64_t mCntTail_;
    uint64_t nCntTail_;
    uint64_t round_;
    uint64_t realRound_;
    uint64_t preCoreNum_;
    uint64_t blockIdxStart_;
    uint64_t blockIdxEnd_;
    uint32_t blockIdx_;
    const TCubeTiling *matmulTilingData_;
};

__aicore__ inline void Mc2QuantBatchMatmulV3BaseBlock::Init(const Mc2QuantBatchMatmulV3TilingData *tilingData)
{
    this->Init(&(tilingData->tileL2cacheTiling), &(tilingData->matmulTiling));
}

__aicore__ inline void Mc2QuantBatchMatmulV3BaseBlock::Init(const Mc2L2cacheTileParam *tilingL2,
                                                            const TCubeTiling *matmulTilingData)
{
    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    if ASCEND_IS_AIV {
        blockIdx_ = blockIdx_ / GetTaskRation();
    }
    matmulTilingData_ = matmulTilingData;
    params_.mTileCntL2 = static_cast<uint64_t>(tilingL2->mTileCntL2); // M方向的Tile份数
    params_.nTileCntL2 = static_cast<uint64_t>(tilingL2->nTileCntL2); // N方向的Tile份数
    params_.mTotalCnt = (static_cast<uint64_t>(matmulTilingData_->M) + matmulTilingData_->singleCoreM - 1) /
                        matmulTilingData_->singleCoreM; // 总的m方向base块个数
    params_.nTotalCnt = (static_cast<uint64_t>(matmulTilingData_->N) + matmulTilingData_->singleCoreN - 1) /
                        matmulTilingData_->singleCoreN; // 总的n方向base块个数
    // 当前tiling保证mTileCntL2和nTileCntL2合法性
    // 需要保证mTileCntL2和nTileCntL2的切分策略正好满足整块+尾块的处理
    mCnt_ = tilingL2->mTileBlock; // 每一份mTile包含的base块个数
    nCnt_ = tilingL2->nTileBlock; // 每一份nTile包含的base块个数
    params_.totalTileCnt = mCnt_ * nCnt_;

    mCntTail_ = params_.mTotalCnt - (params_.mTileCntL2 - 1) * mCnt_; // M方向上mTile尾块里的base块的个数
    nCntTail_ = params_.nTotalCnt - (params_.nTileCntL2 - 1) * nCnt_; // M方向上nTile尾块里的base块的个数
    round_ = (params_.totalTileCnt + matmulTilingData_->usedCoreNum - 1) / matmulTilingData_->usedCoreNum;

    preCoreNum_ = params_.totalTileCnt % matmulTilingData_->usedCoreNum;
    params_.mCntUse = mCnt_;
    params_.nCntUse = nCnt_;

    // calOrder在tiling里没有被默认赋值
    rowOrder_ = tilingL2->calOrder > 0 ? tilingL2->calOrder : ROW_FIRST; // 默认行优先
}

__aicore__ inline void Mc2QuantBatchMatmulV3BaseBlock::UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex)
{
    params_.mTileAddrOffset = mTileIndex * mCnt_ * matmulTilingData_->singleCoreM;
    params_.nTileAddrOffset = nTileIndex * nCnt_ * matmulTilingData_->singleCoreN;

    if ((mTileIndex == (params_.mTileCntL2 - 1)) && (nTileIndex == (params_.nTileCntL2 - 1))) {
        params_.totalTileCnt = mCntTail_ * nCntTail_;
        params_.mCntUse = mCntTail_;
        params_.nCntUse = nCntTail_;
    } else if (mTileIndex == (params_.mTileCntL2 - 1)) {
        params_.totalTileCnt = mCntTail_ * nCnt_;
        params_.mCntUse = mCntTail_;
        params_.nCntUse = nCnt_;
    } else if (nTileIndex == (params_.nTileCntL2 - 1)) {
        params_.totalTileCnt = mCnt_ * nCntTail_;
        params_.mCntUse = mCnt_;
        params_.nCntUse = nCntTail_;
    } else {
        params_.totalTileCnt = mCnt_ * nCnt_;
        params_.mCntUse = mCnt_;
        params_.nCntUse = nCnt_;
    }

    round_ = DequantBmm::CeilDiv(params_.totalTileCnt, static_cast<uint64_t>(matmulTilingData_->usedCoreNum));
    preCoreNum_ = params_.totalTileCnt - (round_ - 1) * matmulTilingData_->usedCoreNum;
    if (preCoreNum_ == 0) {
        preCoreNum_ = static_cast<uint64_t>(matmulTilingData_->usedCoreNum);
    }
}

__aicore__ inline void Mc2QuantBatchMatmulV3BaseBlock::InitFirstTileBlockIndex()
{
    params_.mTileAddrOffset = 0;
    params_.nTileAddrOffset = 0;
    indexInit_ = true; // 负载均衡初始化标志位
    blockIdxEnd_ = preCoreNum_; // 结束运算时，尾核的索引；下一次L2切分区域多一轮开始计算的index，负载均衡
    if (preCoreNum_ == 0) {
        preCoreNum_ = matmulTilingData_->usedCoreNum;
    }
    uint64_t preTotalBlock = 0U;
    if (blockIdx_ < preCoreNum_) {
        if (rowOrder_ == ROW_FIRST) {
            params_.index = blockIdx_ * round_;
        } else {
            preTotalBlock = blockIdx_ * round_;
            params_.index = preTotalBlock / mCnt_ + (preTotalBlock % mCnt_) * nCnt_;
        }
        realRound_ = round_;
    } else {
        if (rowOrder_ == ROW_FIRST) {
            params_.index = blockIdx_ * (round_ - 1) + preCoreNum_;
        } else {
            preTotalBlock = blockIdx_ * (round_ - 1) + preCoreNum_;
            params_.index = preTotalBlock / mCnt_ + (preTotalBlock % mCnt_) * nCnt_;
        }
        realRound_ = round_ - 1;
    }
}

__aicore__ inline void Mc2QuantBatchMatmulV3BaseBlock::InitBlockIndex()
{
    if (indexInit_) {
        blockIdxStart_ = blockIdxEnd_; // 开始运算时，首核的索引
    } else {
        blockIdxStart_ = 0; // 开始运算时，首核的索引; 最开始计算从0核开始
        indexInit_ = true;
    }
    blockIdxEnd_ = (blockIdxStart_ + preCoreNum_) % matmulTilingData_->usedCoreNum; // 结束运算时，尾核的索引
    uint64_t indexStart = blockIdxStart_;
    uint64_t indexEnd = blockIdxEnd_;

    // 利用roudCnt来解决尾块负载均衡问题
    if (indexStart < indexEnd) {
        // 正常排序, preCore在整个Cores的中间
        if (blockIdx_ < indexStart) {
            params_.index = blockIdx_ * (round_ - 1);
            realRound_ = round_ - 1;
        } else if (blockIdx_ < indexEnd) {
            params_.index = indexStart * (round_ - 1) + (blockIdx_ - indexStart) * round_;
            realRound_ = round_;
        } else {
            params_.index = (indexStart * (round_ - 1) + preCoreNum_ * round_ + (blockIdx_ - indexEnd) * (round_ - 1));
            realRound_ = round_ - 1;
        }
        if (rowOrder_ == COL_FIRST) {
            // 列优先分配
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    } else if (indexEnd < indexStart) {
        // indexEnd会翻转
        if (blockIdx_ < indexEnd) {
            params_.index = blockIdx_ * round_;
            realRound_ = round_;
        } else if (blockIdx_ < indexStart) {
            params_.index = indexEnd * round_ + (blockIdx_ - indexEnd) * (round_ - 1);
            realRound_ = round_ - 1;
        } else {
            params_.index =
                (indexEnd * round_ + (indexStart - indexEnd) * (round_ - 1) + (blockIdx_ - indexStart) * round_);
            realRound_ = round_;
        }
        if (rowOrder_ == COL_FIRST) {
            // 列优先分配
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    } else {
        // 不存在尾核，基本块对齐
        params_.index = blockIdx_ * round_;
        realRound_ = round_;
        if (rowOrder_ == COL_FIRST) {
            // 列优先分配
            params_.index = params_.index / params_.mCntUse + params_.index % params_.mCntUse * params_.nCntUse;
        }
    }
}

__aicore__ inline void Mc2QuantBatchMatmulV3BaseBlock::UpdateBlockIndex()
{
    if (rowOrder_ == ROW_FIRST) {
        params_.index += 1;
    } else if (rowOrder_ == COL_FIRST) {
        params_.index += params_.nCntUse;
        if (params_.index >= params_.totalTileCnt) {
            params_.index = params_.index % params_.totalTileCnt + 1;
        }
    }
}
} // namespace AscendC
#endif // QUANT_BATCH_MATMUL_V3_BLOCK_H
