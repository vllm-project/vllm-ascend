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
 * \file scatter_nd_update_asc_pure_copy.h
 * \brief
 */

#ifndef SCATTER_ND_UPDATE_ASC_PURE_COPY_H
#define SCATTER_ND_UPDATE_ASC_PURE_COPY_H

#include "kernel_operator.h"

namespace ScatterNdUpdateAsc {
using namespace AscendC;

// ---- 非连续首轴(view)适配，与 ops-nn a2(arch22) 方案一致 ----
__aicore__ inline uint64_t ComputeViewedRowOffset(uint64_t linearIndex, uint64_t firstDimStrideRows,
                                                  uint64_t varStride0Elements, uint64_t scatterLength)
{
    if (firstDimStrideRows == 0) {
        return linearIndex * scatterLength;
    }
    uint64_t i0 = linearIndex / firstDimStrideRows;
    uint64_t rest = linearIndex - i0 * firstDimStrideRows;
    return i0 * varStride0Elements + rest * scatterLength;
}

// 统一处理 view-stride0 与连续两条路径的输出偏移计算。
template <bool isViewStride0>
__aicore__ inline uint64_t ResolveOutOffset(uint64_t linearIndex, uint64_t scatterLength, uint64_t firstDimStrideRows,
                                            uint64_t varStride0Elements, uint64_t tileOffsetElements)
{
    if constexpr (isViewStride0) {
        return ComputeViewedRowOffset(linearIndex, firstDimStrideRows, varStride0Elements, scatterLength) +
               tileOffsetElements;
    } else {
        return linearIndex * scatterLength + tileOffsetElements;
    }
}

template <typename DVAR, typename DINDICES, bool isViewStride0 = false>
class ScatterNdUpdateAscPureCopy {
public:
    // static constexpr bool ifGroupAlpha_ = true;
    __aicore__ inline ScatterNdUpdateAscPureCopy(TPipe* pipe)
    {
        pipe_ = pipe;
    };

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR update, GM_ADDR y, const ScatterNdUpdateAscTilingData* tilingData) {
        // init tilingdata
        a_ = tilingData->a;
        b_ = tilingData->b;
        bAlign_ = tilingData->bAlign;
        c_ = tilingData->c;
        ubFactor_ = tilingData->ubFactor;
        blockFactor_ = tilingData->blockFactor;
        blockFactorTail_ = tilingData->blockFactorTail;
        blockNum_ = tilingData->blockNum;
        varStride0Elements_ = tilingData->varStride0Elements;
        firstDimStrideRows_ = tilingData->firstDimStrideRows;
        indexDim_ = tilingData->indexDim;
        varDim1_ = tilingData->varDim1;

        // init
        blockIdx_ = GetBlockIdx();
        curBlockFactor_ = blockFactor_;
        if (blockIdx_ == (blockNum_ - 1)) {
            curBlockFactor_ = blockFactorTail_;
        }
        curUBLoops_ = (curBlockFactor_ + ubFactor_ - 1) / ubFactor_;
        curUbFactorTail_ = curBlockFactor_ - (curUBLoops_ - 1) * ubFactor_;
        
        // int gm tensor
        varGm_.SetGlobalBuffer((__gm__ DVAR*)var);
        indicesGm_.SetGlobalBuffer((__gm__ DINDICES*)indices + blockIdx_ * blockFactor_ * indexDim_);
        updateGm_.SetGlobalBuffer((__gm__ DVAR*)update + blockIdx_ * blockFactor_ * b_);
        yGm_.SetGlobalBuffer((__gm__ DVAR*)y);

        // init ub tensor
        pipe_->InitBuffer(updateQueue_, 2, ubFactor_ * bAlign_ * sizeof(DVAR));
    }

    // 读取第 linIdx 个 indices 行并线性化为 var 中的"行号"（拷贝块起点所在的行）：
    //   K=1: row = indices[i,0]
    //   K=2: row = indices[i,0]*varDim1_ + indices[i,1]
    // 任一分量 < 0（padding 槽位）时返回 -1，由调用方跳过，避免写出界。
    __aicore__ inline int64_t ResolveLinearRow(int64_t linIdx)
    {
        int64_t raw0 = static_cast<int64_t>(indicesGm_(linIdx * indexDim_));
        if (indexDim_ == 1) {
            return raw0;
        }
        int64_t raw1 = static_cast<int64_t>(indicesGm_(linIdx * indexDim_ + 1));
        if (raw0 < 0 || raw1 < 0) {
            return -1;
        }
        return raw0 * varDim1_ + raw1;
    }

    __aicore__ inline void Process() {

        for(int64_t i = 0; i < curUBLoops_; i++) {
            curUbFactor_ = (i == (curUBLoops_ - 1)) ? curUbFactorTail_ : ubFactor_;

            // main process
            LocalTensor<DVAR> updateLocal = updateQueue_.AllocTensor<DVAR>();
            DataCopyPadExtParams<DVAR> dataCopyPadExtParamsVar;
            dataCopyPadExtParamsVar.isPad = false;
            dataCopyPadExtParamsVar.leftPadding = 0;
            dataCopyPadExtParamsVar.rightPadding = 0;
            dataCopyPadExtParamsVar.paddingValue = 0;
            DataCopyExtParams copyInParamsVar;
            copyInParamsVar.blockCount = curUbFactor_;
            copyInParamsVar.blockLen =  b_ * sizeof(DVAR);
            copyInParamsVar.srcStride = 0;
            copyInParamsVar.dstStride = 0;
            DataCopyPad(updateLocal, updateGm_[i * ubFactor_ * b_], copyInParamsVar, dataCopyPadExtParamsVar);

            updateQueue_.EnQue(updateLocal);
            updateQueue_.DeQue<DVAR>();

            DataCopyExtParams copyInParamsY;
            copyInParamsY.blockCount = 1;
            copyInParamsY.blockLen =  b_ * sizeof(DVAR);
            copyInParamsY.srcStride = 0;
            copyInParamsY.dstStride = 0;
            
            int64_t fourLoops = curUbFactor_ / UNROLL;
            int64_t fourLoopstIails = curUbFactor_ % UNROLL;
            for (int64_t j = 0; j < fourLoops; j++) {
                int64_t idx0 = ResolveLinearRow(i * ubFactor_ + j * UNROLL);
                int64_t idx1 = ResolveLinearRow(i * ubFactor_ + j * UNROLL + UNROLL_IDX1);
                int64_t idx2 = ResolveLinearRow(i * ubFactor_ + j * UNROLL + UNROLL_IDX2);
                int64_t idx3 = ResolveLinearRow(i * ubFactor_ + j * UNROLL + UNROLL_IDX3);
                if (idx0 >= 0) {
                    uint64_t updateoffset0 = ResolveOutOffset<isViewStride0>(static_cast<uint64_t>(idx0),
                        static_cast<uint64_t>(b_), firstDimStrideRows_, varStride0Elements_, 0);
                    DataCopyPad(yGm_[updateoffset0], updateLocal[(j * UNROLL) * bAlign_], copyInParamsY);
                }
                if (idx1 >= 0) {
                    uint64_t updateoffset1 = ResolveOutOffset<isViewStride0>(static_cast<uint64_t>(idx1),
                        static_cast<uint64_t>(b_), firstDimStrideRows_, varStride0Elements_, 0);
                    DataCopyPad(yGm_[updateoffset1], updateLocal[(j * UNROLL + UNROLL_IDX1) * bAlign_], copyInParamsY);
                }
                if (idx2 >= 0) {
                    uint64_t updateoffset2 = ResolveOutOffset<isViewStride0>(static_cast<uint64_t>(idx2),
                        static_cast<uint64_t>(b_), firstDimStrideRows_, varStride0Elements_, 0);
                    DataCopyPad(yGm_[updateoffset2], updateLocal[(j * UNROLL + UNROLL_IDX2) * bAlign_], copyInParamsY);
                }
                if (idx3 >= 0) {
                    uint64_t updateoffset3 = ResolveOutOffset<isViewStride0>(static_cast<uint64_t>(idx3),
                        static_cast<uint64_t>(b_), firstDimStrideRows_, varStride0Elements_, 0);
                    DataCopyPad(yGm_[updateoffset3], updateLocal[(j * UNROLL + UNROLL_IDX3) * bAlign_], copyInParamsY);
                }
            }

            for (int64_t k = 0; k < fourLoopstIails; k++) {
                int64_t idx = ResolveLinearRow(i * ubFactor_ + fourLoops * UNROLL + k);
                if (idx >= 0) {
                    uint64_t updateoffset = ResolveOutOffset<isViewStride0>(static_cast<uint64_t>(idx),
                        static_cast<uint64_t>(b_), firstDimStrideRows_, varStride0Elements_, 0);
                    DataCopyPad(yGm_[updateoffset], updateLocal[(fourLoops * UNROLL + k) * bAlign_], copyInParamsY);
                }
            }
            updateQueue_.FreeTensor(updateLocal);
        }

    }

protected:
    // global mem
    GlobalTensor<DVAR> varGm_;
    GlobalTensor<DINDICES> indicesGm_;
    GlobalTensor<DVAR> updateGm_;
    GlobalTensor<DVAR> yGm_;

    // ub memory tensor
    TPipe* pipe_ = nullptr;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> updateQueue_;

    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 0;
    int64_t a_ = 0;
    int64_t b_ = 0;
    int64_t bAlign_ = 0;
    int64_t c_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockFactorTail_ = 0;
    int64_t ubFactor_ = 0;
    // 非连续首轴(view)适配，与 ops-nn a2(arch22) 方案一致
    uint64_t varStride0Elements_ = 0;
    uint64_t firstDimStrideRows_ = 1;
    // indices 第二维（scatter_nd K，1/2）与 var dim1（K=2 计算 linearRow）
    int64_t indexDim_ = 1;
    int64_t varDim1_ = 1;

    int64_t curBlockFactor_ = 0;
    int64_t curUbFactor_ = 0;
    int64_t curUbFactorTail_ = 0;
    int64_t curUBLoops_ = 0;

    static constexpr int64_t UNROLL = 4;
    static constexpr int64_t UNROLL_IDX1 = 1;
    static constexpr int64_t UNROLL_IDX2 = 2;
    static constexpr int64_t UNROLL_IDX3 = 3;
};

} 
#endif