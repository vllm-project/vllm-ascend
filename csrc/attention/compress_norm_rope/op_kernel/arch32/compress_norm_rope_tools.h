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
 * \file compress_norm_rope_tools.h
 * \brief 组迭代器：把"全局压缩组序号"映射到 (batch, 组内序号)，并维护 compressedCnt
 *
 * 设计要点（相对 fused compressor 的 slice 迭代器的简化）：
 *   - 任务配额直接以"组"为单位（fused 以 token 行为单位 + tailH/空洞对齐，~300 行）；
 *   - seqused 空洞语义自然包含：组覆盖 [P, P+seqUsed) 之外无工作；
 *   - 全部标量计算，每核一次 Locate + 逐组 Next。
 *
 * 每 batch b（start_pos=P, seq_used=S）：
 *   gBase       = P / r                       （首个工作组的全局组号）
 *   workGroups  = CeilDiv(P+S, r) - gBase     （工作组数 = 本核可能处理的组数）
 *   compGroups  = (P+S) / r - gBase           （实际产出行数 = 完整覆盖的组数）
 *   工作组 w（0 基）：gStart = (gBase+w)*r
 *     tokStart  = max(gStart, P)              （本 call 首个 token 的全局位置）
 *     nTok      = min(gStart+r, P+S) - tokStart
 *     headHolder= tokStart - gStart           （窗口前部来自 state 的历史行数）
 *     produce   = (gStart + r <= P + S)
 */

#ifndef COMPRESS_NORM_ROPE_TOOLS_H
#define COMPRESS_NORM_ROPE_TOOLS_H

#include "compress_norm_rope_comm.h"

namespace CompressNormRope {

struct GroupInfo {
    uint32_t bIdx = 0;
    uint32_t gStart = 0;     // 组起始全局位置（r 对齐）
    uint32_t tokStart = 0;   // 本 call 首个 token 全局位置
    uint32_t nTok = 0;       // 本组本 call token 数
    uint32_t headHolder = 0; // 窗口前部历史行数（仅每 batch 首组可能 > 0）
    uint32_t P = 0;          // start_pos
    uint32_t xRow = 0;       // 本组首个 token 在 mm GM 中的行号
    bool produce = false;    // 是否产出压缩行
};

class GroupIterator {
public:
    __aicore__ inline void Init(__gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos,
                                uint32_t batchSize, uint32_t cmpRatio)
    {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
        isExistSeqUsed_ = (seqUsed != nullptr);
        if (isExistSeqUsed_) {
            seqUsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
        }
        isExistStartPos_ = (startPos != nullptr);
        if (isExistStartPos_) {
            startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
        }
        batchSize_ = batchSize;
        cmpRatio_ = cmpRatio;
    }

    // 定位到全局组序号 targetGroup（0 基，跨 batch 按序展开），
    // 返回该组之前已产出的压缩行数（= 该组的输出行号）。
    // 同时把迭代器游标置于该组（bIdx_/wIdx_），并返回前置 compressedCnt。
    __aicore__ inline uint32_t Locate(uint32_t targetGroup)
    {
        groupAcc_ = 0;
        compAcc_ = 0;
        bIdx_ = 0;
        wIdx_ = 0;
        LoadBatch(0);
        while (bIdx_ < batchSize_) {
            if (workGroups_ > 0 && targetGroup < groupAcc_ + workGroups_) {
                wIdx_ = targetGroup - groupAcc_;
                return CompCntBefore();
            }
            groupAcc_ += workGroups_;
            compAcc_ += compGroups_;
            LoadBatch(bIdx_ + 1);
        }
        // targetGroup 越界（超出实际总组数）：游标置于末尾，无工作
        wIdx_ = workGroups_;
        return compAcc_;
    }

    // 当前组的 compressedCnt（输出行号）
    __aicore__ inline uint32_t CompCntBefore() const
    {
        uint32_t inBatch = wIdx_ < compGroups_ ? wIdx_ : compGroups_;
        return compAcc_ + inBatch;
    }

    __aicore__ inline bool IsEnd() const
    {
        return bIdx_ >= batchSize_ || wIdx_ >= workGroups_;
    }

    // 当前组信息
    __aicore__ inline void GetGroupInfo(GroupInfo &info) const
    {
        uint32_t gStart = (gBase_ + wIdx_) * cmpRatio_;
        uint32_t tokStart = gStart > P_ ? gStart : P_;
        uint32_t tokEnd = gStart + cmpRatio_ < P_ + S_ ? gStart + cmpRatio_ : P_ + S_;
        info.bIdx = bIdx_;
        info.gStart = gStart;
        info.tokStart = tokStart;
        info.nTok = tokEnd > tokStart ? tokEnd - tokStart : 0;
        info.headHolder = tokStart - gStart;
        info.P = P_;
        info.xRow = cuStart_ + (tokStart - P_);
        info.produce = (gStart + cmpRatio_ <= P_ + S_) && (info.nTok > 0);
    }

    // 前进一个工作组（跨 batch 时跳到下一个有工作的 batch）
    __aicore__ inline void Next()
    {
        wIdx_++;
        while (bIdx_ < batchSize_ && wIdx_ >= workGroups_) {
            groupAcc_ += workGroups_;
            compAcc_ += compGroups_;
            LoadBatch(bIdx_ + 1);
            wIdx_ = 0;
        }
    }

    __aicore__ inline uint32_t GetBIdx() const
    {
        return bIdx_;
    }

    __aicore__ inline uint32_t GetStartPos() const
    {
        return P_;
    }

private:
    __aicore__ inline void LoadBatch(uint32_t bIdx)
    {
        bIdx_ = bIdx;
        workGroups_ = 0;
        compGroups_ = 0;
        P_ = 0;
        S_ = 0;
        gBase_ = 0;
        while (bIdx_ < batchSize_) {
            cuStart_ = (uint32_t)cuSeqlensGm_.GetValue(bIdx_);
            S_ = isExistSeqUsed_ ? (uint32_t)seqUsedGm_.GetValue(bIdx_)
                                 : (uint32_t)(cuSeqlensGm_.GetValue(bIdx_ + 1) - cuStart_);
            P_ = isExistStartPos_ ? (uint32_t)startPosGm_.GetValue(bIdx_) : 0;
            gBase_ = P_ / cmpRatio_;
            // S==0 的 batch 无任何工作（其唯一工作组 nTok=0 是纯 no-op），直接跳过，
            // 避免产生 phantom 组（nTok==0 会打断 ping-pong buffer 配对链）
            if (S_ == 0) {
                workGroups_ = 0;
                compGroups_ = 0;
                bIdx_++;
                continue;
            }
            workGroups_ = CeilDivT(P_ + S_, cmpRatio_) - gBase_;
            compGroups_ = (P_ + S_) / cmpRatio_ - gBase_;
            if (workGroups_ > 0) {
                return;
            }
            bIdx_++;
        }
    }

    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> seqUsedGm_;
    GlobalTensor<int32_t> startPosGm_;
    bool isExistSeqUsed_ = false;
    bool isExistStartPos_ = false;
    uint32_t batchSize_ = 0;
    uint32_t cmpRatio_ = 0;

    // 游标
    uint32_t bIdx_ = 0;
    uint32_t wIdx_ = 0;
    uint32_t P_ = 0;
    uint32_t S_ = 0;
    uint32_t gBase_ = 0;
    uint32_t workGroups_ = 0;
    uint32_t compGroups_ = 0;
    uint32_t groupAcc_ = 0;
    uint32_t compAcc_ = 0;
    uint32_t cuStart_ = 0;
};

} // namespace CompressNormRope

#endif // COMPRESS_NORM_ROPE_TOOLS_H
