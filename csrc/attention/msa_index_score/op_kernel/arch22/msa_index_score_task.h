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
 * \file msa_index_score_task.h
 * \brief varlen + sparse_mode 因果裁剪的任务枚举器。
 *
 * atten_mask（sparse_mode=3）采用与 LightningIndexer 相同的 rightDownCausal 解析：
 *   visible_key_end(q) = kv_len - q_len + t_off + 1
 * 不物化 / 不加载 [2048,2048] 模板。
 *
 * start_loc[b] 为当前 query 所在逻辑 block 索引，仅用于 local_mask（强制高分），
 * 不再作为 token 级 prefix 偏移。
 */

#ifndef MSA_INDEX_SCORE_TASK_H
#define MSA_INDEX_SCORE_TASK_H

#include "kernel_operator.h"
#include "../msa_index_score_common.h"

namespace MsaIndexScoreNs {

__aicore__ inline uint32_t MsaCeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__aicore__ inline uint32_t MsaMinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

__aicore__ inline int32_t MsaClampI32(int32_t v, int32_t lo, int32_t hi)
{
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

/// 一个任务 = 一个请求内的一个 M-tile。
/// M 维 = 请求内 (token, head) 扁平行索引，head 为低位。
struct MsaTask {
    uint32_t batchIdx;
    uint32_t cuQStart;
    uint32_t mStart;
    uint32_t mActual;
    uint32_t globalRowBase;
    int32_t startLoc; // 当前 query 所在逻辑 block 索引（local_mask）
    int32_t kvLen;
    int32_t qLen;
    uint32_t cuKStart; // TND：本请求在 packed key 中的 token 起点；PA：0
    uint32_t sparseMode;
    uint32_t initBlocks;
    uint32_t localBlocks;
    uint32_t fullEndBlk;
    uint32_t visibleEndBlk;
    uint32_t numSTiles;        // 含不可见尾，必须写 -inf 的 S-tile 数
    uint32_t numComputeSTiles; // AIC 真正做 QKᵀ 的 S-tile 数（<= numSTiles）
};

class MsaTaskScheduler {
public:
    __aicore__ inline MsaTaskScheduler() {}

    __aicore__ inline void Init(uint32_t batch, uint32_t numQHeads, uint32_t maxBlocksPerBatch,
                                uint32_t scoreBlockStride, uint32_t sparseMode, uint32_t initBlocks,
                                uint32_t localBlocks, uint32_t keyLayout,
                                const AscendC::GlobalTensor<int32_t> &actualSeqQlen,
                                const AscendC::GlobalTensor<int32_t> &actualSeqKlen,
                                const AscendC::GlobalTensor<int32_t> &startLoc)
    {
        batch_ = batch;
        numQHeads_ = numQHeads;
        maxBlocksPerBatch_ = maxBlocksPerBatch;
        scoreBlockStride_ = scoreBlockStride;
        sparseMode_ = sparseMode;
        initBlocks_ = initBlocks;
        localBlocks_ = localBlocks;
        keyLayout_ = keyLayout;
        gActualSeqQlen_ = actualSeqQlen;
        gActualSeqKlen_ = actualSeqKlen;
        gStartLoc_ = startLoc;

        totalTasks_ = 0;
        for (uint32_t b = 0; b < batch_; ++b) {
            const uint32_t qLen = static_cast<uint32_t>(gActualSeqQlen_.GetValue(b + 1) - gActualSeqQlen_.GetValue(b));
            totalTasks_ += MsaCeilDiv(qLen * numQHeads_, MSA_ROW_TILE_M);
        }
        Reset();
    }

    __aicore__ inline uint32_t TotalTasks() const { return totalTasks_; }

    __aicore__ inline void Reset()
    {
        curBatch_ = 0;
        taskBase_ = 0;
        LoadBatch(0);
    }

    /// 按 sparse_mode 计算某 query token 的可见 key 上界（半开区间右端）。
    __aicore__ inline int32_t VisibleKeyEndOf(int32_t tOff) const
    {
        if (sparseMode_ == MSA_SPARSE_MODE_RIGHT_DOWN) {
            // rightDownCausal：j <= kvLen - qLen + tOff  <=>  end = kvLen - qLen + tOff + 1
            return MsaClampI32(kvLen_ - qLen_ + tOff + 1, 0, kvLen_ < 0 ? 0 : kvLen_);
        }
        // defaultMask：仅受 kv_len 截断
        return kvLen_ < 0 ? 0 : kvLen_;
    }

    __aicore__ inline void Decode(uint32_t taskIdx, MsaTask &task)
    {
        while (taskIdx >= taskBase_ + curTasks_) {
            taskBase_ += curTasks_;
            ++curBatch_;
            LoadBatch(curBatch_);
        }

        task.batchIdx = curBatch_;
        task.cuQStart = cuQStart_;
        task.startLoc = startLoc_;
        task.kvLen = kvLen_;
        task.qLen = qLen_;
        task.cuKStart = cuKStart_;
        task.sparseMode = sparseMode_;
        task.initBlocks = initBlocks_;
        task.localBlocks = localBlocks_;
        task.mStart = (taskIdx - taskBase_) * MSA_ROW_TILE_M;
        task.mActual = MsaMinU32(MSA_ROW_TILE_M, curRows_ - task.mStart);
        // Cube L0A fractal 为 MSA_M_ALIGN 行。整请求不足对齐宽度时保持原样（L0 已覆盖）；
        // 否则把末尾短 tile 向前重叠到 MSA_M_ALIGN 行，避免 mActual∈(0,16) 的 int8 路径出错。
        if (task.mActual > 0U && task.mActual < MSA_M_ALIGN && task.mStart >= MSA_M_ALIGN) {
            task.mStart -= (MSA_M_ALIGN - task.mActual);
            task.mActual = MSA_M_ALIGN;
        }
        task.globalRowBase = cuQStart_ * numQHeads_ + task.mStart;

        const uint32_t tokenLo = task.mStart / numQHeads_;
        const uint32_t tokenHi = (task.mStart + task.mActual - 1) / numQHeads_;
        const int32_t visibleKeyEndHi = VisibleKeyEndOf(static_cast<int32_t>(tokenHi));
        const int32_t visibleKeyEndLo = VisibleKeyEndOf(static_cast<int32_t>(tokenLo));

        uint32_t visibleEndBlk = MsaCeilDiv(static_cast<uint32_t>(visibleKeyEndHi), MSA_BLOCK_SIZE);
        visibleEndBlk = MsaMinU32(visibleEndBlk, maxBlocksPerBatch_);

        const uint32_t causalFull = static_cast<uint32_t>(visibleKeyEndLo) / MSA_BLOCK_SIZE;
        const uint32_t seqFull = (kvLen_ < 0 ? 0U : static_cast<uint32_t>(kvLen_)) / MSA_BLOCK_SIZE;
        uint32_t fullEndBlk = MsaMinU32(causalFull, seqFull);

        task.visibleEndBlk = visibleEndBlk;
        task.fullEndBlk = MsaMinU32(fullEndBlk, visibleEndBlk);

        // 不可见尾一律写 -inf，末维对齐到 scoreBlockStride
        task.numSTiles = MsaCeilDiv(scoreBlockStride_, MSA_BLOCKS_PER_STILE);
        task.numComputeSTiles = MsaCeilDiv(task.visibleEndBlk, MSA_BLOCKS_PER_STILE);
        if (task.numComputeSTiles > task.numSTiles) {
            task.numComputeSTiles = task.numSTiles;
        }
    }

private:
    __aicore__ inline void LoadBatch(uint32_t b)
    {
        if (b >= batch_) {
            curTasks_ = 0;
            curRows_ = 0;
            cuQStart_ = 0;
            kvLen_ = 0;
            qLen_ = 0;
            startLoc_ = 0;
            cuKStart_ = 0;
            return;
        }
        cuQStart_ = static_cast<uint32_t>(gActualSeqQlen_.GetValue(b));
        qLen_ = static_cast<int32_t>(gActualSeqQlen_.GetValue(b + 1)) - static_cast<int32_t>(cuQStart_);
        if (qLen_ < 0) {
            qLen_ = 0;
        }
        curRows_ = static_cast<uint32_t>(qLen_) * numQHeads_;
        curTasks_ = MsaCeilDiv(curRows_, MSA_ROW_TILE_M);
        if (keyLayout_ == MSA_KEY_LAYOUT_TND) {
            cuKStart_ = static_cast<uint32_t>(gActualSeqKlen_.GetValue(b));
            kvLen_ = static_cast<int32_t>(gActualSeqKlen_.GetValue(b + 1)) - static_cast<int32_t>(cuKStart_);
            if (kvLen_ < 0) {
                kvLen_ = 0;
            }
        } else {
            cuKStart_ = 0;
            kvLen_ = gActualSeqKlen_.GetValue(b);
        }
        startLoc_ = gStartLoc_.GetValue(b);
    }

    AscendC::GlobalTensor<int32_t> gActualSeqQlen_;
    AscendC::GlobalTensor<int32_t> gActualSeqKlen_;
    AscendC::GlobalTensor<int32_t> gStartLoc_;

    uint32_t batch_ = 0;
    uint32_t numQHeads_ = 1;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t scoreBlockStride_ = 0;
    uint32_t sparseMode_ = MSA_SPARSE_MODE_RIGHT_DOWN;
    uint32_t initBlocks_ = MSA_DEFAULT_INIT_BLOCKS;
    uint32_t localBlocks_ = MSA_DEFAULT_LOCAL_BLOCKS;
    uint32_t keyLayout_ = MSA_KEY_LAYOUT_BBND;

    uint32_t totalTasks_ = 0;
    uint32_t curBatch_ = 0;
    uint32_t taskBase_ = 0;
    uint32_t curTasks_ = 0;
    uint32_t curRows_ = 0;
    uint32_t cuQStart_ = 0;
    int32_t kvLen_ = 0;
    int32_t qLen_ = 0;
    int32_t startLoc_ = 0;
    uint32_t cuKStart_ = 0;
};

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_TASK_H
