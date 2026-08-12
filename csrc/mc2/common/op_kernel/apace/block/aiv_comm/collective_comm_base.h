/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file collective_comm_base.h
 * \brief 基于 hcomm 的集合通信基类 - CRTP 模式 + 统一状态管理
 */

#pragma once

#include "collective_comm_context.h"
#include "apace/tiling/comm_tiling_data.h"
#include "tensor_api/tensor/pointer.h"
#include "barrier/barrier_ubmem.h"

namespace Apace {
namespace AivComm {

using namespace AscendC;

constexpr uint8_t BARRIER_NONE = 0;
constexpr uint8_t BARRIER_DEVICE = 1;
constexpr uint8_t BARRIER_CORE = 2;
constexpr uint8_t BARRIER_BOTH = BARRIER_DEVICE | BARRIER_CORE;

template<typename Impl, typename Dtype, typename Barrier>
class CollectiveCommBase {
    friend Impl;
public:
    __aicore__ inline CollectiveCommBase() {}

    template<uint8_t BarrierMode = BARRIER_BOTH>
    __aicore__ inline void Init(
        __gm__ CommUdmaContext* udmaCtx,
        Barrier& barrier,
        const CommTilingData& tilingData,
        GM_ADDR localAddr,
        __ubuf__ uint8_t* commbuf,
        uint32_t totalJobs,
        uint32_t jobIndex,
        uint64_t winOffset = 0)
    {
        if (jobIndex >= totalJobs) {
            return;
        }
        udmaCtx_ = udmaCtx;
        barrier_ = barrier;
        localAddr_ = localAddr;
        tilingData_ = &tilingData;
        commBuf_ = commbuf;
        winOffset_ = winOffset;

        comm_.Init(commBuf_, COMM_WORKSPACE_SIZE);

        uint32_t rankSize = udmaCtx_->rankSize;
        uint32_t targetRankPerCore = (rankSize + totalJobs - 1) / totalJobs;
        uint32_t targetRankStart = jobIndex * targetRankPerCore;

        uint32_t targetRankCnt;
        if (targetRankStart + targetRankPerCore <= rankSize) {
            targetRankCnt = targetRankPerCore;
        } else if (targetRankStart < rankSize) {
            targetRankCnt = rankSize - targetRankStart;
        } else {
            targetRankCnt = 0;
        }

        targetRankStart_ = targetRankStart;
        targetRankCnt_ = targetRankCnt;

        uint64_t chunkSize = tilingData.splitAxisTileSize * tilingData.splitAxisTileCnt +
                             tilingData.splitAxisTailSize * tilingData.splitAxisTailCnt;
        uint64_t nonSplitAxisBytes = tilingData.nonSplitAxisSize * sizeof(Dtype);

        chunkBytes_ = chunkSize * nonSplitAxisBytes;
        tileMaxByteSize_ = (tilingData_->splitAxisTileSize > tilingData_->splitAxisTailSize ?
            tilingData_->splitAxisTileSize : tilingData_->splitAxisTailSize) * nonSplitAxisBytes;
        currentTileIdx_ = 0;
        tileByteOffset_ = 0;
        remainingChunkSize_ = chunkSize;
        slotByteOffset_ = 0;
        chunkByteOffset_ = 0;
        static_cast<Impl*>(this)->template PostInit<BarrierMode>();
    }

    template<uint8_t BarrierMode = BARRIER_BOTH>
    __aicore__ inline void Commit()
    {
        if (remainingChunkSize_ <= 0) {
            return;
        }

        uint64_t currentTileSize;
        if (currentTileIdx_ < tilingData_->splitAxisTileCnt) {
            currentTileSize = tilingData_->splitAxisTileSize;
        } else {
            currentTileSize = tilingData_->splitAxisTailSize;
        }
        if (currentTileSize > remainingChunkSize_) {
            currentTileSize = remainingChunkSize_;
        }

        uint64_t nonSplitAxisBytes = tilingData_->nonSplitAxisSize * sizeof(Dtype);
        uint64_t currentTileByteSize = currentTileSize * nonSplitAxisBytes;
        uint64_t slotBytes = udmaCtx_->rankSize * tileMaxByteSize_;
        if (targetRankCnt_ > 0) {
            const uint32_t targetRankStart = targetRankStart_;
            const uint32_t targetRankCnt = targetRankCnt_;

            for (uint32_t i = 0; i < targetRankCnt; i++) {
                uint32_t targetRankId = targetRankStart + i;
                static_cast<Impl*>(this)->template DoCommit<BarrierMode>(
                    targetRankId, currentTileByteSize);
            }
        }

        currentTileIdx_++;
        slotByteOffset_ += slotBytes;
        tileByteOffset_ += currentTileByteSize;
        chunkByteOffset_ += chunkBytes_;
        remainingChunkSize_ -= currentTileSize;
    }

    template<uint8_t BarrierMode = BARRIER_BOTH>
    __aicore__ inline void Wait(bool waitLast=false)
    {
        uint64_t totalTiles = tilingData_->splitAxisTileCnt + tilingData_->splitAxisTailCnt;
        if (waitLast && currentTileIdx_ != totalTiles - 1) {
            return;
        }
        for (uint32_t i = 0; i < targetRankCnt_; i++) {
            uint32_t targetRankId = targetRankStart_ + i;
            static_cast<Impl*>(this)->template DoWait<BarrierMode>(targetRankId);
        }
    }

    __aicore__ inline uint64_t GetCommByteSize() const
    {
        return chunkBytes_ * udmaCtx_->rankSize;
    }

    __aicore__ inline uint64_t GetCommTurn() const
    {
        return tilingData_->splitAxisTileCnt + tilingData_->splitAxisTailCnt;
    }

    template<uint8_t BarrierMode = BARRIER_BOTH>
    __aicore__ inline void Finalize()
    {
        static_cast<Impl*>(this)->template DoFinalize<BarrierMode>();
    }

protected:
    __gm__ CommUdmaContext* udmaCtx_;
    const CommTilingData* tilingData_;
    Barrier barrier_;

    Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> comm_;
    GM_ADDR localAddr_;
    __ubuf__ uint8_t* commBuf_;
    uint64_t winOffset_;

    uint64_t chunkBytes_;
    uint64_t currentTileIdx_;
    uint64_t tileByteOffset_;
    uint64_t remainingChunkSize_;
    uint64_t tileMaxByteSize_;
    uint64_t slotByteOffset_;
    uint64_t chunkByteOffset_;

    uint32_t targetRankStart_;
    uint32_t targetRankCnt_;
};

} // namespace AivComm
} // namespace Apace
