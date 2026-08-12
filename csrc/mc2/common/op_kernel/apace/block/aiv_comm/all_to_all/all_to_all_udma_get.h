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
 * \file all_to_all_udma_get.h
 * \brief AllToAll GET 模式实现 - 基于 hcomm ReadNbi
 */

#pragma once

#include "adv_api/hcomm/hcomm.h"
#include "../collective_comm_base.h"

namespace Apace {
namespace AivComm {

using namespace AscendC;

template<typename Dtype, typename Barrier>
class AllToAllCommGetImpl : public CollectiveCommBase<AllToAllCommGetImpl<Dtype, Barrier>, Dtype, Barrier> {
    friend class CollectiveCommBase<AllToAllCommGetImpl<Dtype, Barrier>, Dtype, Barrier>;

private:
    template<uint8_t BarrierMode>
    __aicore__ inline void PostInit() {}

    template<uint8_t BarrierMode>
    __aicore__ inline void DoCommit(uint32_t targetRankId, uint64_t tileByteSize)
    {
        if constexpr (BarrierMode & BARRIER_CORE) {
            this->barrier_.CrossCore();
        }
        if constexpr (BarrierMode & BARRIER_DEVICE) {
            this->barrier_.CrossDevice();
        }
        if (targetRankId == this->udmaCtx_->rankId) {
            return;
        }

        GM_ADDR srcAddr = reinterpret_cast<GM_ADDR>(this->udmaCtx_->commBufferAddrs[targetRankId] + this->winOffset_) +
            this->slotByteOffset_ + this->udmaCtx_->rankId * this->tileMaxByteSize_;

        GM_ADDR dstAddr = this->localAddr_ + targetRankId * this->chunkBytes_ + this->tileByteOffset_;

        int32_t ret = this->comm_.ReadNbi(
            static_cast<ChannelHandle>(this->udmaCtx_->channelHandles[targetRankId]), dstAddr, srcAddr, tileByteSize);
        ascendc_assert(ret == 0, "Urma readNbi failed, ret=%d, targetRankId=%u", ret, targetRankId);
    }

    template<uint8_t BarrierMode>
    __aicore__ inline void DoWait(uint32_t targetRankId)
    {
        if (targetRankId == this->udmaCtx_->rankId) {
            return;
        }
        int32_t ret = this->comm_.Drain(static_cast<ChannelHandle>(this->udmaCtx_->channelHandles[targetRankId]));
        ascendc_assert(ret == 0, "Urma drain failed, ret=%d, targetRankId=%u", ret, targetRankId);
    }

    template<uint8_t BarrierMode>
    __aicore__ inline void DoFinalize()
    {
        if constexpr (BarrierMode & BARRIER_CORE) {
            this->barrier_.CrossCore();
        }
        if constexpr (BarrierMode & BARRIER_DEVICE) {
            this->barrier_.CrossDevice();
        }
    }
};

} // namespace AivComm
} // namespace Apace
