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
 * \file barrier_ubmem.h
 * \brief
 */
#pragma once
#include "../collective_comm_context.h"
#include "basic_api/kernel_basic_intf.h"
#include "tensor_api/tensor.h"

namespace Apace {
namespace AivComm {
using namespace AscendC;

constexpr uint32_t BARRIER_FLAG_SIZE = 32;
constexpr uint32_t UB_SIZE = BARRIER_FLAG_SIZE;
constexpr uint32_t BARRIER_FLAG_ELEMS = BARRIER_FLAG_SIZE / sizeof(int32_t);

class TeamBarrier {
public:
    __aicore__ inline TeamBarrier()
    {}

    __aicore__ inline void Init(
        __ubuf__ uint8_t* syncBuf, __gm__ CommUbmemContext* ctx, uint32_t totalJobs, uint32_t jobIndex);

    __aicore__ inline void CrossDevice();

    __aicore__ inline void CrossCore();

private:
    __ubuf__ uint8_t* syncBuf_;
    __gm__ CommUbmemContext* ctx_;
    uint32_t totalJobs_;
    uint32_t jobIndex_;

    __aicore__ inline void CrossDeviceExecute(int64_t count);
};

__aicore__ inline void TeamBarrier::Init(
    __ubuf__ uint8_t* syncBuf, __gm__ CommUbmemContext* ctx, uint32_t totalJobs, uint32_t jobIndex)
{
    syncBuf_ = syncBuf;
    ctx_ = ctx;
    totalJobs_ = totalJobs;
    jobIndex_ = jobIndex;
}

__aicore__ inline void TeamBarrier::CrossDevice()
{
    if ASCEND_IS_AIV {
        if (jobIndex_ >= totalJobs_) {
            return;
        }

        __gm__ int32_t* teamSyncCounter = (__gm__ int32_t*)(ctx_->commBufferAddrs[ctx_->rankId]
            + BARRIER_FLAG_SIZE + jobIndex_ * BARRIER_FLAG_SIZE);

        auto copyGM2UB = Te::MakeCopy(Te::CopyGM2UB{});
        auto copyUB2GM = Te::MakeCopy(Te::CopyUB2GM{});
        auto ubTmp = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, int32_t>(reinterpret_cast<uint64_t>(syncBuf_)),
            Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, BARRIER_FLAG_ELEMS));
        auto teamSyncGmTensor = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>((__gm__ int32_t*)(teamSyncCounter)),
            Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, BARRIER_FLAG_ELEMS));

        SetFlag<HardEvent::S_MTE2>(1);
        WaitFlag<HardEvent::S_MTE2>(1);
        Te::Copy(copyGM2UB, ubTmp, teamSyncGmTensor);
        SetFlag<HardEvent::MTE2_S>(1);
        WaitFlag<HardEvent::MTE2_S>(1);
        int64_t count = *reinterpret_cast<__ubuf__ int64_t*>(syncBuf_) + 1;
        CrossDeviceExecute(count);

        *reinterpret_cast<__ubuf__ int64_t*>(syncBuf_) = count;
        SetFlag<HardEvent::S_MTE3>(1);
        WaitFlag<HardEvent::S_MTE3>(1);
        Te::Copy(copyUB2GM, teamSyncGmTensor, ubTmp);
        SetFlag<HardEvent::MTE3_S>(1);
        WaitFlag<HardEvent::MTE3_S>(1);
    }
}

__aicore__ inline void TeamBarrier::CrossCore()
{
    if ASCEND_IS_AIV {
        if (jobIndex_ >= totalJobs_) {
            return;
        }
        uint64_t crossCoreBase = reinterpret_cast<uint64_t>(ctx_->commBufferAddrs[ctx_->rankId])
            + BARRIER_FLAG_SIZE + totalJobs_ * BARRIER_FLAG_SIZE;
        __gm__ int32_t* localFlag = (__gm__ int32_t*)(crossCoreBase + jobIndex_ * BARRIER_FLAG_SIZE);

        auto copyGM2UB = Te::MakeCopy(Te::CopyGM2UB{});
        auto copyUB2GM = Te::MakeCopy(Te::CopyUB2GM{});
        auto ubTmp = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, int32_t>(reinterpret_cast<uint64_t>(syncBuf_)),
            Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, BARRIER_FLAG_ELEMS));
        auto gmLocal = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(localFlag),
            Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, BARRIER_FLAG_ELEMS));

        SetFlag<HardEvent::S_MTE2>(1);
        WaitFlag<HardEvent::S_MTE2>(1);
        Te::Copy(copyGM2UB, ubTmp, gmLocal);
        SetFlag<HardEvent::MTE2_S>(1);
        WaitFlag<HardEvent::MTE2_S>(1);
        int64_t count = *reinterpret_cast<__ubuf__ int64_t*>(syncBuf_) + 1;

        *reinterpret_cast<__ubuf__ int64_t*>(syncBuf_) = count;
        SetFlag<HardEvent::S_MTE3>(1);
        WaitFlag<HardEvent::S_MTE3>(1);
        Te::Copy(copyUB2GM, gmLocal, ubTmp);
        SetFlag<HardEvent::MTE3_S>(1);
        WaitFlag<HardEvent::MTE3_S>(1);

        for (uint32_t i = 0; i < totalJobs_; i++) {
            if (i == jobIndex_) {
                continue;
            }
            __gm__ int32_t* remotePtr = (__gm__ int32_t*)(crossCoreBase + i * BARRIER_FLAG_SIZE);
            auto gmRemote = Te::MakeTensor(
                Te::MakeMemPtr<Te::Location::GM>(remotePtr),
                Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, BARRIER_FLAG_ELEMS));
            do {
                SetFlag<HardEvent::S_MTE2>(1);
                WaitFlag<HardEvent::S_MTE2>(1);
                Te::Copy(copyGM2UB, ubTmp, gmRemote);
                SetFlag<HardEvent::MTE2_S>(1);
                WaitFlag<HardEvent::MTE2_S>(1);
                if (*reinterpret_cast<__ubuf__ int64_t*>(syncBuf_) >= count) {
                    break;
                }
            } while (true);
        }
    }
}

__aicore__ inline void TeamBarrier::CrossDeviceExecute(int64_t count)
{
    uint32_t nranks = ctx_->rankSize;
    uint32_t myRank = ctx_->rankId;
    int32_t step = totalJobs_ < nranks ? (int32_t)totalJobs_ : (int32_t)nranks;
    auto localFlag = (__gm__ int32_t*)(ctx_->commBufferAddrs[ctx_->rankId]);

    auto copyGM2UB = Te::MakeCopy(Te::CopyGM2UB{});
    auto copyUB2GM = Te::MakeCopy(Te::CopyUB2GM{});
    auto ubTmp = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::UB, int32_t>(reinterpret_cast<uint64_t>(syncBuf_)),
        Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, BARRIER_FLAG_ELEMS));
    auto gmFlag = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>((__gm__ int32_t*)(localFlag)),
        Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, BARRIER_FLAG_ELEMS));
    *reinterpret_cast<__ubuf__ int64_t*>(syncBuf_) = count;
    SetFlag<HardEvent::S_MTE3>(1);
    WaitFlag<HardEvent::S_MTE3>(1);
    Te::Copy(copyUB2GM, gmFlag, ubTmp);
    SetFlag<HardEvent::MTE3_S>(1);
    WaitFlag<HardEvent::MTE3_S>(1);

    for (uint32_t i = jobIndex_; i < nranks; i += (uint32_t)step) {
        if (i == myRank) {
            continue;
        }
        __gm__ int32_t* remotePtr = (__gm__ int32_t*)(ctx_->commBufferAddrs[i]);
        auto gmRemote = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(remotePtr),
            Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<int32_t>>{}(1, BARRIER_FLAG_ELEMS));
        do {
            SetFlag<HardEvent::S_MTE2>(1);
            WaitFlag<HardEvent::S_MTE2>(1);
            Te::Copy(copyGM2UB, ubTmp, gmRemote);
            SetFlag<HardEvent::MTE2_S>(1);
            WaitFlag<HardEvent::MTE2_S>(1);
            if (*reinterpret_cast<__ubuf__ int64_t*>(syncBuf_) >= count) {
                break;
            }
        } while (true);
    }
}
} // namespace AivComm
} // namespace Apace