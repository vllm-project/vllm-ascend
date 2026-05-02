/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file semaphore.h
 * \brief
 */

#ifndef MATMUL_KERNEL_SEMAPHORE_H
#define MATMUL_KERNEL_SEMAPHORE_H

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "../utils/common_utils.h"
#include "../utils/tuple_utils.h"
#include "../epilogue/block_epilogue_empty.h"

namespace Act {
namespace Gemm {
namespace Kernel {
constexpr uint16_t C2V_PING_FLAG = 0x4;
constexpr uint16_t C2V_PONG_FLAG = 0x5;
constexpr uint16_t V2C_PING_FLAG = 0x6;
constexpr int64_t PINGPONG_FLAG = 1;
constexpr int64_t FIRST_PINGPONG = 2;
constexpr int64_t AIC_PING_START_IDX = 1;
constexpr int64_t AIC_PONG_START_IDX = 2;

__aicore__ inline int64_t GetBlockWorkspaceSize(bool isPingPong, int64_t l1M, int64_t l1N)
{
    int64_t bufferNum = isPingPong ? DOUBLE_BUFFER_COUNT : 1;
    return bufferNum * l1M * l1N;
}

__aicore__ inline int64_t GetWorkspaceOffset(int64_t pingPongIdx, int64_t curBlockIdx, int64_t l1M, int64_t l1N)
{
    return curBlockIdx * GetBlockWorkspaceSize(true, l1M, l1N) + pingPongIdx * GetBlockWorkspaceSize(false, l1M, l1N);
}

template <typename BlockEpilogue_>
__aicore__ inline void AicWaitAiv(const int64_t idx)
{
    if constexpr (!AscendC::Std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>) {
        uint16_t pingPongFlag = static_cast<uint16_t>(idx & PINGPONG_FLAG);
        uint16_t c2vSyncFlag = C2V_PING_FLAG | pingPongFlag;
        AscendC::WaitEvent(c2vSyncFlag);
    }
}

template <typename BlockEpilogue_>
__aicore__ inline void AicNotifyAiv(const int64_t idx)
{
    if constexpr (!AscendC::Std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>) {
        uint16_t pingPongFlag = static_cast<uint16_t>(idx & PINGPONG_FLAG);
        uint16_t v2cSyncFlag = V2C_PING_FLAG | pingPongFlag;
        AscendC::NotifyEvent<PIPE_FIX>(v2cSyncFlag);
    }
}

template <typename BlockEpilogue_>
__aicore__ inline void AivWaitAic(const int64_t idx)
{
    if constexpr (!AscendC::Std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>) {
        uint16_t pingPongFlag = static_cast<uint16_t>(idx & PINGPONG_FLAG);
        uint16_t v2cSyncFlag = V2C_PING_FLAG | pingPongFlag;
        AscendC::WaitEvent(v2cSyncFlag);
    }
}

template <typename BlockEpilogue_>
__aicore__ inline void AivNotifyAic(const int64_t idx)
{
    if constexpr (!AscendC::Std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>) {
        uint16_t pingPongFlag = static_cast<uint16_t>(idx & PINGPONG_FLAG);
        uint16_t c2vSyncFlag = C2V_PING_FLAG | pingPongFlag;
        AscendC::NotifyEvent<PIPE_MTE2>(c2vSyncFlag);
    }
}

template <typename BlockEpilogue_>
__aicore__ inline void AicWaitEvent(const int64_t loopIdx)
{
    if constexpr (!AscendC::Std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>) {
        if (loopIdx >= AIC_PING_START_IDX) {
            AscendC::WaitEvent(C2V_PING_FLAG);
        }
        if (loopIdx >= AIC_PONG_START_IDX) {
            AscendC::WaitEvent(C2V_PONG_FLAG);
        }
    }
}

template <typename BlockEpilogue_>
__aicore__ inline void SendEvent(int64_t curBlockIdx, int64_t tileNum, int64_t blockNum)
{
    int64_t loopIdx = 0;
    for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
        AivWaitAic<BlockEpilogue_>(loopIdx);
        AivNotifyAic<BlockEpilogue_>(loopIdx);
        loopIdx += 1;
    }
    return;
}
} // namespace Kernel
} // namespace Gemm
} // namespace Act
#endif
