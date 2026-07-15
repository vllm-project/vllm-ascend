/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ARCH_CROSS_CORE_SYNC_HPP
#define ARCH_CROSS_CORE_SYNC_HPP

#include "../../attn_infra/base_defs.hpp"

namespace NpuArch::Arch
{

constexpr uint32_t MAX_REVERSE_DEPTH = 16;

using FlagID = uint16_t;
constexpr FlagID AIV_INTER_BLOCK_BARRIER = 8;
constexpr FlagID AIC_INTER_BLOCK_BARRIER = 9;
constexpr FlagID AIV_INTER_SUBBLOCK_BARRIER = 10;
constexpr FlagID FFTS_MAX_FLAG = 7;

struct CrossCoreFlag {
    __aicore__ inline
    CrossCoreFlag() : id(0) {}

    __aicore__ inline
    CrossCoreFlag(FlagID id) : id(id) {}

    FlagID id;
};

template <uint32_t REVERSE_DEPTH_ = MAX_REVERSE_DEPTH>
struct CrossCoreFlagWithReverse {
    __aicore__ inline
    CrossCoreFlagWithReverse() : id(0), reverseId(0) {}

    __aicore__ inline
    CrossCoreFlagWithReverse(FlagID id, FlagID reverseId) : id(id), reverseId(reverseId) {}

    FlagID id;
    FlagID reverseId;
    uint32_t count{ 0 };
};

template <uint8_t MODE, int32_t CORE_TYPE>
struct BarrierFlag {
    static_assert(MODE != MODE, "Unsupported cross core barrier flag, can not find the specialization.");
};

template <>
struct BarrierFlag<0x0, AscendC::AIV> {
    static constexpr FlagID ID = AIV_INTER_BLOCK_BARRIER;
};

template <>
struct BarrierFlag<0x0, AscendC::AIC> {
    static constexpr FlagID ID = AIC_INTER_BLOCK_BARRIER;
};

template <>
struct BarrierFlag<0x1, AscendC::AIV> {
    static constexpr FlagID ID = AIV_INTER_SUBBLOCK_BARRIER;
};

template <uint8_t MODE, pipe_t PIPE>
__aicore__ inline
void CrossCoreBarrier()
{
    FlagID flagId;
    if (g_coreType == AscendC::AIC) {
        flagId = BarrierFlag<MODE, AscendC::AIC>::ID;
    } else if (g_coreType == AscendC::AIV) {
        flagId = BarrierFlag<MODE, AscendC::AIV>::ID;
    }
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}

template <uint8_t MODE, pipe_t PIPE>
__aicore__ inline
void CrossCoreSetFlag(CrossCoreFlag &flag)
{
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
}

__aicore__ inline void CrossCoreWaitFlag(CrossCoreFlag &flag)
{
    AscendC::CrossCoreWaitFlag(flag.id);
}

template <uint8_t MODE, pipe_t PIPE, uint32_t REVERSE_DEPTH>
__aicore__ inline
void CrossCoreSetFlagWithReverse(CrossCoreFlagWithReverse<REVERSE_DEPTH> &flag)
{
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
    if (++flag.count >= REVERSE_DEPTH) {
        AscendC::CrossCoreWaitFlag(flag.reverseId);
        flag.count = 0;
    }
}

template <uint8_t MODE, pipe_t PIPE, uint32_t REVERSE_DEPTH>
__aicore__ inline
void CrossCoreWaitFlagWithReverse(CrossCoreFlagWithReverse<REVERSE_DEPTH> &flag)
{
    AscendC::CrossCoreWaitFlag(flag.id);
    if (++flag.count >= REVERSE_DEPTH) {
        AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.reverseId);
        flag.count = 0;
    }
}

}  // namespace NpuArch::Arch

#endif // ARCH_CROSS_CORE_SYNC_HPP