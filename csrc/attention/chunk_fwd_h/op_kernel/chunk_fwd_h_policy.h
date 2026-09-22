/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_FWD_H_POLICY_H
#define CHUNK_FWD_H_POLICY_H

#include <cstdint>

#include "kernel_operator.h"

namespace GDN {

enum class FwdHGateMode : uint8_t {
    SCALAR_G = 0,
    KEY_GK = 1,
};

template <FwdHGateMode GATE_MODE_, bool USE_EXP2_, bool STATE_FP32_>
struct FwdHCompilePolicy {
    static constexpr FwdHGateMode GATE_MODE = GATE_MODE_;
    static constexpr bool USE_EXP2 = USE_EXP2_;
    static constexpr bool STATE_FP32 = STATE_FP32_;
};

constexpr uint32_t FWD_H_CHUNK = 64;
constexpr uint32_t FWD_H_K = 128;
constexpr uint32_t FWD_H_V = 128;
constexpr uint32_t FWD_H_AIC_HEAD_SLOTS = 4;
constexpr uint32_t FWD_H_AIV_COUNT = 2;
constexpr uint32_t FWD_H_AIV_HEAD_SLOTS = 2;
constexpr uint32_t FWD_H_FLAG_SUBBLOCK_STRIDE = 16;

constexpr uint32_t FWD_H_TOKEN_MATRIX_BF16_BYTES = FWD_H_CHUNK * FWD_H_V * sizeof(bfloat16_t);
constexpr uint32_t FWD_H_TOKEN_MATRIX_FP32_BYTES = FWD_H_CHUNK * FWD_H_V * sizeof(float);
constexpr uint32_t FWD_H_STATE_BF16_BYTES = FWD_H_K * FWD_H_V * sizeof(bfloat16_t);
constexpr uint32_t FWD_H_STATE_FP32_BYTES = FWD_H_K * FWD_H_V * sizeof(float);

// AIC L1 固定布局。四个 W、四个 H/right、四个 kg 槽都按 roundHead 绑定，round 内不复用。
constexpr uint32_t FWD_H_L1_W_BASE = 0;
constexpr uint32_t FWD_H_L1_W_SLOT_BYTES = 16 * 1024;
constexpr uint32_t FWD_H_L1_H_RIGHT_BASE = 128 * 1024;
constexpr uint32_t FWD_H_L1_H_RIGHT_SLOT_BYTES = 32 * 1024;
constexpr uint32_t FWD_H_L1_KG_BASE = 256 * 1024;
constexpr uint32_t FWD_H_L1_KG_SLOT_BYTES = 16 * 1024;
constexpr uint32_t FWD_H_L1_USED_BYTES = 320 * 1024;

// 每个 AIV 的两个 local slot 固定为 64 KiB。P 使用低 32 KiB，Stage1 可在高 32 KiB
// 生成 BF16 右操作数；D 使用完整 64 KiB。
constexpr uint32_t FWD_H_UB_LOCAL_SLOT_BYTES = 64 * 1024;
constexpr uint32_t FWD_H_UB_LOCAL0_BASE = 0;
constexpr uint32_t FWD_H_UB_LOCAL1_BASE = 64 * 1024;
constexpr uint32_t FWD_H_UB_BF16_STATE_BASE = 128 * 1024;
constexpr uint32_t FWD_H_UB_FP32_STATE_BASE = 128 * 1024;
constexpr uint32_t FWD_H_UB_BF16_WORK_BASE = 192 * 1024;
constexpr uint32_t FWD_H_UB_GATE_BASE = 224 * 1024;
constexpr uint32_t FWD_H_UB_SHARE_BASE = 226 * 1024;

// ready/free 使用同一 ID 的双向计数器，且复用前必须由上一代 wait 消费。
// 目标 CANN 9.1 的 A5 mode=0x4 仅支持 AIV 本地 ID 0..10；AIC 侧通过 16-ID
// 步长选择 AIV0/AIV1。各协议使用互不重叠的本地 ID，最大 ID 为 8。
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
constexpr uint32_t FWD_H_AIV_FLAG_STRIDE = 16;
#else
constexpr uint32_t FWD_H_AIV_FLAG_STRIDE = 0;
#endif
constexpr uint32_t FWD_H_P_FREE_FLAG = 0;
constexpr uint32_t FWD_H_P_READY_FLAG = 0;
constexpr uint32_t FWD_H_D_FREE_FLAG = 2;
constexpr uint32_t FWD_H_D_READY_FLAG = 2;
constexpr uint32_t FWD_H_RIGHT_FREE_FLAG = 4;
constexpr uint32_t FWD_H_RIGHT_READY_FLAG = 4;
constexpr uint32_t FWD_H_H_READY_FLAG = 6;
constexpr uint32_t FWD_H_ROUND_DONE_FLAG = 8;
constexpr uint32_t FWD_H_ROUND_ACK_FLAG = 8;

struct FwdHSequenceSpan {
    uint32_t sequence = 0;
    uint32_t physicalBatch = 0;
    uint32_t tokenBegin = 0;
    uint32_t length = 0;
    uint32_t chunkPrefix = 0;
    uint32_t chunkCount = 0;
    uint32_t totalChunks = 0;
};

struct FwdHChunkSpan {
    uint32_t chunk = 0;
    uint32_t globalChunk = 0;
    uint32_t tokenBegin = 0;
    uint32_t validTokens = 0;
    bool first = false;
    bool last = false;
};

struct FwdHHeadBinding {
    uint32_t hv = 0;
    uint32_t kh = 0;
    uint8_t roundHead = 0;
    uint8_t kgSlot = 0;
    uint8_t aiv = 0;
    uint8_t localSlot = 0;
};

struct FwdHKgBinding {
    uint32_t kh = 0;
    uint8_t slot = 0;
    uint8_t firstConsumer = 0;
    uint8_t lastConsumer = 0;
    uint8_t reserved = 0;
};

struct FwdHHeadRoundPlan {
    uint8_t activeHeadCount = 0;
    uint8_t requiredKhCount = 0;
    FwdHHeadBinding heads[FWD_H_AIC_HEAD_SLOTS]{};
};

struct FwdHWorkUnit {
    FwdHSequenceSpan sequence{};
    FwdHHeadRoundPlan headRound{};
};

struct FwdHCoreHeadRange {
    uint32_t begin = 0;
    uint32_t end = 0;
};

struct FwdHWorkspace {
    __gm__ uint8_t *base = nullptr;
    int64_t pOffset = 0;
    int64_t rightOffset = 0;
    int64_t stateOffset = 0;
    int64_t dOffset = 0;
};

// Kernel 入口一次性从 GM tiling 读取这些标量，后续调度和各 stage 只访问寄存器/栈上的副本。
struct FwdHRuntimeTiling {
    uint32_t batch = 0;
    uint32_t seqlen = 0;
    uint32_t kNumHead = 0;
    uint32_t vNumHead = 0;
    uint32_t kHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t chunkSize = 0;
    uint32_t shapeBatch = 0;
    uint32_t tokenBatch = 0;
    uint64_t vWorkspaceOffset = 0;
    uint64_t vUpdateWorkspaceOffset = 0;
    uint64_t kDecayWorkspaceOffset = 0;
    uint64_t hWorkspaceOffset = 0;
    bool useInitialState = false;
    bool storeFinalState = false;
    bool isVariedLen = false;
};

struct FwdHKernelArgs {
    GM_ADDR k = nullptr;
    GM_ADDR w = nullptr;
    GM_ADDR u = nullptr;
    GM_ADDR g = nullptr;
    GM_ADDR gk = nullptr;
    GM_ADDR initialState = nullptr;
    GM_ADDR cuSeqlens = nullptr;
    GM_ADDR chunkIndices = nullptr;
    GM_ADDR h = nullptr;
    GM_ADDR vNew = nullptr;
    GM_ADDR finalState = nullptr;
    GM_ADDR workspace = nullptr;
    FwdHRuntimeTiling tiling{};
};

// A5 mode=0x4 按 subblock stride 选择配对 AIV；A2/A3 mode=0x2 是 AIC:2*AIV
// 集合同步，同一 localSlot 的两个 AIV 必须使用同一 ID。
__aicore__ inline uint32_t FwdHAicPeerFlag(uint32_t base, uint32_t localSlot, uint32_t aiv)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return base + localSlot + aiv * FWD_H_AIV_FLAG_STRIDE;
#else
    (void)aiv;
    return base + localSlot;
#endif
}

// AIV 侧已经处于具体 subblock；两种架构都按 localSlot 选择本地 ID。A2/A3 的
// 两个 subblock 必须各自执行同一 ID 的 wait/set，缺失 head 的 subblock 也做 dummy 同步。
__aicore__ inline uint32_t FwdHAivLocalFlag(uint32_t base, uint32_t localSlot)
{
    return base + localSlot;
}

__aicore__ inline uint32_t FwdHLocalSlotBase(uint32_t localSlot)
{
    return localSlot == 0 ? FWD_H_UB_LOCAL0_BASE : FWD_H_UB_LOCAL1_BASE;
}

__aicore__ inline uint32_t FwdHAivHeadCount(uint32_t activeHeadCount, uint32_t aiv)
{
    return activeHeadCount > aiv ? (activeHeadCount - aiv + 1) / 2 : 0;
}

__aicore__ inline uint32_t FwdHMode2PairCount(uint32_t activeHeadCount)
{
    return (activeHeadCount + 1U) / 2U;
}

__aicore__ inline bool FwdHMode2PairHasHead(uint32_t activeHeadCount, uint32_t pairSlot,
                                           uint32_t aiv)
{
    return pairSlot * 2U + aiv < activeHeadCount;
}

} // namespace GDN

#endif // CHUNK_FWD_H_POLICY_H
