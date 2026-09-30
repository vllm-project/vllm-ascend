/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_FWD_H_UTILS_H
#define CHUNK_FWD_H_UTILS_H

#include "chunk_fwd_h_policy.h"

namespace GDN {

__aicore__ inline uint32_t FwdHCeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

__aicore__ inline uint32_t FwdHAlignCube(uint32_t value)
{
    return (value + 15U) & ~15U;
}

__aicore__ inline FwdHSequenceSpan FwdHResolveSequence(const FwdHKernelArgs &args, uint32_t sequence)
{
    FwdHSequenceSpan span{};
    span.sequence = sequence;
    if (args.tiling.isVariedLen == 0) {
        span.physicalBatch = sequence;
        span.tokenBegin = 0;
        span.length = static_cast<uint32_t>(args.tiling.seqlen);
        span.chunkCount = FwdHCeilDiv(span.length, FWD_H_CHUNK);
        span.chunkPrefix = sequence * span.chunkCount;
        span.totalChunks = static_cast<uint32_t>(args.tiling.shapeBatch) * span.chunkCount;
        return span;
    }

    AscendC::GlobalTensor<int64_t> cu;
    cu.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(args.cuSeqlens));
    const int64_t begin = cu.GetValue(sequence);
    const int64_t end = cu.GetValue(sequence + 1);
    span.physicalBatch = 0;
    span.tokenBegin = static_cast<uint32_t>(begin);
    span.length = static_cast<uint32_t>(end - begin);
    span.chunkCount = FwdHCeilDiv(span.length, FWD_H_CHUNK);
    uint32_t prefix = 0;
    uint32_t totalChunks = 0;
    for (uint32_t seq = 0; seq < static_cast<uint32_t>(args.tiling.tokenBatch); ++seq) {
        const int64_t seqBegin = cu.GetValue(seq);
        const int64_t seqEnd = cu.GetValue(seq + 1);
        const uint32_t chunks = FwdHCeilDiv(static_cast<uint32_t>(seqEnd - seqBegin), FWD_H_CHUNK);
        if (seq < sequence) {
            prefix += chunks;
        }
        totalChunks += chunks;
    }
    span.chunkPrefix = prefix;
    span.totalChunks = totalChunks;
    return span;
}

template <FwdHGateMode GATE_MODE>
__aicore__ inline uint32_t FwdHHeadRoundsPerSequence(const FwdHRuntimeTiling &tiling)
{
    if constexpr (GATE_MODE == FwdHGateMode::KEY_GK) {
        return FwdHCeilDiv(static_cast<uint32_t>(tiling.vNumHead), FWD_H_AIC_HEAD_SLOTS);
    }

    const uint32_t hk = static_cast<uint32_t>(tiling.kNumHead);
    const uint32_t groupSize = static_cast<uint32_t>(tiling.vNumHead) / hk;
    if (groupSize >= FWD_H_AIC_HEAD_SLOTS) {
        return hk * FwdHCeilDiv(groupSize, FWD_H_AIC_HEAD_SLOTS);
    }
    const uint32_t keysPerRound = FWD_H_AIC_HEAD_SLOTS / groupSize;
    return FwdHCeilDiv(hk, keysPerRound);
}

template <FwdHGateMode GATE_MODE>
__aicore__ inline FwdHHeadRoundPlan FwdHBuildHeadRange(const FwdHRuntimeTiling &tiling,
    uint32_t hvBegin, uint32_t headCount)
{
    FwdHHeadRoundPlan plan{};
    const uint32_t hv = static_cast<uint32_t>(tiling.vNumHead);
    const uint32_t hk = static_cast<uint32_t>(tiling.kNumHead);
    const uint32_t groupSize = GATE_MODE == FwdHGateMode::SCALAR_G ? hv / hk : 1;
    const uint32_t remainingHeads = hv > hvBegin ? hv - hvBegin : 0;
    plan.activeHeadCount = headCount < remainingHeads ? headCount : remainingHeads;
    if (plan.activeHeadCount > FWD_H_AIC_HEAD_SLOTS) {
        plan.activeHeadCount = FWD_H_AIC_HEAD_SLOTS;
    }

    for (uint32_t roundHead = 0; roundHead < plan.activeHeadCount; ++roundHead) {
        FwdHHeadBinding &head = plan.heads[roundHead];
        head.roundHead = roundHead;
        head.hv = hvBegin + roundHead;
        head.kh = GATE_MODE == FwdHGateMode::SCALAR_G ? head.hv / groupSize : head.hv;
        head.aiv = roundHead & 1U;
        head.localSlot = roundHead >> 1U;

        uint32_t kgSlot = plan.requiredKhCount;
        for (uint32_t consumer = 0; consumer < roundHead; ++consumer) {
            if (plan.heads[consumer].kh == head.kh) {
                kgSlot = plan.heads[consumer].kgSlot;
                break;
            }
        }
        if (kgSlot == plan.requiredKhCount) {
            ++plan.requiredKhCount;
        }
        head.kgSlot = kgSlot;
    }
    return plan;
}

template <FwdHGateMode GATE_MODE>
__aicore__ inline FwdHHeadRoundPlan FwdHBuildHeadRound(const FwdHRuntimeTiling &tiling,
    uint32_t round)
{
    const uint32_t hv = static_cast<uint32_t>(tiling.vNumHead);
    const uint32_t hk = static_cast<uint32_t>(tiling.kNumHead);
    const uint32_t groupSize = GATE_MODE == FwdHGateMode::SCALAR_G ? hv / hk : 1;
    uint32_t hvBegin = round * FWD_H_AIC_HEAD_SLOTS;
    uint32_t activeHeadCount = 0;
    if constexpr (GATE_MODE == FwdHGateMode::SCALAR_G) {
        if (groupSize >= FWD_H_AIC_HEAD_SLOTS) {
            const uint32_t roundsPerKey = FwdHCeilDiv(groupSize, FWD_H_AIC_HEAD_SLOTS);
            const uint32_t kh = round / roundsPerKey;
            const uint32_t groupRound = round % roundsPerKey;
            hvBegin = kh * groupSize + groupRound * FWD_H_AIC_HEAD_SLOTS;
            const uint32_t groupRemain = groupSize - groupRound * FWD_H_AIC_HEAD_SLOTS;
            activeHeadCount = groupRemain < FWD_H_AIC_HEAD_SLOTS
                                  ? groupRemain
                                  : FWD_H_AIC_HEAD_SLOTS;
        } else {
            const uint32_t keysPerRound = FWD_H_AIC_HEAD_SLOTS / groupSize;
            const uint32_t khBegin = round * keysPerRound;
            const uint32_t activeKeys = hk - khBegin < keysPerRound ? hk - khBegin : keysPerRound;
            hvBegin = khBegin * groupSize;
            activeHeadCount = activeKeys * groupSize;
        }
    } else {
        activeHeadCount = hv > hvBegin ? hv - hvBegin : 0;
    }
    return FwdHBuildHeadRange<GATE_MODE>(tiling, hvBegin, activeHeadCount);
}

__aicore__ inline FwdHKgBinding FwdHBuildKgBinding(const FwdHHeadRoundPlan &plan,
                                                   uint32_t kgSlot)
{
    FwdHKgBinding binding{};
    binding.slot = static_cast<uint8_t>(kgSlot);
    bool found = false;
    for (uint32_t consumer = 0; consumer < plan.activeHeadCount; ++consumer) {
        const FwdHHeadBinding &head = plan.heads[consumer];
        if (head.kgSlot != kgSlot) {
            continue;
        }
        if (!found) {
            binding.kh = head.kh;
            binding.firstConsumer = static_cast<uint8_t>(consumer);
            found = true;
        }
        binding.lastConsumer = static_cast<uint8_t>(consumer);
    }
    return binding;
}

__aicore__ inline FwdHChunkSpan FwdHBuildChunk(const FwdHSequenceSpan &sequence, uint32_t chunk)
{
    FwdHChunkSpan span{};
    span.chunk = chunk;
    span.globalChunk = sequence.chunkPrefix + chunk;
    span.tokenBegin = sequence.tokenBegin + chunk * FWD_H_CHUNK;
    span.validTokens = sequence.length - chunk * FWD_H_CHUNK;
    if (span.validTokens > FWD_H_CHUNK) {
        span.validTokens = FWD_H_CHUNK;
    }
    span.first = chunk == 0;
    span.last = chunk + 1 == sequence.chunkCount;
    return span;
}

template <FwdHGateMode GATE_MODE>
__aicore__ inline FwdHWorkUnit FwdHResolveWorkUnit(const FwdHKernelArgs &args, uint32_t workUnitId)
{
    const uint32_t rounds = FwdHHeadRoundsPerSequence<GATE_MODE>(args.tiling);
    const uint32_t sequence = workUnitId / rounds;
    const uint32_t round = workUnitId % rounds;
    return {FwdHResolveSequence(args, sequence), FwdHBuildHeadRound<GATE_MODE>(args.tiling, round)};
}

template <FwdHGateMode GATE_MODE>
__aicore__ inline uint32_t FwdHTotalWorkUnits(const FwdHRuntimeTiling &tiling)
{
    const uint32_t sequenceCount = tiling.isVariedLen != 0
                                       ? static_cast<uint32_t>(tiling.tokenBatch)
                                       : static_cast<uint32_t>(tiling.shapeBatch);
    return sequenceCount * FwdHHeadRoundsPerSequence<GATE_MODE>(tiling);
}

__aicore__ inline uint32_t FwdHSequenceCount(const FwdHRuntimeTiling &tiling)
{
    return tiling.isVariedLen != 0 ? static_cast<uint32_t>(tiling.tokenBatch)
                                   : static_cast<uint32_t>(tiling.shapeBatch);
}

__aicore__ inline uint32_t FwdHTotalHeadTasks(const FwdHRuntimeTiling &tiling)
{
    return FwdHSequenceCount(tiling) * static_cast<uint32_t>(tiling.vNumHead);
}

__aicore__ inline FwdHCoreHeadRange FwdHResolveCoreHeadRange(const FwdHRuntimeTiling &tiling,
                                                              uint32_t coreIdx,
                                                              uint32_t coreNum)
{
    const uint32_t totalHeads = FwdHTotalHeadTasks(tiling);
    if (coreNum == 0 || coreIdx >= coreNum || totalHeads == 0) {
        return {};
    }
    const uint32_t headsPerCore = FwdHCeilDiv(totalHeads, coreNum);
    const uint32_t begin = coreIdx * headsPerCore;
    uint32_t end = begin + headsPerCore;
    if (end > totalHeads) {
        end = totalHeads;
    }
    return {begin, end};
}

__aicore__ inline uint64_t FwdHInputOffset(const FwdHRuntimeTiling &tiling,
                                           uint32_t physicalBatch, uint32_t head,
                                           uint32_t tokenBegin, uint32_t dim)
{
    return ((static_cast<uint64_t>(physicalBatch) * tiling.vNumHead + head) * tiling.seqlen +
            tokenBegin) * dim;
}

__aicore__ inline uint64_t FwdHKOffset(const FwdHRuntimeTiling &tiling,
                                       uint32_t physicalBatch, uint32_t kh,
                                       uint32_t tokenBegin)
{
    return ((static_cast<uint64_t>(physicalBatch) * tiling.kNumHead + kh) * tiling.seqlen +
            tokenBegin) * FWD_H_K;
}

__aicore__ inline uint64_t FwdHHOffset(const FwdHRuntimeTiling &tiling,
                                       const FwdHSequenceSpan &sequence, uint32_t hv,
                                       uint32_t globalChunk)
{
    if (tiling.isVariedLen != 0) {
        return (static_cast<uint64_t>(hv) * sequence.totalChunks +
                globalChunk) * FWD_H_K * FWD_H_V;
    }
    const uint32_t chunksPerSequence = FwdHCeilDiv(static_cast<uint32_t>(tiling.seqlen), FWD_H_CHUNK);
    return ((static_cast<uint64_t>(sequence.physicalBatch) * tiling.vNumHead + hv) *
                chunksPerSequence + globalChunk - sequence.chunkPrefix) * FWD_H_K * FWD_H_V;
}

template <bool STATE_V_FIRST>
__aicore__ inline uint64_t FwdHStateOffset(const FwdHRuntimeTiling &tiling,
                                           uint32_t sequence, uint32_t hv, uint32_t k, uint32_t v)
{
    const uint64_t base = (static_cast<uint64_t>(sequence) * tiling.vNumHead + hv) * FWD_H_K * FWD_H_V;
    if constexpr (STATE_V_FIRST) {
        return base + static_cast<uint64_t>(v) * FWD_H_K + k;
    }
    return base + static_cast<uint64_t>(k) * FWD_H_V + v;
}

__aicore__ inline uint64_t FwdHCoreSlotOffset(uint32_t core, uint32_t slot, uint32_t slotElements)
{
    return (static_cast<uint64_t>(core) * FWD_H_AIC_HEAD_SLOTS + slot) * slotElements;
}

} // namespace GDN

#endif // CHUNK_FWD_H_UTILS_H
