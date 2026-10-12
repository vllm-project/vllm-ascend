#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_WIDE_VICTIM_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_WIDE_VICTIM_H

#include "kernel_operator.h"
#include "fused_quant_lightning_indexer_manage_workspace.h"
#include "lim_cache_budget.h"
#include "lim_payload_layout.h"

namespace fused_quant_li_manage_wide_victim {
using namespace AscendC;

constexpr uint32_t CHUNK = 512U;
constexpr uint32_t LANES32 = 64U;
constexpr uint32_t LANES16 = 128U;
constexpr uint32_t TOPK = lim_request_limits::TOPK;
constexpr uint32_t PAIR_WORDS = TOPK * 2U;
constexpr uint32_t PAIR_BUFFER_BYTES = 4U * TOPK * 2U * sizeof(float);
static_assert(lim_payload_layout::PACKED_SOURCE_CAPACITY % CHUNK == 0U,
              "victim chunks must not cross a packed source page");

// The whole protection table occupies pairInputBuf_ and the scan uses
// pairOutputBuf_. This still fits at the largest A3 cache budget.
static_assert(lim_cache_budget::MAX_CACHE_TOKENS * sizeof(uint16_t) <=
                  PAIR_BUFFER_BYTES,
              "wide victim protection table exceeds pairInputBuf_");

namespace vf {

// Keep the exact scalar validity test before narrowing to uint16. Otherwise
// an invalid 32-bit pool entry could wrap into a valid slot during Cast.
__simd_vf__ inline void ClampPoolSlots(
    __ubuf__ uint32_t *slots, uint32_t budget, uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> limit;
    MicroAPI::RegTensor<uint32_t> invalid;
    MicroAPI::Duplicate(limit, budget);
    MicroAPI::Duplicate(invalid, 0xffffU);
    for (uint32_t loop = 0U; loop < loops; ++loop) {
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            slot, slots + loop * LANES32);
        MicroAPI::MaskReg valid;
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(valid, slot, limit, all);
        MicroAPI::RegTensor<uint32_t> normalized;
        MicroAPI::Select(normalized, slot, invalid, valid);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            slots + loop * LANES32, normalized, all);
    }
}

// A zero flag means that every visited route is strictly below its TopK
// threshold. Process four score rows per pass so Q14 needs no Q-sized UB row.
__simd_vf__ inline void AccumulateUnsafeFlags(
    __ubuf__ uint32_t *flags, __ubuf__ uint16_t *score0,
    __ubuf__ uint16_t *score1, __ubuf__ uint16_t *score2,
    __ubuf__ uint16_t *score3, uint16_t threshold0,
    uint16_t threshold1, uint16_t threshold2, uint16_t threshold3,
    uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> kth0, kth1, kth2, kth3, zero, one;
    MicroAPI::Duplicate(kth0, static_cast<uint32_t>(threshold0));
    MicroAPI::Duplicate(kth1, static_cast<uint32_t>(threshold1));
    MicroAPI::Duplicate(kth2, static_cast<uint32_t>(threshold2));
    MicroAPI::Duplicate(kth3, static_cast<uint32_t>(threshold3));
    MicroAPI::Duplicate(zero, 0U);
    MicroAPI::Duplicate(one, 1U);
    for (uint32_t loop = 0U; loop < loops; ++loop) {
        const uint32_t offset = loop * LANES32;
        MicroAPI::RegTensor<uint16_t> s0, s1, s2, s3;
        MicroAPI::RegTensor<uint32_t> previous;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            s0, score0 + offset);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            s1, score1 + offset);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            s2, score2 + offset);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            s3, score3 + offset);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            previous, flags + offset);
        MicroAPI::MaskReg safe0, safe1, safe2, safe3, wasSafe, safe;
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            safe0, (MicroAPI::RegTensor<uint32_t>&)s0, kth0, all);
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            safe1, (MicroAPI::RegTensor<uint32_t>&)s1, kth1, all);
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            safe2, (MicroAPI::RegTensor<uint32_t>&)s2, kth2, all);
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            safe3, (MicroAPI::RegTensor<uint32_t>&)s3, kth3, all);
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(
            wasSafe, previous, zero, all);
        MicroAPI::And(safe, safe0, safe1, all);
        MicroAPI::And(safe, safe, safe2, all);
        MicroAPI::And(safe, safe, safe3, all);
        MicroAPI::And(safe, safe, wasSafe, all);
        MicroAPI::RegTensor<uint32_t> result;
        MicroAPI::Select(result, zero, one, safe);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            flags + offset, result, all);
    }
}

// uint16 Gather addresses the table by slot index. Invalid slots gather
// entry zero safely and are marked protected in the output.
__simd_vf__ inline void GatherProtectedFlags(
    __ubuf__ uint16_t *dst, __ubuf__ uint16_t *protectedSlots,
    __ubuf__ uint16_t *slots, uint16_t budget, uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint16_t> limit;
    MicroAPI::RegTensor<uint16_t> zero;
    MicroAPI::RegTensor<uint16_t> one;
    MicroAPI::Duplicate(limit, budget);
    MicroAPI::Duplicate(zero, static_cast<uint16_t>(0));
    MicroAPI::Duplicate(one, static_cast<uint16_t>(1));
    for (uint32_t loop = 0U; loop < loops; ++loop) {
        MicroAPI::RegTensor<uint16_t> slot;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(
            slot, slots + loop * LANES16);
        MicroAPI::MaskReg valid;
        MicroAPI::Compare<uint16_t, CMPMODE::LT>(valid, slot, limit, all);
        MicroAPI::RegTensor<uint16_t> safeSlot;
        MicroAPI::Select(safeSlot, slot, zero, valid);
        MicroAPI::RegTensor<uint16_t> protectedFlag;
        MicroAPI::Gather(protectedFlag, protectedSlots, safeSlot, all);
        MicroAPI::Select(protectedFlag, protectedFlag, one, valid);
        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(
            dst + loop * LANES16, protectedFlag, all);
    }
}

// Compact in increasing source order. Payloads carry the 17 low source bits;
// the caller restores the common high bits of this 512-source chunk.
__simd_vf__ inline void CompactUnprotected(
    __ubuf__ uint32_t *dst, __ubuf__ uint16_t *slots,
    __ubuf__ uint16_t *protectedFlags, uint32_t sourceBase,
    uint32_t budget, uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> limit;
    MicroAPI::RegTensor<uint32_t> zero;
    MicroAPI::RegTensor<uint32_t> sourceMask;
    MicroAPI::Duplicate(limit, budget);
    MicroAPI::Duplicate(zero, 0U);
    MicroAPI::Duplicate(sourceMask, lim_payload_layout::SOURCE_MASK);
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;
    for (uint32_t loop = 0U; loop < loops; ++loop) {
        const uint32_t offset = loop * LANES32;
        MicroAPI::RegTensor<uint16_t> packedSlot;
        MicroAPI::RegTensor<uint16_t> packedFlag;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            packedSlot, slots + offset);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            packedFlag, protectedFlags + offset);
        auto &slot = (MicroAPI::RegTensor<uint32_t>&)packedSlot;
        auto &flag = (MicroAPI::RegTensor<uint32_t>&)packedFlag;
        MicroAPI::MaskReg ownsSlot;
        MicroAPI::MaskReg unprotected;
        MicroAPI::MaskReg eligible;
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            ownsSlot, slot, limit, all);
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(
            unprotected, flag, zero, all);
        MicroAPI::And(eligible, ownsSlot, unprotected, all);

        MicroAPI::RegTensor<int32_t> source;
        MicroAPI::RegTensor<uint32_t> encodedSlot;
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::Arange(source, static_cast<int32_t>(sourceBase + offset));
        MicroAPI::And((MicroAPI::RegTensor<uint32_t>&)source,
                      (MicroAPI::RegTensor<uint32_t>&)source,
                      sourceMask, all);
        MicroAPI::ShiftLefts(
            encodedSlot, slot,
            static_cast<int16_t>(lim_payload_layout::SOURCE_BITS), all);
        MicroAPI::Add(payload, encodedSlot,
                      (MicroAPI::RegTensor<uint32_t>&)source, all);
        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
            compact, payload, eligible);
        MicroAPI::StoreUnAlign<uint32_t,
                               MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
}

__simd_vf__ inline void UnpackPayloads(
    __ubuf__ uint32_t *victimSources, __ubuf__ uint32_t *destinations,
    __ubuf__ uint32_t *payloads, uint32_t count,
    uint32_t sourcePage, uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> page;
    MicroAPI::RegTensor<uint32_t> sourceMask;
    MicroAPI::Duplicate(page, sourcePage);
    MicroAPI::Duplicate(sourceMask, lim_payload_layout::SOURCE_MASK);
    for (uint32_t loop = 0U; loop < loops; ++loop) {
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            payload, payloads + loop * LANES32);
        MicroAPI::RegTensor<uint32_t> source;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::And(source, payload, sourceMask, all);
        MicroAPI::Add(source, source, page, all);
        MicroAPI::ShiftRights(
            slot, payload,
            static_cast<int16_t>(lim_payload_layout::SOURCE_BITS), all);
        MicroAPI::RegTensor<int32_t> positions;
        MicroAPI::Arange(positions, static_cast<int32_t>(loop * LANES32));
        MicroAPI::MaskReg valid;
        MicroAPI::Compares<uint32_t, CMPMODE::LT>(
            valid, (MicroAPI::RegTensor<uint32_t>&)positions,
            count, all);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            victimSources + loop * LANES32, source, valid);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            destinations + loop * LANES32, slot, valid);
    }
}

} // namespace vf

template <HardEvent event>
__aicore__ inline void Sync(HardEvent value)
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(value));
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

// The common Q>4 case needs no TopK-slot table. A score strictly below every
// route threshold cannot be selected by any route, so compact those residents
// before paying for exact TopK protection. Keep the score rows at four lanes
// regardless of Q, as in the existing Q1-4 victim scan.
__aicore__ inline uint32_t CompactSafeVictims(
    GlobalTensor<int32_t> cacheSlotsGm,
    GlobalTensor<uint16_t> scoreWorkspaceGm,
    GlobalTensor<uint16_t> routeThresholdsGm,
    GlobalTensor<int32_t> routePairsGm,
    GlobalTensor<int32_t> unionDestinationsGm,
    uint32_t queryStart, uint32_t routeCount, uint64_t routeStorageBase,
    uint32_t scoreRowStride, uint32_t candidate, uint32_t poolRow,
    uint32_t poolStride, uint32_t budget, uint32_t required,
    uint32_t scanSeed, uint64_t missBase,
    LocalTensor<uint16_t> thresholdStorage, LocalTensor<uint8_t> scratch)
{
    constexpr uint32_t SCORE_LANES = 4U;
    constexpr uint32_t THRESHOLD_STRIDE = static_cast<uint32_t>(
        fused_quant_li_manage_workspace::THRESHOLD_STRIDE);
    LocalTensor<uint16_t> scores = scratch.ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> score1 = scores[CHUNK];
    LocalTensor<uint16_t> score2 = scores[CHUNK * 2U];
    LocalTensor<uint16_t> score3 = scores[CHUNK * 3U];
    LocalTensor<int32_t> unsafe32 =
        scratch[SCORE_LANES * CHUNK * sizeof(uint16_t)]
            .ReinterpretCast<int32_t>();
    LocalTensor<int32_t> slots32 =
        scratch[SCORE_LANES * CHUNK * sizeof(uint16_t) +
                CHUNK * sizeof(int32_t)].ReinterpretCast<int32_t>();
    LocalTensor<int16_t> slots16 =
        scratch[SCORE_LANES * CHUNK * sizeof(uint16_t) +
                CHUNK * 2U * sizeof(int32_t)].ReinterpretCast<int16_t>();
    LocalTensor<int16_t> unsafe16 =
        scratch[SCORE_LANES * CHUNK * sizeof(uint16_t) +
                CHUNK * (2U * sizeof(int32_t) + sizeof(int16_t))]
            .ReinterpretCast<int16_t>();
    LocalTensor<uint32_t> compact =
        scratch[SCORE_LANES * CHUNK * sizeof(uint16_t) +
                CHUNK * (2U * sizeof(int32_t) + 2U * sizeof(int16_t))]
            .ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> victimSources =
        scratch[SCORE_LANES * CHUNK * sizeof(uint16_t) +
                CHUNK * (3U * sizeof(int32_t) + 2U * sizeof(int16_t))]
            .ReinterpretCast<int32_t>();
    LocalTensor<int32_t> destinations =
        scratch[SCORE_LANES * CHUNK * sizeof(uint16_t) +
                CHUNK * (4U * sizeof(int32_t) + 2U * sizeof(int16_t))]
            .ReinterpretCast<int32_t>();
    static_assert(SCORE_LANES * CHUNK * sizeof(uint16_t) +
                      CHUNK * (5U * sizeof(int32_t) +
                               2U * sizeof(int16_t)) <= PAIR_BUFFER_BYTES,
                  "wide score victim scratch exceeds pairOutputBuf_");

    for (uint32_t route = 0U; route < routeCount; ++route) {
        DataCopyPad(
            thresholdStorage[route * THRESHOLD_STRIDE],
            routeThresholdsGm[(routeStorageBase + route) * THRESHOLD_STRIDE],
            {1, THRESHOLD_STRIDE * static_cast<uint32_t>(sizeof(uint16_t)),
             0, 0, 0}, {false, 0, 0, 0});
    }
    Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    uint16_t thresholds[14] = {0U};
    for (uint32_t route = 0U; route < routeCount; ++route)
        thresholds[route] = thresholdStorage.GetValue(
            route * THRESHOLD_STRIDE);

    const uint32_t chunks = (candidate + CHUNK - 1U) / CHUNK;
    const uint32_t firstChunk = scanSeed % chunks;
    const uint64_t cacheBase = static_cast<uint64_t>(poolRow) * poolStride;
    const uint64_t victimScratch =
        static_cast<uint64_t>(queryStart) * PAIR_WORDS;
    uint32_t written = 0U;
    for (uint32_t visit = 0U;
         visit < chunks && written < required; ++visit) {
        const uint32_t base = ((firstChunk + visit) % chunks) * CHUNK;
        const uint32_t length = candidate - base < CHUNK
            ? candidate - base : CHUNK;
        Duplicate(unsafe32, 0, length);
        PipeBarrier<PIPE_V>();
        for (uint32_t group = 0U; group < routeCount;
             group += SCORE_LANES) {
            uint16_t kth[SCORE_LANES] = {0xffffU, 0xffffU,
                                         0xffffU, 0xffffU};
            for (uint32_t lane = 0U; lane < SCORE_LANES; ++lane) {
                LocalTensor<uint16_t> score = scores[lane * CHUNK];
                const uint32_t route = group + lane;
                if (route < routeCount) {
                    kth[lane] = thresholds[route];
                    DataCopyPad(
                        score,
                        scoreWorkspaceGm[(routeStorageBase + route) *
                                             scoreRowStride + base],
                        {1, length * static_cast<uint32_t>(sizeof(uint16_t)),
                         0, 0, 0}, {false, 0, 0, 0});
                } else {
                    Duplicate(score, static_cast<uint16_t>(0), length);
                }
            }
            Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            PipeBarrier<PIPE_V>();
            asc_vf_call<vf::AccumulateUnsafeFlags>(
                (__ubuf__ uint32_t *)unsafe32.GetPhyAddr(),
                (__ubuf__ uint16_t *)scores.GetPhyAddr(),
                (__ubuf__ uint16_t *)score1.GetPhyAddr(),
                (__ubuf__ uint16_t *)score2.GetPhyAddr(),
                (__ubuf__ uint16_t *)score3.GetPhyAddr(),
                kth[0], kth[1], kth[2], kth[3], length / LANES32);
            PipeBarrier<PIPE_V>();
            // The next group reuses all four score rows for MTE2.
            Sync<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        }

        DataCopyPad(slots32, cacheSlotsGm[cacheBase + base],
                    {1, length * static_cast<uint32_t>(sizeof(int32_t)),
                     0, 0, 0}, {false, 0, 0, 0});
        Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        asc_vf_call<vf::ClampPoolSlots>(
            (__ubuf__ uint32_t *)slots32.GetPhyAddr(),
            budget, length / LANES32);
        PipeBarrier<PIPE_V>();
        Cast(slots16, slots32, RoundMode::CAST_NONE, length);
        Cast(unsafe16, unsafe32, RoundMode::CAST_NONE, length);
        PipeBarrier<PIPE_V>();
        asc_vf_call<vf::CompactUnprotected>(
            (__ubuf__ uint32_t *)compact.GetPhyAddr(),
            (__ubuf__ uint16_t *)slots16.GetPhyAddr(),
            (__ubuf__ uint16_t *)unsafe16.GetPhyAddr(), base, budget,
            length / LANES32);
        const uint32_t compactCount = static_cast<uint32_t>(
            GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        const uint32_t remaining = required - written;
        const uint32_t keep = compactCount < remaining
            ? compactCount : remaining;
        if (keep == 0U) continue;
        asc_vf_call<vf::UnpackPayloads>(
            (__ubuf__ uint32_t *)victimSources.GetPhyAddr(),
            (__ubuf__ uint32_t *)destinations.GetPhyAddr(),
            (__ubuf__ uint32_t *)compact.GetPhyAddr(), keep,
            base & ~lim_payload_layout::SOURCE_MASK,
            (keep + LANES32 - 1U) / LANES32);
        Sync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(routePairsGm[victimScratch + written], victimSources,
                    {1, static_cast<uint16_t>(keep * sizeof(int32_t)),
                     0, 0});
        DataCopyPad(unionDestinationsGm[missBase + written], destinations,
                    {1, static_cast<uint16_t>(keep * sizeof(int32_t)),
                     0, 0});
        Sync<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        written += keep;
    }
    Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    return written;
}

// Exact fallback for threshold ties or when conservative score screening does
// not find enough candidates. It also excludes the slots already selected by
// CompactSafeVictims, without modifying the pool until all victims are found.
__aicore__ inline uint32_t AppendExactVictims(
    GlobalTensor<int32_t> cacheSlotsGm,
    GlobalTensor<int32_t> topkSlotsGm,
    GlobalTensor<int32_t> routePairsGm,
    GlobalTensor<int32_t> unionDestinationsGm,
    uint32_t queryStart, uint32_t routeCount,
    uint32_t candidate, uint32_t poolRow, uint32_t poolStride,
    uint32_t budget, uint32_t required, uint64_t missBase,
    uint32_t written, uint32_t scanSeed,
    LocalTensor<uint16_t> protectedSlots,
    LocalTensor<uint8_t> scratch)
{
    if (written >= required) return written;
    // pairInputBuf_: budget * 2 bytes, at most 64768 of 65536.
    Duplicate(protectedSlots, static_cast<uint16_t>(0), budget);
    PipeBarrier<PIPE_V>();
    Sync<HardEvent::V_S>(HardEvent::V_S);

    // pairOutputBuf_ layout, all offsets 32-byte aligned.
    LocalTensor<int32_t> slots32 = scratch.ReinterpretCast<int32_t>();
    LocalTensor<int16_t> slots16 =
        scratch[CHUNK * sizeof(int32_t)].ReinterpretCast<int16_t>();
    LocalTensor<uint16_t> flags16 =
        scratch[CHUNK * (sizeof(int32_t) + sizeof(uint16_t))]
            .ReinterpretCast<uint16_t>();
    LocalTensor<uint32_t> compact =
        scratch[CHUNK * (sizeof(int32_t) + 2U * sizeof(uint16_t))]
            .ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> victimSources =
        scratch[CHUNK * (2U * sizeof(int32_t) +
                         2U * sizeof(uint16_t))]
            .ReinterpretCast<int32_t>();
    LocalTensor<int32_t> destinations =
        scratch[CHUNK * (3U * sizeof(int32_t) +
                         2U * sizeof(uint16_t))]
            .ReinterpretCast<int32_t>();
    LocalTensor<int32_t> topkSlots =
        scratch[CHUNK * (4U * sizeof(int32_t) +
                         2U * sizeof(uint16_t))]
            .ReinterpretCast<int32_t>();
    static_assert(CHUNK * (4U * sizeof(int32_t) +
                           2U * sizeof(uint16_t)) +
                      TOPK * sizeof(int32_t) <= PAIR_BUFFER_BYTES,
                  "wide victim scratch exceeds pairOutputBuf_");

    for (uint32_t index = 0U; index < written; ++index) {
        const int32_t slot = unionDestinationsGm.GetValue(missBase + index);
        if (slot >= 0 && static_cast<uint32_t>(slot) < budget)
            protectedSlots.SetValue(static_cast<uint32_t>(slot), 1U);
    }
    for (uint32_t route = 0U; route < routeCount; ++route) {
        const uint64_t topkBase =
            static_cast<uint64_t>(queryStart + route) * TOPK;
        DataCopyPad(topkSlots, topkSlotsGm[topkBase],
                    {1, TOPK * static_cast<uint32_t>(sizeof(int32_t)),
                     0, 0, 0},
                    {false, 0, 0, 0});
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        for (uint32_t index = 0U; index < TOPK; ++index) {
            const int32_t slot = topkSlots.GetValue(index);
            if (slot >= 0 && static_cast<uint32_t>(slot) < budget)
                protectedSlots.SetValue(static_cast<uint32_t>(slot), 1U);
        }
    }
    Sync<HardEvent::S_V>(HardEvent::S_V);

    const uint64_t cacheBase = static_cast<uint64_t>(poolRow) * poolStride;
    const uint64_t victimScratch =
        static_cast<uint64_t>(queryStart) * PAIR_WORDS;
    const uint32_t chunks = (candidate + CHUNK - 1U) / CHUNK;
    const uint32_t firstChunk = scanSeed % chunks;
    for (uint32_t visit = 0U;
         visit < chunks && written < required; ++visit) {
        const uint32_t base = ((firstChunk + visit) % chunks) * CHUNK;
        const uint32_t length = candidate - base < CHUNK
            ? candidate - base : CHUNK;
        // ValidateSparse requires a 128-aligned candidate, so each chunk
        // contains whole B16/B32 vector loops without padding.
        DataCopyPad(slots32, cacheSlotsGm[cacheBase + base],
                    {1, length * static_cast<uint32_t>(sizeof(int32_t)),
                     0, 0, 0},
                    {false, 0, 0, 0});
        Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        asc_vf_call<vf::ClampPoolSlots>(
            (__ubuf__ uint32_t *)slots32.GetPhyAddr(), budget,
            length / LANES32);
        PipeBarrier<PIPE_V>();
        Cast(slots16, slots32, RoundMode::CAST_NONE, length);
        PipeBarrier<PIPE_V>();
        asc_vf_call<vf::GatherProtectedFlags>(
            (__ubuf__ uint16_t *)flags16.GetPhyAddr(),
            (__ubuf__ uint16_t *)protectedSlots.GetPhyAddr(),
            (__ubuf__ uint16_t *)slots16.GetPhyAddr(),
            static_cast<uint16_t>(budget), length / LANES16);
        PipeBarrier<PIPE_V>();
        asc_vf_call<vf::CompactUnprotected>(
            (__ubuf__ uint32_t *)compact.GetPhyAddr(),
            (__ubuf__ uint16_t *)slots16.GetPhyAddr(),
            (__ubuf__ uint16_t *)flags16.GetPhyAddr(),
            base, budget, length / LANES32);
        const uint32_t compactCount = static_cast<uint32_t>(
            GetSpr<SpecialPurposeReg::AR>() / sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        const uint32_t remaining = required - written;
        const uint32_t keep = compactCount < remaining
            ? compactCount : remaining;
        if (keep == 0U)
            continue;
        asc_vf_call<vf::UnpackPayloads>(
            (__ubuf__ uint32_t *)victimSources.GetPhyAddr(),
            (__ubuf__ uint32_t *)destinations.GetPhyAddr(),
            (__ubuf__ uint32_t *)compact.GetPhyAddr(),
            keep, base & ~lim_payload_layout::SOURCE_MASK,
            (keep + LANES32 - 1U) / LANES32);
        Sync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(routePairsGm[victimScratch + written], victimSources,
                    {1, static_cast<uint16_t>(keep * sizeof(int32_t)),
                     0, 0});
        DataCopyPad(unionDestinationsGm[missBase + written], destinations,
                    {1, static_cast<uint16_t>(keep * sizeof(int32_t)),
                     0, 0});
        Sync<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        written += keep;
    }
    Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    return written;
}

// Only Q>4 calls this entry point. The exact pass is cold with respect to
// TopK protection and runs only when strict score screening is insufficient.
__aicore__ inline uint32_t FindVictims(
    GlobalTensor<int32_t> cacheSlotsGm,
    GlobalTensor<uint16_t> scoreWorkspaceGm,
    GlobalTensor<uint16_t> routeThresholdsGm,
    GlobalTensor<int32_t> topkSlotsGm,
    GlobalTensor<int32_t> routePairsGm,
    GlobalTensor<int32_t> unionDestinationsGm,
    uint32_t queryStart, uint32_t routeCount, uint64_t routeStorageBase,
    uint32_t scoreRowStride, uint32_t candidate, uint32_t poolRow,
    uint32_t poolStride, uint32_t budget, uint32_t required,
    uint32_t scanSeed, uint64_t missBase,
    LocalTensor<uint16_t> protectedSlots, LocalTensor<uint8_t> scratch)
{
    uint32_t written = CompactSafeVictims(
        cacheSlotsGm, scoreWorkspaceGm, routeThresholdsGm, routePairsGm,
        unionDestinationsGm, queryStart, routeCount, routeStorageBase,
        scoreRowStride, candidate, poolRow, poolStride, budget, required,
        scanSeed, missBase, protectedSlots, scratch);
    if (written < required) {
        written = AppendExactVictims(
            cacheSlotsGm, topkSlotsGm, routePairsGm, unionDestinationsGm,
            queryStart, routeCount, candidate, poolRow, poolStride, budget,
            required, missBase, written, scanSeed, protectedSlots, scratch);
    }
    return written;
}

} // namespace fused_quant_li_manage_wide_victim

#endif // FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_WIDE_VICTIM_H
