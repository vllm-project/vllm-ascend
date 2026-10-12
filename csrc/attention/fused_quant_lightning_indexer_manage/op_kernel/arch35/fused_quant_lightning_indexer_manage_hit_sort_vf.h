#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_HIT_SORT_VF_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_HIT_SORT_VF_H

#include "kernel_operator.h"
#include "lim_payload_layout.h"

namespace fused_quant_li_manage_hit_sort_vf {
using namespace AscendC;

constexpr uint32_t LANES = 64U;
constexpr uint32_t SOURCE_BITS = lim_payload_layout::SOURCE_BITS;
constexpr uint32_t SOURCE_MASK = lim_payload_layout::SOURCE_MASK;
constexpr uint32_t LONG_SOURCE_MASK = lim_payload_layout::LONG_SOURCE_MASK;
constexpr uint32_t LONG_HIT_BASE = 0x3f800000U + LONG_SOURCE_MASK;
constexpr uint32_t LONG_MISS_BASE = 0x40000000U + LONG_SOURCE_MASK;

__simd_vf__ inline void DecodeDenseLong(
    __ubuf__ uint32_t *sources, __ubuf__ uint32_t *packed,
    __ubuf__ uint16_t *tags, uint32_t loops)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> sourceMask;
    MicroAPI::Duplicate(sourceMask, SOURCE_MASK);
    for (uint32_t index = 0U; index < loops; ++index) {
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::RegTensor<uint16_t> tag;
        MicroAPI::RegTensor<uint32_t> low;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            payload, packed + index * LANES);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            tag, tags + index * LANES);
        MicroAPI::And(low, payload, sourceMask, all);
        MicroAPI::ShiftLefts((MicroAPI::RegTensor<uint32_t>&)tag,
            (MicroAPI::RegTensor<uint32_t>&)tag,
            static_cast<int16_t>(SOURCE_BITS), all);
        MicroAPI::Add(low, low, (MicroAPI::RegTensor<uint32_t>&)tag, all);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            sources + index * LANES, low, all);
    }
}

__simd_vf__ inline void PackClassifiedLong(
    __ubuf__ uint32_t *keys, __ubuf__ uint32_t *payloads,
    __ubuf__ uint32_t *packed, __ubuf__ uint16_t *tags,
    uint32_t loops)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> sourceMask;
    MicroAPI::RegTensor<uint32_t> missSlot;
    MicroAPI::RegTensor<uint32_t> hitBase;
    MicroAPI::RegTensor<uint32_t> missBase;
    MicroAPI::Duplicate(sourceMask, SOURCE_MASK);
    MicroAPI::Duplicate(missSlot, lim_payload_layout::MISS_SLOT);
    MicroAPI::Duplicate(hitBase, LONG_HIT_BASE);
    MicroAPI::Duplicate(missBase, LONG_MISS_BASE);
    for (uint32_t index = 0U; index < loops; ++index) {
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::RegTensor<uint16_t> tag;
        MicroAPI::RegTensor<uint32_t> source;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> key;
        MicroAPI::RegTensor<uint32_t> missKey;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            payload, packed + index * LANES);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            tag, tags + index * LANES);
        MicroAPI::And(source, payload, sourceMask, all);
        MicroAPI::ShiftLefts((MicroAPI::RegTensor<uint32_t>&)tag,
            (MicroAPI::RegTensor<uint32_t>&)tag,
            static_cast<int16_t>(SOURCE_BITS), all);
        MicroAPI::Add(source, source, (MicroAPI::RegTensor<uint32_t>&)tag, all);
        MicroAPI::ShiftRights(slot, payload, static_cast<int16_t>(SOURCE_BITS), all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(isMiss, slot, missSlot, all);
        MicroAPI::Sub(key, hitBase, source, all);
        MicroAPI::Sub(missKey, missBase, source, all);
        MicroAPI::Select(key, missKey, key, isMiss);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            keys + index * LANES, key, all);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            payloads + index * LANES, payload, all);
    }
}

__simd_vf__ inline void UnpackClassifiedLong(
    __ubuf__ uint32_t *sources, __ubuf__ uint32_t *slots,
    __ubuf__ uint32_t *sortedPairs, uint32_t loops)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> one;
    MicroAPI::RegTensor<uint32_t> missSlot;
    MicroAPI::RegTensor<uint32_t> invalid;
    MicroAPI::RegTensor<uint32_t> hitBase;
    MicroAPI::RegTensor<uint32_t> missBase;
    MicroAPI::Duplicate(one, 1U);
    MicroAPI::Duplicate(missSlot, lim_payload_layout::MISS_SLOT);
    MicroAPI::Duplicate(invalid, 0xffffffffU);
    MicroAPI::Duplicate(hitBase, LONG_HIT_BASE);
    MicroAPI::Duplicate(missBase, LONG_MISS_BASE);
    for (uint32_t index = 0U; index < loops; ++index) {
        MicroAPI::RegTensor<int32_t> positions;
        MicroAPI::Arange(positions, static_cast<int32_t>(index * LANES));
        auto &offsets = (MicroAPI::RegTensor<uint32_t>&)positions;
        MicroAPI::ShiftLefts(offsets, offsets, static_cast<int16_t>(1), all);
        MicroAPI::RegTensor<uint32_t> key;
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::Gather(key, sortedPairs, offsets, all);
        MicroAPI::Add(offsets, offsets, one, all);
        MicroAPI::Gather(payload, sortedPairs, offsets, all);
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> source;
        MicroAPI::RegTensor<uint32_t> missSource;
        MicroAPI::ShiftRights(slot, payload, static_cast<int16_t>(SOURCE_BITS), all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(isMiss, slot, missSlot, all);
        MicroAPI::Sub(source, hitBase, key, all);
        MicroAPI::Sub(missSource, missBase, key, all);
        MicroAPI::Select(source, missSource, source, isMiss);
        MicroAPI::Select(slot, invalid, slot, isMiss);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            sources + index * LANES, source, all);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            slots + index * LANES, slot, all);
    }
}

// Sort a complete TopK row in one pass. The miss key range is above the
// hit range; both ranges order by ascending source ID. The original packed
// payload retains the destination slot through the hardware sort.
__simd_vf__ inline void PackClassified(
    __ubuf__ uint32_t *keys, __ubuf__ uint32_t *payloads,
    __ubuf__ uint32_t *packed, uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> sourceMask;
    MicroAPI::RegTensor<uint32_t> missSlot;
    MicroAPI::RegTensor<uint32_t> hitBase;
    MicroAPI::RegTensor<uint32_t> missBase;
    MicroAPI::Duplicate(sourceMask, SOURCE_MASK);
    MicroAPI::Duplicate(missSlot, lim_payload_layout::MISS_SLOT);
    MicroAPI::Duplicate(hitBase, 0x3f800000U + SOURCE_MASK);
    MicroAPI::Duplicate(missBase, 0x40000000U + SOURCE_MASK);
    for (uint32_t index = 0U; index < loops; ++index) {
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::RegTensor<uint32_t> source;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> key;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            payload, packed + index * LANES);
        MicroAPI::And(source, payload, sourceMask, all);
        MicroAPI::ShiftRights(slot, payload,
            static_cast<int16_t>(SOURCE_BITS), all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(
            isMiss, slot, missSlot, all);
        MicroAPI::Sub(key, hitBase, source, all);
        MicroAPI::RegTensor<uint32_t> missKey;
        MicroAPI::Sub(missKey, missBase, source, all);
        MicroAPI::Select(key, missKey, key, isMiss);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            keys + index * LANES, key, all);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            payloads + index * LANES, payload, all);
    }
}

__simd_vf__ inline void UnpackClassified(
    __ubuf__ uint32_t *sources, __ubuf__ uint32_t *slots,
    __ubuf__ uint32_t *sortedPairs, uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> one;
    MicroAPI::RegTensor<uint32_t> sourceMask;
    MicroAPI::RegTensor<uint32_t> missSlot;
    MicroAPI::RegTensor<uint32_t> invalid;
    MicroAPI::Duplicate(one, 1U);
    MicroAPI::Duplicate(sourceMask, SOURCE_MASK);
    MicroAPI::Duplicate(missSlot, lim_payload_layout::MISS_SLOT);
    MicroAPI::Duplicate(invalid, 0xffffffffU);
    for (uint32_t index = 0U; index < loops; ++index) {
        MicroAPI::RegTensor<int32_t> positions;
        MicroAPI::Arange(positions, static_cast<int32_t>(index * LANES));
        auto &offsets = (MicroAPI::RegTensor<uint32_t>&)positions;
        MicroAPI::ShiftLefts(offsets, offsets, static_cast<int16_t>(1), all);
        MicroAPI::Add(offsets, offsets, one, all);
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::RegTensor<uint32_t> source;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::Gather(payload, sortedPairs, offsets, all);
        MicroAPI::And(source, payload, sourceMask, all);
        MicroAPI::ShiftRights(slot, payload,
            static_cast<int16_t>(SOURCE_BITS), all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(
            isMiss, slot, missSlot, all);
        MicroAPI::Select(slot, invalid, slot, isMiss);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            sources + index * LANES, source, all);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            slots + index * LANES, slot, all);
    }
}

// The first-fill publisher has an explicit miss-prefix length after slot repair.
// Encode that boundary in the key, then sort the entire 2048-element row.
__simd_vf__ inline void PackRow(
    __ubuf__ uint32_t *keys, __ubuf__ uint32_t *payloads,
    __ubuf__ uint32_t *sources, __ubuf__ uint32_t *slots,
    uint32_t missCount, uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> missLimit;
    MicroAPI::RegTensor<uint32_t> hitBase;
    MicroAPI::RegTensor<uint32_t> missBase;
    MicroAPI::Duplicate(missLimit, missCount);
    MicroAPI::Duplicate(hitBase, LONG_HIT_BASE);
    MicroAPI::Duplicate(missBase, LONG_MISS_BASE);
    for (uint32_t index = 0U; index < loops; ++index) {
        MicroAPI::RegTensor<int32_t> positions;
        MicroAPI::Arange(positions, static_cast<int32_t>(index * LANES));
        MicroAPI::RegTensor<uint32_t> source;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> key;
        MicroAPI::RegTensor<uint32_t> missKey;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            source, sources + index * LANES);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            slot, slots + index * LANES);
        MicroAPI::Sub(key, hitBase, source, all);
        MicroAPI::Sub(missKey, missBase, source, all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            isMiss, (MicroAPI::RegTensor<uint32_t>&)positions,
            missLimit, all);
        MicroAPI::Select(key, missKey, key, isMiss);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            keys + index * LANES, key, all);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            payloads + index * LANES, slot, all);
    }
}

// These sort keys retain all 21 source bits. The payload carries only
// the destination slot, so full source IDs need no auxiliary tag here.
__simd_vf__ inline void Pack(
    __ubuf__ uint32_t *keys, __ubuf__ uint32_t *payloads,
    __ubuf__ uint32_t *sources, __ubuf__ uint32_t *slots,
    uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> keyBase;
    MicroAPI::Duplicate(keyBase, LONG_MISS_BASE);
    for (uint32_t index = 0U; index < loops; ++index) {
        MicroAPI::RegTensor<uint32_t> source;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> key;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            source, sources + index * LANES);
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            slot, slots + index * LANES);
        MicroAPI::Sub(key, keyBase, source, all);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            keys + index * LANES, key, all);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            payloads + index * LANES, slot, all);
    }
}

__simd_vf__ inline void Unpack(
    __ubuf__ uint32_t *sources, __ubuf__ uint32_t *slots,
    __ubuf__ uint32_t *sortedPairs, uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> one;
    MicroAPI::RegTensor<uint32_t> hitBase;
    MicroAPI::RegTensor<uint32_t> missBase;
    MicroAPI::RegTensor<uint32_t> missBoundary;
    MicroAPI::Duplicate(one, 1U);
    MicroAPI::Duplicate(hitBase, LONG_HIT_BASE);
    MicroAPI::Duplicate(missBase, LONG_MISS_BASE);
    MicroAPI::Duplicate(missBoundary, 0x40000000U);
    for (uint32_t index = 0U; index < loops; ++index) {
        MicroAPI::RegTensor<int32_t> positions;
        MicroAPI::Arange(positions, static_cast<int32_t>(index * LANES));
        auto &offsets = (MicroAPI::RegTensor<uint32_t>&)positions;
        MicroAPI::ShiftLefts(offsets, offsets, static_cast<int16_t>(1), all);
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::RegTensor<uint32_t> source;
        MicroAPI::RegTensor<uint32_t> missSource;
        MicroAPI::RegTensor<uint32_t> key;
        MicroAPI::Gather(key, sortedPairs, offsets, all);
        MicroAPI::Add(offsets, offsets, one, all);
        MicroAPI::Gather(payload, sortedPairs, offsets, all);
        MicroAPI::Sub(source, hitBase, key, all);
        MicroAPI::Sub(missSource, missBase, key, all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(isMiss, key, missBoundary, all);
        MicroAPI::Select(source, missSource, source, isMiss);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            sources + index * LANES, source, all);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            slots + index * LANES, payload, all);
    }
}

} // namespace fused_quant_li_manage_hit_sort_vf

#endif // FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_HIT_SORT_VF_H
