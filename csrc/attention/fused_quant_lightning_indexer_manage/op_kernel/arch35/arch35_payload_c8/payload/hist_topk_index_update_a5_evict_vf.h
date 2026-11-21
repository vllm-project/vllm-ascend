/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef VLLM_FAST_LIGHTNING_INDEXER_PAYLOAD_EVICT_VF_H
#define VLLM_FAST_LIGHTNING_INDEXER_PAYLOAD_EVICT_VF_H

#include "../../lim_payload_layout.h"

#include "kernel_operator.h"

namespace VllmFastLightningIndexerPayloadEvictVF {

constexpr uint32_t VF_B16_LANES = 128;

__simd_vf__ void CompactScores(__ubuf__ uint16_t *dst,
                               __ubuf__ uint16_t *src,
                               __ubuf__ int16_t *slots,
                               uint32_t vecLoopNum)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<int16_t> signedZero;
    MicroAPI::RegTensor<uint16_t> allOnes;
    MicroAPI::Duplicate(signedZero, static_cast<int16_t>(0));
    MicroAPI::Duplicate(allOnes, static_cast<uint16_t>(0xffff));
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t i = 0; i < vecLoopNum; ++i) {
        MicroAPI::RegTensor<uint16_t> key;
        MicroAPI::RegTensor<int16_t> slot;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(
            key, src + i * VF_B16_LANES);
        MicroAPI::LoadAlign<int16_t, MicroAPI::LoadDist::DIST_NORM>(
            slot, slots + i * VF_B16_LANES);
        MicroAPI::MaskReg ownsSlot;
        MicroAPI::Compare<int16_t, CMPMODE::GE>(ownsSlot, slot, signedZero, all);
        MicroAPI::RegTensor<uint16_t> reversed;
        MicroAPI::RegTensor<uint16_t> compact;
        MicroAPI::Xor(reversed, key, allOnes, all);
        MicroAPI::Squeeze<uint16_t, MicroAPI::GatherMaskMode::STORE_REG>(
            compact, reversed, ownsSlot);
        MicroAPI::StoreUnAlign<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
}

__simd_vf__ void CompactPayloads(__ubuf__ uint32_t *dst,
                                 __ubuf__ uint16_t *slots,
                                 uint32_t tokenBase,
                                 uint32_t vecLoopNum)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> invalidBit;
    MicroAPI::Duplicate(invalidBit, static_cast<uint32_t>(0x8000));
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t i = 0; i < vecLoopNum; ++i) {
        MicroAPI::RegTensor<uint16_t> slotPacked;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            slotPacked, slots + i * 64);
        MicroAPI::MaskReg ownsSlot;
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            ownsSlot, (MicroAPI::RegTensor<uint32_t>&)slotPacked, invalidBit, all);

        MicroAPI::RegTensor<int32_t> token;
        MicroAPI::Arange(token, static_cast<int32_t>(tokenBase + i * 64));
        MicroAPI::RegTensor<uint32_t> encodedSlot;
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::ShiftLefts(encodedSlot,
                             (MicroAPI::RegTensor<uint32_t>&)slotPacked,
                             static_cast<int16_t>(lim_payload_layout::SOURCE_BITS), all);
        MicroAPI::Add(payload, encodedSlot,
                      (MicroAPI::RegTensor<uint32_t>&)token, all);

        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
            compact, payload, ownsSlot);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
}

// Convert sortable score keys into bottom-K eviction keys. Only cached tokens
// strictly below the protected TopK boundary remain eligible.
__simd_vf__ void BuildEvictKeys(__ubuf__ uint16_t *scoreKeys,
                                __ubuf__ int16_t *slots,
                                uint16_t kthValue,
                                uint32_t vecLoopNum)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint16_t> kth;
    MicroAPI::RegTensor<uint16_t> zero;
    MicroAPI::RegTensor<uint16_t> allOnes;
    MicroAPI::RegTensor<int16_t> signedZero;
    MicroAPI::Duplicate(kth, kthValue);
    MicroAPI::Duplicate(zero, static_cast<uint16_t>(0));
    MicroAPI::Duplicate(allOnes, static_cast<uint16_t>(0xffff));
    MicroAPI::Duplicate(signedZero, static_cast<int16_t>(0));

    for (uint32_t i = 0; i < vecLoopNum; ++i) {
        MicroAPI::RegTensor<uint16_t> key;
        MicroAPI::RegTensor<int16_t> slot;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_NORM>(
            key, scoreKeys + i * VF_B16_LANES);
        MicroAPI::LoadAlign<int16_t, MicroAPI::LoadDist::DIST_NORM>(
            slot, slots + i * VF_B16_LANES);
        MicroAPI::MaskReg belowBoundary;
        MicroAPI::MaskReg ownsSlot;
        MicroAPI::MaskReg eligible;
        MicroAPI::Compare<uint16_t, CMPMODE::LT>(belowBoundary, key, kth, all);
        MicroAPI::Compare<int16_t, CMPMODE::GE>(ownsSlot, slot, signedZero, all);
        MicroAPI::And(eligible, belowBoundary, ownsSlot, all);

        MicroAPI::RegTensor<uint16_t> reversed;
        MicroAPI::RegTensor<uint16_t> evictKey;
        MicroAPI::Xor(reversed, key, allOnes, all);
        MicroAPI::Select(evictKey, reversed, zero, eligible);
        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(
            scoreKeys + i * VF_B16_LANES, evictKey, all);
    }
}

// Stable compact of unique cached tokens below the protected TopK boundary.
// The payload layout is {slot15, token17}; streaming chunks preserves token
// uniqueness without a bitmap or a second random-write marking pass.
__simd_vf__ void CompactEligiblePayloads(__ubuf__ uint32_t *dst,
                                         __ubuf__ uint16_t *scoreKeys,
                                         __ubuf__ uint16_t *slots,
                                         uint16_t kthValue,
                                         uint32_t tokenBase,
                                         uint32_t vecLoopNum)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> kth;
    MicroAPI::RegTensor<uint32_t> invalidBit;
    MicroAPI::Duplicate(kth, static_cast<uint32_t>(kthValue));
    MicroAPI::Duplicate(invalidBit, static_cast<uint32_t>(0x8000));
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t i = 0; i < vecLoopNum; ++i) {
        MicroAPI::RegTensor<uint16_t> scorePacked;
        MicroAPI::RegTensor<uint16_t> slotPacked;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            scorePacked, scoreKeys + i * 64);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            slotPacked, slots + i * 64);
        MicroAPI::MaskReg belowBoundary;
        MicroAPI::MaskReg ownsSlot;
        MicroAPI::MaskReg eligible;
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            belowBoundary, (MicroAPI::RegTensor<uint32_t>&)scorePacked, kth, all);
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            ownsSlot, (MicroAPI::RegTensor<uint32_t>&)slotPacked, invalidBit, all);
        MicroAPI::And(eligible, belowBoundary, ownsSlot, all);

        MicroAPI::RegTensor<int32_t> token;
        MicroAPI::Arange(token, static_cast<int32_t>(tokenBase + i * 64));
        MicroAPI::RegTensor<uint32_t> encodedSlot;
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::ShiftLefts(encodedSlot,
                             (MicroAPI::RegTensor<uint32_t>&)slotPacked,
                             static_cast<int16_t>(lim_payload_layout::SOURCE_BITS), all);
        MicroAPI::Add(payload, encodedSlot,
                      (MicroAPI::RegTensor<uint32_t>&)token, all);
        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
            compact, payload, eligible);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
}

} // namespace VllmFastLightningIndexerPayloadEvictVF

#endif // VLLM_FAST_LIGHTNING_INDEXER_PAYLOAD_EVICT_VF_H
