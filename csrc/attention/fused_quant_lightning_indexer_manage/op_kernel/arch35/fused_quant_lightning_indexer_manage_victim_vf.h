#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_VICTIM_VF_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_VICTIM_VF_H

#include "kernel_operator.h"
#include "lim_payload_layout.h"

namespace FusedQuantLiManageVictimVF {

constexpr uint32_t B32_LANES = 64U;

// Compact cached tokens that are strictly below all four route thresholds.
// Threshold ties are conservatively excluded and handled by the exact scalar
// fallback in the request-owner code.
__simd_vf__ void CompactEligiblePayloads(
    __ubuf__ uint32_t *dst,
    __ubuf__ uint16_t *score0,
    __ubuf__ uint16_t *score1,
    __ubuf__ uint16_t *score2,
    __ubuf__ uint16_t *score3,
    __ubuf__ uint16_t *slots,
    uint16_t threshold0,
    uint16_t threshold1,
    uint16_t threshold2,
    uint16_t threshold3,
    uint16_t validSlotLimit,
    uint32_t tokenBase,
    uint32_t vecLoopNum)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> kth0;
    MicroAPI::RegTensor<uint32_t> kth1;
    MicroAPI::RegTensor<uint32_t> kth2;
    MicroAPI::RegTensor<uint32_t> kth3;
    MicroAPI::RegTensor<uint32_t> invalidSlot;
    MicroAPI::RegTensor<uint32_t> sourceMask;
    MicroAPI::Duplicate(kth0, static_cast<uint32_t>(threshold0));
    MicroAPI::Duplicate(kth1, static_cast<uint32_t>(threshold1));
    MicroAPI::Duplicate(kth2, static_cast<uint32_t>(threshold2));
    MicroAPI::Duplicate(kth3, static_cast<uint32_t>(threshold3));
    MicroAPI::Duplicate(
        invalidSlot, static_cast<uint32_t>(validSlotLimit));
    MicroAPI::Duplicate(sourceMask, lim_payload_layout::SOURCE_MASK);
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t loop = 0U; loop < vecLoopNum; ++loop) {
        const uint32_t offset = loop * B32_LANES;
        MicroAPI::RegTensor<uint16_t> score0Packed;
        MicroAPI::RegTensor<uint16_t> score1Packed;
        MicroAPI::RegTensor<uint16_t> score2Packed;
        MicroAPI::RegTensor<uint16_t> score3Packed;
        MicroAPI::RegTensor<uint16_t> slotPacked;
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            score0Packed, score0 + offset);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            score1Packed, score1 + offset);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            score2Packed, score2 + offset);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            score3Packed, score3 + offset);
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            slotPacked, slots + offset);

        MicroAPI::MaskReg eligible0;
        MicroAPI::MaskReg eligible1;
        MicroAPI::MaskReg eligible2;
        MicroAPI::MaskReg eligible3;
        MicroAPI::MaskReg ownsSlot;
        MicroAPI::MaskReg eligible;
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            eligible0, (MicroAPI::RegTensor<uint32_t>&)score0Packed, kth0, all);
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            eligible1, (MicroAPI::RegTensor<uint32_t>&)score1Packed, kth1, all);
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            eligible2, (MicroAPI::RegTensor<uint32_t>&)score2Packed, kth2, all);
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            eligible3, (MicroAPI::RegTensor<uint32_t>&)score3Packed, kth3, all);
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(
            ownsSlot, (MicroAPI::RegTensor<uint32_t>&)slotPacked,
            invalidSlot, all);
        MicroAPI::And(eligible, eligible0, eligible1, all);
        MicroAPI::And(eligible, eligible, eligible2, all);
        MicroAPI::And(eligible, eligible, eligible3, all);
        MicroAPI::And(eligible, eligible, ownsSlot, all);

        MicroAPI::RegTensor<int32_t> source;
        MicroAPI::RegTensor<uint32_t> encodedSlot;
        MicroAPI::RegTensor<uint32_t> payload;
        MicroAPI::Arange(source, static_cast<int32_t>(tokenBase + offset));
        MicroAPI::And((MicroAPI::RegTensor<uint32_t>&)source,
            (MicroAPI::RegTensor<uint32_t>&)source, sourceMask, all);
        MicroAPI::ShiftLefts(
            encodedSlot, (MicroAPI::RegTensor<uint32_t>&)slotPacked,
            static_cast<int16_t>(lim_payload_layout::SOURCE_BITS), all);
        MicroAPI::Add(
            payload, encodedSlot,
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

} // namespace FusedQuantLiManageVictimVF

#endif
