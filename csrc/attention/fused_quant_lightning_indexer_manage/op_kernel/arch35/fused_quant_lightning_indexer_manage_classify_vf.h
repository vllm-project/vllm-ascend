/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_CLASSIFY_VF_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_CLASSIFY_VF_H

#include "kernel_operator.h"
#include "lim_payload_layout.h"

namespace VllmFastTopkIndexerClassifyVF {
using namespace AscendC;

constexpr uint32_t CHUNK_SIZE = 64;
constexpr uint32_t INDEXER_TOKEN_BITS = lim_payload_layout::SOURCE_BITS;
constexpr uint32_t INDEXER_TOKEN_MASK = lim_payload_layout::SOURCE_MASK;
constexpr uint32_t INDEXER_MISS_SLOT = lim_payload_layout::MISS_SLOT;

__simd_vf__ void SqueezeIndexerMissTokenIds(__ubuf__ uint32_t *dst,
                                             __ubuf__ uint32_t *payload,
                                             uint32_t vecLoopNum)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> tokenMask;
    MicroAPI::RegTensor<uint32_t> missSlot;
    MicroAPI::Duplicate(tokenMask, INDEXER_TOKEN_MASK);
    MicroAPI::Duplicate(missSlot, INDEXER_MISS_SLOT);
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t i = 0; i < vecLoopNum; ++i) {
        MicroAPI::RegTensor<uint32_t> packed;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> token;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            packed, payload + i * CHUNK_SIZE);
        MicroAPI::ShiftRights(slot, packed, static_cast<int16_t>(INDEXER_TOKEN_BITS), all);
        MicroAPI::And(token, packed, tokenMask, all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(isMiss, slot, missSlot, all);
        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(compact, token, isMiss);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                          AscendC::MicroAPI::MemType::VEC_LOAD>();
}

__simd_vf__ void SqueezeIndexerHitTokenIds(__ubuf__ uint32_t *dst,
                                            __ubuf__ uint32_t *payload,
                                            uint32_t vecLoopNum)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> tokenMask;
    MicroAPI::RegTensor<uint32_t> missSlot;
    MicroAPI::Duplicate(tokenMask, INDEXER_TOKEN_MASK);
    MicroAPI::Duplicate(missSlot, INDEXER_MISS_SLOT);
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t i = 0; i < vecLoopNum; ++i) {
        MicroAPI::RegTensor<uint32_t> packed;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::RegTensor<uint32_t> token;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            packed, payload + i * CHUNK_SIZE);
        MicroAPI::ShiftRights(slot, packed, static_cast<int16_t>(INDEXER_TOKEN_BITS), all);
        MicroAPI::And(token, packed, tokenMask, all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::MaskReg isHit;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(isMiss, slot, missSlot, all);
        MicroAPI::Xor(isHit, isMiss, all, all);
        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(compact, token, isHit);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                          AscendC::MicroAPI::MemType::VEC_LOAD>();
}

__simd_vf__ void SqueezeIndexerHitSlots(__ubuf__ uint32_t *dst,
                                         __ubuf__ uint32_t *payload,
                                         uint32_t vecLoopNum)
{
    MicroAPI::MaskReg all = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> missSlot;
    MicroAPI::Duplicate(missSlot, INDEXER_MISS_SLOT);
    MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t i = 0; i < vecLoopNum; ++i) {
        MicroAPI::RegTensor<uint32_t> packed;
        MicroAPI::RegTensor<uint32_t> slot;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            packed, payload + i * CHUNK_SIZE);
        MicroAPI::ShiftRights(slot, packed, static_cast<int16_t>(INDEXER_TOKEN_BITS), all);
        MicroAPI::MaskReg isMiss;
        MicroAPI::MaskReg isHit;
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(isMiss, slot, missSlot, all);
        MicroAPI::Xor(isHit, isMiss, all, all);
        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(compact, slot, isHit);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
    MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                          AscendC::MicroAPI::MemType::VEC_LOAD>();
}

} // namespace VllmFastTopkIndexerClassifyVF

#endif
